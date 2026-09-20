"""Immutable exploratory RL runner over a frozen Slice 1 reference run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import subprocess
import time
import uuid

import numpy as np
import polars as pl
import yaml

from .dataset import prepare_dataset, sha256
from .environment import StrategySelectorEnv
from .policies import evaluate, fixed_action, model_action, random_action
from .protocol import ACTION_IDS, POLICY_IDS, load_protocol
from .device import resolve_device
from trading_pipeline.tracking.telemetry import (
    TELEMETRY_SCHEMA_VERSION, utc_now, write_training_telemetry,
)


def _revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dirty():
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_hashes():
    paths = sorted(Path("src/trading_pipeline/rl").glob("*.py")) + [
        Path("src/trading_pipeline/portfolio/backtest.py")
    ]
    return {path.as_posix(): sha256(path) for path in paths}


def _run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]


def _versions():
    import gymnasium, stable_baselines3, torch
    return {
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "polars": pl.__version__,
        "numpy": np.__version__,
    }


def _determinism(seed: int, actual_device: str) -> dict:
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if actual_device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "remaining_nondeterminism": (
            "CUDA kernels and library versions may still vary across hardware/runtime builds"
            if actual_device == "cuda" else
            "CPU numerical results may vary across library builds and instruction sets"
        ),
    }


def _training_callback(run_id, seed, requested_device, actual_device, started, rows):
    from stable_baselines3.common.callbacks import BaseCallback

    class TelemetryCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.episode_reward = 0.0

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards")
            dones = self.locals.get("dones")
            if rewards is not None:
                self.episode_reward += float(np.asarray(rewards).reshape(-1)[0])
            if dones is not None and bool(np.asarray(dones).reshape(-1)[0]):
                rows.append({
                    "schema_version": TELEMETRY_SCHEMA_VERSION,
                    "run_id": run_id, "experiment_id": None,
                    "policy_id": "RL1_DQN_SELECTOR", "model_family": "DQN",
                    "seed": seed, "requested_device": requested_device,
                    "actual_device": actual_device, "phase": "training",
                    "step": self.num_timesteps, "epoch": None,
                    "metric_name": "rollout/episode_reward",
                    "metric_value": self.episode_reward,
                    "elapsed_seconds": time.perf_counter() - started,
                    "learning_rate": float(self.model.lr_schedule(1.0)),
                    "timestamp": utc_now(),
                })
                self.episode_reward = 0.0
            if self.num_timesteps % 250:
                return True
            values = dict(getattr(self.logger, "name_to_value", {}))
            values.setdefault("rollout/exploration_rate", float(self.model.exploration_rate))
            for name in ("train/loss", "rollout/exploration_rate", "rollout/ep_rew_mean"):
                value = values.get(name)
                if value is None:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value):
                    continue
                rows.append({
                    "schema_version": TELEMETRY_SCHEMA_VERSION,
                    "run_id": run_id, "experiment_id": None,
                    "policy_id": "RL1_DQN_SELECTOR", "model_family": "DQN",
                    "seed": seed, "requested_device": requested_device,
                    "actual_device": actual_device, "phase": "training",
                    "step": self.num_timesteps, "epoch": None,
                    "metric_name": name, "metric_value": value,
                    "elapsed_seconds": time.perf_counter() - started,
                    "learning_rate": float(self.model.lr_schedule(1.0)),
                    "timestamp": utc_now(),
                })
            return True

    return TelemetryCallback()


def _benchmark_dqn(DQN, train_data, evaluation_data, config, devices):
    """Short fixed-budget timing/parity diagnostic; never used for selection."""
    rows = []
    base = dict(config["dqn"])
    base.pop("total_timesteps")
    policy = base.pop("policy")
    base.pop("device", None)
    steps = min(512, int(config["dqn"]["total_timesteps"]))
    base["learning_starts"] = min(int(base.get("learning_starts", 0)), 64)
    base["buffer_size"] = max(512, min(int(base.get("buffer_size", 10000)), 2048))
    for device in devices:
        started = time.perf_counter()
        model = DQN(policy, StrategySelectorEnv(train_data, config["cost_bps_one_way"]),
                    seed=int(config["seeds"][0]), verbose=0, device=device, **base)
        model.learn(total_timesteps=steps, progress_bar=False)
        duration = time.perf_counter() - started
        _, _, metrics = evaluate(
            StrategySelectorEnv(evaluation_data, config["cost_bps_one_way"]),
            model_action(model), "RL1_DQN_SELECTOR", int(config["seeds"][0])
        )
        rows.append({
            "benchmark_kind": "short_dqn_diagnostic_not_model_selection",
            "requested_device": device, "actual_device": str(model.device),
            "steps": steps, "duration_seconds": duration,
            "steps_per_second": steps / duration,
            "evaluation_reward_sum": metrics["reward_sum"],
            "seed": int(config["seeds"][0]),
        })
    return pl.DataFrame(rows)


def run_pilot(config_path: str | Path, *, skip_dqn: bool = False) -> Path:
    config_path = Path(config_path).resolve()
    config = load_protocol(config_path)
    output = Path(config["output_root"]).resolve() / _run_id()
    run_id = output.name
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    (output / "models").mkdir()
    shutil.copy2(config_path, output / "config.yaml")
    source_features = Path(config["reference_features"]).resolve()
    if sha256(source_features) != config["expected_feature_sha256"]:
        raise ValueError("Feature hash failed before snapshot")
    snapshot = output / "inputs" / "features.parquet"
    shutil.copy2(source_features, snapshot)
    if sha256(snapshot) != config["expected_feature_sha256"]:
        raise ValueError("Feature snapshot hash differs from validated source")
    reference = Path(config["reference_run"]).resolve()
    for name in ("dataset_manifest.json", "split_manifest.json", "metadata.json", "selection.json"):
        shutil.copy2(reference / name, output / "inputs" / name)

    train_data = prepare_dataset(reference, snapshot, config["expected_feature_sha256"], config["train_split"])
    evaluation_data = prepare_dataset(reference, snapshot, config["expected_feature_sha256"], config["evaluation_split"])
    device = resolve_device(config["dqn"].get("device", "cpu"))
    requested_device, actual_device = device["requested_device"], device["actual_device"]
    all_actions, all_curves, all_metrics, training_rows = [], [], [], []
    trace_rows, summary_rows = [], []
    evaluation_env = lambda: StrategySelectorEnv(evaluation_data, config["cost_bps_one_way"])
    for index in range(6):
        policy_id = f"RL_B{index}_ALWAYS_E{index}"
        actions, curve, metrics = evaluate(evaluation_env(), fixed_action(index + 1), policy_id, 42)
        all_actions.append(actions); all_curves.append(curve); all_metrics.append(metrics)
    for seed in config["seeds"]:
        actions, curve, metrics = evaluate(
            evaluation_env(), random_action(seed), "RL0_RANDOM_SELECTOR", seed
        )
        all_actions.append(actions); all_curves.append(curve); all_metrics.append(metrics)

    if not skip_dqn:
        from stable_baselines3 import DQN
        params = dict(config["dqn"])
        timesteps, policy = params.pop("total_timesteps"), params.pop("policy")
        params["device"] = actual_device
        benchmark_devices = ["cpu"] + (["cuda"] if device["cuda_available"] else [])
        benchmark = _benchmark_dqn(DQN, train_data, evaluation_data, config, benchmark_devices)
        benchmark.write_parquet(output / "device_benchmark.parquet")
        for seed in config["seeds"]:
            import torch
            settings = _determinism(seed, actual_device)
            if actual_device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started_at, started = utc_now(), time.perf_counter()
            model = DQN(policy, StrategySelectorEnv(train_data, config["cost_bps_one_way"]),
                        seed=seed, verbose=0, **params)
            callback = _training_callback(
                run_id, seed, requested_device, actual_device, started, trace_rows
            )
            model.learn(total_timesteps=timesteps, progress_bar=False, callback=callback)
            duration, completed_at = time.perf_counter() - started, utc_now()
            model_path = output / "models" / f"RL1_DQN_SELECTOR_seed_{seed}"
            model.save(model_path)
            actions, curve, metrics = evaluate(
                evaluation_env(), model_action(model), "RL1_DQN_SELECTOR", seed
            )
            all_actions.append(actions); all_curves.append(curve); all_metrics.append(metrics)
            training_rows.append({
                "policy_id": "RL1_DQN_SELECTOR", "seed": seed,
                "total_timesteps": timesteps, "actual_device": str(model.device),
                "model_path": f"models/{model_path.name}.zip",
            })
            trace_rows.append({
                "schema_version": TELEMETRY_SCHEMA_VERSION, "run_id": run_id,
                "experiment_id": None, "policy_id": "RL1_DQN_SELECTOR",
                "model_family": "DQN", "seed": seed,
                "requested_device": requested_device, "actual_device": actual_device,
                "phase": "descriptive_evaluation", "step": timesteps, "epoch": None,
                "metric_name": "evaluation/reward_sum",
                "metric_value": float(metrics["reward_sum"]),
                "elapsed_seconds": duration, "learning_rate": float(params["learning_rate"]),
                "timestamp": completed_at,
            })
            summary_rows.append({
                "schema_version": TELEMETRY_SCHEMA_VERSION, "run_id": run_id,
                "experiment_id": None, "policy_id": "RL1_DQN_SELECTOR",
                "model_family": "DQN", "seed": seed,
                "requested_device": requested_device, "actual_device": actual_device,
                "fallback_reason": device["fallback_reason"], "duration_seconds": duration,
                "data_rows": len(train_data.signal_dates),
                "data_columns": int(train_data.base_observations.shape[1]),
                "iterations": timesteps, "epochs": None,
                "stopping_reason": "fixed_predeclared_timestep_budget_reached",
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated()) if actual_device == "cuda" else None
                ),
                "package_versions_json": json.dumps(_versions(), sort_keys=True),
                "cuda_versions_json": json.dumps(device, sort_keys=True),
                "determinism_json": json.dumps(settings, sort_keys=True),
                "started_at": started_at, "completed_at": completed_at,
            })
    else:
        pl.DataFrame(schema={
            "benchmark_kind": pl.String, "requested_device": pl.String,
            "actual_device": pl.String, "steps": pl.Int64,
            "duration_seconds": pl.Float64, "steps_per_second": pl.Float64,
            "evaluation_reward_sum": pl.Float64, "seed": pl.Int64,
        }).write_parquet(output / "device_benchmark.parquet")

    write_training_telemetry(output, trace_rows, summary_rows)

    actions = pl.concat(all_actions)
    curves = pl.concat(all_curves)
    metrics = pl.DataFrame(all_metrics)
    frequencies = actions.group_by("policy_id", "seed", "action_id").agg(
        pl.len().alias("count")
    ).with_columns((pl.col("count") / pl.col("count").sum().over("policy_id", "seed")).alias("frequency"))
    actions.write_parquet(output / "actions.parquet")
    actions.select(
        "policy_id", "seed", "signal_date", "execution_date", "action_id",
        "traded_value", "turnover", "cost"
    ).write_parquet(output / "trades.parquet")
    curves.write_parquet(output / "equity_curve.parquet")
    metrics.write_parquet(output / "metrics.parquet")
    frequencies.write_parquet(output / "action_frequencies.parquet")
    pl.DataFrame(training_rows, schema={
        "policy_id": pl.String, "seed": pl.Int64, "total_timesteps": pl.Int64,
        "actual_device": pl.String, "model_path": pl.String,
    }).write_parquet(output / "training_log.parquet")
    (output / "policy_registry.json").write_text(
        json.dumps({key: {"description": value} for key, value in POLICY_IDS.items()}, indent=2),
        encoding="utf-8",
    )
    (output / "environment_manifest.json").write_text(json.dumps({
        "action_mapping": ACTION_IDS,
        "observation_fields": StrategySelectorEnv(train_data).observation_fields,
        "reward": config["reward"], "execution": config["execution"],
        "cost_bps_one_way": config["cost_bps_one_way"],
        "terminated": "last complete weekly transition in selected split",
        "truncated": "never in the fixed historical episode",
    }, indent=2), encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps({
        "train": {"source_split": train_data.split, "start": str(train_data.signal_dates[0]),
                  "end": str(train_data.end_dates[-1]), "transitions": len(train_data.signal_dates)},
        "evaluation": {"source_split": evaluation_data.split, "start": str(evaluation_data.signal_dates[0]),
                       "end": str(evaluation_data.end_dates[-1]), "transitions": len(evaluation_data.signal_dates)},
        "boundary_rule": "episodes are constructed independently and cannot cross source split boundaries",
    }, indent=2), encoding="utf-8")
    inputs = sorted((output / "inputs").iterdir())
    (output / "input_manifest.json").write_text(json.dumps({
        "reference_run": str(reference),
        "files": [{"path": f"inputs/{path.name}", "sha256": sha256(path), "bytes": path.stat().st_size} for path in inputs],
    }, indent=2), encoding="utf-8")
    median = metrics.group_by("policy_id").agg(
        pl.col("sharpe").median().alias("median_sharpe"),
        pl.col("total_return").median().alias("median_total_return"),
        pl.col("seed").sort_by("sharpe").get(pl.len() // 2).alias("reported_median_seed"),
        pl.len().alias("seed_count"),
    ).sort("policy_id")
    median.write_parquet(output / "policy_summary.parquet")
    metadata = {
        "run_id": output.name, "status": "complete", "research_status": config["research_status"],
        "generated_at": datetime.now(timezone.utc).isoformat(), "code_revision": _revision(),
        "code_dirty": _dirty(), "source_sha256": _source_hashes(),
        "reference_run_id": reference.name, "versions": _versions(),
        "requested_device": requested_device, "actual_device": actual_device,
        "device_fallback_reason": device["fallback_reason"], "device_preflight": device,
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        "seeds": config["seeds"], "dqn_skipped": skip_dqn,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary_lines = [
        "# Exploratory Phase 2 RL pilot", "", f"Run: `{output.name}`.", "",
        "This generated evidence is exploratory and not confirmatory. The completed Slice 1 final test was not used to tune the frozen protocol.", "",
        "## Generated policy summary", "", "| Policy | Median seed | Seeds | Median total return | Median Sharpe |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in median.iter_rows(named=True):
        summary_lines.append(f"| {row['policy_id']} | {row['reported_median_seed']} | {row['seed_count']} | {row['median_total_return']:.6f} | {row['median_sharpe']:.6f} |")
    summary_lines += ["", "No superiority claim is made. A fresh data vintage and predeclared walk-forward protocol are required for confirmatory evidence.", ""]
    (output / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    required = ["config.yaml", "metadata.json", "input_manifest.json", "split_manifest.json",
                "environment_manifest.json", "policy_registry.json", "training_log.parquet",
                "training_trace.parquet", "training_summary.parquet", "device_benchmark.parquet",
                "actions.parquet", "trades.parquet", "equity_curve.parquet", "metrics.parquet",
                "action_frequencies.parquet", "policy_summary.parquet", "summary.md"]
    model_files = sorted((output / "models").glob("*.zip"))
    manifest_files = [output / name for name in required] + model_files
    (output / "artifact_manifest.json").write_text(json.dumps({
        "files": [{"relative_path": path.relative_to(output).as_posix(),
                   "sha256": sha256(path), "bytes": path.stat().st_size}
                  for path in manifest_files],
    }, indent=2), encoding="utf-8")
    required.append("artifact_manifest.json")
    checks = {
        **{f"file:{name}": (output / name).is_file() for name in required},
        "feature_snapshot_hash": sha256(snapshot) == config["expected_feature_sha256"],
        "t_plus_one": actions.filter(pl.col("execution_date") <= pl.col("signal_date")).is_empty(),
        "costs_nonnegative": actions.filter(pl.col("cost") < 0).is_empty(),
        "exposure_long_only_no_leverage": actions.filter(
            (pl.col("gross_exposure") < -1e-12) | (pl.col("gross_exposure") > 1 + 1e-12)
        ).is_empty(),
        "metrics_finite": metrics.select(pl.exclude("policy_id")).select(
            pl.all().is_finite().all()
        ).row(0) == tuple(True for _ in metrics.select(pl.exclude("policy_id")).columns),
        "actions_registered": set(actions["action_id"].unique()) <= set(ACTION_IDS.values()),
        "dqn_models_present": skip_dqn or len(model_files) == len(config["seeds"]),
        "telemetry_present": skip_dqn or len(summary_rows) == len(config["seeds"]),
    }
    audit = {"checks": checks}
    audit["passed"] = all(audit["checks"].values())
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError("RL artefact audit failed")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run the frozen exploratory Phase 2 selector pilot")
    parser.add_argument("--config", default="configs/rl_pilot.yaml")
    parser.add_argument("--skip-dqn", action="store_true", help="Produce fixed/random evidence without optional DQN training")
    args = parser.parse_args()
    print(run_pilot(args.config, skip_dqn=args.skip_dqn))


if __name__ == "__main__":
    main()
