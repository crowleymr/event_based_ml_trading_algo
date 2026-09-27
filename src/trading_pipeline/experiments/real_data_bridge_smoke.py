"""Score-blind engineering smoke for a metadata-declared prepared draft.

Only fit-partition targets are projected from the canonical snapshot.  The
evaluation partition is always feature-only; predictions and policy actions
are checked in memory and discarded.  This module has no study-run entry point.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from trading_pipeline.experiments import FitContext
from trading_pipeline.experiments.calibration import _contains_outcome
from trading_pipeline.experiments.default_registry import default_registry
from trading_pipeline.experiments.deep_study_adapter import fit_score_deep_study
from trading_pipeline.experiments.rl_episode_producer import (
    MODEL_SLEEVES, produce_selector_episode, selector_feature_frame,
)
from trading_pipeline.experiments.schema import load_study
from trading_pipeline.features import F1
from trading_pipeline.optimisation.search import random_proposals
from trading_pipeline.rl.environment import StrategySelectorEnv


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pinned(root: Path, relative: str, expected: str) -> Path:
    if (not isinstance(relative, str) or relative.startswith("pending")
            or not isinstance(expected, str) or len(expected) != 64):
        raise ValueError("Smoke input needs a resolved path and SHA-256")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or _sha(path) != expected:
        raise ValueError(f"Missing or changed smoke input: {relative}")
    return path


def validate_inputs(root: Path, study_path: Path):
    """Reject incomplete protocol before opening a feature or target parquet."""
    study = load_study(study_path, allow_engineering_draft=True)
    config = study.config
    if not study.engineering_only or config["search"].get("real_data_execution") == "enabled":
        raise PermissionError("Smoke requires a non-executable prepared draft")
    arms = config["experiment_arms"]
    registry = default_registry()
    if not arms or len({arm["id"] for arm in arms}) != len(arms):
        raise ValueError("Smoke requires unique metadata-declared arms")
    for arm in arms:
        spec = registry.spec(arm["component_id"])
        if spec.interface != arm.get("interface") or spec.interface not in {"SupervisedModel", "RLPolicy"}:
            raise ValueError("Smoke arm interface differs from trusted registry metadata")
        if not spec.capabilities.get("study_adapter"):
            raise ValueError("Smoke arm has no trusted study adapter")
    if any(arm.get("feature_set_id") != "F1_CAUSAL_STACK_V1" for arm in arms):
        raise ValueError("Smoke requires the common causal-stack view")
    pinned = config["authority"].get("input_sha256", {})
    for key in ("snapshot_manifest",):
        relative = config["data"].get(key)
        if relative not in pinned:
            raise ValueError(f"Unpinned {key}")
        _pinned(root, relative, pinned[relative])
    inner_name = config["validation"].get("inner_windows_manifest")
    if inner_name not in pinned:
        raise ValueError("Unpinned inner windows")
    inner_path = _pinned(root, inner_name, pinned[inner_name])
    snapshot = json.loads((root / config["data"]["snapshot_manifest"]).read_text(encoding="utf-8"))
    feature_path = _pinned(root, snapshot["feature_path"], snapshot["feature_sha256"])
    contract = config["data"].get("augmented_feature_contract")
    required = {"feature_path", "feature_sha256", "feature_columns",
                "source_manifest", "source_manifest_sha256"}
    if not isinstance(contract, dict) or not required <= contract.keys():
        raise ValueError("Missing augmented causal-stack contract")
    columns = contract["feature_columns"]
    if (not isinstance(columns, list) or len(columns) != len(set(columns))
            or not set(F1) < set(columns)
            or {"forward_return_5d", "label_end_date", "split"} & set(columns)):
        raise ValueError("Invalid causal-stack feature columns")
    augmented = _pinned(root, contract["feature_path"], contract["feature_sha256"])
    source = _pinned(root, contract["source_manifest"], contract["source_manifest_sha256"])
    lineage = json.loads(source.read_text(encoding="utf-8"))
    if (lineage.get("study_id") != config["study_id"]
            or lineage.get("feature_sha256") != snapshot["feature_sha256"]
            or lineage.get("table_sha256") != contract["feature_sha256"]):
        raise ValueError("Causal-stack lineage does not bind to the draft")
    rl = config["data"].get("rl_inputs")
    if not isinstance(rl, dict) or set(rl.get("model_sleeve_components", {})) != set(MODEL_SLEEVES):
        raise ValueError("Missing complete RL input contract")
    bars = _pinned(root, rl.get("bars_path"), rl.get("bars_sha256"))
    outputs = _pinned(root, rl.get("model_outputs_path"), rl.get("model_outputs_sha256"))
    if bars != feature_path or rl["bars_sha256"] != snapshot["feature_sha256"]:
        raise ValueError("RL bars must be the canonical snapshot")
    portfolio = config.get("portfolio", {})
    if (portfolio.get("e5_source") not in MODEL_SLEEVES
            or not isinstance(portfolio.get("volatility_lookback_sessions"), int)
            or not isinstance(portfolio.get("top_k"), int)
            or set(portfolio.get("risk_scenarios", {})) != {"conservative", "balanced", "aggressive"}):
        raise ValueError("Missing RL episode protocol fields")
    training = config.get("search", {}).get("deep_training", {})
    if not {"epochs", "patience"} <= training.keys():
        raise ValueError("Missing deep training protocol fields")
    folds = json.loads(inner_path.read_text(encoding="utf-8"))["folds"]
    if not folds:
        raise ValueError("No inner fold")
    return study, folds[0], feature_path, augmented, outputs, inner_path


def _sample(study, fold, feature_path: Path, augmented_path: Path, *, securities: int,
            fit_sessions: int = 80):
    if securities < 2:
        raise ValueError("At least two securities are required")
    config = study.config
    fit_dates = [date.fromisoformat(value) for value in fold["fit_dates"]]
    stop_dates = [date.fromisoformat(value) for value in fold["stopping_dates"]]
    score_dates = [date.fromisoformat(value) for value in fold["score_dates"]]
    if not max(fit_dates) < min(stop_dates) or not max(stop_dates) < min(score_dates):
        raise ValueError("Nonchronological smoke partitions")
    # A fixed tail retains sequence history and enough weekly RL transitions.
    lookback = int(config["portfolio"]["volatility_lookback_sessions"])
    if fit_sessions < 0:
        raise ValueError("fit_sessions must be zero or positive")
    if fit_sessions:
        fit_dates = fit_dates[-max(fit_sessions, lookback + 40):]
    wanted = [*fit_dates, *stop_dates, *score_dates]
    canonical = (pl.scan_parquet(feature_path)
                 .filter(pl.col("session_date").is_in(wanted))
                 .select("security_id", "ticker", "session_date", *F1).collect())
    candidates = (canonical.group_by("security_id").len()
                  .filter(pl.col("len") == len(wanted))
                  .sort("security_id").head(securities)
                  .get_column("security_id").to_list())
    if len(candidates) != securities:
        raise ValueError("Insufficient stable securities")
    canonical = canonical.filter(pl.col("security_id").is_in(candidates))
    extra = [name for name in config["data"]["augmented_feature_contract"]["feature_columns"] if name not in F1]
    augmented = (pl.scan_parquet(augmented_path)
                 .filter(pl.col("session_date").is_in(wanted) & pl.col("security_id").is_in(candidates))
                 .select("security_id", "session_date", *F1, *extra).collect())
    keys = ["security_id", "session_date"]
    if (augmented.height != canonical.height or augmented.select(keys).is_duplicated().any()
            or not augmented.select([*keys, *F1]).sort(keys).equals(
                canonical.select([*keys, *F1]).sort(keys))):
        raise ValueError("Smoke causal stack differs from canonical F1")
    joined = canonical.select([*keys, "ticker"]).join(augmented, on=keys, validate="1:1")
    if joined.height != len(wanted) * securities:
        raise ValueError("Incomplete deterministic smoke panel")
    # The target projection is restricted to fitting/stopping dates. Evaluation
    # targets and the sealed final-holdout partition are never opened.
    labels = (pl.scan_parquet(feature_path)
              .filter(pl.col("session_date").is_in([*fit_dates, *stop_dates])
                      & pl.col("security_id").is_in(candidates))
              .select(*keys, "forward_return_5d", "label_end_date").collect())
    labelled = joined.filter(pl.col("session_date").is_in([*fit_dates, *stop_dates])).join(
        labels, on=keys, validate="1:1")
    fit = labelled.filter(pl.col("session_date").is_in(fit_dates)).to_dicts()
    stop = labelled.filter(pl.col("session_date").is_in(stop_dates)).to_dicts()
    score = joined.filter(pl.col("session_date").is_in(score_dates)).to_dicts()
    pool = joined.to_dicts()
    return fit, stop, score, pool, candidates, wanted


def _parameters(registry, component: str, seed: int, device: str = "cpu"):
    if registry.spec(component).interface == "RLPolicy":
        model = registry.create(component, total_timesteps=1, device=device)
        space = model.search_space()["parameters"]
        for item in random_proposals(space, 100, seed):
            try:
                registry.create(component, parameters=item, total_timesteps=1, device=device)
                return item
            except (TypeError, ValueError):
                continue
        raise ValueError(f"No valid policy parameters: {component}")
    space = registry.create(component).search_space()["parameters"]
    return random_proposals(space, 1, seed)[0]


def _causal_rl_signal_dates(fit: list[dict], available: set[date]) -> tuple[date, ...]:
    """Apply the authoritative RL causal-output eligibility rule to smoke dates."""
    dates = tuple(day for day in sorted({row["session_date"] for row in fit})
                  if day in available)
    if len(dates) < 2:
        raise ValueError("RL smoke lacks two causal-output signal sessions")
    return dates


def _supervised(arm, config, registry, params, fit, stop, score, pool, device="cpu"):
    columns = tuple(config["data"]["augmented_feature_contract"]["feature_columns"])
    if registry.spec(arm["component_id"]).capabilities.get("sequence_view"):
        trained = fit_score_deep_study(
            component_id=arm["component_id"], feature_set_id="F1_CAUSAL_STACK_V1",
            parameters=params, seed=41, study_id=config["study_id"], trial_id="smoke",
            fold_id="inner_smoke", feature_pool=pool, fit_rows=fit,
            stopping_rows=stop, score_rows=[{**row, "split": "inner_smoke"} for row in score],
            feature_columns=columns, requested_device=device,
            epochs=int(config["search"]["deep_training"]["epochs"]),
            patience=int(config["search"]["deep_training"]["patience"]))
        values = trained.predictions
        device = trained.telemetry.get("actual_device", "cpu")
    else:
        def matrix(rows):
            return np.asarray([[np.nan if row.get(name) is None else float(row[name])
                                for name in columns] for row in rows], dtype=np.float64)
        x, y = matrix(fit), np.asarray([row["forward_return_5d"] for row in fit])
        if not np.isfinite(y).all():
            raise ValueError("Fit labels must be finite")
        spec = registry.spec(arm["component_id"])
        kwargs = {"params": params}
        if spec.capabilities.get("cuda"):
            kwargs["device"] = device
        model = registry.create(arm["component_id"], **kwargs)
        stopping = (matrix(stop), np.asarray([row["forward_return_5d"] for row in stop])) if model.spec.capabilities.get("stopping_data") else None
        model.fit(x, y, context=FitContext(config["study_id"], "smoke", "inner_smoke", 41, {}),
                  stopping_data=stopping)
        values = model.predict(matrix(score))
        device = model.telemetry().get("actual_device", "cpu")
    if len(values) != len(score) or not np.isfinite(values).all():
        raise ValueError("Smoke prediction shape or finiteness failed")
    return str(device), len(values)


def _rl(arm, config, params, fit, score, candidates, feature_path, outputs_path,
        feature_sha, output_sha, fold_sha, protocol_sha, device="cpu"):
    # Learning necessarily consumes the declared development reward internally,
    # but no reward, return, action ranking or performance metric is persisted.
    source = (pl.scan_parquet(outputs_path)
              .filter(pl.col("security_id").is_in(candidates))
              .collect())
    required = {"upstream_component_id", "security_id", "session_date", "prediction",
                "fit_cutoff_date", "fit_max_label_end_date", "prediction_role"}
    if not required <= set(source.columns):
        raise ValueError("Incomplete causal model-output schema")
    # Match the authoritative runner: RL signals begin only once every causal
    # sleeve output exists. Earlier canonical rows remain available solely as
    # point-in-time volatility history and must not become policy observations.
    available = set(source["session_date"].unique().to_list())
    dates = _causal_rl_signal_dates(fit, available)
    first = dates[0]
    lookback = config["portfolio"]["volatility_lookback_sessions"]
    historical = (pl.scan_parquet(feature_path)
                  .filter(pl.col("session_date") < first)
                  .select("session_date").unique().sort("session_date").tail(lookback)
                  .collect().get_column("session_date").to_list())
    if len(historical) != lookback:
        raise ValueError("Insufficient pre-signal calendar for RL volatility")
    calendar = tuple(sorted({*historical, *(row["session_date"] for row in [*fit, *score])}))
    bars = (pl.scan_parquet(feature_path)
            .filter(pl.col("security_id").is_in(candidates)
                    & pl.col("session_date").is_in(calendar))
            .select("security_id", "session_date", "adjusted_close").collect())
    source = source.filter(pl.col("session_date").is_in(dates))
    mapping = config["data"]["rl_inputs"]["model_sleeve_components"]
    source = source.with_columns(
        pl.col("upstream_component_id").replace({value: key for key, value in mapping.items()}).alias("model_id"),
        pl.col("prediction").alias("predicted_return_5d"))
    portfolio = config["portfolio"]
    scenario = portfolio["risk_scenarios"]["balanced"]
    episode = produce_selector_episode(
        split="engineering_smoke", signal_sessions=dates, calendar=calendar,
        features=selector_feature_frame(fit).filter(pl.col("session_date").is_in(dates)),
        model_outputs=source, bars=bars,
        source_hashes={"canonical_features": feature_sha, "model_outputs": output_sha,
                       "adjusted_close": feature_sha, "fold_manifest": fold_sha,
                       "protocol": protocol_sha},
        e5_source=portfolio["e5_source"], top_k=portfolio["top_k"],
        gross_exposure_cap=scenario["gross_exposure_cap"],
        max_position=scenario["max_position"],
        annualised_volatility_target=scenario["annualised_volatility_target"],
        volatility_lookback_sessions=portfolio["volatility_lookback_sessions"],
        allowed_prediction_roles=("inner_oof",),
        price_eligibility_policy=config["data"]["rl_inputs"].get(
            "price_eligibility_policy", "complete_daily_valuation_v1"))
    env = StrategySelectorEnv(episode.dataset, portfolio["cost_bps_one_way"])
    observation, _ = env.reset(seed=41)
    total_timesteps = int(config["search"]["rl_training"]["total_timesteps"])
    policy = default_registry().create(arm["component_id"], parameters=params,
                                       total_timesteps=total_timesteps, device=device)
    if not env.observation_space.contains(observation) or env.action_space.n != 7:
        raise ValueError("RL environment shape/action contract failed")
    policy._validate_discrete_environment(env)
    policy.learn(env, context=FitContext(config["study_id"], "smoke", "inner_smoke", 41,
                                         {"environment_steps": total_timesteps}))
    action = policy.act(observation, deterministic=True)
    if not env.action_space.contains(action):
        raise ValueError("Policy produced an invalid selector action")
    return str(policy.telemetry().get("actual_device", "unknown")), len(episode.transitions), len(observation)


def run_smoke(*, repository_root: str | Path, prepared_study: str | Path,
              output_dir: str | Path, records_path: str | Path,
              securities: int = 8, mapping_path: str | Path | None = None,
              device: str = "cpu", fit_sessions: int = 80,
              arm_ids: tuple[str, ...] | None = None) -> list[dict]:
    root = Path(repository_root).resolve()
    study_path = Path(prepared_study).resolve()
    study, fold, feature_path, augmented_path, outputs_path, inner_path = validate_inputs(root, study_path)
    config = study.config
    if device not in {"cpu", "cuda", "auto", "protocol"}:
        raise ValueError("device must be cpu, cuda, auto or protocol")
    requested_device = device
    fit, stop, score, pool, candidates, _ = _sample(study, fold, feature_path, augmented_path,
                                                     securities=securities,
                                                     fit_sessions=fit_sessions)
    output = Path(output_dir).resolve()
    records_file = Path(records_path).resolve()
    mapping_file = Path(mapping_path).resolve() if mapping_path is not None else None
    if output.exists() or records_file.exists() or (mapping_file is not None and mapping_file.exists()):
        raise FileExistsError("Smoke output is immutable; choose new paths")
    registry = default_registry()
    prepared = []
    declared = {arm["id"]: arm for arm in config["experiment_arms"]}
    selected_ids = tuple(declared) if arm_ids is None else arm_ids
    if (not selected_ids or len(set(selected_ids)) != len(selected_ids)
            or not set(selected_ids) <= set(declared)):
        raise ValueError("arm_ids must be a unique nonempty subset of declared study arms")
    for arm_id in selected_ids:
        arm = declared[arm_id]
        started = time.perf_counter()
        arm_device = (arm.get("execution_device",
                      config["reproducibility"].get("requested_device", "cpu"))
                      if requested_device == "protocol" else requested_device)
        params = _parameters(registry, arm["component_id"], 41, arm_device)
        if registry.spec(arm["component_id"]).interface == "RLPolicy":
            actual_device, evaluation_rows, width = _rl(
                arm, config, params, fit, score, candidates, feature_path, outputs_path,
                _sha(feature_path), _sha(outputs_path), _sha(inner_path), study.sha256,
                device=arm_device)
        else:
            actual_device, evaluation_rows = _supervised(
                arm, config, registry, params, fit, stop, score, pool,
                device=arm_device)
            width = len(config["data"]["augmented_feature_contract"]["feature_columns"])
        elapsed = max(time.perf_counter() - started, 1e-9)
        evidence = {"schema_version": 1, "study_id": config["study_id"],
                    "arm_id": arm["id"], "component_id": arm["component_id"],
                    "evidence_kind": "real_data_bridge_smoke", "status": "passed",
                    "score_blind": True, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "prepared_study_sha256": _sha(study_path),
                    "input_sha256": {"canonical_features": _sha(feature_path),
                                     "augmented_features": _sha(augmented_path),
                                     "model_outputs": _sha(outputs_path),
                                     "inner_windows": _sha(inner_path)},
                    "fit_rows": len(fit), "evaluation_rows": evaluation_rows,
                    "feature_width": width, "securities": len(candidates),
                    "requested_device": arm_device,
                    "device": actual_device, "wall_seconds": elapsed,
                    "bridge_scope": ("resource_timed_policy_learning_without_outcome_persistence"
                                     if registry.spec(arm["component_id"]).interface == "RLPolicy"
                                     else "fit_predict")}
        if _contains_outcome(evidence):
            raise ValueError("Outcome-like telemetry is forbidden")
        prepared.append(evidence)
    output.mkdir(parents=True)
    records = []
    for evidence in prepared:
        path = output / f"{evidence['arm_id']}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        records.append({"arm_id": evidence["arm_id"], "component_id": evidence["component_id"],
                        "wall_seconds": evidence["wall_seconds"], "fit_rows": evidence["fit_rows"],
                        "evaluation_rows": evidence["evaluation_rows"], "device": evidence["device"],
                        "status": "complete", "evidence_sha256": _sha(path)})
    records_file.parent.mkdir(parents=True, exist_ok=True)
    with records_file.open("x", encoding="utf-8") as stream:
        json.dump(records, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    if mapping_file is not None:
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        paths = {evidence["arm_id"]: (output / f"{evidence['arm_id']}.json").as_posix()
                 for evidence in prepared}
        with mapping_file.open("x", encoding="utf-8") as stream:
            json.dump(paths, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    return records


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--prepared-study", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--mapping")
    parser.add_argument("--securities", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto", "protocol"), default="cpu")
    parser.add_argument("--fit-sessions", type=int, default=80,
                        help="Development fit-session tail; zero uses the full declared fit window")
    parser.add_argument("--arm-id", action="append", dest="arm_ids",
                        help="Declared arm to benchmark; repeat to run a subset")
    args = parser.parse_args()
    records = run_smoke(repository_root=args.repository_root,
        prepared_study=args.prepared_study, output_dir=args.output_dir,
        records_path=args.records, securities=args.securities, mapping_path=args.mapping,
        device=args.device, fit_sessions=args.fit_sessions,
        arm_ids=tuple(args.arm_ids) if args.arm_ids else None)
    print(json.dumps({"records": len(records), "output_dir": args.output_dir}, sort_keys=True))


if __name__ == "__main__":
    _cli()
