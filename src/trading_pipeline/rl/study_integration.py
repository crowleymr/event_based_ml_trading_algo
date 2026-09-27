"""Study-owned DQN/PPO trial bridge over frozen, point-in-time selector episodes.

Only the authoritative study runner should call this module. It receives already
partitioned episodes; constructing sleeves or choosing folds is outside its scope.
The direct pilot runner remains restricted to synthetic engineering verification.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

from trading_pipeline.experiments import FitContext
from trading_pipeline.experiments.authority import VerifiedStudyAuthority
from trading_pipeline.experiments.default_registry import default_registry
from trading_pipeline.experiments.schema import ResolvedStudy, protocol_content_sha256
from trading_pipeline.rl.dataset import PilotDataset
from trading_pipeline.rl.environment import StrategySelectorEnv
from trading_pipeline.rl.protocol import ACTION_IDS
from trading_pipeline.rl.runner import PolicyTrialResult
from trading_pipeline.portfolio import financial_metrics


def build_study_selector_dataset(
    *, split: str, transitions: list[Mapping[str, Any]],
    observations: list[Mapping[str, Any]], sleeves: list[Mapping[str, Any]],
    prices: list[Mapping[str, Any]], calendar: list[date],
    observation_fields: tuple[str, ...], source_hashes: Mapping[str, str],
    valuation_policy: str = "complete_daily_valuation_v1",
) -> PilotDataset:
    """Build a fold episode from explicit central-runner inputs, without legacy artefacts.

    Required row fields are:
    transitions: signal_date, execution_date, end_date;
    observations: signal_date plus every named observation field;
    sleeves: signal_date, action_id (E0..E5), security_id, target_weight;
    prices: session_date, security_id, adjusted_close.
    The caller supplies causally generated observations and frozen sleeve weights.
    Every named action must have at least one weight row at each signal; cash is
    implicit action 0. All source hashes must identify immutable upstream files.
    """
    if not observation_fields or len(set(observation_fields)) != len(observation_fields):
        raise ValueError("RL observation fields must be nonempty and unique")
    signals = tuple(row["signal_date"] for row in transitions)
    obs_by_date = {row["signal_date"]: row for row in observations}
    if len(obs_by_date) != len(observations) or set(obs_by_date) != set(signals):
        raise ValueError("RL observations must match transition signals one to one")
    matrix = np.asarray([[obs_by_date[day][field] for field in observation_fields]
                         for day in signals], dtype=np.float32)
    target_lookup: dict[tuple[int, int], dict[str, float]] = {}
    step_by_signal = {day: index for index, day in enumerate(signals)}
    if len(step_by_signal) != len(signals):
        raise ValueError("RL transition signals must be unique")
    target_lookup.update({(step, 0): {} for step in range(len(signals))})
    action_by_id = {name: action for action, name in ACTION_IDS.items() if action != 0}
    for row in sleeves:
        signal, action_id = row["signal_date"], row["action_id"]
        if signal not in step_by_signal or action_id not in action_by_id:
            raise ValueError("RL sleeve references undeclared signal or action")
        key = (step_by_signal[signal], action_by_id[action_id])
        weights = target_lookup.setdefault(key, {})
        security = row["security_id"]
        if security in weights:
            raise ValueError("Duplicate RL sleeve security weight")
        weights[security] = float(row["target_weight"])
    price_lookup: dict[tuple[date, str], float] = {}
    for row in prices:
        key = (row["session_date"], row["security_id"])
        if key in price_lookup:
            raise ValueError("Duplicate RL valuation bar")
        price_lookup[key] = float(row["adjusted_close"])
    dataset = PilotDataset(
        split=split, signal_dates=signals,
        execution_dates=tuple(row["execution_date"] for row in transitions),
        end_dates=tuple(row["end_date"] for row in transitions),
        base_observations=matrix, observation_fields=observation_fields,
        targets=target_lookup, prices=price_lookup, calendar=tuple(calendar),
        reference_equity={}, source_hashes=dict(source_hashes),
        valuation_policy=valuation_policy,
    )
    _episode(dataset, label=split)
    return dataset


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _append(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str, allow_nan=False) + "\n")


def _episode(dataset: PilotDataset, *, label: str) -> None:
    count = len(dataset.signal_dates)
    if count == 0 or len(dataset.execution_dates) != count or len(dataset.end_dates) != count:
        raise ValueError(f"{label} episode has incomplete transition dates")
    if dataset.base_observations.shape != (count, len(dataset.observation_fields)):
        raise ValueError(f"{label} observation shape differs from declared fields")
    if not np.isfinite(dataset.base_observations).all():
        raise ValueError(f"{label} observations must be finite")
    if not dataset.source_hashes or any(len(value) != 64 for value in dataset.source_hashes.values()):
        raise ValueError(f"{label} requires source SHA-256 lineage")
    calendar = {day: index for index, day in enumerate(dataset.calendar)}
    if dataset.valuation_policy not in {"complete_daily_valuation_v1",
                                        "observed_history_and_endpoint_valuation_v1"}:
        raise ValueError(f"{label} has an unsupported valuation policy")
    if len(calendar) != len(dataset.calendar) or tuple(sorted(calendar)) != dataset.calendar:
        raise ValueError(f"{label} calendar must be sorted and unique")
    for step, (signal, execution, end) in enumerate(zip(dataset.signal_dates,
                                                         dataset.execution_dates,
                                                         dataset.end_dates, strict=True)):
        if (signal not in calendar or execution not in calendar or end not in calendar
                or calendar[execution] != calendar[signal] + 1
                or not calendar[signal] < calendar[execution] <= calendar[end]):
            raise ValueError(f"{label} transition {step} violates T+1 chronology")
        for action in ACTION_IDS:
            target = dataset.targets.get((step, action))
            if target is None or any(not math.isfinite(weight) or weight < 0
                                     for weight in target.values()) or sum(target.values()) > 1 + 1e-12:
                raise ValueError(f"{label} has missing or invalid frozen sleeve {action}")
            if action != 0 and not target:
                raise ValueError(f"{label} has empty frozen sleeve {action}")
            for security in target:
                valuation_dates = (dataset.calendar[calendar[execution]:calendar[end] + 1]
                                   if dataset.valuation_policy == "complete_daily_valuation_v1"
                                   else (execution, end))
                if any((day, security) not in dataset.prices for day in valuation_dates):
                    raise ValueError(f"{label} has a missing selected-security endpoint valuation")
                if step + 1 < count and (dataset.execution_dates[step + 1], security) not in dataset.prices:
                    raise ValueError(f"{label} has a missing carried-security T+1 valuation")
    if tuple(sorted(dataset.signal_dates)) != dataset.signal_dates:
        raise ValueError(f"{label} signals must be chronological")


def _scenario_dataset(dataset: PilotDataset, *, gross_cap: float, max_position: float) -> PilotDataset:
    """Apply the same predeclared constraints to every frozen DQN/PPO sleeve."""
    targets = {}
    for key, weights in dataset.targets.items():
        gross = sum(weights.values())
        scale = min(1.0, gross_cap / gross) if gross else 1.0
        if weights:
            scale = min(scale, min(max_position / weight for weight in weights.values()
                                   if weight > 0) if any(weight > 0 for weight in weights.values()) else 1.0)
        targets[key] = {security: weight * scale for security, weight in weights.items()}
    return replace(dataset, targets=targets)


class _CertaintyEquivalentEnv(StrategySelectorEnv):
    """Reward is net weekly log return less half lambda times squared log return."""

    def __init__(self, dataset: PilotDataset, cost_bps: float, risk_aversion: float):
        super().__init__(dataset, cost_bps)
        self.risk_aversion = risk_aversion

    def step(self, action):
        observation, net_log_return, terminated, truncated, info = super().step(action)
        reward = net_log_return - 0.5 * self.risk_aversion * net_log_return ** 2
        return observation, float(reward), terminated, truncated, {
            **info, "net_log_return": net_log_return, "certainty_equivalent_reward": reward,
        }


def _learning_state(policy: Any) -> tuple[Any, ...]:
    backend = policy._model
    replay = getattr(backend, "replay_buffer", None)
    normaliser = getattr(backend, "obs_rms", None)
    if normaliser is not None:
        mean = np.asarray(normaliser.mean).copy()
        var = np.asarray(normaliser.var).copy()
        normaliser_state = (mean.tobytes(), var.tobytes(), getattr(normaliser, "count", None))
    else:
        normaliser_state = None
    parameters = getattr(getattr(backend, "policy", None), "parameters", None)
    parameter_state = (tuple(parameter.detach().cpu().numpy().tobytes()
                             for parameter in parameters()) if callable(parameters) else None)
    return (getattr(backend, "num_timesteps", None), getattr(backend, "_n_updates", None),
            getattr(replay, "pos", None), getattr(replay, "full", None),
            normaliser_state, parameter_state)


def _evaluate(policy: Any, environment: _CertaintyEquivalentEnv, context: FitContext,
              component_id: str) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    before = _learning_state(policy)
    observation, _ = environment.reset(seed=context.seed)
    actions, equity = [], []
    done = False
    while not done:
        action = policy.act(observation, deterministic=True)
        observation, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        actions.append({"component_id": component_id, "trial_id": context.trial_id,
                        "fold_id": context.fold_id, "seed": context.seed, "reward": reward,
                        **info})
        equity.append({"component_id": component_id, "trial_id": context.trial_id,
                       "fold_id": context.fold_id, "seed": context.seed,
                       "session_date": info["end_date"], "equity": info["equity"],
                       "daily_return": float(np.expm1(info["net_log_return"])),
                       "turnover": info["turnover"], "cost": info["cost"]})
    if _learning_state(policy) != before:
        raise RuntimeError("RL evaluation mutated policy, replay or normaliser state")
    action_frame, equity_frame = pl.DataFrame(actions), pl.DataFrame(equity)
    metrics = financial_metrics(equity_frame, periods_per_year=52)
    metrics.update({"periods_per_year": 52,
                    "certainty_equivalent_mean": float(action_frame["reward"].mean()),
                    "certainty_equivalent_sum": float(action_frame["reward"].sum()),
                    "reward_convention": "net_log_return_minus_half_lambda_squared_log_return"})
    return action_frame, equity_frame, metrics


def execute_study_rl_trial(
    *, study: ResolvedStudy, authority: VerifiedStudyAuthority, component_id: str,
    context: FitContext, train_dataset: PilotDataset, score_dataset: PilotDataset,
    risk_scenario: str, parameters: Mapping[str, Any], total_timesteps: int,
    output_dir: str | Path, backend_factory=None,
) -> PolicyTrialResult:
    """Fit and persist one complete authorised RL fit/evaluation cell.

    The caller is the central study runner after verifying manifests and assigning
    fold data. A failed cell remains in the append-only trial ledger and cannot be
    silently dropped from complete-fold/seed checks.
    """
    config = study.config
    if (study.engineering_only or authority.study_id != config.get("study_id")
            or context.study_id != authority.study_id
            or authority.protocol_sha256 != protocol_content_sha256(config)
            or authority.data_role != "expanded_fresh_vintage"
            or config.get("search", {}).get("real_data_execution") != "enabled"):
        raise PermissionError("Verified approved expanded-study authority is required")
    registry = default_registry()
    spec = registry.spec(component_id)
    if (spec.interface != "RLPolicy" or not spec.capabilities.get("study_adapter")
            or not spec.capabilities.get("discrete_actions")) or not any(
            arm.get("component_id") == component_id and arm.get("interface") == "RLPolicy"
            for arm in config.get("experiment_arms", [])):
        raise ValueError("RL component is not a declared study arm")
    seeds = config.get("reproducibility", {}).get("seeds", [])
    if context.seed not in seeds or type(context.seed) is not int:
        raise ValueError("RL fit seed is not declared by the study")
    objective = config.get("objectives", {}).get("rl_certainty_equivalent_v1", {})
    if objective.get("primary") != "weekly_net_certainty_equivalent_return":
        raise ValueError("Unsupported RL objective")
    scenarios = objective.get("risk_scenarios", {})
    portfolio_scenarios = config.get("portfolio", {}).get("risk_scenarios", {})
    if set(scenarios) != {"conservative", "balanced", "aggressive"} or set(portfolio_scenarios) != set(scenarios):
        raise ValueError("All three declared RL risk scenarios are required")
    if risk_scenario not in scenarios:
        raise ValueError("Undeclared RL risk scenario")
    coefficients = [item.get("risk_aversion") for item in scenarios.values()]
    if any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0
           for value in coefficients) or len(set(coefficients)) != 3:
        raise ValueError("Risk coefficients must be distinct finite positive numbers")
    risk_aversion = float(scenarios[risk_scenario]["risk_aversion"])
    constraints = portfolio_scenarios[risk_scenario]
    gross_cap, max_position = (constraints.get(key) for key in
                               ("gross_exposure_cap", "max_position"))
    if any(type(value) not in {int, float} or not 0 < value <= 1
           for value in (gross_cap, max_position)):
        raise ValueError("RL scenario requires valid exposure and position limits")
    if config.get("portfolio", {}).get("execution") != "action_at_T_filled_at_T_plus_1_close":
        raise ValueError("RL requires the study T+1 execution convention")
    cost_bps = config["portfolio"].get("cost_bps_one_way")
    if type(cost_bps) not in {int, float} or not math.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("RL requires a finite declared one-way cost")
    _episode(train_dataset, label="train")
    _episode(score_dataset, label="score")
    if train_dataset.observation_fields != score_dataset.observation_fields:
        raise ValueError("Train/score RL observation fields differ")
    if train_dataset.source_hashes != score_dataset.source_hashes:
        raise ValueError("Train/score RL source lineage differs")
    if max(train_dataset.end_dates) >= min(score_dataset.signal_dates):
        raise ValueError("RL train episode overlaps score episode")
    policy = registry.create(
        component_id, parameters=parameters, total_timesteps=total_timesteps,
        device=next(
            arm.get("execution_device", config["reproducibility"].get("requested_device", "cpu"))
            for arm in config["experiment_arms"]
            if arm.get("component_id") == component_id
        ),
        backend_factory=backend_factory)
    if context.fidelity and set(context.fidelity) != {"environment_steps"}:
        raise ValueError("RL fidelity requires environment_steps only")
    root = Path(output_dir)
    if not root.is_dir():
        raise ValueError("Central runner must create the trial output directory")
    identity = {"study_id": context.study_id, "component_id": component_id,
                "trial_id": context.trial_id, "fold_id": context.fold_id,
                "seed": context.seed, "risk_scenario": risk_scenario}
    cell_id = _digest(identity)[:20]
    cell_dir = root / "rl_cells" / cell_id
    if cell_dir.exists():
        raise FileExistsError(f"RL study cell already exists: {cell_id}")
    cell_dir.mkdir(parents=True)
    base = {**identity, "cell_id": cell_id, "protocol_sha256": authority.protocol_sha256,
            "train_source_sha256": _digest(train_dataset.source_hashes),
            "score_source_sha256": _digest(score_dataset.source_hashes),
            "parameters": dict(parameters), "risk_aversion": risk_aversion,
            "cost_bps_one_way": cost_bps,
            "started_at": datetime.now(timezone.utc).isoformat()}
    _append(root / "rl_trial_ledger.jsonl", {**base, "status": "started"})
    train = _scenario_dataset(train_dataset, gross_cap=gross_cap, max_position=max_position)
    score = _scenario_dataset(score_dataset, gross_cap=gross_cap, max_position=max_position)
    try:
        policy.learn(_CertaintyEquivalentEnv(train, cost_bps, risk_aversion), context=context)
        actions, curve, metrics = _evaluate(
            policy, _CertaintyEquivalentEnv(score, cost_bps, risk_aversion),
            context, component_id)
        actions.write_parquet(cell_dir / "actions.parquet")
        curve.write_parquet(cell_dir / "equity_curve.parquet")
        (cell_dir / "metrics.json").write_text(json.dumps(metrics, sort_keys=True,
                                                           default=str, allow_nan=False), encoding="utf-8")
        telemetry = {**policy.telemetry(), "evaluation_mode": "deterministic_no_learning",
                     "evaluation_state_unchanged": True,
                     "risk_scenario": risk_scenario, "risk_aversion": risk_aversion,
                     "reward_convention": metrics["reward_convention"]}
        (cell_dir / "telemetry.json").write_text(json.dumps(telemetry, sort_keys=True,
                                                             default=str, allow_nan=False), encoding="utf-8")
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in cell_dir.iterdir() if path.is_file()}
        _append(root / "rl_resource_ledger.jsonl", {**identity,
                "environment_steps_requested": telemetry.get("environment_steps_requested"),
                "environment_steps_completed": telemetry.get("environment_steps_completed"),
                "gradient_updates": telemetry.get("gradient_updates"),
                "parameter_count": telemetry.get("parameter_count"),
                "duration_seconds": telemetry.get("duration_seconds"),
                "actual_device": telemetry.get("actual_device"),
                "backend_metrics": telemetry.get("backend_metrics")})
        _append(root / "rl_trial_ledger.jsonl", {**identity, "cell_id": cell_id,
                "status": "complete", "artefact_sha256": hashes,
                "completed_at": datetime.now(timezone.utc).isoformat()})
        return PolicyTrialResult(component_id, context.study_id, context.trial_id,
                                 context.fold_id, context.seed, actions, curve, metrics,
                                 telemetry)
    except Exception as exc:
        _append(root / "rl_trial_ledger.jsonl", {**identity, "cell_id": cell_id,
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "failed_at": datetime.now(timezone.utc).isoformat()})
        raise
