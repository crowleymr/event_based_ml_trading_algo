"""Read-only expanded-study evidence from one completed, audited run.

This command never opens a holdout for selection and never writes under runs/.
It intentionally refuses partial matrices and unpinned source data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import polars as pl

from trading_pipeline.portfolio.backtest import financial_metrics, solve_rebalance
from trading_pipeline.portfolio.frontier import (
    RISK_SCENARIOS, model_conditioned_frontier, realised_risk_return_curve,
)
from trading_pipeline.reporting.generate import OPTIONAL_RESEARCH_COLUMNS


TABLES = tuple(name.removesuffix(".parquet") for name in OPTIONAL_RESEARCH_COLUMNS)
SUPERVISED = "SupervisedModel"
RL = "RLPolicy"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _lines(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or out-of-root input: {relative}")
    return path


def _verified_run(root: Path, run: Path) -> tuple[dict, dict, dict[str, str]]:
    if not run.is_relative_to(root / "runs"):
        raise ValueError("Source must be under repository runs/")
    metadata, audit, completion = (_json(run / name) for name in
                                   ("metadata.json", "audit.json", "completion.json"))
    if (metadata.get("status") != "complete" or audit.get("status") != "passed"
            or completion.get("status") != "complete"
            or completion.get("run_id") != metadata.get("run_id")
            or metadata.get("run_id") != run.name
            or completion.get("audit_sha256") != _hash(run / "audit.json")
            or metadata.get("holdout_role") != "descriptive_only"):
        raise ValueError("Run is not a completed audited descriptive study")
    expected = audit.get("artefact_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Audit lacks artifact hashes")
    actual = {p.relative_to(run).as_posix(): _hash(p) for p in run.rglob("*")
              if p.is_file() and p.name not in {"audit.json", "completion.json"}}
    if actual != expected:
        raise ValueError("Run audit hash inventory mismatch")
    if (run / "failure.json").exists():
        raise ValueError("Completed run also has failure marker")
    return metadata, _json(run / "protocol.json"), actual


def _protocol_inputs(
    root: Path, metadata: dict, protocol: dict,
) -> tuple[pl.DataFrame, pl.DataFrame, dict, dict[str, str]]:
    pinned = metadata.get("input_sha256")
    if (not isinstance(pinned, dict) or not pinned
            or pinned != protocol.get("authority", {}).get("input_sha256")
            or metadata.get("protocol_sha256") != protocol.get("authority", {}).get("protocol_sha256")):
        raise ValueError("Protocol/input authority differs from completed run")
    sources: dict[str, str] = {}
    for relative, sha in pinned.items():
        path = _inside(root, relative)
        if _hash(path) != sha:
            raise ValueError(f"Pinned authority hash mismatch: {relative}")
        sources[relative] = sha
    snapshot = _json(_inside(root, protocol["data"]["snapshot_manifest"]))
    source_relative = snapshot["source_manifest_path"]
    source_path = _inside(root, source_relative)
    if _hash(source_path) != snapshot["source_manifest_sha256"]:
        raise ValueError("Source admission manifest hash mismatch")
    source = _json(source_path)
    snapshot["requested_count"] = source["requested_count"]
    snapshot["mapped_count"] = source["admitted_count"]
    sources[source_relative] = snapshot["source_manifest_sha256"]
    feature_relative = snapshot["feature_path"]
    feature_path = _inside(root, feature_relative)
    metadata_feature_relative = metadata.get("feature_path")
    if (not isinstance(metadata_feature_relative, str)
            or snapshot["feature_sha256"] != _hash(feature_path)
            or _inside(root, metadata_feature_relative) != feature_path
            or metadata.get("feature_sha256") != snapshot["feature_sha256"]):
        raise ValueError("Canonical feature snapshot is not hash-pinned")
    sources[feature_relative] = snapshot["feature_sha256"]
    bars_contract = protocol["data"].get("rl_inputs", {})
    if (bars_contract.get("bars_path") != feature_relative
            or bars_contract.get("bars_sha256") != snapshot["feature_sha256"]):
        raise ValueError("Portfolio bars must be the pinned feature snapshot")
    frame = pl.read_parquet(feature_path)
    required = {"security_id", "ticker", "session_date", "adjusted_close", "vol_20d"}
    if not required <= set(frame.columns):
        raise ValueError(f"Canonical snapshot lacks {sorted(required - set(frame.columns))}")
    if frame.select("security_id", "session_date").is_duplicated().any():
        raise ValueError("Duplicate canonical security/session keys")
    benchmark_contract = protocol["data"].get("benchmark_contract")
    if not isinstance(benchmark_contract, dict):
        raise ValueError("Approved protocol lacks a pinned benchmark contract")
    manifest_relative = benchmark_contract.get("manifest_path")
    bars_relative = benchmark_contract.get("bars_path")
    benchmark_manifest = _json(_inside(root, manifest_relative))
    benchmark_path = _inside(root, bars_relative)
    if (benchmark_contract.get("benchmark_id") != "B0-SPY"
            or benchmark_manifest.get("benchmark_id") != "B0-SPY"
            or benchmark_manifest.get("ticker") != "SPY"
            or benchmark_manifest.get("bars_path") != bars_relative
            or _hash(benchmark_path) != benchmark_contract.get("bars_sha256")
            or _hash(benchmark_path) != benchmark_manifest.get("bars_sha256")):
        raise ValueError("Pinned SPY benchmark contract is inconsistent")
    benchmark = pl.read_parquet(benchmark_path)
    required_benchmark = {"security_id", "ticker", "session_date", "adjusted_close"}
    if (not required_benchmark <= set(benchmark.columns)
            or benchmark.select("security_id", "session_date").is_duplicated().any()):
        raise ValueError("Pinned SPY benchmark violates the daily-bar contract")
    if _hash(_inside(root, manifest_relative)) != benchmark_contract.get("manifest_sha256"):
        raise ValueError("Pinned SPY benchmark manifest hash mismatch")
    sources[manifest_relative] = benchmark_contract["manifest_sha256"]
    sources[bars_relative] = benchmark_contract["bars_sha256"]
    return frame, benchmark, snapshot, sources


def _trial_tables(run: Path, protocol: dict) -> tuple[list[dict], list[dict], list[dict]]:
    arms = {arm["id"]: arm for arm in protocol["experiment_arms"]}
    inner_rows = _json(_inside(run.parents[2],
        protocol["validation"]["inner_windows_manifest"]))["folds"]
    inner_by_outer = defaultdict(set)
    for fold in inner_rows:
        inner_by_outer[fold["outer_fold_id"]].add(fold["fold_id"])
    seeds = set(protocol["reproducibility"]["seeds"])
    proposed = {}
    terminal = {}
    for row in _lines(run / "trial_ledger.jsonl"):
        trial = row["trial_id"]
        if row["status"] == "proposed":
            if trial in proposed:
                raise ValueError(f"Duplicate proposal {trial}")
            proposed[trial] = row
        elif row["status"] in {"complete", "failed", "pruned"}:
            if trial in terminal:
                raise ValueError(f"Duplicate trial terminal status {trial}")
            terminal[trial] = row
        else:
            raise ValueError(f"Unknown trial status {trial}")
    if not proposed or set(proposed) != set(terminal):
        raise ValueError("Incomplete trial ledger")
    fit_rows = _lines(run / "fit_ledger.jsonl")
    by_trial = defaultdict(list)
    for row in fit_rows:
        by_trial[row["trial_id"]].append((row["inner_fold_id"], row["seed"]))
    for trial, proposal in proposed.items():
        if terminal[trial]["status"] != "complete":
            continue
        expected = {(fold, seed) for fold in inner_by_outer[proposal["outer_fold_id"]]
                    for seed in seeds}
        observed = by_trial[trial]
        if set(observed) != expected or len(observed) != len(expected):
            raise ValueError(f"Incomplete inner fold/seed evidence for {trial}")
    outer_rows = _lines(run / "outer_fit_ledger.jsonl")
    expected_outer = {(arm_id, fold, seed,
                       scenario if arm["interface"] == RL else None)
                      for arm_id, arm in arms.items() for fold in inner_by_outer
                      for seed in seeds
                      for scenario in (RISK_SCENARIOS if arm["interface"] == RL else (None,))}
    observed_outer = [(row["arm_id"], row["outer_fold_id"], row["seed"],
                       row.get("risk_scenario")) for row in outer_rows]
    if set(observed_outer) != expected_outer or len(observed_outer) != len(expected_outer):
        raise ValueError("Incomplete outer fold/seed/risk matrix")
    locks = _json(run / "family_locks.json")["final_choices"]
    hpo, architecture = [], []
    for trial, proposal in sorted(proposed.items()):
        arm_id = proposal["arm_id"]
        if arm_id not in arms:
            raise ValueError(f"Unknown trial arm {arm_id}")
        state = terminal[trial]
        row = {"model_id": arm_id, "trial_id": trial, "status": state["status"],
               "family": arms[arm_id]["component_id"],
               "feature_set_id": arms[arm_id]["feature_set_id"],
               "outer_fold_id": proposal["outer_fold_id"],
               "risk_scenario": proposal.get("risk_scenario"),
               "proposal_index": proposal["proposal_index"],
               "parameters_json": json.dumps(proposal["parameters"], sort_keys=True),
               "mean_ic": state.get("mean_ic"), "mean_rmse": state.get("mean_rmse"),
               "mean_certainty_equivalent": state.get("mean_certainty_equivalent"),
               "resource_seconds": state.get("resource_seconds"),
               "reason": state.get("reason"),
               "selected_global": locks.get(arm_id + ("::" + proposal["risk_scenario"]
                                  if proposal.get("risk_scenario") else ""), {}).get("proposal_index")
                                  == proposal["proposal_index"]}
        hpo.append(row)
        if arms[arm_id]["component_id"] in {
            "supervised.lstm.v1", "supervised.causal_transformer.v1",
            "rl_dqn_sb3_v1", "rl_ppo_categorical_sb3_v1",
        }:
            architecture.append(dict(row))
    if not architecture:
        raise ValueError("Architecture evidence missing")
    return hpo, architecture, _lines(run / "holdout_fit_ledger.jsonl")


def _holdout_predictions(run: Path, protocol: dict, holdout: list, features: pl.DataFrame):
    arms = {arm["id"]: arm for arm in protocol["experiment_arms"]}
    seeds = set(protocol["reproducibility"]["seeds"])
    supervised = {key for key, arm in arms.items() if arm["interface"] == SUPERVISED}
    preds = pl.read_parquet(run / "predictions.parquet").filter(
        pl.col("partition") == "descriptive_holdout")
    required = {"arm_id", "seed", "session_date", "security_id", "predicted_return_5d"}
    if not required <= set(preds.columns) or preds.is_empty():
        raise ValueError("Holdout prediction contract incomplete")
    if preds.select("arm_id", "seed", "session_date", "security_id").is_duplicated().any():
        raise ValueError("Duplicate holdout predictions")
    calendar = tuple(sorted(set(holdout)))
    if set(preds["arm_id"].unique()) != supervised:
        raise ValueError("Missing supervised finalist predictions")
    if set(preds["seed"].unique()) != seeds:
        raise ValueError("Undeclared or missing prediction seed")
    feature_keys = features.filter(pl.col("session_date").is_in(calendar)).select(
        "session_date", "security_id").sort("session_date", "security_id")
    for arm in supervised:
        for seed in seeds:
            cell = preds.filter((pl.col("arm_id") == arm) & (pl.col("seed") == seed))
            if cell.select("session_date", "security_id").sort(
                    "session_date", "security_id").to_dicts() != feature_keys.to_dicts():
                raise ValueError(f"Incomplete prediction cell {arm}/{seed}")
            if not np.isfinite(cell["predicted_return_5d"].to_numpy()).all():
                raise ValueError(f"Nonfinite prediction cell {arm}/{seed}")
    return preds, arms, seeds, calendar


def _weekly_signals(calendar: tuple) -> tuple:
    seen, days = set(), []
    for day in calendar:
        key = day.isocalendar()[:2]
        if key not in seen:
            seen.add(key)
            days.append(day)
    return tuple(days)


def _supervised_portfolio(preds: pl.DataFrame, features: pl.DataFrame, calendar: tuple,
                          scenario: dict, config: dict) -> tuple[pl.DataFrame, list[dict], list[dict], list[dict]]:
    """Apply declared caps and trailing volatility target at T, then T+1 close."""
    lookback = config["volatility_lookback_sessions"]
    top_k = config["top_k"]
    cost_bps = config["cost_bps_one_way"]
    rows = features.select("session_date", "security_id", "adjusted_close").to_dicts()
    prices = {(r["session_date"], r["security_id"]): r["adjusted_close"] for r in rows}
    histories = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["session_date"]):
        histories[row["security_id"]].append((row["session_date"], row["adjusted_close"]))
    by_signal = {day: preds.filter(pl.col("session_date") == day).sort(
        ["predicted_return_5d", "security_id"], descending=[True, False]).head(top_k)
        for day in _weekly_signals(calendar)}
    executions = {}
    for day, cross in by_signal.items():
        ix = calendar.index(day)
        if ix + 1 >= len(calendar):
            continue
        names = cross["security_id"].to_list()
        if len(names) != top_k:
            raise ValueError(f"Fewer than top K predictions on {day}")
        cap, max_pos = scenario["gross_exposure_cap"], scenario["max_position"]
        base = min(cap / top_k, max_pos)
        target = {security: base for security in names}
        # Ex-ante volatility uses only complete adjusted-price history through T.
        returns = []
        for security in names:
            prior = [price for stamp, price in histories[security] if stamp <= day]
            if len(prior) < lookback + 1 or any(not np.isfinite(p) or p <= 0 for p in prior[-lookback-1:]):
                raise ValueError(f"Insufficient PIT covariance bars: {security} {day}")
            returns.append(np.diff(np.log(prior[-lookback-1:])))
        matrix = np.asarray(returns)
        covariance = np.atleast_2d(np.cov(matrix, ddof=1)) * 252
        raw = np.full(len(names), base)
        vol = math.sqrt(max(float(raw @ covariance @ raw), 0))
        scale = min(1.0, scenario["annualised_volatility_target"] / vol) if vol else 1.0
        target = {security: weight * scale for security, weight in target.items()}
        executions[calendar[ix + 1]] = (day, target)
    holdings, last_prices, cash, old_equity = {}, {}, 1.0, 1.0
    curves, positions, trades, contributions = [], [], [], []
    for day in calendar:
        for security, value in list(holdings.items()):
            price = prices.get((day, security))
            if price is None or not np.isfinite(price) or price <= 0:
                raise ValueError(f"Missing held price {security}/{day}")
            holdings[security] = value * price / last_prices[security]
            contributions.append({"session_date": day, "security_id": security,
                                  "gross_contribution": holdings[security] - value})
            last_prices[security] = price
        before = cash + sum(holdings.values())
        turnover = cost = 0.0
        if day in executions:
            signal, target = executions[day]
            if any((day, security) not in prices for security in target):
                raise ValueError(f"Missing T+1 execution bar {day}")
            holdings, cash, turnover, cost, dollars = solve_rebalance(
                before, holdings, target, cost_bps)
            last_prices = {security: prices[(day, security)] for security in holdings}
            trades.extend({"session_date": day, "signal_date": signal,
                           "security_id": security, "traded_value": value,
                           "cost": abs(value) * cost_bps / 10000}
                          for security, value in dollars.items())
        equity = cash + sum(holdings.values())
        if not np.isfinite(equity) or equity <= 0:
            raise ValueError("Invalid portfolio equity")
        positions.extend({"session_date": day, "security_id": security,
                          "weight": value / equity, "position_value": value}
                         for security, value in holdings.items())
        curves.append({"session_date": day, "equity": equity,
                       "daily_return": equity / old_equity - 1,
                       "turnover": turnover, "cost": cost})
        old_equity = equity
    return pl.DataFrame(curves), positions, trades, contributions


def _rl_holdout(run: Path, arms: dict, seeds: set[int],
                calendar: tuple) -> dict[tuple[str, int, str], pl.DataFrame]:
    expected = {(arm_id, seed, scenario) for arm_id, arm in arms.items()
                if arm["interface"] == RL for seed in seeds for scenario in RISK_SCENARIOS}
    rows = {}
    for row in _lines(run / "rl_trial_ledger.jsonl"):
        if row.get("fold_id") != "holdout" or row.get("status") != "complete":
            continue
        key = (row["component_id"], row["seed"], row["risk_scenario"])
        arm_id = next((name for name, arm in arms.items()
                       if arm["component_id"] == row["component_id"]), None)
        if arm_id is None:
            raise ValueError("Undeclared RL holdout component")
        cell = (arm_id, row["seed"], row["risk_scenario"])
        if cell in rows:
            raise ValueError(f"Duplicate RL holdout cell {cell}")
        directory = run / "rl_cells" / row["cell_id"]
        hashes = row.get("artefact_sha256", {})
        if set(hashes) != {"actions.parquet", "equity_curve.parquet", "metrics.json", "telemetry.json"}:
            raise ValueError("RL cell lacks complete hash inventory")
        if any(_hash(directory / name) != sha for name, sha in hashes.items()):
            raise ValueError("RL cell hash mismatch")
        curve = pl.read_parquet(directory / "equity_curve.parquet")
        actions = pl.read_parquet(directory / "actions.parquet")
        metrics = _json(directory / "metrics.json")
        index = {day: i for i, day in enumerate(calendar)}
        if (curve.is_empty() or curve.height != actions.height
                or curve["session_date"].to_list() != actions["end_date"].to_list()
                or len(set(curve["session_date"].to_list())) != curve.height
                or curve["session_date"].to_list() != sorted(curve["session_date"].to_list())
                or any(actions["signal_date"][i] not in index
                       or actions["execution_date"][i] not in index
                       or actions["end_date"][i] not in index
                       or index[actions["execution_date"][i]]
                          != index[actions["signal_date"][i]] + 1
                       or index[actions["end_date"][i]]
                          <= index[actions["execution_date"][i]]
                       for i in range(actions.height))):
            raise ValueError("RL cell has incomplete or invalid T+1 chronology")
        computed = financial_metrics(curve, periods_per_year=52)
        for field in ("total_return", "total_turnover", "cumulative_transaction_cost"):
            if not math.isclose(computed[field], metrics[field], abs_tol=1e-8):
                raise ValueError(f"RL cell metric mismatch: {field}")
        rows[cell] = curve.select("session_date", "equity", "daily_return", "turnover", "cost")
    if set(rows) != expected:
        raise ValueError(f"Incomplete RL holdout matrix: {sorted(expected - set(rows))}")
    return rows


def _sample_weekly(daily: pl.DataFrame, marks: tuple) -> pl.DataFrame:
    rows = daily.to_dicts()
    if any(day not in set(daily["session_date"].to_list()) for day in marks):
        raise ValueError("Unaligned supervised/RL valuation calendars")
    sampled, previous = [], 1.0
    pending_turnover = pending_cost = 0.0
    mark_set = set(marks)
    for row in rows:
        pending_turnover += row["turnover"]
        pending_cost += row["cost"]
        if row["session_date"] in mark_set:
            sampled.append({"session_date": row["session_date"], "equity": row["equity"],
                            "daily_return": row["equity"] / previous - 1,
                            "turnover": pending_turnover, "cost": pending_cost})
            previous = row["equity"]
            pending_turnover = pending_cost = 0.0
    if tuple(row["session_date"] for row in sampled) != marks:
        raise ValueError("Weekly sampling omitted a common mark")
    if any(row["session_date"] > marks[-1] and (row["turnover"] or row["cost"])
           for row in rows):
        raise ValueError("Trading after last aligned RL mark")
    return pl.DataFrame(sampled)


def _profile(curve: pl.DataFrame, model: str, seed: int | None, scenario: str,
             control: float, split: str, benchmark_available: bool) -> tuple[dict, list[dict]]:
    dates = curve["session_date"].to_list()
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Invalid final test bench dates")
    values = curve["equity"].to_numpy()
    returns = curve["daily_return"].to_numpy()
    if (not np.isfinite(values).all() or (values <= 0).any()
            or not np.isfinite(returns).all()
            or not np.allclose(np.cumprod(1 + returns), values, atol=1e-8)):
        raise ValueError("Final equity does not reconcile with period returns")
    metrics = financial_metrics(curve, periods_per_year=52)
    calendar_sha = hashlib.sha256("\n".join(str(day) for day in dates).encode()).hexdigest()
    row = {"model_id": model, "experiment_id": model, "seed": seed,
           "risk_scenario": scenario, "risk_control_value": control,
           "split": split, "periods_per_year": 52, "start_date": dates[0],
           "end_date": dates[-1], "calendar_sha256": calendar_sha,
           "benchmark_available": benchmark_available, **metrics}
    peak = 1.0
    equity_rows = []
    for point in curve.to_dicts():
        peak = max(peak, point["equity"])
        equity_rows.append({"model_id": model, "experiment_id": model,
                            "display_label": model, "seed": seed,
                            "risk_scenario": scenario, "split": split,
                            "session_date": point["session_date"],
                            "equity": point["equity"], "drawdown": point["equity"] / peak - 1,
                            "period_return": point["daily_return"],
                            "turnover": point["turnover"], "cost": point["cost"],
                            "calendar_sha256": calendar_sha})
    return row, equity_rows


def _security_and_stage(features: pl.DataFrame, predictions: pl.DataFrame,
                        positions: list[dict], trades: list[dict], contributions: list[dict],
                        snapshot: dict,
                        calendar: tuple) -> tuple[list[dict], list[dict]]:
    frame = features.filter(pl.col("session_date").is_in(calendar))
    prediction_keys = set(zip(predictions["security_id"].to_list(),
                              predictions["session_date"].to_list()))
    selected = defaultdict(set)
    traded = defaultdict(int)
    costs = defaultdict(float)
    gross = defaultdict(float)
    for row in positions:
        selected[row["security_id"]].add(row["session_date"])
    for row in trades:
        traded[row["security_id"]] += 1
        costs[row["security_id"]] += row["cost"]
    for row in contributions:
        gross[row["security_id"]] += row["gross_contribution"]
    feature_cols = [name for name in ("return_5d", "return_10d", "return_20d",
                    "vol_5d", "vol_20d", "latest_eps", "latest_net_income")
                    if name in frame.columns]
    security_rows = []
    for security, group in frame.partition_by("security_id", as_dict=True).items():
        security_id = security[0] if isinstance(security, tuple) else security
        days = group["session_date"].to_list()
        ready = group.select(pl.all_horizontal(*(pl.col(col).is_not_null()
            & pl.col(col).is_finite() for col in feature_cols)).sum()).item() if feature_cols else 0
        covered = group.filter(pl.col("latest_eps").is_not_null()
                             & pl.col("latest_net_income").is_not_null()).height
        security_rows.append({"security_id": security_id,
                              "ticker": group["ticker"][0], "first_date": min(days),
                              "last_date": max(days), "priced_rows": group.height,
                              "fundamental_covered_rows": covered,
                              "feature_ready_rows": ready,
                              "labelled_rows": group.filter(pl.col("forward_return_5d").is_not_null()).height,
                              "predicted_rows": sum((security_id, day) in prediction_keys for day in days),
                              "held_rows": len(selected[security_id]),
                              "trade_rows": traded[security_id],
                              "gross_contribution": gross[security_id],
                              "transaction_cost": costs[security_id],
                              "net_contribution": gross[security_id] - costs[security_id]})
    stages = [
        ("requested", snapshot.get("requested_count")),
        ("mapped", snapshot.get("mapped_count")),
        ("priced", sum(row["priced_rows"] for row in security_rows)),
        ("fundamental_covered", sum(row["fundamental_covered_rows"] for row in security_rows)),
        ("feature_ready", sum(row["feature_ready_rows"] for row in security_rows)),
        ("labelled", sum(row["labelled_rows"] for row in security_rows)),
        ("predicted", sum(row["predicted_rows"] for row in security_rows)),
        ("held", sum(row["held_rows"] for row in security_rows)),
    ]
    if any(value is None for _, value in stages):
        raise ValueError("Pipeline manifest lacks admission count")
    return ([{"stage": name, "count": value, "count_unit":
              "securities" if name in {"requested", "mapped"} else "security_sessions",
              "partition": "descriptive_holdout"} for name, value in stages], security_rows)


def _frontiers(predictions: pl.DataFrame, features: pl.DataFrame, calendar: tuple,
               protocol: dict, seeds: set[int]) -> tuple[list[dict], list[dict]]:
    portfolio = protocol["portfolio"]
    if portfolio.get("covariance_estimator") != "trailing_sample_covariance_v1":
        raise ValueError("Unsupported frontier covariance estimator")
    lookback = portfolio["volatility_lookback_sessions"]
    history = defaultdict(list)
    for row in features.select("security_id", "session_date", "adjusted_close").sort(
            "session_date").to_dicts():
        history[row["security_id"]].append((row["session_date"], row["adjusted_close"]))
    points, weights = [], []
    for model in sorted(predictions["arm_id"].unique().to_list()):
        for seed in sorted(seeds):
            cell = predictions.filter((pl.col("arm_id") == model) & (pl.col("seed") == seed))
            for day in _weekly_signals(calendar):
                cross = cell.filter(pl.col("session_date") == day).sort(
                    ["predicted_return_5d", "security_id"], descending=[True, False]
                ).head(portfolio["top_k"])
                names = cross["security_id"].to_list()
                if len(names) != portfolio["top_k"]:
                    raise ValueError("Incomplete frontier expected-return vector")
                series = []
                for security in names:
                    prior = [price for stamp, price in history[security] if stamp <= day]
                    if len(prior) < lookback + 1 or any(not np.isfinite(p) or p <= 0 for p in prior[-lookback-1:]):
                        raise ValueError("Incomplete point-in-time frontier covariance")
                    series.append(np.diff(np.log(prior[-lookback-1:])))
                covariance = np.atleast_2d(np.cov(np.asarray(series), ddof=1)) * 252
                vector = cross["predicted_return_5d"].to_numpy() * (252 / 5)
                input_sha = hashlib.sha256(json.dumps({"model": model, "seed": seed,
                    "date": str(day), "names": names, "means": vector.tolist(),
                    "covariance": covariance.tolist()}, sort_keys=True).encode()).hexdigest()
                for scenario in RISK_SCENARIOS:
                    constraint = portfolio["risk_scenarios"][scenario]
                    aversion = protocol["objectives"]["rl_certainty_equivalent_v1"]["risk_scenarios"][scenario]["risk_aversion"]
                    point, weight = model_conditioned_frontier(
                        model_id=f"{model}|seed={seed}|scenario={scenario}",
                        rebalance_date=day, security_ids=names,
                        expected_returns=vector, covariance=covariance,
                        risk_aversions=[aversion],
                        gross_exposure_cap=constraint["gross_exposure_cap"],
                        max_position=constraint["max_position"],
                        annualised_volatility_cap=constraint["annualised_volatility_target"],
                        covariance_id="trailing_sample_covariance_v1",
                        input_sha256=input_sha)
                    points.extend({**row, "model_id": model, "seed": seed,
                                   "risk_scenario": scenario,
                                   "covariance_lookback_sessions": lookback,
                                   "volatility_target": constraint["annualised_volatility_target"]}
                                  for row in point.to_dicts())
                    weights.extend({**row, "model_id": model, "seed": seed,
                                    "risk_scenario": scenario}
                                   for row in weight.to_dicts())
    return points, weights


def generate_expanded_closeout(*, repository_root: str | Path, run_id: str) -> Path:
    """Generate eleven hash-declared tables from one immutable expanded run."""
    root = Path(repository_root).resolve()
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    run = (root / "runs" / "expanded_closeout" / run_id).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    metadata, protocol, run_hashes = _verified_run(root, run)
    if (protocol.get("study_id") != metadata.get("study_id")
            or protocol.get("status") != "approved"
            or protocol.get("search", {}).get("real_data_execution") != "enabled"
            or metadata.get("claim_status") != "exploratory_closeout_not_confirmatory"):
        raise ValueError("Source is not an approved expanded exploratory run")
    features, benchmark_bars, snapshot, sources = _protocol_inputs(root, metadata, protocol)
    holdout_manifest = _json(_inside(root, protocol["validation"]["final_holdout"]["manifest"]))
    holdout = tuple(__import__("datetime").date.fromisoformat(day)
                    for day in holdout_manifest["session_dates"])
    if not holdout_manifest.get("sealed") or not holdout or holdout != tuple(sorted(set(holdout))):
        raise ValueError("Invalid sealed holdout calendar")
    hpo, architecture, holdout_fits = _trial_tables(run, protocol)
    predictions, arms, seeds, calendar = _holdout_predictions(run, protocol, holdout, features)
    if calendar != holdout:
        raise ValueError("Predictions do not match the sealed holdout calendar")
    expected_fits = {(arm, seed, scenario if detail["interface"] == RL else None)
                     for arm, detail in arms.items() for seed in seeds
                     for scenario in (RISK_SCENARIOS if detail["interface"] == RL else (None,))}
    observed_fits = {(row["arm_id"], row["seed"], row.get("risk_scenario"))
                     for row in holdout_fits}
    if observed_fits != expected_fits or len(holdout_fits) != len(expected_fits):
        raise ValueError("Incomplete holdout fit matrix")
    rl_curves = _rl_holdout(run, arms, seeds, calendar)
    rl_marks = {tuple(curve["session_date"].to_list()) for curve in rl_curves.values()}
    if len(rl_marks) != 1:
        raise ValueError("RL holdout calendars differ")
    marks = next(iter(rl_marks))
    if marks != _weekly_signals(calendar)[1:]:
        raise ValueError("RL marks do not align to the sealed holdout")
    valuation_calendar = tuple(day for day in calendar if day <= marks[-1])
    portfolio = protocol["portfolio"]
    if (portfolio.get("execution") != "action_at_T_filled_at_T_plus_1_close"
            or portfolio.get("cost_bps_one_way") is None
            or set(portfolio.get("risk_scenarios", {})) != set(RISK_SCENARIOS)):
        raise ValueError("Unsupported or incomplete portfolio protocol")
    metric_rows, equity_rows = [], []
    positions, trades, contributions = [], [], []
    curves = {}
    for model, detail in sorted(arms.items()):
        for seed in sorted(seeds):
            for scenario in RISK_SCENARIOS:
                if detail["interface"] == SUPERVISED:
                    cell = predictions.filter((pl.col("arm_id") == model) & (pl.col("seed") == seed))
                    daily, held, traded, earned = _supervised_portfolio(
                        cell, features, valuation_calendar,
                        portfolio["risk_scenarios"][scenario], portfolio)
                    positions.extend({**row, "model_id": model, "seed": seed,
                                      "risk_scenario": scenario} for row in held)
                    trades.extend({**row, "model_id": model, "seed": seed,
                                   "risk_scenario": scenario} for row in traded)
                    contributions.extend({**row, "model_id": model, "seed": seed,
                                          "risk_scenario": scenario} for row in earned)
                    curve = _sample_weekly(daily, marks)
                else:
                    curve = rl_curves[(model, seed, scenario)]
                control = float(portfolio["risk_scenarios"][scenario]["annualised_volatility_target"])
                if detail["interface"] == RL:
                    control = 1 / float(protocol["objectives"]["rl_certainty_equivalent_v1"]["risk_scenarios"][scenario]["risk_aversion"])
                profile, series = _profile(curve, model, seed, scenario, control,
                                           "descriptive_holdout", False)
                curves[(model, seed, scenario)] = curve
                metric_rows.append(profile)
                equity_rows.extend(series)
    # SPY comes from the separately acquired, protocol-pinned benchmark vintage.
    spy = benchmark_bars.filter(pl.col("ticker") == "SPY")
    benchmark_available = False
    benchmark_reason = "Protocol-pinned SPY benchmark is unavailable"
    if not spy.is_empty():
        ids = spy["security_id"].unique().to_list()
        if len(ids) != 1:
            raise ValueError("Ambiguous pinned SPY identifier")
        spy = spy.filter(pl.col("session_date").is_in(calendar)).sort("session_date")
        if spy["session_date"].to_list() != list(calendar):
            raise ValueError("Pinned SPY has incomplete holdout valuation bars")
        prices = spy["adjusted_close"].to_numpy()
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Pinned SPY prices are invalid")
        entry = calendar[1]
        cost = portfolio["cost_bps_one_way"] / 10000
        benchmark_daily = []
        entry_price = float(spy.filter(pl.col("session_date") == entry)["adjusted_close"][0])
        for day, price in zip(calendar, prices, strict=True):
            equity = 1.0 if day < entry else (1 - cost) * float(price) / entry_price
            previous = benchmark_daily[-1]["equity"] if benchmark_daily else 1.0
            benchmark_daily.append({"session_date": day, "equity": equity,
                                    "daily_return": equity / previous - 1,
                                    "turnover": 1.0 if day == entry else 0.0,
                                    "cost": cost if day == entry else 0.0})
        benchmark = _sample_weekly(pl.DataFrame(benchmark_daily), marks)
        profile, series = _profile(benchmark, "B0_SPY", None, "benchmark", 0.0,
                                   "descriptive_holdout", True)
        metric_rows.append(profile)
        equity_rows.extend(series)
        benchmark_available = True
        benchmark_reason = None
        bench_equity = benchmark["equity"].to_numpy()
        for row in equity_rows:
            ix = marks.index(row["session_date"])
            row["benchmark_equity"] = float(bench_equity[ix])
            row["relative_return"] = row["equity"] / bench_equity[ix] - 1
        for row in metric_rows:
            row["benchmark_available"] = True
            row["benchmark_total_return"] = profile["total_return"]
            row["benchmark_relative_total_return"] = row["total_return"] - profile["total_return"]
    else:
        for row in metric_rows:
            row["benchmark_available"] = False
            row["benchmark_total_return"] = None
            row["benchmark_relative_total_return"] = None
        for row in equity_rows:
            row["benchmark_equity"] = None
            row["relative_return"] = None
    stage, security = _security_and_stage(features, predictions, positions, trades,
                                           contributions,
                                           snapshot, calendar)
    expected_net = sum(row["total_return"] for row in metric_rows
                       if arms.get(row["model_id"], {}).get("interface") == SUPERVISED)
    actual_net = sum(row["net_contribution"] for row in security)
    if not math.isclose(actual_net, expected_net, abs_tol=1e-7):
        raise ValueError("Security contributions do not reconcile to portfolio return")
    frontier_points, frontier_weights = _frontiers(predictions, features, calendar,
                                                   protocol, seeds)
    # realised_risk_return_curve expects one three-scenario profile per model.
    # Preserve seed identity by extending its model key; retain the original arm ID.
    realised_input = pl.DataFrame([{**row, "model_id": f"{row['model_id']}|seed={row['seed']}"}
                                   for row in metric_rows if row["model_id"] != "B0_SPY"])
    realised = realised_risk_return_curve(realised_input).to_dicts()
    for row in realised:
        model, seed = row["model_id"].split("|seed=")
        row["model_id"] = model
        row["seed"] = int(seed)
    risk = [dict(row) for row in metric_rows if row["model_id"] != "B0_SPY"]
    questions = [{"item_id": "benchmark_availability", "status":
                  "available" if benchmark_available else "unavailable",
                  "kind": "generated_limitation", "fact": benchmark_reason or
                  "SPY was loaded from the protocol-pinned benchmark vintage and aligned to the holdout"},
                 {"item_id": "holdout_role", "status": "descriptive_only",
                  "kind": "protocol_fact", "fact":
                  "Completed expanded-study holdout is descriptive and cannot select a model or scenario"}]
    table_rows = {
        "pipeline_stage_summary": stage,
        "pipeline_security_summary": security,
        "architecture_trial_summary": architecture,
        "hpo_trial_summary": hpo,
        "risk_scenario_summary": risk,
        "model_conditioned_frontier_points": frontier_points,
        "model_conditioned_frontier_weights": frontier_weights,
        "realised_risk_return_curve": realised,
        "final_testbench_metrics": metric_rows,
        "final_testbench_equity_curve": equity_rows,
        "question_and_assumption_register": questions,
    }
    destination = (root / "reports" / "expanded_closeout" / run_id).resolve()
    if not destination.is_relative_to(root / "reports") or destination.exists():
        raise FileExistsError(f"Report destination must be new: {destination}")
    # Validate every table before the first write, preserving fail-closed generation.
    frames = {}
    for name in TABLES:
        frame = pl.DataFrame(table_rows[name])
        missing = OPTIONAL_RESEARCH_COLUMNS[f"{name}.parquet"] - set(frame.columns)
        if frame.is_empty() or missing:
            raise ValueError(f"Incomplete generated table {name}: {sorted(missing)}")
        frames[name] = frame
    destination.mkdir(parents=True, exist_ok=False)
    outputs = []
    for name, frame in frames.items():
        path = destination / f"{name}.parquet"
        frame.write_parquet(path)
        outputs.append({"relative_path": path.name, "sha256": _hash(path),
                        "rows": frame.height})
    generation = {"schema_version": 1, "source_run_id": run_id,
                  "source_run_path": run.relative_to(root).as_posix(),
                  "source_run_audit_sha256": _hash(run / "audit.json"),
                  "source_run_completion_sha256": _hash(run / "completion.json"),
                  "source_completed_at_utc": _json(run / "completion.json")["completed_at"],
                  "source_git_commit": metadata.get("git_commit"),
                  "source_vintage_id": metadata.get("vintage_id"),
                  "protocol_sha256": metadata["protocol_sha256"],
                  "comparison_contract": {
                      "metric_definition_id": "expanded_closeout_weekly_after_cost_v1",
                      "calendar_sha256": hashlib.sha256(json.dumps(
                          sorted({str(row["session_date"]) for row in equity_rows}),
                          separators=(",", ":")
                      ).encode("utf-8")).hexdigest(),
                      "execution": protocol["portfolio"]["execution"],
                      "cost_bps_one_way": protocol["portfolio"]["cost_bps_one_way"],
                      "annualisation_periods_per_year": 52,
                      "benchmark_symbol": protocol["inference"]["benchmark"],
                  },
                  "protocol_summary": {
                      "study_id": protocol["study_id"],
                      "budget_tier": protocol["search"]["policy_id"],
                      "budgets": protocol["search"]["budgets"],
                      "outer_folds": protocol["validation"]["window_policy"]["outer_folds"],
                      "inner_folds": protocol["validation"]["window_policy"]["inner_folds"],
                      "seeds": protocol["reproducibility"]["seeds"],
                      "requested_device": protocol["reproducibility"]["requested_device"],
                      "arms": [{"id": arm["id"], "component_id": arm["component_id"],
                                "interface": arm["interface"],
                                "requested_device": arm.get("execution_device",
                                    protocol["reproducibility"]["requested_device"])}
                               for arm in protocol["experiment_arms"]],
                  },
                  "code_version": subprocess.check_output(["git", "rev-parse", "HEAD"],
                      cwd=root, text=True).strip(),
                  "generator_module_sha256": _hash(Path(__file__)),
                  "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "claim_status": metadata["claim_status"],
                  "input_artifacts": [{"relative_path": name, "sha256": sha}
                                      for name, sha in sorted({**sources, **{
                                          f"{run.relative_to(root).as_posix()}/{name}": sha
                                          for name, sha in run_hashes.items()}}.items())],
                  "benchmark": {"symbol": "SPY", "available": benchmark_available,
                                "reason": benchmark_reason},
                  "outputs": outputs}
    (destination / "research_evidence_manifest.json").write_text(
        json.dumps(generation, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    root = Path(args.repository_root).resolve() if args.repository_root else _repository_root()
    run_id = args.run_id or _newest_completed_run_id(root)
    destination = root / "reports" / "expanded_closeout" / run_id
    if destination.exists() and args.run_id is None:
        from trading_pipeline.dashboard.loader import load_expanded_report
        load_expanded_report(destination)
        print(destination)
        return
    print(generate_expanded_closeout(repository_root=root, run_id=run_id))


def _repository_root() -> Path:
    for start in (Path.cwd(), Path(__file__).resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "trading_pipeline").is_dir():
                return candidate
    raise FileNotFoundError("Repository root not found; supply --repository-root")


def _newest_completed_run_id(root: Path) -> str:
    eligible = []
    for run in (root / "runs" / "expanded_closeout").glob("*"):
        if not run.is_dir():
            continue
        try:
            _verified_run(root, run)
            completed = _json(run / "completion.json")["completed_at"]
            stamp = datetime.fromisoformat(completed.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                continue
            eligible.append((stamp, run.name))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    if not eligible:
        raise FileNotFoundError("No completed audited expanded study run found under runs/expanded_closeout")
    return max(eligible)[1]


if __name__ == "__main__":
    main()
