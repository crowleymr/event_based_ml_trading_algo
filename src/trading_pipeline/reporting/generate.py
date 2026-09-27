"""Versioned reports derived only from a specified immutable run contract."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import subprocess

import polars as pl

from trading_pipeline.data.schemas import digest
from trading_pipeline.features import F0, F1
from trading_pipeline.tracking.telemetry import TRACE_SCHEMA, SUMMARY_SCHEMA
from .registry import registry

REPORT_SCHEMA_VERSION = 2

REQUIRED = (
    "metadata.json",
    "selection.json",
    "split_manifest.json",
    "dataset_manifest.json",
    "metrics.json",
    "experiment_comparison.parquet",
    "equity_curve.parquet",
    "positions.parquet",
    "trades.parquet",
    "daily_ic.parquet",
    "predictions.parquet",
    "feature_importance.parquet",
    "datasets/security_master.parquet",
    "datasets/market_bars.parquet",
    "datasets/fundamental_facts.parquet",
)
OPTIONAL_REPORT_INPUTS = (
    "training_trace.parquet",
    "training_summary.parquet",
    "device_benchmark.parquet",
)
OPTIONAL_RESEARCH_INPUTS = (
    "pipeline_stage_summary.parquet",
    "pipeline_security_summary.parquet",
    "architecture_trial_summary.parquet",
    "hpo_trial_summary.parquet",
    "risk_scenario_summary.parquet",
    "model_conditioned_frontier_points.parquet",
    "model_conditioned_frontier_weights.parquet",
    "realised_risk_return_curve.parquet",
    "final_testbench_metrics.parquet",
    "final_testbench_equity_curve.parquet",
    "question_and_assumption_register.parquet",
)

# Stable consumer contracts for optional expanded-study evidence. These are
# deliberately limited to fields used by the dashboard/notebook views.
OPTIONAL_RESEARCH_COLUMNS = {
    "pipeline_stage_summary.parquet": {"stage"},
    "pipeline_security_summary.parquet": {"security_id"},
    "architecture_trial_summary.parquet": {"model_id", "trial_id", "status"},
    "hpo_trial_summary.parquet": {"model_id", "trial_id", "status"},
    "risk_scenario_summary.parquet": {"model_id", "risk_scenario"},
    "model_conditioned_frontier_points.parquet": {
        "rebalance_date", "model_id", "status", "expected_volatility", "expected_return",
    },
    "model_conditioned_frontier_weights.parquet": {
        "rebalance_date", "model_id", "security_id", "weight",
    },
    "realised_risk_return_curve.parquet": {
        "model_id", "risk_order", "risk_scenario", "annualised_volatility",
        "annualised_return", "risk_control_monotonic",
    },
    "final_testbench_metrics.parquet": {"experiment_id", "split"},
    "final_testbench_equity_curve.parquet": {
        "experiment_id", "display_label", "split", "session_date", "equity", "drawdown",
    },
    "question_and_assumption_register.parquet": {"item_id", "status"},
}

METRICS = {
    "total_return": ("Compounded end equity minus one", "decimal fraction"),
    "annualised_return": ("Compounded return annualised over 252 sessions", "decimal fraction per year"),
    "annualised_volatility": ("Sample standard deviation of daily returns times sqrt(252)", "decimal fraction per year"),
    "sharpe": ("Annualised mean daily return divided by annualised volatility; zero risk-free rate", "dimensionless"),
    "maximum_drawdown": ("Minimum equity divided by prior running peak minus one", "decimal fraction"),
    "average_turnover": ("Mean daily absolute traded value divided by pre-trade equity", "decimal fraction"),
    "total_turnover": ("Sum of daily turnover", "decimal fraction"),
    "cumulative_transaction_cost": ("Sum of deducted transaction costs", "NAV units"),
    "mae": ("Mean absolute five-session return prediction error", "decimal return"),
    "rmse": ("Root mean squared five-session return prediction error", "decimal return"),
    "mean_ic": ("Mean finite daily cross-sectional Spearman rank correlation", "dimensionless"),
    "directional_accuracy": ("Share of predictions with the same sign as realised five-session return", "decimal fraction"),
    "active_return": ("Strategy daily return minus SPY daily return on the same session", "decimal return"),
    "tracking_error": ("Sample standard deviation of daily active return times sqrt(252)", "decimal fraction per year"),
    "information_ratio": ("Annualised mean active return divided by annualised tracking error", "dimensionless"),
    "beta": ("Sample covariance of strategy and SPY daily returns divided by SPY variance", "dimensionless"),
    "downside_capture": ("Mean strategy return on negative-SPY sessions divided by mean SPY return on those sessions", "ratio"),
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_run(root: Path) -> dict:
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run contract missing required inputs: {', '.join(missing)}")
    metadata = _load_json(root / "metadata.json")
    if metadata.get("status") != "complete":
        raise ValueError("Reporting requires a completed immutable run")
    if metadata.get("run_id") != root.name:
        raise ValueError("Run directory name does not match metadata run_id")
    return metadata


def _labels(frame: pl.DataFrame, labels: dict, id_column: str) -> pl.DataFrame:
    if frame.is_empty():
        renamed = frame.rename({id_column: "experiment_id"})
        return renamed.with_columns(*[
            pl.lit(None, dtype=pl.String).alias(name) for name in (
                "display_label", "feature_set_label", "estimator_label",
                "portfolio_label", "selection_role",
            )
        ])
    ids = set(frame[id_column].unique().to_list())
    unknown = ids - set(labels)
    if unknown:
        raise ValueError(f"Unregistered experiment IDs: {sorted(unknown)}")
    lookup = pl.DataFrame([
        {"experiment_id": key, **value} for key, value in labels.items() if key in ids
    ])
    renamed = frame.rename({id_column: "experiment_id"})
    return renamed.join(lookup, on="experiment_id", how="left")


def _selected_parameters(experiment: str, selection: dict) -> str:
    if experiment in selection["models"]:
        return json.dumps(
            selection["models"][experiment]["parameters"],
            sort_keys=True,
            separators=(",", ":"),
        )
    if experiment == "E5":
        return json.dumps({"prediction_source": selection["e5_source"]}, separators=(",", ":"))
    return "{}"


def _experiment_comparison(root: Path, selection: dict, labels: dict) -> pl.DataFrame:
    source = pl.read_parquet(root / "experiment_comparison.parquet")
    frame = _labels(source, labels, "experiment").with_columns(
        pl.col("experiment_id").map_elements(
            lambda value: _selected_parameters(value, selection),
            return_dtype=pl.String,
        ).alias("parameters_provenance")
    )
    columns = [
        "experiment_id", "display_label", "feature_set_label", "estimator_label",
        "portfolio_label", "selection_role", "parameters_provenance", "split",
    ] + [name for name in source.columns if name not in ("experiment", "split")]
    return frame.select(columns).sort(["split", "experiment_id"])


def _universe_tables(root: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    master = pl.read_parquet(root / "datasets/security_master.parquet")
    bars = pl.read_parquet(root / "datasets/market_bars.parquet")
    facts = pl.read_parquet(root / "datasets/fundamental_facts.parquet")
    per_market = bars.group_by("security_id").agg(
        pl.len().alias("market_sessions"),
        pl.col("session_date").min().alias("market_start"),
        pl.col("session_date").max().alias("market_end"),
    )
    per_fact = facts.group_by("security_id").agg(
        pl.len().alias("fact_records"),
        pl.col("fact_name").filter(pl.col("fact_name") == "EarningsPerShareBasic").len().alias("eps_records"),
        pl.col("fact_name").filter(pl.col("fact_name") == "NetIncomeLoss").len().alias("net_income_records"),
        pl.col("filed_date").min().alias("first_filed_date"),
        pl.col("filed_date").max().alias("last_filed_date"),
    )
    security = master.join(per_market, on="security_id", how="left").join(
        per_fact, on="security_id", how="left"
    ).sort("security_id")
    summary = pl.DataFrame([{
        "security_count": master.height,
        "market_row_count": bars.height,
        "market_start": bars["session_date"].min(),
        "market_end": bars["session_date"].max(),
        "fundamental_fact_count": facts.height,
        "eps_security_count": facts.filter(
            pl.col("fact_name") == "EarningsPerShareBasic"
        )["security_id"].n_unique(),
        "net_income_security_count": facts.filter(
            pl.col("fact_name") == "NetIncomeLoss"
        )["security_id"].n_unique(),
        "currency": "USD",
        "market_frequency": "daily sessions",
        "fundamental_frequency": "irregular SEC filings",
    }])
    return summary, security


def _security_lookup(root: Path) -> pl.DataFrame:
    master = pl.read_parquet(root / "datasets/security_master.parquet").select(
        "security_id", "ticker"
    )
    positions = pl.read_parquet(root / "positions.parquet")
    missing = set(positions["security_id"].unique()) - set(master["security_id"])
    extras = [
        {"security_id": item, "ticker": "SPY" if item == "BENCHMARK-SPY" else item}
        for item in sorted(missing)
    ]
    return pl.concat([master, pl.DataFrame(extras, schema=master.schema)]) if extras else master


def _security_summaries(root: Path, labels: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    lookup = _security_lookup(root)
    positions = pl.read_parquet(root / "positions.parquet")
    holdings = positions.group_by("experiment", "split", "security_id").agg(
        pl.len().alias("held_sessions"),
        pl.col("session_date").min().alias("first_held_date"),
        pl.col("session_date").max().alias("last_held_date"),
        pl.col("weight").mean().alias("average_weight_when_held"),
        pl.col("weight").max().alias("maximum_weight"),
        pl.col("position_value").last().alias("ending_position_value"),
    )
    trades = pl.read_parquet(root / "trades.parquet")
    trading = trades.group_by("experiment", "split", "security_id").agg(
        pl.len().alias("trade_count"),
        pl.col("trade_value").filter(pl.col("trade_value") > 0).sum().alias("buy_value"),
        (-pl.col("trade_value").filter(pl.col("trade_value") < 0).sum()).alias("sell_value"),
        pl.col("trade_value").abs().sum().alias("gross_traded_value"),
        pl.col("turnover").sum().alias("turnover_sum"),
        pl.col("cost").sum().alias("transaction_cost"),
    )
    def enrich(frame: pl.DataFrame) -> pl.DataFrame:
        return _labels(frame, labels, "experiment").join(
            lookup, on="security_id", how="left"
        ).sort(["split", "experiment_id", "security_id"])
    return enrich(holdings), enrich(trading)


def _contributions(root: Path, labels: dict) -> pl.DataFrame:
    curves = pl.read_parquet(root / "equity_curve.parquet").sort(
        ["experiment", "split", "session_date"]
    )
    positions = pl.read_parquet(root / "positions.parquet")
    trades = pl.read_parquet(root / "trades.parquet")
    bars = pl.read_parquet(root / "datasets/market_bars.parquet").select(
        "session_date", "security_id", "adjusted_close"
    )
    benchmark_path = root / "datasets/benchmark.parquet"
    if benchmark_path.is_file():
        bars = pl.concat([
            bars,
            pl.read_parquet(benchmark_path).select(
                "session_date", "security_id", "adjusted_close"
            ),
        ])
    prices = {
        (row["session_date"], row["security_id"]): row["adjusted_close"]
        for row in bars.to_dicts()
    }
    next_dates = {}
    prior_equity = {}
    expected = {}
    for group in curves.partition_by(["experiment", "split"], maintain_order=True):
        experiment, split = group["experiment"][0], group["split"][0]
        dates = group["session_date"].to_list()
        equities = group["equity"].to_list()
        returns = group["daily_return"].to_list()
        for index, day in enumerate(dates):
            prior_equity[(experiment, split, day)] = 1.0 if index == 0 else equities[index - 1]
            expected[(experiment, split, day)] = returns[index]
            if index + 1 < len(dates):
                next_dates[(experiment, split, day)] = dates[index + 1]
    totals = defaultdict(lambda: {"gross_contribution": 0.0, "cost_contribution": 0.0})
    daily = defaultdict(float)
    for row in positions.select(
        "experiment", "split", "session_date", "security_id", "position_value"
    ).to_dicts():
        source_key = (row["experiment"], row["split"], row["session_date"])
        next_day = next_dates.get(source_key)
        if next_day is None:
            continue
        old_price = prices.get((row["session_date"], row["security_id"]))
        new_price = prices.get((next_day, row["security_id"]))
        if old_price is None or new_price is None:
            raise ValueError(
                f"Cannot attribute missing price: {row['security_id']} {next_day}"
            )
        contribution = (
            row["position_value"]
            * (new_price / old_price - 1.0)
            / prior_equity[(row["experiment"], row["split"], next_day)]
        )
        key = (row["experiment"], row["split"], row["security_id"])
        totals[key]["gross_contribution"] += contribution
        daily[(row["experiment"], row["split"], next_day)] += contribution
    for row in trades.select(
        "experiment", "split", "execution_date", "security_id", "cost"
    ).to_dicts():
        contribution = -row["cost"] / prior_equity[
            (row["experiment"], row["split"], row["execution_date"])
        ]
        key = (row["experiment"], row["split"], row["security_id"])
        totals[key]["cost_contribution"] += contribution
        daily[(row["experiment"], row["split"], row["execution_date"])] += contribution
    for key, value in expected.items():
        if not math.isclose(daily[key], value, rel_tol=1e-9, abs_tol=1e-11):
            raise ValueError(
                f"Security contributions do not reconcile for {key}: {daily[key]} != {value}"
            )
    rows = []
    for (experiment, split, security_id), values in totals.items():
        rows.append({
            "experiment": experiment,
            "split": split,
            "security_id": security_id,
            **values,
            "net_contribution": values["gross_contribution"] + values["cost_contribution"],
        })
    result = pl.DataFrame(rows)
    return _labels(result, labels, "experiment").join(
        _security_lookup(root), on="security_id", how="left"
    ).sort(["split", "experiment_id", "security_id"])


def _series(root: Path, labels: dict) -> dict[str, pl.DataFrame]:
    curves = pl.read_parquet(root / "equity_curve.parquet").sort(
        ["experiment", "split", "session_date"]
    ).with_columns(
        (
            pl.col("equity")
            / pl.col("equity").cum_max().over(["experiment", "split"]).clip(lower_bound=1.0)
            - 1
        ).alias("drawdown")
    )
    equity = _labels(
        curves.select("experiment", "split", "session_date", "equity", "daily_return", "drawdown"),
        labels,
        "experiment",
    ).sort(["split", "experiment_id", "session_date"])
    costs = _labels(
        curves.select("experiment", "split", "session_date", "turnover", "cost").with_columns(
            pl.col("turnover").cum_sum().over(["experiment", "split"]).alias("cumulative_turnover"),
            pl.col("cost").cum_sum().over(["experiment", "split"]).alias("cumulative_cost"),
        ),
        labels,
        "experiment",
    ).sort(["split", "experiment_id", "session_date"])
    ic = _labels(pl.read_parquet(root / "daily_ic.parquet"), labels, "experiment").sort(
        ["split", "experiment_id", "session_date"]
    )
    return {
        "equity_drawdown_series": equity,
        "turnover_cost_series": costs,
        "ic_series": ic,
    }


def _feature_snapshot(root: Path) -> tuple[pl.DataFrame, dict]:
    manifest = _load_json(root / "dataset_manifest.json")
    path = Path(manifest["feature_path"]).resolve()
    expected = manifest["feature_sha256"]
    if not path.is_file() or digest(path) != expected:
        raise ValueError("Hash-verified feature snapshot is unavailable or differs from the run manifest")
    return pl.read_parquet(path), {
        "path": str(path), "sha256": expected, "role": "external_hash_verified_feature_snapshot"
    }


def _data_diagnostics(features: pl.DataFrame) -> dict[str, pl.DataFrame]:
    usable = features.filter(pl.col("split").is_in(["train", "validation", "test"]))
    split_profile = usable.group_by("split").agg(
        pl.len().alias("rows"), pl.col("security_id").n_unique().alias("securities"),
        pl.col("session_date").n_unique().alias("dates"),
        pl.col("session_date").min().alias("start_date"),
        pl.col("session_date").max().alias("end_date"),
        pl.col("forward_return_5d").is_not_null().sum().alias("target_available_rows"),
        pl.col("forward_return_5d").is_null().sum().alias("target_unavailable_rows"),
    ).sort("split")
    rows = []
    for split in ("train", "validation", "test"):
        frame = usable.filter(pl.col("split") == split)
        for name in list(dict.fromkeys(F0 + F1)):
            values = frame[name]
            rows.append({
                "split": split, "feature": name,
                "semantic_label": name.replace("_", " ").title(),
                "rows": frame.height, "missing_count": values.null_count(),
                "missing_fraction": values.null_count() / frame.height if frame.height else None,
                "minimum": values.min(), "q05": values.quantile(.05),
                "q25": values.quantile(.25), "median": values.median(),
                "q75": values.quantile(.75), "q95": values.quantile(.95),
                "maximum": values.max(),
            })
    feature_summary = pl.DataFrame(rows).sort(["split", "feature"])
    target_summary = usable.group_by("split").agg(
        pl.col("forward_return_5d").count().alias("available_rows"),
        pl.col("forward_return_5d").null_count().alias("missing_rows"),
        pl.col("forward_return_5d").mean().alias("mean"),
        pl.col("forward_return_5d").std().alias("standard_deviation"),
        pl.col("forward_return_5d").quantile(.05).alias("q05"),
        pl.col("forward_return_5d").quantile(.25).alias("q25"),
        pl.col("forward_return_5d").median().alias("median"),
        pl.col("forward_return_5d").quantile(.75).alias("q75"),
        pl.col("forward_return_5d").quantile(.95).alias("q95"),
    ).sort("split")
    return {"split_profile": split_profile, "feature_summary": feature_summary,
            "target_summary": target_summary}


def _prediction_diagnostics(root: Path, labels: dict) -> dict[str, pl.DataFrame]:
    predictions = pl.read_parquet(root / "predictions.parquet").filter(
        pl.col("actual_forward_return_5d").is_not_null()
    ).with_columns(
        (pl.col("predicted_return_5d") - pl.col("actual_forward_return_5d")).alias("residual"),
        (pl.col("predicted_return_5d").sign() == pl.col("actual_forward_return_5d").sign()).alias("direction_correct"),
        pl.len().over(["model_id", "split", "session_date"]).alias("cross_section_rows"),
    ).with_columns(
        (((pl.col("predicted_rank") - 1) * 10 / pl.col("cross_section_rows")).floor() + 1)
        .clip(1, 10).cast(pl.Int8).alias("prediction_decile")
    )
    residual = predictions.group_by("model_id", "split").agg(
        pl.len().alias("rows"), pl.col("residual").mean().alias("mean_residual"),
        pl.col("residual").std().alias("residual_standard_deviation"),
        pl.col("residual").quantile(.05).alias("residual_q05"),
        pl.col("residual").median().alias("residual_median"),
        pl.col("residual").quantile(.95).alias("residual_q95"),
        pl.col("residual").abs().mean().alias("mae"),
        (pl.col("residual").pow(2).mean().sqrt()).alias("rmse"),
        pl.col("direction_correct").mean().alias("directional_accuracy"),
    ).rename({"model_id": "experiment"})
    deciles = predictions.group_by("model_id", "split", "prediction_decile").agg(
        pl.len().alias("rows"),
        pl.col("predicted_return_5d").mean().alias("mean_prediction"),
        pl.col("actual_forward_return_5d").mean().alias("mean_realised_return"),
        pl.col("residual").mean().alias("mean_residual"),
        pl.col("direction_correct").mean().alias("directional_accuracy"),
    ).rename({"model_id": "experiment"})
    return {
        "prediction_diagnostics": _labels(residual, labels, "experiment").sort(["split", "experiment_id"]),
        "prediction_deciles": _labels(deciles, labels, "experiment").sort(["split", "experiment_id", "prediction_decile"]),
    }


def _feature_importance(root: Path, labels: dict) -> pl.DataFrame:
    value = pl.read_parquet(root / "feature_importance.parquet").rename({"model_id": "experiment"})
    return _labels(value, labels, "experiment").with_columns(
        pl.col("feature").str.replace_all("_", " ").str.to_titlecase().alias("semantic_label")
    ).sort(["experiment_id", "importance"], descending=[False, True])


def _benchmark_relative(root: Path, labels: dict) -> dict[str, pl.DataFrame]:
    curves = pl.read_parquet(root / "equity_curve.parquet").sort(["split", "experiment", "session_date"])
    rows, metric_rows = [], []
    for split in curves["split"].unique().sort().to_list():
        benchmark = curves.filter((pl.col("split") == split) & (pl.col("experiment") == "B0")).select(
            "session_date", pl.col("daily_return").alias("benchmark_return"),
            pl.col("equity").alias("benchmark_equity")
        )
        if benchmark.is_empty():
            continue
        for experiment in curves.filter(pl.col("split") == split)["experiment"].unique().sort().to_list():
            if experiment == "B0":
                continue
            joined = curves.filter(
                (pl.col("split") == split) & (pl.col("experiment") == experiment)
            ).join(benchmark, on="session_date", how="inner").with_columns(
                (pl.col("daily_return") - pl.col("benchmark_return")).alias("active_return"),
                (pl.col("equity") / pl.col("benchmark_equity") - 1).alias("relative_return"),
            ).with_columns(
                (pl.col("active_return").rolling_std(63) * math.sqrt(252)).alias("rolling_63d_tracking_error")
            )
            active, strategy, bench = joined["active_return"], joined["daily_return"], joined["benchmark_return"]
            te = active.std() * math.sqrt(252) if active.len() > 1 else None
            strategy_values, benchmark_values = strategy.to_numpy(), bench.to_numpy()
            benchmark_variance = float(np.var(benchmark_values, ddof=1)) if len(benchmark_values) > 1 else None
            beta = (
                float(np.cov(strategy_values, benchmark_values, ddof=1)[0, 1]) / benchmark_variance
                if benchmark_variance not in (None, 0) else None
            )
            downside = joined.filter(pl.col("benchmark_return") < 0)
            down_bench = downside["benchmark_return"].mean() if downside.height else None
            metric_rows.append({
                "split": split, "experiment": experiment,
                "annualised_active_return": active.mean() * 252,
                "tracking_error": te,
                "information_ratio": active.mean() * 252 / te if te else None,
                "beta": beta,
                "downside_capture": downside["daily_return"].mean() / down_bench if down_bench else None,
                "sessions": joined.height,
            })
            rows.append(joined.select(
                pl.lit(experiment).alias("experiment"), "split", "session_date",
                pl.col("daily_return").alias("strategy_return"), "benchmark_return",
                "active_return", "relative_return", "rolling_63d_tracking_error"
            ))
    series = pl.concat(rows) if rows else pl.DataFrame(schema={
        "experiment": pl.String, "split": pl.String, "session_date": pl.Date,
        "strategy_return": pl.Float64, "benchmark_return": pl.Float64,
        "active_return": pl.Float64, "relative_return": pl.Float64,
        "rolling_63d_tracking_error": pl.Float64,
    })
    metrics = pl.DataFrame(metric_rows) if metric_rows else pl.DataFrame(schema={
        "split": pl.String, "experiment": pl.String,
        "annualised_active_return": pl.Float64, "tracking_error": pl.Float64,
        "information_ratio": pl.Float64, "beta": pl.Float64,
        "downside_capture": pl.Float64, "sessions": pl.Int64,
    })
    return {
        "benchmark_relative_series": _labels(series, labels, "experiment").sort(["split", "experiment_id", "session_date"]),
        "benchmark_relative_metrics": _labels(metrics, labels, "experiment").sort(["split", "experiment_id"]),
    }


def _training_tables(root: Path) -> dict[str, pl.DataFrame]:
    trace_path, summary_path = root / "training_trace.parquet", root / "training_summary.parquet"
    trace = pl.read_parquet(trace_path) if trace_path.is_file() else pl.DataFrame(schema=TRACE_SCHEMA)
    summary = pl.read_parquet(summary_path) if summary_path.is_file() else pl.DataFrame(schema=SUMMARY_SCHEMA)
    availability = pl.DataFrame([
        {"model_family": "Elastic Net", "state": "not recorded for this run",
         "reason": "The immutable Slice 1 run predates training telemetry; Elastic Net has iterations, not epochs."},
        {"model_family": "Histogram GBT", "state": "not recorded for this run",
         "reason": "The immutable Slice 1 run predates staged training telemetry."},
        {"model_family": "XGBoost", "state": "not recorded for this run",
         "reason": "The immutable Slice 1 run predates the separately registered XGBoost family."},
    ]) if summary.is_empty() else pl.DataFrame([
        {
            "model_family": family,
            "state": "summary recorded; no conventional learning curve" if family == "Elastic Net" else "recorded",
            "reason": (
                "Elastic Net exposes solver iterations and final diagnostics, not epoch/boosting stages; no curve is fabricated."
                if family == "Elastic Net" else None
            ),
        }
        for family in summary["model_family"].unique().sort().to_list()
    ])
    benchmark_path = root / "device_benchmark.parquet"
    device_benchmark = pl.read_parquet(benchmark_path) if benchmark_path.is_file() else pl.DataFrame(schema={
        "experiment_id": pl.String, "requested_device": pl.String,
        "actual_device": pl.String, "status": pl.String,
        "fallback_reason": pl.String, "duration_seconds": pl.Float64,
        "validation_rmse": pl.Float64, "selected_iteration": pl.Int64,
        "max_abs_prediction_delta_vs_cpu": pl.Float64,
        "research_status": pl.String,
    })
    return {"training_trace": trace, "training_summary": summary,
            "training_availability": availability,
            "device_benchmark": device_benchmark}


def _optional_research_tables(root: Path) -> tuple[dict[str, pl.DataFrame], list[dict]]:
    """Carry forward only audited, manifest-bound research evidence."""
    present = [name for name in OPTIONAL_RESEARCH_INPUTS if (root / name).is_file()]
    if not present:
        return {}, []
    audit_path = root / "audit.json"
    manifest_path = root / "research_evidence_manifest.json"
    if not audit_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Optional research evidence requires audit.json and research_evidence_manifest.json"
        )
    audit = _load_json(audit_path)
    manifest = _load_json(manifest_path)
    if not audit.get("passed") or manifest.get("schema_version") != 1:
        raise ValueError("Optional research evidence must be audited under manifest schema v1")
    declared = {
        item.get("relative_path"): item.get("sha256")
        for item in manifest.get("files", []) if isinstance(item, dict)
    }
    for name in present:
        if declared.get(name) != digest(root / name):
            raise ValueError(f"Optional research evidence hash mismatch or undeclared: {name}")
    dependencies = [
        {"relative_path": "audit.json", "sha256": digest(audit_path),
         "role": "research_evidence_audit"},
        {"relative_path": "research_evidence_manifest.json", "sha256": digest(manifest_path),
         "role": "research_evidence_manifest"},
    ] + [
        {"relative_path": name, "sha256": digest(root / name),
         "role": "completed_research_evidence"}
        for name in present
    ]
    tables = {}
    for name in present:
        frame = pl.read_parquet(root / name)
        missing_columns = OPTIONAL_RESEARCH_COLUMNS[name] - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"Optional research table schema invalid for {name}; missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )
        tables[Path(name).stem] = frame
    return tables, dependencies


def _rl_tables(rl_root: Path | None) -> tuple[dict[str, pl.DataFrame], list[dict]]:
    names = ("actions", "equity_curve", "metrics", "policy_summary", "action_frequencies",
             "training_trace", "training_summary", "device_benchmark")
    if rl_root is None:
        return {}, []
    metadata = _load_json(rl_root / "metadata.json")
    audit = _load_json(rl_root / "audit.json")
    if metadata.get("status") != "complete" or not audit.get("passed"):
        raise ValueError("RL report input must be completed and audited")
    tables = {f"rl_{name}": pl.read_parquet(rl_root / f"{name}.parquet") for name in names}
    inputs = [{"path": str(rl_root / f"{name}.parquet"), "sha256": digest(rl_root / f"{name}.parquet"),
               "role": "audited_rl_evidence"} for name in names]
    inputs += [{"path": str(rl_root / name), "sha256": digest(rl_root / name), "role": "audited_rl_contract"}
               for name in ("metadata.json", "audit.json")]
    return tables, inputs


def _field_definitions(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    precise = {
        "residual": "Predicted five-session return minus realised five-session return",
        "prediction_decile": "Within-date predicted-rank bucket; 1 is the highest predicted-return decile",
        "directional_accuracy": METRICS["directional_accuracy"][0],
        "active_return": METRICS["active_return"][0],
        "relative_return": "Strategy equity divided by SPY equity minus one",
        "rolling_63d_tracking_error": "63-session sample standard deviation of active return times sqrt(252)",
        "tracking_error": METRICS["tracking_error"][0],
        "information_ratio": METRICS["information_ratio"][0],
        "beta": METRICS["beta"][0],
        "downside_capture": METRICS["downside_capture"][0],
        "metric_value": "Recorded value emitted by the named training metric; unavailable values are omitted",
        "peak_gpu_memory_bytes": "Peak allocated PyTorch CUDA memory during this fit, when CUDA executed",
        "selected_iteration": "One-based best boosting round selected using validation evidence only",
        "max_abs_prediction_delta_vs_cpu": "Maximum absolute validation-prediction difference from the paired CPU diagnostic fit",
    }
    rows = []
    base_tables = (
        "split_profile", "feature_summary", "target_summary", "prediction_diagnostics",
        "prediction_deciles", "feature_importance", "benchmark_relative_series",
        "benchmark_relative_metrics", "training_trace", "training_summary", "device_benchmark",
    )
    optional_tables = tuple(
        Path(name).stem for name in OPTIONAL_RESEARCH_INPUTS if Path(name).stem in tables
    )
    for table in base_tables + optional_tables:
        for field in tables[table].columns:
            unit = "identifier/text"
            if any(token in field for token in ("return", "accuracy", "missing_fraction", "drawdown")):
                unit = "decimal fraction"
            elif field.endswith("_seconds"):
                unit = "seconds"
            elif field.endswith("_bytes"):
                unit = "bytes"
            elif field in {"rows", "dates", "securities", "step", "epoch", "iterations", "epochs"} or field.endswith("_rows"):
                unit = "count"
            elif tables[table].schema[field].is_numeric():
                unit = "numeric; see definition"
            rows.append({
                "table": table, "field": field,
                "definition": precise.get(field, field.replace("_", " ").capitalize()),
                "unit": unit,
                "limitations": (
                    "Descriptive diagnostic; the observed final test cannot be used for model selection"
                    if table in {"prediction_diagnostics", "prediction_deciles", "benchmark_relative_series", "benchmark_relative_metrics"}
                    else "Unavailable source values remain null and are never estimated"
                ),
                "provenance": (
                    "Hash-verified external feature snapshot named by dataset_manifest.json"
                    if table in {"split_profile", "feature_summary", "target_summary"}
                    else "Immutable source run artefacts and deterministic report code"
                ),
            })
    return pl.DataFrame(rows).sort(["table", "field"])


def build_report(root: str | Path, rl_run: str | Path | None = None) -> tuple[dict[str, pl.DataFrame], str, dict]:
    """Derive deterministic tables and Markdown without writing to the source run."""
    root = Path(root).resolve()
    metadata = _validate_run(root)
    selection = _load_json(root / "selection.json")
    labels = registry(selection)
    universe, universe_security = _universe_tables(root)
    holdings, trades = _security_summaries(root, labels)
    features, feature_dependency = _feature_snapshot(root)
    rl_tables, rl_inputs = _rl_tables(Path(rl_run).resolve() if rl_run else None)
    optional_research, optional_research_inputs = _optional_research_tables(root)
    tables = {
        "experiment_comparison": _experiment_comparison(root, selection, labels),
        "metric_definitions": pl.DataFrame([
            {"metric": name, "definition": definition, "unit": unit}
            for name, (definition, unit) in METRICS.items()
        ]),
        "universe_summary": universe,
        "universe_security_summary": universe_security,
        "security_holdings_summary": holdings,
        "security_trades_summary": trades,
        "security_contribution_summary": _contributions(root, labels),
        "feature_importance": _feature_importance(root, labels),
        "deferred_fields": pl.DataFrame([
            {
                "field": name,
                "status": "blocked/deferred",
                "reason": "No approved point-in-time-safe source and policy",
            }
            for name in ("industry", "historical_market_cap", "price_to_earnings")
        ]),
        **_series(root, labels),
        **_data_diagnostics(features),
        **_prediction_diagnostics(root, labels),
        **_benchmark_relative(root, labels),
        **_training_tables(root),
        **optional_research,
        **rl_tables,
    }
    tables["field_definitions"] = _field_definitions(tables)
    report = _markdown(metadata, selection, tables)
    provenance = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "source_run_id": metadata["run_id"],
        "source_run_research_status": metadata.get("research_status", "not_recorded"),
        "source_run_path": str(root),
        "source_run_git_commit": metadata.get("git_commit"),
        "inputs": [
            {"relative_path": name, "sha256": digest(root / name)}
            for name in REQUIRED
        ] + (
            [{"relative_path": "datasets/benchmark.parquet", "sha256": digest(root / "datasets/benchmark.parquet")}]
            if (root / "datasets/benchmark.parquet").is_file() else []
        ) + [
            {"relative_path": name, "sha256": digest(root / name)}
            for name in OPTIONAL_REPORT_INPUTS if (root / name).is_file()
        ] + [feature_dependency] + optional_research_inputs + rl_inputs,
    }
    return tables, report, provenance


def _format(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def _markdown(metadata: dict, selection: dict, tables: dict[str, pl.DataFrame]) -> str:
    universe = tables["universe_summary"].row(0, named=True)
    comparison = tables["experiment_comparison"]
    lines = [
        "# Versioned evidence report",
        "",
        f"Source run: `{metadata['run_id']}`. Report schema: v{REPORT_SCHEMA_VERSION}.",
        "",
        "This report is deterministically derived from immutable run artefacts. Final-test results are descriptive and were not used for selection.",
        "",
        "## Universe",
        "",
        "| Securities | Market rows | Market period | SEC fact rows | EPS coverage | Net Income coverage |",
        "|---:|---:|---|---:|---:|---:|",
        (
            f"| {universe['security_count']} | {universe['market_row_count']} | "
            f"{universe['market_start']} to {universe['market_end']} | "
            f"{universe['fundamental_fact_count']} | {universe['eps_security_count']} | "
            f"{universe['net_income_security_count']} |"
        ),
        "",
        "## Experiment comparison",
        "",
        f"E5 reuses the frozen predictions of `{selection['e5_source']}`, chosen on validation data.",
        "",
        "| Split | ID | Semantic label | Total return | Sharpe | Max drawdown | Mean IC | RMSE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.iter_rows(named=True):
        lines.append(
            f"| {row['split']} | {row['experiment_id']} | {row['display_label']} | "
            f"{_format(row['total_return'])} | {_format(row['sharpe'])} | "
            f"{_format(row['maximum_drawdown'])} | {_format(row['mean_ic'])} | "
            f"{_format(row['rmse'])} |"
        )
    lines += [
        "",
        "## Generated tables and series",
        "",
        "- `experiment_comparison`: model, feature, portfolio and split metrics with immutable IDs and semantic labels.",
        "- `universe_summary` and `universe_security_summary`: source coverage without unavailable classifications.",
        "- `security_holdings_summary`, `security_trades_summary` and `security_contribution_summary`: defensible security-level aggregates. Net contribution is prior-day holding return less execution-day cost, reconciled to daily portfolio returns.",
        "- `equity_drawdown_series`, `turnover_cost_series` and `ic_series`: chart-ready time series.",
        "- `split_profile`, `feature_summary` and `target_summary`: hash-verified train/validation/test coverage and distributions.",
        "- `prediction_diagnostics`, `prediction_deciles` and `feature_importance`: error, ranking behavior and semantic model diagnostics.",
        "- `benchmark_relative_series` and `benchmark_relative_metrics`: same-session SPY-relative evidence with explicitly defined risk measures.",
        "- `training_trace`, `training_summary` and `training_availability`: recorded telemetry or explicit not-recorded states; no history is reconstructed.",
        "- `field_definitions`: definitions, units, limitations and provenance for every v2 diagnostic field.",
        "",
        "## Metric definitions and units",
        "",
        "| Metric | Definition | Unit |",
        "|---|---|---|",
    ]
    lines += [f"| `{name}` | {definition} | {unit} |" for name, (definition, unit) in METRICS.items()]
    lines += [
        "",
        "## Blocked or deferred fields",
        "",
        "Industry, historical market capitalization, P/E and other valuation/classification fields are not present. They remain blocked pending an approved point-in-time-safe source and policy; no values are invented or inferred.",
        "",
        "## Interpretation limits",
        "",
        "The fixed current-survivor universe, retrospective price adjustments, limited SEC concepts, one chronological split, fixed cost model and T-to-T+5 label versus T+1 execution gap constrain inference. This report does not establish persistent alpha, production readiness or investment advice.",
        "",
        "See `provenance.json` for source hashes, source/code revisions and generation time.",
        "",
    ]
    return "\n".join(lines)


def _code_revision() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def generate_report(root: str | Path, output: str | Path, rl_run: str | Path | None = None) -> Path:
    """Create a new versioned output directory; never mutate the source run."""
    source = Path(root).resolve()
    output = Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Report output must be outside the immutable source run")
    if output.exists():
        raise FileExistsError(f"Report output already exists: {output}")
    tables, report, provenance = build_report(source, rl_run=rl_run)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, frame in tables.items():
            frame.write_parquet(output / f"{name}.parquet")
            frame.write_csv(output / f"{name}.csv")
        (output / "report.md").write_text(report, encoding="utf-8")
        commit, dirty = _code_revision()
        output_files = sorted(
            path for path in output.iterdir() if path.name != "provenance.json"
        )
        provenance.update({
            "report_code_revision": commit,
            "report_code_dirty": dirty,
            "reporting_source_sha256": {
                path.as_posix(): digest(path)
                for path in sorted(Path("src/trading_pipeline/reporting").rglob("*.py"))
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "outputs": [
                {
                    "relative_path": path.name,
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                }
                for path in output_files
            ],
        })
        (output / "provenance.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    except Exception:
        # Leave the new directory as failure evidence; source run remains untouched.
        raise
    return output
