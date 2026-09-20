"""Schema-versioned, model-agnostic training telemetry contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import polars as pl

TELEMETRY_SCHEMA_VERSION = 1

TRACE_SCHEMA = {
    "schema_version": pl.Int32,
    "run_id": pl.String,
    "experiment_id": pl.String,
    "policy_id": pl.String,
    "model_family": pl.String,
    "seed": pl.Int64,
    "requested_device": pl.String,
    "actual_device": pl.String,
    "phase": pl.String,
    "step": pl.Int64,
    "epoch": pl.Int64,
    "metric_name": pl.String,
    "metric_value": pl.Float64,
    "elapsed_seconds": pl.Float64,
    "learning_rate": pl.Float64,
    "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
}

SUMMARY_SCHEMA = {
    "schema_version": pl.Int32,
    "run_id": pl.String,
    "experiment_id": pl.String,
    "policy_id": pl.String,
    "model_family": pl.String,
    "seed": pl.Int64,
    "requested_device": pl.String,
    "actual_device": pl.String,
    "fallback_reason": pl.String,
    "duration_seconds": pl.Float64,
    "data_rows": pl.Int64,
    "data_columns": pl.Int64,
    "iterations": pl.Int64,
    "epochs": pl.Int64,
    "stopping_reason": pl.String,
    "peak_gpu_memory_bytes": pl.Int64,
    "package_versions_json": pl.String,
    "cuda_versions_json": pl.String,
    "determinism_json": pl.String,
    "started_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "completed_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def frame(rows: Iterable[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Build a stable nullable frame, including a correctly typed empty frame."""
    return pl.DataFrame(list(rows), schema=schema, strict=False)


def validate_trace(value: pl.DataFrame) -> None:
    if value.schema != TRACE_SCHEMA:
        raise ValueError(f"Training trace schema mismatch: {value.schema}")
    if value.filter(pl.col("metric_value").is_null() | ~pl.col("metric_value").is_finite()).height:
        raise ValueError("Training trace metric values must be finite and recorded, not estimated")
    if value.filter((pl.col("step") < 0) | (pl.col("elapsed_seconds") < 0)).height:
        raise ValueError("Training trace steps and elapsed times must be non-negative")


def validate_summary(value: pl.DataFrame) -> None:
    if value.schema != SUMMARY_SCHEMA:
        raise ValueError(f"Training summary schema mismatch: {value.schema}")
    if value.filter(pl.col("duration_seconds") < 0).height:
        raise ValueError("Training duration must be non-negative")
    if value.filter(pl.col("actual_device").is_null()).height:
        raise ValueError("Every training summary must record an actual device")


def write_training_telemetry(
    root: Path, trace_rows: Iterable[dict[str, Any]], summary_rows: Iterable[dict[str, Any]]
) -> tuple[Path, Path]:
    trace = frame(trace_rows, TRACE_SCHEMA)
    summary = frame(summary_rows, SUMMARY_SCHEMA)
    validate_trace(trace)
    validate_summary(summary)
    trace_path, summary_path = root / "training_trace.parquet", root / "training_summary.parquet"
    trace.write_parquet(trace_path)
    summary.write_parquet(summary_path)
    return trace_path, summary_path
