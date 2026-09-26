"""Strict read-only schema loaders; no training or report regeneration."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import polars as pl

REPORT_TABLES_V1 = (
    "experiment_comparison", "equity_drawdown_series", "turnover_cost_series",
    "ic_series", "security_holdings_summary", "security_trades_summary",
    "security_contribution_summary",
)
REPORT_TABLES = REPORT_TABLES_V1 + (
    "split_profile", "feature_summary", "target_summary", "prediction_diagnostics",
    "prediction_deciles", "feature_importance", "benchmark_relative_series",
    "benchmark_relative_metrics", "training_trace", "training_summary",
    "training_availability", "metric_definitions", "universe_summary",
    "field_definitions",
)
OPTIONAL_REPORT_TABLES = (
    "pipeline_stage_summary", "pipeline_security_summary",
    "architecture_trial_summary", "hpo_trial_summary", "risk_scenario_summary",
    "model_conditioned_frontier_points", "model_conditioned_frontier_weights",
    "realised_risk_return_curve", "final_testbench_metrics", "final_testbench_equity_curve",
    "question_and_assumption_register",
)
RL_TABLES = ("policy_summary", "metrics", "equity_curve", "actions", "action_frequencies")
RL_TELEMETRY_TABLES = ("training_trace", "training_summary", "device_benchmark")


def _tables(root: Path, names: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    missing = [f"{name}.parquet" for name in names if not (root / f"{name}.parquet").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing dashboard inputs: {', '.join(missing)}")
    return {name: pl.read_parquet(root / f"{name}.parquet") for name in names}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_generated_outputs(root: Path, provenance: dict) -> None:
    """Verify new reports while keeping legacy output-manifest-free reports readable."""
    for item in provenance.get("outputs", []):
        path = root / item["relative_path"]
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            raise ValueError(f"Generated report checksum mismatch: {item['relative_path']}")


def load_report(path: str | Path) -> dict:
    root = Path(path).resolve()
    provenance_path = root / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("Generated report provenance.json is required")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    version = provenance.get("report_schema_version")
    if version not in {1, 2}:
        raise ValueError("Unsupported report schema version")
    names = REPORT_TABLES if version == 2 else REPORT_TABLES_V1
    _verify_generated_outputs(root, provenance)
    tables = _tables(root, names)
    declared_outputs = {
        item.get("relative_path"): item.get("sha256")
        for item in provenance.get("outputs", [])
    }
    for name in OPTIONAL_REPORT_TABLES:
        path = root / f"{name}.parquet"
        if path.is_file():
            if declared_outputs.get(path.name) != _sha256(path):
                raise ValueError(
                    f"Optional report table is not hash-declared: {path.name}"
                )
            tables[name] = pl.read_parquet(path)
    benchmark = root / "device_benchmark.parquet"
    if benchmark.is_file():
        tables["device_benchmark"] = pl.read_parquet(benchmark)
    return {"root": root, "provenance": provenance, "tables": tables}


def load_rl_run(path: str | Path) -> dict:
    root = Path(path).resolve()
    metadata_path, audit_path = root / "metadata.json", root / "audit.json"
    if not metadata_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("RL metadata.json and audit.json are required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or not audit.get("passed"):
        raise ValueError("Dashboard only reads completed, audited RL runs")
    if metadata.get("research_status") not in {
        "exploratory_not_confirmatory", "diagnostic_reproduction_not_model_selection"
    }:
        raise ValueError("RL run lacks the required non-confirmatory status")
    tables = _tables(root, RL_TABLES)
    for name in RL_TELEMETRY_TABLES:
        path = root / f"{name}.parquet"
        if path.is_file():
            tables[name] = pl.read_parquet(path)
    return {"root": root, "metadata": metadata, "audit": audit, "tables": tables}


def load_catalog(path: str | Path) -> dict:
    """Load a generated multi-run catalogue without touching source runs."""
    root = Path(path).resolve()
    provenance_path = root / "provenance.json"
    catalog_path = root / "run_catalog.parquet"
    if not provenance_path.is_file() or not catalog_path.is_file():
        raise FileNotFoundError("Catalogue provenance and run_catalog.parquet are required")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema_version") != 1:
        raise ValueError("Unsupported catalogue schema version")
    if _sha256(catalog_path) != provenance.get("run_catalog_sha256"):
        raise ValueError("Catalogue checksum mismatch: run_catalog.parquet")
    tables = {"runs": pl.read_parquet(catalog_path)}
    metrics = root / "metric_catalog.parquet"
    if metrics.is_file():
        if _sha256(metrics) != provenance.get("metric_catalog_sha256"):
            raise ValueError("Catalogue checksum mismatch: metric_catalog.parquet")
        tables["metrics"] = pl.read_parquet(metrics)
    return {"root": root, "provenance": provenance, "tables": tables}
