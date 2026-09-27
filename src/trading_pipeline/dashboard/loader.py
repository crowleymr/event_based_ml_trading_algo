"""Strict read-only schema loaders; no training or report regeneration."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json
import polars as pl
from trading_pipeline.reporting.generate import OPTIONAL_RESEARCH_COLUMNS

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
EXPANDED_REPORT_TABLES = tuple(
    name.removesuffix(".parquet") for name in OPTIONAL_RESEARCH_COLUMNS
)
EXPANDED_VIEW_COLUMNS = {
    "final_testbench_metrics": {"model_id", "risk_scenario"},
    "final_testbench_equity_curve": {"model_id", "risk_scenario"},
}


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
            frame = pl.read_parquet(path)
            missing_columns = OPTIONAL_RESEARCH_COLUMNS[f"{name}.parquet"] - set(frame.columns)
            if missing_columns:
                raise ValueError(
                    f"Optional report table schema invalid for {path.name}; missing columns: "
                    f"{', '.join(sorted(missing_columns))}"
                )
            tables[name] = frame
    benchmark = root / "device_benchmark.parquet"
    if benchmark.is_file():
        tables["device_benchmark"] = pl.read_parquet(benchmark)
    return {"root": root, "provenance": provenance, "tables": tables}


def load_expanded_report(path: str | Path) -> dict:
    """Read the complete, hash-sealed WP7 report without opening source runs."""
    root = Path(path).resolve()
    manifest_path = root / "research_evidence_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Expanded research_evidence_manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported expanded evidence schema version")
    run_id = manifest.get("source_run_id")
    if not isinstance(run_id, str) or not run_id or root.name != run_id:
        raise ValueError("Expanded report directory and source run ID disagree")
    if manifest.get("source_run_path") != f"runs/expanded_closeout/{run_id}":
        raise ValueError("Expanded report source run path is invalid")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(EXPANDED_REPORT_TABLES):
        raise ValueError("Expanded report requires all eleven declared outputs")
    declared = {item.get("relative_path"): item for item in outputs if isinstance(item, dict)}
    expected = {f"{name}.parquet" for name in EXPANDED_REPORT_TABLES}
    if set(declared) != expected or len(declared) != len(outputs):
        raise ValueError("Expanded report output inventory is invalid")
    tables = {}
    for name in EXPANDED_REPORT_TABLES:
        filename = f"{name}.parquet"
        source = root / filename
        if not source.is_file() or _sha256(source) != declared[filename].get("sha256"):
            raise ValueError(f"Expanded report checksum mismatch: {filename}")
        frame = pl.read_parquet(source)
        missing = (OPTIONAL_RESEARCH_COLUMNS[filename] |
                   EXPANDED_VIEW_COLUMNS.get(name, set())) - set(frame.columns)
        if frame.is_empty() or missing or frame.height != declared[filename].get("rows"):
            raise ValueError(f"Expanded report table schema or row count invalid: {filename}")
        tables[name] = frame
    contract = manifest.get("comparison_contract")
    if contract is not None:
        if not isinstance(contract, dict) or contract.get("metric_definition_id") != "expanded_closeout_weekly_after_cost_v1":
            raise ValueError("Expanded comparison contract is invalid")
        dates = sorted({str(day) for day in tables["final_testbench_equity_curve"]["session_date"].to_list()})
        digest = hashlib.sha256(json.dumps(dates, separators=(",", ":")).encode("utf-8")).hexdigest()
        if contract.get("calendar_sha256") != digest:
            raise ValueError("Expanded comparison calendar does not match evidence")
    return {"root": root, "provenance": manifest, "tables": tables}


def repository_root() -> Path:
    """Resolve the checkout from the operator's cwd or this installed source tree."""
    for start in (Path.cwd(), Path(__file__).resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "trading_pipeline").is_dir():
                return candidate
    raise FileNotFoundError("Repository root not found")


def discover_expanded_reports(root: str | Path) -> list[dict]:
    """Return only complete, locally bound, untampered reports, newest first."""
    project = Path(root).resolve()
    available = []
    for candidate in (project / "reports" / "expanded_closeout").glob("*"):
        if not candidate.is_dir():
            continue
        try:
            report = load_expanded_report(candidate)
            provenance = report["provenance"]
            run = project / provenance["source_run_path"]
            metadata_path = run / "metadata.json"
            protocol_path = run / "protocol.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            audit_path = run / "audit.json"
            completion_path = run / "completion.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            inventory = audit.get("artefact_sha256", {})
            contract = provenance.get("comparison_contract")
            if (metadata.get("status") != "complete" or audit.get("status") != "passed"
                    or completion.get("status") != "complete"
                    or metadata.get("run_id") != candidate.name
                    or completion.get("run_id") != candidate.name
                    or metadata.get("protocol_sha256") != provenance.get("protocol_sha256")
                    or completion.get("audit_sha256") != _sha256(audit_path)
                    or provenance.get("source_run_audit_sha256") != _sha256(audit_path)
                    or provenance.get("source_run_completion_sha256") != _sha256(completion_path)
                    or inventory.get("metadata.json") != _sha256(metadata_path)
                    or inventory.get("protocol.json") != _sha256(protocol_path)
                    or provenance.get("source_completed_at_utc", completion.get("completed_at"))
                       != completion.get("completed_at")):
                continue
            if contract and (contract.get("execution") != protocol.get("portfolio", {}).get("execution")
                             or contract.get("cost_bps_one_way") != protocol.get("portfolio", {}).get("cost_bps_one_way")
                             or contract.get("benchmark_symbol") != protocol.get("inference", {}).get("benchmark")
                             or provenance.get("protocol_sha256") != protocol.get("authority", {}).get("protocol_sha256")):
                continue
            stamp = provenance.get("source_completed_at_utc") or completion.get("completed_at")
            timestamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                continue
            available.append((timestamp, candidate.name, report))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    available.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in available]


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
