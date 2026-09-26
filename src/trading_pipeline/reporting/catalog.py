"""Read-only catalogue of completed, audited immutable runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import polars as pl


def _bytes_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _canonical_sha256(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _candidate_roots(roots: Iterable[str | Path]) -> list[Path]:
    candidates: set[Path] = set()
    for item in roots:
        root = Path(item)
        if (root / "metadata.json").is_file():
            candidates.add(root.resolve())
        elif root.is_dir():
            candidates.update(path.parent.resolve() for path in root.rglob("metadata.json"))
    return sorted(candidates, key=lambda value: value.as_posix())


def _catalog_row(root: Path) -> dict | None:
    metadata_path, audit_path = root / "metadata.json", root / "audit.json"
    if not audit_path.is_file():
        return None
    metadata, audit = _json(metadata_path), _json(audit_path)
    if metadata.get("status") != "complete" or audit.get("passed") is not True:
        return None
    dataset_path, split_path = root / "dataset_manifest.json", root / "split_manifest.json"
    dataset = _json(dataset_path) if dataset_path.is_file() else {}
    splits = _json(split_path) if split_path.is_file() else metadata.get("split_dates", {})
    selection_path = root / "selection.json"
    selection = _json(selection_path) if selection_path.is_file() else {}
    config_path = root / "config.yaml"
    experiments = metadata.get("experiments", {})
    if isinstance(experiments, dict):
        experiment_ids = sorted(experiments)
    else:
        experiment_ids = []
    compatibility = {
        "dataset_feature_sha256": dataset.get("feature_sha256"),
        "label_horizon": metadata.get("label_horizon"),
        "cost_bps_one_way": metadata.get("cost_bps_one_way"),
        "execution": metadata.get("execution"),
        "splits": splits,
        "selection_criterion": selection.get("criterion"),
    }
    return {
        "run_id": str(metadata.get("run_id", root.name)),
        "run_path": str(root),
        "timestamp": metadata.get("timestamp"),
        "research_status": metadata.get("research_status", "not_recorded"),
        "mode": metadata.get("mode", "not_recorded"),
        "git_commit": metadata.get("git_commit"),
        "git_dirty": metadata.get("git_dirty"),
        "config_sha256": _bytes_sha256(config_path),
        "metadata_sha256": _bytes_sha256(metadata_path),
        "audit_sha256": _bytes_sha256(audit_path),
        "dataset_manifest_sha256": _bytes_sha256(dataset_path),
        "split_manifest_sha256": _bytes_sha256(split_path),
        "compatibility_key": _canonical_sha256(compatibility),
        "compatibility_json": json.dumps(compatibility, sort_keys=True, default=str),
        "experiment_ids_json": json.dumps(experiment_ids),
        "audit_check_count": audit.get("check_count"),
    }


def build_run_catalog(roots: Iterable[str | Path]) -> pl.DataFrame:
    """Index completed audited runs without writing to or modifying source roots."""
    rows = [row for root in _candidate_roots(roots) if (row := _catalog_row(root))]
    if not rows:
        return pl.DataFrame(schema={
            "run_id": pl.String, "run_path": pl.String, "timestamp": pl.String,
            "research_status": pl.String, "mode": pl.String, "git_commit": pl.String,
            "git_dirty": pl.Boolean, "config_sha256": pl.String,
            "metadata_sha256": pl.String, "audit_sha256": pl.String,
            "dataset_manifest_sha256": pl.String, "split_manifest_sha256": pl.String,
            "compatibility_key": pl.String, "compatibility_json": pl.String,
            "experiment_ids_json": pl.String, "audit_check_count": pl.Int64,
        })
    return pl.DataFrame(rows).sort(["timestamp", "run_id"], nulls_last=True)


def build_metric_catalog(roots: Iterable[str | Path]) -> pl.DataFrame:
    """Combine persisted comparison rows with run/protocol identity."""
    catalog = build_run_catalog(roots)
    rows = []
    for item in catalog.iter_rows(named=True):
        path = Path(item["run_path"]) / "experiment_comparison.parquet"
        if not path.is_file():
            continue
        comparison = pl.read_parquet(path)
        if "experiment" in comparison.columns and "experiment_id" not in comparison.columns:
            comparison = comparison.rename({"experiment": "experiment_id"})
        comparison = comparison.with_columns(
            pl.lit(item["run_id"]).alias("run_id"),
            pl.lit(item["timestamp"]).alias("run_timestamp"),
            pl.lit(item["config_sha256"]).alias("config_sha256"),
            pl.lit(item["compatibility_key"]).alias("compatibility_key"),
            pl.lit(item["research_status"]).alias("research_status"),
        )
        rows.append(comparison)
    return pl.concat(rows, how="diagonal_relaxed") if rows else pl.DataFrame(schema={
        "run_id": pl.String, "run_timestamp": pl.String,
        "config_sha256": pl.String, "compatibility_key": pl.String,
        "research_status": pl.String, "experiment_id": pl.String, "split": pl.String,
    })


def write_run_catalog(
    catalog: pl.DataFrame, output: str | Path, metrics: pl.DataFrame | None = None
) -> Path:
    """Write a new versioned catalogue and provenance; never overwrite."""
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite catalogue output: {root}")
    root.mkdir(parents=True)
    table_path = root / "run_catalog.parquet"
    catalog.write_parquet(table_path)
    metric_path = root / "metric_catalog.parquet"
    if metrics is not None:
        metrics.write_parquet(metric_path)
    provenance = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": catalog.height,
        "run_catalog_sha256": _bytes_sha256(table_path),
        "metric_catalog_sha256": _bytes_sha256(metric_path),
        "source_metadata_sha256": sorted(
            value for value in catalog.get_column("metadata_sha256").drop_nulls().to_list()
        ) if "metadata_sha256" in catalog.columns else [],
    }
    (root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root
