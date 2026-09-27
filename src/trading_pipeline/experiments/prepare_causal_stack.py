"""Materialise the fixed causal helper layer for an expanded study vintage.

This is feature preparation, not model comparison: it uses the protocol-declared
fixed base layer, emits no performance metric and never fits on protected-period
labels.  Outputs are new and immutable.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import polars as pl
import yaml

from trading_pipeline.experiments.causal_stack import (
    augmented_tabular_view, build_stack_store, closeout_f1_base_layers,
    expanding_crossfit_windows, produce_causal_stack,
)
from trading_pipeline.experiments.default_registry import default_registry
from trading_pipeline.features import F1


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str, allow_nan=False)
        stream.write("\n")


def prepare_causal_stack(
    *, repository_root: str | Path, prepared_study: str | Path,
    feature_manifest: str | Path, output_dir: str | Path,
) -> dict:
    root = Path(repository_root).resolve()
    study_path = Path(prepared_study).resolve()
    feature_manifest_path = Path(feature_manifest).resolve()
    destination = Path(output_dir).resolve()
    for path in (study_path, feature_manifest_path, destination):
        if not path.is_relative_to(root):
            raise ValueError(f"Path escapes repository: {path}")
    if destination.exists():
        raise FileExistsError(destination)
    config = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    if config.get("status") != "draft" or config.get("study_id") != "expanded_closeout_exploratory_v1":
        raise ValueError("Causal stack preparation requires the expanded prepared draft")
    declaration = config.get("data", {}).get("upstream_layer", {})
    cross = declaration.get("cross_fit", {})
    layers, ensemble = closeout_f1_base_layers()
    declared_outputs = declaration.get("outputs")
    expected = [
        {"id": "E1", "component_id": layers[0].component_id,
         "parameters": dict(layers[0].parameters)},
        {"id": "E2", "component_id": layers[1].component_id,
         "parameters": dict(layers[1].parameters)},
        {"id": "E3", "component_id": layers[2].component_id,
         "parameters": dict(layers[2].parameters)},
        {"id": "E4", "derived_from": ["E1", "E2", "E3"],
         "aggregation": "equal_weight_mean"},
    ]
    if (declaration.get("feature_set_id") != "F1" or declaration.get("seed") != 41
            or declared_outputs != expected
            or cross.get("scheme") != "expanding_chronological_blocks"
            or cross.get("label_purge") != "information_interval_overlap"):
        raise ValueError("Prepared protocol upstream declaration differs from implemented fixed layer")
    feature_info = _json(feature_manifest_path)
    artifact = feature_info.get("artifacts", {}).get("features", {})
    feature_path = (root / artifact.get("relative_path", "")).resolve()
    if (not feature_path.is_file() or _hash(feature_path) != artifact.get("sha256")
            or feature_info.get("feature_columns", {}).get("F1") != list(F1)):
        raise ValueError("Feature manifest does not bind the canonical F1 snapshot")
    snapshot = _json(root / config["data"]["snapshot_manifest"])
    if snapshot.get("feature_sha256") != _hash(feature_path):
        raise ValueError("Prepared study snapshot differs from feature manifest")
    holdout = _json(root / config["validation"]["final_holdout"]["manifest"])
    holdout_dates = tuple(date.fromisoformat(value) for value in holdout["session_dates"])
    protected_start = holdout_dates[0]
    frame = pl.read_parquet(feature_path)
    usable = frame.filter(pl.col("label_end_date").is_not_null()).filter(
        pl.col("session_date") <= holdout_dates[-1])
    exposure = usable.group_by("session_date").agg(
        pl.col("label_end_date").max().alias("max_label_end_date")).sort("session_date")
    calendar = tuple(exposure["session_date"].to_list())
    ends = {row["session_date"]: row["max_label_end_date"] for row in exposure.to_dicts()}
    windows = expanding_crossfit_windows(
        calendar=calendar, max_label_end_by_session=ends,
        protected_start_date=protected_start,
        minimum_fit_sessions=int(cross["min_fit_sessions"]),
        stopping_sessions=int(config["validation"]["window_policy"]["stopping_sessions"]),
        score_block_sessions=int(cross["score_block_sessions"]),
    )
    destination.mkdir(parents=True)
    producer = produce_causal_stack(
        frame=frame, registry=default_registry(), layers=layers, ensemble=ensemble,
        windows=windows, study_id=config["study_id"],
        protected_start_date=protected_start,
        minimum_fit_sessions=int(cross["min_fit_sessions"]),
        score_block_sessions=int(cross["score_block_sessions"]),
        output_dir=destination / "producer",
    )
    store = build_stack_store(
        producer_manifest=producer, feature_path=feature_path,
        feature_sha256=_hash(feature_path), feature_manifest_path=feature_manifest_path,
        feature_manifest_sha256=_hash(feature_manifest_path),
        output_dir=destination / "store",
    )
    training_keys = frame.filter(
        (pl.col("session_date") < protected_start)
        & pl.col("forward_return_5d").is_not_null()
    ).select("security_id", "session_date")
    augmented, imputation = augmented_tabular_view(
        canonical=frame, store_manifest_path=destination / "store" / "manifest.json",
        feature_sha256=_hash(feature_path), study_id=config["study_id"],
        training_keys=training_keys,
    )
    table_path = destination / "augmented_features.parquet"
    augmented.write_parquet(table_path)
    extra_columns = [column for column in augmented.columns
                     if column.startswith("upstream_prediction_")]
    manifest = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_sha256": _hash(Path(__file__)),
        "prepared_study_path": study_path.relative_to(root).as_posix(),
        "prepared_study_sha256": _hash(study_path),
        "feature_path": feature_path.relative_to(root).as_posix(),
        "feature_sha256": _hash(feature_path),
        "feature_manifest_path": feature_manifest_path.relative_to(root).as_posix(),
        "feature_manifest_sha256": _hash(feature_manifest_path),
        "producer_manifest_path": (destination / "producer" / "manifest.json").relative_to(root).as_posix(),
        "producer_manifest_sha256": _hash(destination / "producer" / "manifest.json"),
        "source_prediction_path": Path(producer["source_prediction_path"]).relative_to(root).as_posix(),
        "source_prediction_sha256": producer["source_prediction_sha256"],
        "store_manifest_path": (destination / "store" / "manifest.json").relative_to(root).as_posix(),
        "store_manifest_sha256": _hash(destination / "store" / "manifest.json"),
        "table_path": table_path.relative_to(root).as_posix(),
        "table_sha256": _hash(table_path),
        "feature_columns": [*F1, *extra_columns],
        "imputation_metadata": imputation,
        "window_count": len(windows),
        "protected_start_date": protected_start.isoformat(),
        "protected_labels_used_for_fit": False,
    }
    _write_new(destination / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--prepared-study", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare_causal_stack(repository_root=args.repository_root,
        prepared_study=args.prepared_study, feature_manifest=args.feature_manifest,
        output_dir=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
