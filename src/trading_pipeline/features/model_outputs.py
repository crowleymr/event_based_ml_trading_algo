"""Versioned, causal upstream prediction features for the expanded study.

The authoritative evaluator supplies predictions from earlier, declared layers.
This module never fits a model or fills an absent prediction with an estimate.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import polars as pl


SCHEMA_VERSION = 1
KEYS = ("security_id", "session_date")
LINEAGE = ("upstream_component_id", "fold_id", "seed", "fit_sha256",
           "fit_cutoff_date", "fit_max_label_end_date", "prediction_role")
INPUT_COLUMNS = (*KEYS, "prediction", *LINEAGE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: str | Path, expected_sha256: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or _sha256(source) != expected_sha256:
        raise ValueError(f"Missing or hash-mismatched input: {source}")
    return source


def _keys(frame: pl.DataFrame, label: str) -> None:
    if frame.is_empty() or frame.select(pl.any_horizontal(*(pl.col(key).is_null() for key in KEYS)).any()).item():
        raise ValueError(f"{label} has empty or null keys")
    if frame.select(KEYS).is_duplicated().any():
        raise ValueError(f"{label} has duplicate security/session keys")


def build_oof_store(
    *, feature_path: str | Path, feature_sha256: str,
    feature_manifest_path: str | Path,
    source_prediction_path: str | Path, source_prediction_sha256: str,
    components: tuple[str, ...], output_dir: str | Path,
    study_id: str, feature_manifest_sha256: str,
) -> dict:
    """Materialise an immutable full-key store from supplied inner OOF predictions.

    Missing upstream outputs are null with ``is_available=False``. Consumers must
    carry the mask and fit any numeric imputation on their training partition only.
    An entirely absent component is an error, never an all-null feature column.
    """
    feature_file = _verified(feature_path, feature_sha256)
    feature_manifest_file = _verified(feature_manifest_path, feature_manifest_sha256)
    feature_manifest = json.loads(feature_manifest_file.read_text(encoding="utf-8"))
    if feature_manifest.get("artifacts", {}).get("features", {}).get("sha256") != feature_sha256:
        raise ValueError("Feature manifest does not pin the canonical feature snapshot")
    source_file = _verified(source_prediction_path, source_prediction_sha256)
    if not study_id or not components or len(set(components)) != len(components):
        raise ValueError("A study ID and unique declared components are required")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Output feature store already exists: {destination}")

    canonical = pl.read_parquet(feature_file, columns=list(KEYS))
    canonical = canonical.with_columns(pl.col("session_date").cast(pl.Date))
    _keys(canonical, "Canonical features")
    source = pl.read_parquet(source_file)
    missing = set(INPUT_COLUMNS) - set(source.columns)
    if missing:
        raise ValueError(f"Upstream source lacks required fields: {sorted(missing)}")
    source = source.select(INPUT_COLUMNS).with_columns(
        pl.col("session_date").cast(pl.Date),
        pl.col("fit_cutoff_date").cast(pl.Date),
        pl.col("fit_max_label_end_date").cast(pl.Date),
        pl.col("prediction").cast(pl.Float64),
        pl.col("seed").cast(pl.Int64),
    )
    if source.is_empty() or source.select(pl.any_horizontal(*(pl.col(c).is_null() for c in INPUT_COLUMNS)).any()).item():
        raise ValueError("Every supplied prediction must be complete with non-null lineage")
    if source.select(pl.col("prediction").is_finite().all()).item() is not True:
        raise ValueError("Upstream predictions must be finite")
    if source.filter(~pl.col("upstream_component_id").is_in(components)).height:
        raise ValueError("Source contains an undeclared upstream component")
    if set(source["upstream_component_id"].to_list()) != set(components):
        raise ValueError("Every declared component must supply at least one OOF prediction")
    if source.select((*KEYS, "upstream_component_id")).is_duplicated().any():
        raise ValueError("Upstream component/security/session predictions must be unique")
    if source.filter(pl.col("prediction_role") != "inner_oof").height:
        raise ValueError("Only inner_oof predictions may become training features")
    if source.filter(
        (pl.col("fit_cutoff_date") >= pl.col("session_date"))
        | (pl.col("fit_max_label_end_date") > pl.col("fit_cutoff_date"))
    ).height:
        raise ValueError("Upstream fit or label information reaches its prediction session")
    if source.filter(
        (pl.col("fit_sha256").str.len_chars() != 64)
        | pl.col("fit_sha256").str.contains(r"^[0-9a-f]{64}$").not_()
        | (pl.col("fold_id").str.len_chars() == 0)
    ).height:
        raise ValueError("Upstream fit hash and fold ID must be valid")
    unknown = source.join(canonical, on=list(KEYS), how="anti")
    if unknown.height:
        raise ValueError("Upstream source contains keys absent from the pinned feature snapshot")

    grid = pl.concat([
        canonical.with_columns(pl.lit(component).alias("upstream_component_id"))
        for component in components
    ])
    coverage = grid.join(source, on=[*KEYS, "upstream_component_id"], how="left")
    coverage = coverage.with_columns(pl.col("prediction").is_not_null().alias("is_available"))
    coverage = coverage.sort(["upstream_component_id", *KEYS])
    destination.mkdir(parents=True)
    table_path = destination / "oof_predictions.parquet"
    manifest_path = destination / "manifest.json"
    coverage.write_parquet(table_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_sha256": _sha256(Path(__file__)),
        "feature_path": str(feature_file),
        "feature_sha256": feature_sha256,
        "feature_manifest_path": str(feature_manifest_file),
        "feature_manifest_sha256": feature_manifest_sha256,
        "source_prediction_path": str(source_file),
        "source_prediction_sha256": source_prediction_sha256,
        "table_path": str(table_path),
        "table_sha256": _sha256(table_path),
        "components": list(components),
        "canonical_rows": canonical.height,
        "store_rows": coverage.height,
        "available_rows": int(coverage["is_available"].sum()),
        "missing_output_policy": "null_plus_mask_train_partition_only_imputation",
        "allowed_prediction_role": "inner_oof",
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def load_oof_store(manifest_path: str | Path, *, expected_feature_sha256: str,
                   expected_study_id: str,
                   expected_manifest_sha256: str | None = None) -> pl.DataFrame:
    """Verify the sealed table and its canonical binding before any join."""
    manifest_file = Path(manifest_path)
    if expected_manifest_sha256 is not None:
        _verified(manifest_file, expected_manifest_sha256)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("study_id") != expected_study_id
            or manifest.get("feature_sha256") != expected_feature_sha256
            or manifest.get("missing_output_policy") != "null_plus_mask_train_partition_only_imputation"):
        raise ValueError("Incompatible OOF feature-store manifest")
    _verified(manifest["feature_path"], expected_feature_sha256)
    _verified(manifest["feature_manifest_path"], manifest["feature_manifest_sha256"])
    _verified(manifest["source_prediction_path"], manifest["source_prediction_sha256"])
    table = pl.read_parquet(_verified(manifest["table_path"], manifest["table_sha256"]))
    if table.height != manifest["store_rows"] or int(table["is_available"].sum()) != manifest["available_rows"]:
        raise ValueError("OOF store count mismatch")
    if table.filter(pl.col("is_available") != pl.col("prediction").is_not_null()).height:
        raise ValueError("OOF availability mask differs from prediction presence")
    if table.filter(pl.col("is_available") & (pl.col("prediction_role") != "inner_oof")).height:
        raise ValueError("Forbidden prediction role in OOF store")
    if table.filter(pl.col("is_available") & (
        (pl.col("fit_cutoff_date") >= pl.col("session_date"))
        | (pl.col("fit_max_label_end_date") > pl.col("fit_cutoff_date"))
    )).height:
        raise ValueError("Noncausal lineage in OOF store")
    return table


def audit_representation_parity(store: pl.DataFrame, views: dict[str, pl.DataFrame]) -> dict:
    """Check that tabular, sequence and RL use identical available outputs at T.

    Each view supplies the exact per-security/session/component prediction keys it
    will consume, with ``prediction``, ``is_available`` and its
    ``consumer_cutoff_date``. Sequence history can repeat historical rows and RL
    can sample weekly; both must preserve the sealed value and causal cutoff.
    """
    if set(views) != {"tabular", "sequence", "rl"}:
        raise ValueError("Parity requires tabular, sequence and rl views")
    columns = [*KEYS, "upstream_component_id", "prediction", "is_available"]
    identity = [*KEYS, "upstream_component_id"]
    expected = store.select(columns).unique()
    if expected.height != store.height:
        raise ValueError("OOF store has duplicate keys")
    result = {}
    for name, view in views.items():
        if not set([*columns, "consumer_cutoff_date"]) <= set(view.columns):
            raise ValueError(f"{name} view lacks parity fields")
        if view.is_empty():
            raise ValueError(f"{name} view is empty")
        if view.filter(pl.col("session_date") > pl.col("consumer_cutoff_date")).height:
            raise ValueError(f"{name} view consumes an output after its cutoff")
        used = view.select(columns).unique()
        if used.height != view.select(identity).unique().height:
            raise ValueError(f"{name} view has conflicting values for one key")
        matched = expected.join(used.select(identity), on=identity, how="semi")
        if used.height != matched.height or not used.sort(identity).equals(matched.sort(identity)):
            raise ValueError(f"{name} representation differs from causal OOF store")
        result[name] = {"rows": used.height, "available_rows": int(used["is_available"].sum()),
                        "unconsumed_canonical_rows": expected.height - used.height}
    return {"schema_version": SCHEMA_VERSION, "representations": result,
            "differences": "sequence lookback and RL aggregation may differ after this shared keyed input"}
