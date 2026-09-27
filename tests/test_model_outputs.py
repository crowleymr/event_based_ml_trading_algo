"""Synthetic contract checks; no research scores are asserted."""

from datetime import date
import hashlib
import json

import polars as pl
import pytest

from trading_pipeline.features.model_outputs import (
    audit_representation_parity, build_oof_store, load_oof_store,
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path, *, role="inner_oof", fit_date=date(2026, 1, 1)):
    features = tmp_path / "features.parquet"
    upstream = tmp_path / "upstream.parquet"
    pl.DataFrame({
        "security_id": ["A", "B", "A"],
        "session_date": [date(2026, 1, 5), date(2026, 1, 5), date(2026, 1, 6)],
    }).write_parquet(features)
    feature_manifest = tmp_path / "feature_manifest.json"
    feature_manifest.write_text(json.dumps({"artifacts": {"features": {
        "sha256": _hash(features)}}}), encoding="utf-8")
    pl.DataFrame({
        "security_id": ["A", "B"],
        "session_date": [date(2026, 1, 5), date(2026, 1, 5)],
        "prediction": [0.1, 0.2],
        "upstream_component_id": ["supervised.elastic_net.v1"] * 2,
        "fold_id": ["inner_0"] * 2,
        "seed": [41] * 2,
        "fit_sha256": ["a" * 64] * 2,
        "fit_cutoff_date": [fit_date] * 2,
        "fit_max_label_end_date": [fit_date] * 2,
        "prediction_role": [role] * 2,
    }).write_parquet(upstream)
    kwargs = dict(feature_path=features, feature_sha256=_hash(features),
                  feature_manifest_path=feature_manifest,
                  source_prediction_path=upstream, source_prediction_sha256=_hash(upstream),
                  components=("supervised.elastic_net.v1",), output_dir=tmp_path / "store",
                  study_id="synthetic_study", feature_manifest_sha256=_hash(feature_manifest))
    return kwargs


def test_oof_store_has_explicit_missing_mask_and_parity(tmp_path):
    kwargs = _inputs(tmp_path)
    build_oof_store(**kwargs)
    store = load_oof_store(tmp_path / "store/manifest.json",
                           expected_feature_sha256=kwargs["feature_sha256"],
                           expected_study_id="synthetic_study")
    assert store.height == 3
    assert store["is_available"].to_list().count(False) == 1
    assert store.filter(~pl.col("is_available"))["prediction"].is_null().all()
    shared = store.select("security_id", "session_date", "upstream_component_id",
                          "prediction", "is_available").with_columns(
                              pl.col("session_date").alias("consumer_cutoff_date"))
    audit = audit_representation_parity(store, {name: shared for name in ("tabular", "sequence", "rl")})
    assert all(view["available_rows"] == 2 for view in audit["representations"].values())
    altered = shared.with_columns(pl.when(pl.col("security_id") == "A").then(0.9)
                                  .otherwise(pl.col("prediction")).alias("prediction"))
    with pytest.raises(ValueError, match="differs"):
        audit_representation_parity(store, {"tabular": shared, "sequence": shared, "rl": altered})
    subset = shared.head(1)
    assert audit_representation_parity(store, {"tabular": shared, "sequence": shared,
                                               "rl": subset})["representations"]["rl"]["unconsumed_canonical_rows"] == 2
    future = shared.with_columns(pl.lit(date(2026, 1, 1)).alias("consumer_cutoff_date"))
    with pytest.raises(ValueError, match="after its cutoff"):
        audit_representation_parity(store, {"tabular": shared, "sequence": shared, "rl": future})


@pytest.mark.parametrize("role", ["outer_score", "final_fit", "holdout", "legacy_observed_final"])
def test_non_oof_prediction_roles_are_rejected(tmp_path, role):
    with pytest.raises(ValueError, match="Only inner_oof"):
        build_oof_store(**_inputs(tmp_path, role=role))


def test_future_fit_cutoff_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="reaches its prediction session"):
        build_oof_store(**_inputs(tmp_path, fit_date=date(2026, 1, 5)))


def test_tampered_source_or_store_is_rejected(tmp_path):
    kwargs = _inputs(tmp_path)
    kwargs["source_prediction_path"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash-mismatched"):
        build_oof_store(**kwargs)
    kwargs = _inputs(tmp_path)
    build_oof_store(**kwargs)
    (tmp_path / "store/oof_predictions.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash-mismatched"):
        load_oof_store(tmp_path / "store/manifest.json",
                       expected_feature_sha256=kwargs["feature_sha256"],
                       expected_study_id="synthetic_study")


def test_pinned_manifest_tamper_is_rejected(tmp_path):
    kwargs = _inputs(tmp_path)
    build_oof_store(**kwargs)
    manifest = tmp_path / "store/manifest.json"
    sealed_hash = _hash(manifest)
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash-mismatched"):
        load_oof_store(manifest, expected_feature_sha256=kwargs["feature_sha256"],
                       expected_study_id="synthetic_study",
                       expected_manifest_sha256=sealed_hash)
