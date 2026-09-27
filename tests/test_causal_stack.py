"""Synthetic upstream producer tests; no expanded research result is computed."""

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trading_pipeline.experiments.causal_stack import (
    BaseLayer, CrossFitWindow, EnsembleLayer, augmented_tabular_view, build_stack_store,
    closeout_f1_base_layers, expanding_crossfit_windows, produce_causal_stack,
)
from trading_pipeline.experiments.contracts import ComponentSpec, SupervisedModel
from trading_pipeline.experiments.registry import ComponentRegistry
from trading_pipeline.features import F1


SPEC = ComponentSpec("synthetic.fixed.v1", "SupervisedModel", "synthetic.Fixed",
                     research_enabled=True, capabilities={"stopping_data": False})


class FixedModel(SupervisedModel):
    def __init__(self, *, params, spec=SPEC):
        self.params = params
        self._spec = spec
        self.value = None

    @property
    def spec(self):
        return self._spec

    def search_space(self):
        return {}

    def fit(self, x, y, *, context, stopping_data=None):
        assert stopping_data is None
        self.value = float(np.mean(y)) + self.params["offset"]
        return self

    def predict(self, x):
        return np.full(len(x), self.value)

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"value": self.value}, sort_keys=True), encoding="utf-8")

    def telemetry(self):
        return {}


def _inputs():
    start = date(2026, 1, 1)
    dates = tuple(start + timedelta(days=2 * i) for i in range(6))
    rows = []
    for day in dates:
        for security in ("A", "B"):
            rows.append({"security_id": security, "session_date": day,
                         "label_end_date": day + timedelta(days=1),
                         "forward_return_5d": 0.01 if security == "A" else 0.02,
                         **{column: 1.0 for column in F1}})
    windows = (
        CrossFitWindow("inner_1", (dates[0],), (dates[1],), (dates[2],)),
        CrossFitWindow("inner_2", (dates[0], dates[1], dates[2]),
                       (dates[3],), (dates[4],)),
    )
    registry = ComponentRegistry()
    registry.register(SPEC, FixedModel)
    return pl.DataFrame(rows), windows, registry, dates


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_producer_store_and_train_only_augmented_view(tmp_path):
    frame, windows, registry, dates = _inputs()
    manifest = produce_causal_stack(frame=frame, windows=windows, registry=registry,
                                    layers=(BaseLayer(SPEC.component_id, {"offset": 0.1}, 41),),
                                    study_id="synthetic", protected_start_date=dates[5],
                                    minimum_fit_sessions=1, score_block_sessions=1,
                                    output_dir=tmp_path / "source")
    assert manifest["source_rows"] == 4
    source = pl.read_parquet(manifest["source_prediction_path"])
    assert source["prediction_role"].unique().to_list() == ["inner_oof"]
    assert source["fit_cutoff_date"].to_list() == [dates[0] + timedelta(days=1)] * 2 + [dates[2] + timedelta(days=1)] * 2
    assert all(len(value) == 64 for value in source["fit_sha256"])
    feature_path = tmp_path / "features.parquet"
    frame.write_parquet(feature_path)
    feature_manifest = tmp_path / "features.json"
    feature_manifest.write_text(json.dumps({"artifacts": {"features": {
        "sha256": _hash(feature_path)}}}), encoding="utf-8")
    build_stack_store(producer_manifest=manifest, feature_path=feature_path,
                      feature_sha256=_hash(feature_path), feature_manifest_path=feature_manifest,
                      feature_manifest_sha256=_hash(feature_manifest), output_dir=tmp_path / "store")
    train_keys = frame.filter(pl.col("session_date") == dates[2]).select("security_id", "session_date")
    view, imputation = augmented_tabular_view(canonical=frame,
        store_manifest_path=tmp_path / "store" / "manifest.json",
        feature_sha256=_hash(feature_path), study_id="synthetic", training_keys=train_keys)
    assert "forward_return_5d" not in view.columns
    assert view.height == frame.height
    assert view.filter(pl.col("session_date") == dates[5])["upstream_prediction_0"].is_null().all()
    assert imputation["columns"]["upstream_prediction_0"]["train_observed_rows"] == 2
    assert imputation["columns"]["upstream_prediction_0"]["train_median"] == pytest.approx(0.115)


def test_rejects_undeclared_or_noncausal_protocol(tmp_path):
    frame, windows, registry, dates = _inputs()
    kwargs = dict(frame=frame, windows=windows, registry=registry,
                  layers=(BaseLayer(SPEC.component_id, {"offset": 0.1}, 41),),
                  study_id="synthetic", protected_start_date=dates[5],
                  minimum_fit_sessions=1, score_block_sessions=1,
                  output_dir=tmp_path / "source")
    with pytest.raises(ValueError, match="base_layers"):
        produce_causal_stack(**(kwargs | {"layers": ()}))
    with pytest.raises(ValueError, match="Protected-period labels"):
        produce_causal_stack(**(kwargs | {"protected_start_date": dates[3]}))
    bad = frame.with_columns(pl.when(pl.col("session_date") == dates[0])
                             .then(pl.lit(dates[1])).otherwise(pl.col("label_end_date"))
                             .alias("label_end_date"))
    with pytest.raises(ValueError, match="protected boundary"):
        produce_causal_stack(**(kwargs | {"frame": bad}))
    assert not (tmp_path / "source").exists()


def test_future_labels_do_not_change_source_predictions(tmp_path):
    frame, windows, registry, dates = _inputs()
    args = dict(windows=windows, registry=registry,
                layers=(BaseLayer(SPEC.component_id, {"offset": 0.1}, 41),),
                study_id="synthetic", protected_start_date=dates[5],
                minimum_fit_sessions=1, score_block_sessions=1)
    a = produce_causal_stack(frame=frame, output_dir=tmp_path / "a", **args)
    changed = frame.with_columns(pl.when(pl.col("session_date") >= dates[2])
                                 .then(pl.lit(100.0)).otherwise(pl.col("forward_return_5d"))
                                 .alias("forward_return_5d"))
    b = produce_causal_stack(frame=changed, output_dir=tmp_path / "b", **args)
    first_a = pl.read_parquet(a["source_prediction_path"]).filter(pl.col("fold_id") == "inner_1")
    first_b = pl.read_parquet(b["source_prediction_path"]).filter(pl.col("fold_id") == "inner_1")
    assert first_a.select("prediction", "fit_sha256").equals(first_b.select("prediction", "fit_sha256"))


def test_derived_equal_weight_ensemble_has_own_lineage(tmp_path):
    frame, windows, registry, dates = _inputs()
    extra = []
    for index in (2, 3):
        spec = ComponentSpec(f"synthetic.fixed{index}.v1", "SupervisedModel",
                             f"synthetic.Fixed{index}", research_enabled=True,
                             capabilities={"stopping_data": False})
        registry.register(spec, lambda *, params, spec=spec: FixedModel(params=params, spec=spec))
        extra.append(spec.component_id)
    members = (SPEC.component_id, *extra)
    manifest = produce_causal_stack(frame=frame, windows=windows, registry=registry,
        layers=tuple(BaseLayer(component, {"offset": float(index)}, 41)
                     for index, component in enumerate(members)),
        ensemble=EnsembleLayer("derived.equal_weight_mean.v1", members),
        study_id="synthetic", protected_start_date=dates[5],
        minimum_fit_sessions=1, score_block_sessions=1, output_dir=tmp_path / "source")
    source = pl.read_parquet(manifest["source_prediction_path"])
    assert len(manifest["components"]) == 4
    for fold in ("inner_1", "inner_2"):
        part = source.filter(pl.col("fold_id") == fold)
        expected = part.filter(pl.col("upstream_component_id").is_in(members))["prediction"].mean()
        derived = part.filter(pl.col("upstream_component_id") == "derived.equal_weight_mean.v1")
        assert derived["prediction"].to_list() == pytest.approx([expected] * 2)
        assert len(derived["fit_sha256"].unique()) == 1


def test_closeout_declaration_is_fixed_f1_layer():
    layers, ensemble = closeout_f1_base_layers()
    assert [layer.parameters for layer in layers] == [
        {"alpha": 0.001, "l1_ratio": 0.5},
        {"max_leaf_nodes": 15, "l2_regularization": 10.0},
        {"max_depth": 3, "reg_lambda": 10.0},
    ]
    assert all(layer.seed == 41 for layer in layers)
    assert ensemble.members == tuple(layer.component_id for layer in layers)


def test_window_builder_covers_protected_dates_without_using_their_labels():
    start = date(2026, 1, 1)
    days = tuple(start + timedelta(days=2 * index) for index in range(12))
    ends = {day: day + timedelta(days=1) for day in days}
    windows = expanding_crossfit_windows(calendar=days,
        max_label_end_by_session=ends, protected_start_date=days[8],
        minimum_fit_sessions=3, stopping_sessions=2, score_block_sessions=2)
    assert {day for window in windows for day in window.score_dates} == set(days[5:])
    protected = [window for window in windows if window.score_dates[0] >= days[8]]
    assert protected
    assert all(max((*window.fit_dates, *window.stopping_dates)) < days[8]
               for window in protected)
