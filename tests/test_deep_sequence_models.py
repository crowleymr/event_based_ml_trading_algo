import importlib.util
import json

import numpy as np
import pytest

from trading_pipeline.experiments import ComponentRegistry, FitContext
from trading_pipeline.features.sequence_view import CausalSequenceView, SequenceBatch
from trading_pipeline.modelling.supervised_models import register_supervised_models
from trading_pipeline.modelling.supervised_models.deep_sequence import (
    CausalTransformerModel, LSTMModel,
)


def _rows():
    ids = ["A", "B", "A", "B", "A", "B", "A", "B"]
    dates = ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02",
             "2026-01-03", "2026-01-03", "2026-01-04", "2026-01-04"]
    values = np.arange(16, dtype=float).reshape(8, 2)
    values[2, 1] = np.nan
    return ids, dates, values


def test_sequence_is_security_isolated_causal_and_train_scaled():
    ids, dates, values = _rows()
    view = CausalSequenceView(3).fit(ids, dates, values, training_indices=[0, 1, 2, 3])
    original_mean = view.mean_.copy()
    before = view.transform(ids, dates, values, target_indices=[2, 4])
    changed = values.copy()
    changed[6:] = 1e9
    after = view.transform(ids, dates, changed, target_indices=[2, 4])
    np.testing.assert_array_equal(before.values, after.values)
    np.testing.assert_array_equal(original_mean, view.mean_)
    assert before.values.shape == (2, 3, 2)
    np.testing.assert_array_equal(before.time_mask[0], [True, True, False])
    assert before.feature_mask[0, 1, 1] == False
    assert not before.feature_mask[0, 2].any()
    assert before.values[0, 1, 1] == 0.0
    # B's values never enter A's history.
    changed[1::2] = -1e9
    other_security = view.transform(ids, dates, changed, target_indices=[2, 4])
    np.testing.assert_array_equal(before.values, other_security.values)


def test_sequence_rejects_invalid_training_boundary_and_keys():
    ids, dates, values = _rows()
    view = CausalSequenceView(2)
    with pytest.raises(RuntimeError, match="fitted"):
        view.transform(ids, dates, values, target_indices=[0])
    with pytest.raises(ValueError, match="unique"):
        view.fit(ids, dates, values, training_indices=[0, 0])
    with pytest.raises(ValueError, match="unique"):
        view.fit(["A"] * 8, ["same"] * 8, values, training_indices=[0])


@pytest.mark.parametrize("cls", [LSTMModel, CausalTransformerModel])
def test_deep_contract_space_gate_and_parameter_validation(cls):
    registry = ComponentRegistry()
    register_supervised_models(registry)
    model = registry.create(cls.SPEC.component_id)
    assert not model.spec.research_enabled
    with pytest.raises(ValueError, match="not enabled for research"):
        registry.create(cls.SPEC.component_id, require_research_enabled=True)
    space = model.search_space()
    json.dumps(space)
    assert {"lookback", "depth", "width", "dropout", "activation",
            "optimizer", "learning_rate", "batch_size"} <= set(space["parameters"])
    if cls is CausalTransformerModel:
        assert {"heads", "feed_forward"} <= set(space["parameters"])
        with pytest.raises(ValueError, match="divisible"):
            cls(params={"width": 33, "heads": 2})
    with pytest.raises(ValueError, match="lookback"):
        cls(params={"lookback": 0})


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch")
@pytest.mark.parametrize("cls", [LSTMModel, CausalTransformerModel])
def test_deep_fit_mask_telemetry_determinism_and_save_load(cls, tmp_path):
    ids, dates, values = _rows()
    view = CausalSequenceView(2).fit(ids, dates, values, training_indices=[0, 1, 2, 3])
    training = view.transform(ids, dates, values, target_indices=[0, 1, 2, 3])
    stopping = view.transform(ids, dates, values, target_indices=[4, 5])
    y_train = np.array([0.1, 0.2, 0.3, 0.4])
    y_stop = np.array([0.5, 0.6])
    ctx = FitContext("synthetic", "trial", "fold", 7, {"epochs": 1})
    params = {"lookback": 2, "width": 8, "depth": 1, "batch_size": 2}
    if cls is CausalTransformerModel:
        params.update(heads=2, feed_forward=16)
    model = cls(params=params, device="cpu", training_config={"epochs": 1, "patience": 1})
    with pytest.raises(ValueError, match="stopping_data"):
        model.fit(training, y_train, context=ctx)
    model.fit(training, y_train, context=ctx, stopping_data=(stopping, y_stop))
    prediction = model.predict(stopping)
    assert prediction.shape == (2,) and np.isfinite(prediction).all()
    modified_values = stopping.values.copy()
    modified_values[~stopping.feature_mask] = 1e6
    masked = SequenceBatch(modified_values, stopping.time_mask, stopping.feature_mask)
    np.testing.assert_allclose(prediction, model.predict(masked), rtol=0, atol=1e-6)
    telemetry = model.telemetry()
    assert telemetry["actual_device"] == "cpu"
    assert telemetry["parameter_count"] > 0
    assert telemetry["training_trace"][0]["epoch"] == 1
    second = cls(params=params, device="cpu", training_config={"epochs": 1, "patience": 1})
    second.fit(training, y_train, context=ctx, stopping_data=(stopping, y_stop))
    np.testing.assert_allclose(prediction, second.predict(stopping), rtol=0, atol=1e-6)
    path = tmp_path / "model.pt"
    model.save(path)
    restored = cls.load(path)
    np.testing.assert_allclose(prediction, restored.predict(stopping), rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="lookback"):
        wrong = SequenceBatch(np.zeros((2, 3, 2), np.float32),
                              np.ones((2, 3), bool), np.ones((2, 3, 2), bool))
        restored.predict(wrong)
