import importlib.util
import json

import numpy as np
import pytest

from trading_pipeline.experiments import ComponentRegistry, FitContext
from trading_pipeline.features.sequence_view import CausalSequenceView, SequenceBatch
from trading_pipeline.modelling.supervised_models import register_supervised_models
from trading_pipeline.modelling.supervised_models.deep_sequence import (
    CausalTransformerModel, LSTMModel, _torch,
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
    assert space["parameters"]["batch_size"]["values"] == [512, 1024]
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


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch")
@pytest.mark.parametrize("cls", [LSTMModel, CausalTransformerModel])
def test_deep_host_to_device_transfers_are_bounded_and_predict_order_matches_full_batch(
        cls, monkeypatch):
    ids, dates, values = _rows()
    view = CausalSequenceView(2).fit(ids, dates, values, training_indices=range(5))
    training = view.transform(ids, dates, values, target_indices=range(5))
    stopping = view.transform(ids, dates, values, target_indices=[5, 6, 7])
    prediction_batch = view.transform(ids, dates, values,
                                      target_indices=[7, 1, 6, 0, 5])
    params = {"lookback": 2, "width": 8, "depth": 1, "batch_size": 2}
    if cls is CausalTransformerModel:
        params.update(heads=2, feed_forward=16)
    model = cls(params=params, device="cpu", training_config={"epochs": 1, "patience": 1})
    transferred = []
    original = model._tensors

    def checked_transfer(batch, target, device, torch, indices):
        transferred.append(len(indices))
        assert 0 < len(indices) <= params["batch_size"]
        return original(batch, target, device, torch, indices)

    monkeypatch.setattr(model, "_tensors", checked_transfer)
    model.fit(training, np.arange(5, dtype=np.float32),
              context=FitContext("synthetic", "trial", "fold", 7, {"epochs": 1}),
              stopping_data=(stopping, np.arange(3, dtype=np.float32)))
    prediction = model.predict(prediction_batch)
    torch, _ = _torch()
    with torch.no_grad():
        expected = model._model(torch.as_tensor(prediction_batch.values),
                                torch.as_tensor(prediction_batch.time_mask),
                                torch.as_tensor(prediction_batch.feature_mask)).numpy()
        train_prediction = model._model(torch.as_tensor(training.values),
                                        torch.as_tensor(training.time_mask),
                                        torch.as_tensor(training.feature_mask))
        full_mse = torch.mean((train_prediction -
                               torch.as_tensor(np.arange(5, dtype=np.float32))) ** 2).item()
        batched_mse = model._mean_squared_error(model._model, training,
                                                np.arange(5, dtype=np.float32),
                                                "cpu", torch)
    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-6)
    assert batched_mse == pytest.approx(full_mse, abs=1e-6)
    assert transferred and max(transferred) == params["batch_size"]
    assert 1 in transferred
    with pytest.raises(ValueError, match="exceeds declared batch_size"):
        original(training, None, "cpu", torch, np.arange(3))
