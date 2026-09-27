from __future__ import annotations

from datetime import date, timedelta
import importlib.util

import numpy as np
import pytest

from trading_pipeline.experiments.deep_study_adapter import fit_score_deep_study
from trading_pipeline.features.sequence_view import CausalSequenceView


def _partitions():
    rows = []
    for day in range(6):
        session = date(2026, 1, 1) + timedelta(days=day)
        for security in ("A", "B"):
            rows.append({"security_id": security, "ticker": security,
                         "session_date": session, "label_end_date": session,
                         "return_5d": float(day + (security == "B")), "vol_20d": 0.2,
                         "forward_return_5d": float(day) / 100,
                         "split": "synthetic"})
    fit, stopping, score = rows[:4], rows[4:8], rows[8:]
    pool = [{key: value for key, value in row.items() if key != "forward_return_5d"}
            for row in rows]
    return pool, fit, stopping, score


def _params(component):
    params = {"lookback": 2, "depth": 1, "width": 8, "dropout": 0.0,
              "activation": "relu", "optimizer": "adamw",
              "learning_rate": 0.001, "batch_size": 2}
    if component.endswith("transformer.v1"):
        params.update(heads=2, feed_forward=16)
    return params


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch")
@pytest.mark.parametrize("component", ["supervised.lstm.v1", "supervised.causal_transformer.v1"])
def test_study_adapter_runs_real_data_shaped_fold_and_records_ledger_fields(component):
    pool, fit, stopping, score = _partitions()
    result = fit_score_deep_study(
        component_id=component, feature_set_id="F1", parameters=_params(component),
        seed=41, study_id="synthetic", trial_id="trial", fold_id="fold",
        feature_pool=pool, fit_rows=fit, stopping_rows=stopping,
        score_rows=list(reversed(score)), feature_columns=["return_5d"],
        requested_device="cpu", epochs=1, patience=1,
    )
    assert result.predictions.shape == (len(score),)
    assert np.isfinite(result.predictions).all()
    assert result.telemetry["parameters"] == _params(component)
    assert result.telemetry["architecture_parameters"] == _params(component)
    assert result.telemetry["actual_device"] == "cpu"
    assert result.telemetry["prediction_mode"] == "frozen_no_learning"
    assert result.telemetry["training_trace"][0]["epoch"] == 1
    assert result.telemetry["fit_rows"] == len(fit)
    assert result.telemetry["stopping_rows"] == len(stopping)
    assert result.telemetry["score_rows"] == len(score)
    assert result.telemetry["wall_seconds"] >= 0
    # Only fit observations determine preprocessing statistics.
    assert result.sequence_scaler["mean"] == (1.0,)


def test_study_adapter_rejects_incomplete_and_invalid_architecture_before_fit():
    pool, fit, stopping, score = _partitions()
    common = dict(component_id="supervised.causal_transformer.v1", feature_set_id="F1",
                  seed=41, study_id="synthetic", trial_id="trial", fold_id="fold",
                  feature_pool=pool, fit_rows=fit, stopping_rows=stopping,
                  score_rows=score, feature_columns=["return_5d"],
                  requested_device="cpu", epochs=1, patience=1)
    with pytest.raises(ValueError, match="exact architecture"):
        fit_score_deep_study(parameters={"lookback": 2}, **common)
    with pytest.raises(ValueError, match="divisible"):
        fit_score_deep_study(parameters={**_params(common["component_id"]), "heads": 3}, **common)


def test_sequence_index_preserves_causal_history_under_future_perturbation():
    ids = ["A", "B"] * 5
    dates = [date(2026, 1, 1) + timedelta(days=day // 2) for day in range(10)]
    x = np.arange(10, dtype=float).reshape(-1, 1)
    view = CausalSequenceView(3).fit(ids, dates, x, training_indices=[0, 1, 2, 3])
    before = view.transform(ids, dates, x, target_indices=[2, 4])
    changed = x.copy()
    changed[6:] = 1e9
    after = view.transform(ids, dates, changed, target_indices=[2, 4])
    np.testing.assert_array_equal(before.values, after.values)


def test_sequence_index_uses_session_order_when_pool_is_shuffled():
    ids = ["A", "B", "A", "A", "B", "A"]
    dates = [date(2026, 1, 3), date(2026, 1, 1), date(2026, 1, 1),
             date(2026, 1, 4), date(2026, 1, 2), date(2026, 1, 2)]
    x = np.array([[3.0], [10.0], [1.0], [4.0], [20.0], [2.0]])
    view = CausalSequenceView(3).fit(ids, dates, x, training_indices=[1, 2, 4, 5])
    batch = view.transform(ids, dates, x, target_indices=[0])
    expected = (np.array([1.0, 2.0, 3.0]) - view.mean_[0]) / view.scale_[0]
    np.testing.assert_allclose(batch.values[0, :, 0], expected, rtol=1e-6)
