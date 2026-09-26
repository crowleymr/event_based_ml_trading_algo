from __future__ import annotations

from datetime import date, timedelta
import importlib.util

import numpy as np
import pytest

from trading_pipeline.experiments import FitContext
from trading_pipeline.optimisation.model_execution import fit_predict_supervised


def _fixture():
    rows = []
    for day in range(6):
        session = date(2026, 1, 1) + timedelta(days=day)
        for ticker in ("A", "B"):
            rows.append({"security_id": ticker, "ticker": ticker,
                         "session_date": session, "label_end_date": session,
                         "f0": float(day + (ticker == "B")), "vol_20d": 0.2,
                         "forward_return_5d": 0.01 * day,
                         "split": "inner_score"})
    return rows, rows[:4], rows[4:8], rows[8:]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch")
@pytest.mark.parametrize("component_id,extra", [
    ("supervised.lstm.v1", {}),
    ("supervised.causal_transformer.v1", {"heads": 2, "feed_forward": 16}),
])
def test_deep_bridge_trains_on_fold_and_emits_canonical_predictions(component_id, extra):
    rows, fit, stop, score = _fixture()
    score = [{**row, "forward_return_5d": None} for row in score]
    result = fit_predict_supervised(
        component_id=component_id,
        params={"lookback": 2, "width": 8, "depth": 1, "batch_size": 2, **extra},
        rows=rows, fit_rows=fit, stopping_rows=stop, score_rows=score,
        context=FitContext("synthetic", "trial", "fold", 3, {"epochs": 1}),
        feature_columns=["f0"], device="cpu", feature_set="F0",
        training_config={"epochs": 1, "patience": 1},
    )
    assert len(result.predictions) == 4
    assert set(result.predictions[0]) == {
        "session_date", "security_id", "ticker", "vol_20d", "split",
        "actual_forward_return_5d", "predicted_return_5d", "model_id",
        "feature_set", "predicted_rank"}
    assert all(row["actual_forward_return_5d"] is None for row in result.predictions)
    assert {row["predicted_rank"] for row in result.predictions if row["session_date"] == score[0]["session_date"]} == {1, 2}
    assert result.telemetry["epochs_ran"] == 1
    assert result.telemetry["prediction_mode"] == "frozen_no_learning"
    assert np.isclose(result.sequence_scaler["mean"][0], 1.0)


def test_deep_bridge_rejects_partition_overlap_before_training():
    rows, fit, stop, score = _fixture()
    with pytest.raises(ValueError, match="disjoint"):
        fit_predict_supervised(
            component_id="supervised.lstm.v1", params={"lookback": 2},
            rows=rows, fit_rows=fit, stopping_rows=stop,
            score_rows=[*score, fit[0]],
            context=FitContext("synthetic", "trial", "fold", 3),
            feature_columns=["f0"])


def test_deep_bridge_rejects_label_overlap_before_training():
    rows, fit, stop, score = _fixture()
    fit = [{**row, "label_end_date": stop[0]["session_date"]} for row in fit]
    with pytest.raises(ValueError, match="labels overlap"):
        fit_predict_supervised(
            component_id="supervised.lstm.v1", params={"lookback": 2},
            rows=rows, fit_rows=fit, stopping_rows=stop, score_rows=score,
            context=FitContext("synthetic", "trial", "fold", 3),
            feature_columns=["f0"])
