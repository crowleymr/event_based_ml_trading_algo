"""Fold-level execution for registered deep supervised candidates.

The temporal evaluator supplies disjoint, chronological fit/stopping/score rows.
``rows`` is the point-in-time feature pool for sequence history, including rows
whose labels are unavailable; only ``fit_rows`` train the scaler and model.
This module does not select trials or authorize research use.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Mapping, Sequence

import numpy as np

from trading_pipeline.experiments import FitContext
from trading_pipeline.experiments.default_registry import default_registry
from trading_pipeline.features.sequence_view import CausalSequenceView


@dataclass(frozen=True)
class SupervisedExecution:
    model: Any
    predictions: tuple[Mapping[str, Any], ...]
    telemetry: Mapping[str, Any]
    sequence_scaler: Mapping[str, tuple[float, ...]]


def _key(row: Mapping[str, Any]) -> tuple[str, date]:
    security, session = row.get("security_id"), row.get("session_date")
    if not isinstance(security, str) or not security or not isinstance(session, date):
        raise ValueError("Every row requires a nonempty security_id and session_date")
    return security, session


def _target(row: Mapping[str, Any]) -> float:
    value = row.get("forward_return_5d")
    if isinstance(value, bool) or value is None or not math.isfinite(float(value)):
        raise ValueError("Fit and stopping rows require finite forward_return_5d")
    return float(value)


def fit_predict_supervised(
    *, component_id: str, params: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    fit_rows: Sequence[Mapping[str, Any]],
    stopping_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    context: FitContext, feature_columns: Sequence[str], device: str = "cpu",
    feature_set: str = "study", training_config: Mapping[str, Any] | None = None,
) -> SupervisedExecution:
    """Fit one deep trial and predict one scoring partition without learning there.

    Each row needs ``security_id`` (str), ``session_date`` (date), and the
    declared numeric feature columns (NaN allowed). Fit/stopping rows additionally
    need finite ``forward_return_5d``; scoring labels may be absent. Scoring rows
    need ``ticker``, ``split`` and finite ``vol_20d`` for the canonical prediction
    schema. Returned prediction records contain session_date, security_id, ticker,
    vol_20d, split, actual_forward_return_5d, predicted_return_5d, model_id,
    feature_set and predicted_rank. The caller persists them in its run artefacts.
    """
    registry = default_registry()
    spec = registry.spec(component_id)
    if spec.interface != "SupervisedModel" or not spec.capabilities.get("sequence_view"):
        raise ValueError("Fold execution requires a registered sequence SupervisedModel")
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Explicit unique feature_columns are required")
    if any(column in {"forward_return_5d", "label_end_date", "split"}
           for column in feature_columns):
        raise ValueError("Labels and partition metadata cannot be feature columns")
    if not all((rows, fit_rows, stopping_rows, score_rows)):
        raise ValueError("Feature pool and each fold partition must be nonempty")
    keys = [_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Feature pool has duplicate security/session keys")
    lookup = {key: index for index, key in enumerate(keys)}
    partitions = [[_key(row) for row in part]
                  for part in (fit_rows, stopping_rows, score_rows)]
    if any(len(set(part)) != len(part) or not set(part) <= lookup.keys()
           for part in partitions):
        raise ValueError("Fold rows must have unique keys present in the feature pool")
    if any(set(partitions[i]) & set(partitions[j])
           for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Fit, stopping and score partitions must be disjoint")
    if not (max(day for _, day in partitions[0]) < min(day for _, day in partitions[1])
            and max(day for _, day in partitions[1]) < min(day for _, day in partitions[2])):
        raise ValueError("Fold partitions must be strictly chronological")
    first_stop = min(day for _, day in partitions[1])
    first_score = min(day for _, day in partitions[2])
    for part, boundary in ((fit_rows, first_stop), (stopping_rows, first_score)):
        for row in part:
            label_end = row.get("label_end_date")
            if not isinstance(label_end, date) or label_end >= boundary:
                raise ValueError("Fit/stopping labels overlap the next partition")
    matrix = np.empty((len(rows), len(feature_columns)), dtype=np.float64)
    for index, row in enumerate(rows):
        for column_index, column in enumerate(feature_columns):
            value = row.get(column)
            try:
                number = np.nan if value is None else float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric feature: {column}") from exc
            if np.isinf(number):
                raise ValueError(f"Infinite feature: {column}")
            matrix[index, column_index] = number
    indices = [[lookup[key] for key in part] for part in partitions]
    ids = [security for security, _ in keys]
    dates = [session for _, session in keys]
    model = registry.create(component_id, params=params, device=device,
                            training_config=training_config)
    view = CausalSequenceView(model._params["lookback"]).fit(
        ids, dates, matrix, training_indices=indices[0])
    batches = [view.transform(ids, dates, matrix, target_indices=part)
               for part in indices]
    model.fit(batches[0], np.asarray([_target(row) for row in fit_rows]),
              context=context,
              stopping_data=(batches[1], np.asarray([_target(row) for row in stopping_rows])))
    prediction = model.predict(batches[2])
    if prediction.shape != (len(score_rows),) or not np.isfinite(prediction).all():
        raise ValueError("Model returned invalid predictions")
    records: list[dict[str, Any]] = []
    for row, estimate in zip(score_rows, prediction, strict=True):
        if not isinstance(row.get("ticker"), str) or not isinstance(row.get("split"), str):
            raise ValueError("Scoring rows require ticker and split")
        volatility = row.get("vol_20d")
        if volatility is None or not math.isfinite(float(volatility)):
            raise ValueError("Scoring rows require finite vol_20d")
        actual = row.get("forward_return_5d")
        if actual is not None and not math.isfinite(float(actual)):
            raise ValueError("Scoring label must be finite when present")
        records.append({"session_date": row["session_date"],
                        "security_id": row["security_id"], "ticker": row["ticker"],
                        "vol_20d": float(volatility), "split": row["split"],
                        "actual_forward_return_5d": None if actual is None else float(actual),
                        "predicted_return_5d": float(estimate), "model_id": component_id,
                        "feature_set": feature_set})
    by_date: dict[date, list[dict[str, Any]]] = {}
    for record in records:
        by_date.setdefault(record["session_date"], []).append(record)
    ranked = []
    for session in sorted(by_date):
        for rank, record in enumerate(sorted(by_date[session], key=lambda item:
                                             (-item["predicted_return_5d"], item["security_id"])), 1):
            ranked.append({**record, "predicted_rank": rank})
    telemetry = {**model.telemetry(), "component_id": component_id,
                 "feature_columns": tuple(feature_columns),
                 "score_rows": len(score_rows), "prediction_mode": "frozen_no_learning"}
    scaler = {"mean": tuple(float(value) for value in view.mean_),
              "scale": tuple(float(value) for value in view.scale_)}
    return SupervisedExecution(model, tuple(ranked), telemetry, scaler)
