"""Fold adapter for registered sequence-view supervised study families.

The authoritative study runner owns authority, labels, selection and persistence.
This adapter receives only its declared fold partitions and emits ordered scores
and JSON-compatible telemetry for the common fit ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import time
from typing import Any, Mapping, Sequence

import numpy as np

from trading_pipeline.experiments import FitContext
from trading_pipeline.experiments.default_registry import default_registry
from trading_pipeline.optimisation.model_execution import fit_predict_supervised


@dataclass(frozen=True)
class DeepStudyFit:
    predictions: np.ndarray
    telemetry: Mapping[str, Any]
    sequence_scaler: Mapping[str, tuple[float, ...]]


def _key(row: Mapping[str, Any]) -> tuple[str, date]:
    return row["security_id"], row["session_date"]


def fit_score_deep_study(
    *, component_id: str, feature_set_id: str,
    parameters: Mapping[str, Any], seed: int, study_id: str,
    trial_id: str, fold_id: str,
    feature_pool: Sequence[Mapping[str, Any]],
    fit_rows: Sequence[Mapping[str, Any]],
    stopping_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str], requested_device: str,
    epochs: int, patience: int,
) -> DeepStudyFit:
    """Execute one declared trial × fold × seed, preserving score-row order.

    Architecture and training controls are explicit inputs. Missing architecture
    fields are rejected before training so ledger parameters fully describe the
    fitted candidate. The lower-level bridge checks disjoint chronological
    partitions, label intervals, train-only scaling and frozen inference.
    """
    registry = default_registry()
    spec = registry.spec(component_id)
    if (spec.interface != "SupervisedModel"
            or not spec.capabilities.get("study_adapter")
            or not spec.capabilities.get("sequence_view")
            or spec.capabilities.get("tabular_view")):
        raise ValueError(f"Unsupported deep study component: {component_id}")
    if not isinstance(parameters, Mapping):
        raise ValueError("Deep study parameters must be an explicit mapping")
    model = registry.create(component_id)
    required = set(model.search_space()["parameters"])
    if set(parameters) != required:
        raise ValueError(f"Deep trial must declare exact architecture and optimisation parameters: {sorted(required)}")
    # Construction validates conditional architecture rules before any fit occurs.
    registry.create(component_id, params=parameters)
    if type(seed) is not int or type(epochs) is not int or epochs < 1 or type(patience) is not int or patience < 1:
        raise ValueError("Seed must be an integer and epochs/patience positive integers")
    if requested_device not in {"cpu", "cuda", "auto"}:
        raise ValueError("requested_device must be cpu, cuda or auto")
    if not feature_pool or not fit_rows or not stopping_rows or not score_rows:
        raise ValueError("Feature pool and every fold partition must be nonempty")
    first_fit = min(row["session_date"] for row in fit_rows)
    last_score = max(row["session_date"] for row in score_rows)
    eligible_ids = {row["security_id"] for part in (fit_rows, stopping_rows, score_rows)
                    for row in part}
    # Historical sequence context is needed before the first fit row. Keep the
    # exact per-security tail and all in-window rows; later rows cannot affect
    # a causal prediction and would waste substantial expanded-vintage memory.
    history = {security: [] for security in eligible_ids}
    history_limit = parameters["lookback"] - 1
    active = []
    for row in feature_pool:
        security = row["security_id"]
        if security not in eligible_ids:
            continue
        session = row["session_date"]
        if session < first_fit:
            if history_limit:
                tail = history[security]
                if len(tail) < history_limit:
                    tail.append(row)
                else:
                    oldest = min(range(len(tail)), key=lambda i: tail[i]["session_date"])
                    if session > tail[oldest]["session_date"]:
                        tail[oldest] = row
        elif session <= last_score:
            active.append(row)
    fold_pool = [row for securities in history.values() for row in securities] + active
    started = time.perf_counter()
    result = fit_predict_supervised(
        component_id=component_id, params=parameters, rows=fold_pool,
        fit_rows=fit_rows, stopping_rows=stopping_rows, score_rows=score_rows,
        context=FitContext(study_id, trial_id, fold_id, seed, {"epochs": epochs}),
        feature_columns=feature_columns, device=requested_device,
        feature_set=feature_set_id,
        training_config={"epochs": epochs, "patience": patience},
    )
    lookup = {_key(row): float(row["predicted_return_5d"]) for row in result.predictions}
    score_keys = [_key(row) for row in score_rows]
    if len(lookup) != len(score_rows) or len(set(score_keys)) != len(score_keys) or set(lookup) != set(score_keys):
        raise ValueError("Deep predictions do not match declared scoring keys")
    predictions = np.asarray([lookup[key] for key in score_keys], dtype=np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError("Deep study predictions must be finite")
    telemetry = {**result.telemetry,
                 "wall_seconds": time.perf_counter() - started,
                 "architecture_parameters": dict(parameters),
                 "training_config": {"epochs": epochs, "patience": patience},
                 "fit_rows": len(fit_rows), "stopping_rows": len(stopping_rows),
                 "score_rows": len(score_rows),
                 "sequence_scaler": result.sequence_scaler}
    return DeepStudyFit(predictions, telemetry, result.sequence_scaler)
