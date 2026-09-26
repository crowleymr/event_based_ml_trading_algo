"""Shared values and validation for causal unsupervised components.

These components fit a frozen representation on a caller-supplied training partition.
They do not create temporal splits, select candidates, or attach economic meanings to
arbitrary state identifiers.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from trading_pipeline.experiments import FitContext


FloatMatrix = NDArray[np.float64]


def numeric_matrix(x: ArrayLike, *, minimum_rows: int = 1) -> FloatMatrix:
    """Return a finite two-dimensional matrix or fail before estimator fitting."""

    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Unsupervised observations must be a two-dimensional matrix")
    if values.shape[0] < minimum_rows:
        raise ValueError(f"At least {minimum_rows} observations are required")
    if values.shape[1] == 0:
        raise ValueError("At least one feature is required")
    if not np.isfinite(values).all():
        raise ValueError("Unsupervised observations must contain only finite values")
    return values


def require_fitted(value: Any, *, component_id: str) -> None:
    if value is None:
        raise RuntimeError(f"{component_id} must be fitted before this operation")


def context_telemetry(context: FitContext | None) -> Mapping[str, Any] | None:
    if context is None:
        return None
    return {
        "study_id": context.study_id,
        "trial_id": context.trial_id,
        "fold_id": context.fold_id,
        "seed": context.seed,
    }
