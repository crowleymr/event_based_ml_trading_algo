"""Causal, per-security sequence view for registered neural predictors.

The evaluator owns chronological partitions. It supplies the exact training row
indices to ``fit`` and target row indices to ``transform``. No label is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SequenceBatch:
    values: np.ndarray  # rows x lookback x features, right padded with zero
    time_mask: np.ndarray  # true only for real historical sessions
    feature_mask: np.ndarray  # true only for observed finite feature values

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("Sequence values must have three dimensions")
        if self.time_mask.shape != self.values.shape[:2]:
            raise ValueError("Time mask shape does not match sequence values")
        if self.feature_mask.shape != self.values.shape:
            raise ValueError("Feature mask shape does not match sequence values")
        if self.time_mask.dtype != np.bool_ or self.feature_mask.dtype != np.bool_:
            raise ValueError("Sequence masks must be boolean")
        if not np.isfinite(self.values).all():
            raise ValueError("Sequence values must be finite")
        if not self.time_mask.any(axis=1).all():
            raise ValueError("Each sequence needs at least one real session")
        if np.any(self.feature_mask & ~self.time_mask[:, :, None]):
            raise ValueError("Padded positions cannot contain observed features")


class CausalSequenceView:
    """Scale from selected training rows and assemble history through each target T."""

    def __init__(self, lookback: int) -> None:
        if not isinstance(lookback, int) or lookback < 1:
            raise ValueError("lookback must be a positive integer")
        self.lookback = lookback
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @staticmethod
    def _inputs(security_ids: Sequence, session_dates: Sequence, features) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = np.asarray(security_ids)
        dates = np.asarray(session_dates)
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
            raise ValueError("Features must be a nonempty two-dimensional matrix")
        if ids.ndim != 1 or dates.ndim != 1 or len(ids) != len(x) or len(dates) != len(x):
            raise ValueError("Security IDs, dates and features must be row-aligned")
        if any(value is None for value in ids) or any(value is None for value in dates):
            raise ValueError("Security IDs and dates cannot be null")
        keys = [(str(ids[i]), str(dates[i])) for i in range(len(x))]
        if len(set(keys)) != len(keys):
            raise ValueError("Security/session keys must be unique")
        if np.isinf(x).any():
            raise ValueError("Infinite feature values are not allowed")
        return ids, dates, x

    def fit(self, security_ids: Sequence, session_dates: Sequence, features, *, training_indices: Sequence[int]) -> "CausalSequenceView":
        _, _, x = self._inputs(security_ids, session_dates, features)
        index = np.asarray(training_indices)
        if index.ndim != 1 or not len(index) or not np.issubdtype(index.dtype, np.integer):
            raise ValueError("training_indices must contain explicit integer row indices")
        if (index < 0).any() or (index >= len(x)).any() or len(np.unique(index)) != len(index):
            raise ValueError("training_indices must be unique and in range")
        train = x[index]
        if not np.isfinite(train).any(axis=0).all():
            raise ValueError("Each feature must have a finite training observation")
        self.mean_ = np.nanmean(train, axis=0)
        scale = np.nanstd(train, axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, security_ids: Sequence, session_dates: Sequence, features, *, target_indices: Sequence[int]) -> SequenceBatch:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Sequence view must be fitted on training rows")
        ids, dates, x = self._inputs(security_ids, session_dates, features)
        if x.shape[1] != len(self.mean_):
            raise ValueError("Feature count differs from fitted scaler")
        targets = np.asarray(target_indices)
        if targets.ndim != 1 or not len(targets) or not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("target_indices must contain explicit integer row indices")
        if (targets < 0).any() or (targets >= len(x)).any():
            raise ValueError("target_indices are out of range")
        values = np.zeros((len(targets), self.lookback, x.shape[1]), dtype=np.float32)
        time_mask = np.zeros((len(targets), self.lookback), dtype=bool)
        feature_mask = np.zeros_like(values, dtype=bool)
        # Index each security once. A per-target full-table scan is quadratic at
        # expanded-universe scale and does not change the causal view semantics.
        histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        grouped: dict[str, list[int]] = {}
        for index, security in enumerate(ids):
            grouped.setdefault(str(security), []).append(index)
        for security, indices in grouped.items():
            security_rows = np.asarray(indices, dtype=np.intp)
            ordered = security_rows[np.argsort(dates[security_rows], kind="stable")]
            histories[security] = ordered, dates[ordered]
        for row, target in enumerate(targets):
            # Sort by actual session, independent of input row order. The target's
            # own session is included because the signal is created after T.
            ordered, ordered_dates = histories[str(ids[target])]
            endpoint = np.searchsorted(ordered_dates, dates[target], side="right")
            history = ordered[max(0, endpoint - self.lookback):endpoint]
            observed = np.isfinite(x[history])
            scaled = (np.where(observed, x[history], self.mean_) - self.mean_) / self.scale_
            size = len(history)
            values[row, :size] = scaled.astype(np.float32)
            time_mask[row, :size] = True
            feature_mask[row, :size] = observed
        return SequenceBatch(values, time_mask, feature_mask)
