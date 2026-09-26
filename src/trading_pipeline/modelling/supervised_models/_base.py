"""Shared mechanics for supervised adapters around the frozen legacy builders."""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
import time
from typing import Any, Mapping
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from trading_pipeline.experiments import ComponentSpec, FitContext, SupervisedModel


class LegacySklearnAdapter(SupervisedModel):
    """Common lifecycle that leaves fold ownership outside the model."""

    SPEC: ComponentSpec
    SEARCH_SPACE: Mapping[str, Any]

    def __init__(self, *, params: Mapping[str, Any] | None = None) -> None:
        self._params = dict(params or self.default_params())
        self._model = None
        self._telemetry: dict[str, Any] = {
            "status": "not_fitted",
            "parameters": dict(self._params),
        }

    @property
    def spec(self) -> ComponentSpec:
        return self.SPEC

    @classmethod
    @abstractmethod
    def default_params(cls) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _build(self, *, seed: int):
        raise NotImplementedError

    def search_space(self) -> Mapping[str, Any]:
        # Return new containers so callers cannot mutate the class declaration.
        return deepcopy(self.SEARCH_SPACE)

    def fit(self, x, y, *, context: FitContext, stopping_data=None):
        if stopping_data is not None:
            raise ValueError(f"{self.spec.component_id} does not support stopping_data")
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        if x_array.ndim != 2:
            raise ValueError("Supervised fit features must be a two-dimensional array")
        if y_array.ndim != 1 or len(x_array) != len(y_array):
            raise ValueError("Supervised fit targets must be one-dimensional and row-aligned")
        if len(x_array) == 0:
            raise ValueError("Supervised fit data must not be empty")

        model = self._build(seed=context.seed)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            model.fit(x_array, y_array)
        duration = time.perf_counter() - started
        fitted = model.named_steps["model"]
        warning_messages = [str(item.message) for item in caught]
        self._model = model
        self._telemetry = {
            "status": "fitted",
            "study_id": context.study_id,
            "trial_id": context.trial_id,
            "fold_id": context.fold_id,
            "seed": context.seed,
            "fidelity": dict(context.fidelity),
            "parameters": dict(self._params),
            "fit_rows": int(x_array.shape[0]),
            "fit_columns": int(x_array.shape[1]),
            "fit_duration_seconds": float(duration),
            "warnings": warning_messages,
            "convergence_warning": any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            ),
            "iterations": int(getattr(fitted, "n_iter_", getattr(fitted, "max_iter", 0))),
        }
        return self

    def predict(self, x):
        if self._model is None:
            raise RuntimeError(f"{self.spec.component_id} must be fitted before prediction")
        with threadpool_limits(limits=1):
            values = self._model.predict(np.asarray(x))
        return np.asarray(values, dtype=np.float64)

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError(f"{self.spec.component_id} must be fitted before saving")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    def telemetry(self) -> Mapping[str, Any]:
        return dict(self._telemetry)

    @property
    def fitted_pipeline(self):
        """Expose the legacy sklearn pipeline for parity checks and persistence."""
        if self._model is None:
            raise RuntimeError(f"{self.spec.component_id} has not been fitted")
        return self._model
