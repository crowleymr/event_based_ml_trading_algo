"""Optional registered adapter for the existing diagnostic XGBoost model."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from trading_pipeline.experiments import ComponentSpec, FitContext, SupervisedModel


XGBOOST_SPEC = ComponentSpec(
    component_id="supervised.xgboost.v1",
    interface="SupervisedModel",
    implementation=(
        "trading_pipeline.modelling.supervised_models.xgboost.XGBoostModel"
    ),
    version=1,
    research_enabled=False,
    capabilities={
        "study_adapter": True,
        "tabular_view": True,
        "train_only_preprocessing": True,
        "stopping_data": True,
        "continuation": False,
        "staged_metrics": True,
        "feature_importance": True,
        "optional_dependency": True,
        "cpu": True,
        "cuda": True,
    },
)

XGBOOST_SEARCH_SPACE: Mapping[str, Any] = {
    "schema_version": 1,
    "strategy": "explicit_grid",
    "parameters": {
        "max_depth": {"type": "categorical", "values": [2, 3, 4, 5, 6, 8]},
        "reg_lambda": {"type": "categorical", "values": [0.1, 1.0, 10.0, 100.0]},
    },
}


class XGBoostModel(SupervisedModel):
    """Adapter that defers importing the optional XGBoost package until fitting."""

    def __init__(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        device: str = "cpu",
        training_config: Mapping[str, Any] | None = None,
    ) -> None:
        self._params = dict(params or {"max_depth": 3, "reg_lambda": 1.0})
        self._device = str(device)
        self._training_config = dict(training_config or {})
        self._model = None
        self._telemetry: dict[str, Any] = {
            "status": "not_fitted",
            "parameters": dict(self._params),
            "requested_device": self._device,
        }

    @property
    def spec(self) -> ComponentSpec:
        return XGBOOST_SPEC

    def search_space(self) -> Mapping[str, Any]:
        return deepcopy(XGBOOST_SEARCH_SPACE)

    def fit(self, x, y, *, context: FitContext, stopping_data=None):
        if stopping_data is None:
            raise ValueError("supervised.xgboost.v1 requires chronological stopping_data")
        try:
            x_stopping, y_stopping = stopping_data
        except (TypeError, ValueError) as exc:
            raise ValueError("stopping_data must be an (x, y) pair") from exc

        x_array = np.asarray(x)
        y_array = np.asarray(y)
        x_stop_array = np.asarray(x_stopping)
        y_stop_array = np.asarray(y_stopping)
        if x_array.ndim != 2 or x_stop_array.ndim != 2:
            raise ValueError("Supervised fit and stopping features must be two-dimensional")
        if y_array.ndim != 1 or len(x_array) != len(y_array):
            raise ValueError("Supervised fit targets must be one-dimensional and row-aligned")
        if y_stop_array.ndim != 1 or len(x_stop_array) != len(y_stop_array):
            raise ValueError("Stopping targets must be one-dimensional and row-aligned")
        if not len(x_array) or not len(x_stop_array):
            raise ValueError("Fit and stopping data must not be empty")
        if x_array.shape[1] != x_stop_array.shape[1]:
            raise ValueError("Fit and stopping data must have identical feature counts")

        # Importing this module is dependency-light; its fit function resolves the
        # optional xgboost package only when this component is actually executed.
        from trading_pipeline.modelling.xgboost_model import (
            fit_xgboost,
            selected_details,
        )

        model = fit_xgboost(
            self._params,
            context.seed,
            self._device,
            self._training_config,
            x_array,
            y_array,
            x_stop_array,
            y_stop_array,
        )
        details = selected_details(model)
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
            "stopping_rows": int(x_stop_array.shape[0]),
            "fit_duration_seconds": float(model._telemetry_fit_duration_seconds),
            "requested_device": details["requested_device"],
            "actual_device": details["actual_device"],
            "fallback_reason": details["fallback_reason"],
            "selected_iteration": details["selected_iteration"],
            "best_score": details["best_score"],
            "build_info": details["build_info"],
            "evals_result": details["evals_result"],
        }
        return self

    def predict(self, x):
        if self._model is None:
            raise RuntimeError("supervised.xgboost.v1 must be fitted before prediction")
        return np.asarray(self._model.predict(np.asarray(x)), dtype=np.float64)

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("supervised.xgboost.v1 must be fitted before saving")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    def telemetry(self) -> Mapping[str, Any]:
        return deepcopy(self._telemetry)

    @property
    def fitted_pipeline(self):
        if self._model is None:
            raise RuntimeError("supervised.xgboost.v1 has not been fitted")
        return self._model


COMPONENT_REGISTRATIONS = ((XGBOOST_SPEC, XGBoostModel),)
