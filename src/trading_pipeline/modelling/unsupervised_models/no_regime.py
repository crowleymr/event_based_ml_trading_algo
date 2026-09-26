"""No-regime control for experiments that require a registered baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray

from trading_pipeline.experiments import ComponentSpec, FitContext, UnsupervisedModel

from .base import context_telemetry, numeric_matrix, require_fitted


NO_REGIME_SPEC = ComponentSpec(
    component_id="no_regime_v1",
    interface="UnsupervisedModel",
    implementation=(
        "trading_pipeline.modelling.unsupervised_models.no_regime.NoRegimeModel"
    ),
    research_enabled=False,
    capabilities={
        "causal_forward_transform": True,
        "probabilistic_states": True,
        "forward_density": False,
    },
)


class NoRegimeModel(UnsupervisedModel):
    """Emit one constant state without learning from observations.

    This is a structural control, not a density model. Its sole state has no economic
    interpretation and its output is independent of both training and future values.
    """

    def __init__(
        self,
        *,
        observation_unit: str = "time_observation",
        information_cutoff: str | None = None,
    ) -> None:
        if not observation_unit.strip():
            raise ValueError("observation_unit must be non-empty")
        self.observation_unit = observation_unit
        self.information_cutoff = information_cutoff
        self._n_features: int | None = None
        self._fit_context: FitContext | None = None
        self._train_rows: int | None = None

    @property
    def spec(self) -> ComponentSpec:
        return NO_REGIME_SPEC

    def search_space(self) -> Mapping[str, Any]:
        return {"parameters": {}, "fixed": {"n_components": 1}}

    def fit(self, x: ArrayLike, *, context: FitContext) -> "NoRegimeModel":
        values = numeric_matrix(x)
        self._n_features = values.shape[1]
        self._train_rows = values.shape[0]
        self._fit_context = context
        return self

    def transform(self, x: ArrayLike) -> NDArray[np.float64]:
        require_fitted(self._n_features, component_id=self.spec.component_id)
        values = numeric_matrix(x)
        if values.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, received {values.shape[1]}"
            )
        return np.ones((values.shape[0], 1), dtype=np.float64)

    def save(self, path: str | Path) -> None:
        require_fitted(self._n_features, component_id=self.spec.component_id)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    def telemetry(self) -> Mapping[str, Any]:
        fitted = self._n_features is not None
        return {
            "component_id": self.spec.component_id,
            "research_enabled": self.spec.research_enabled,
            "fitted": fitted,
            "observation_unit": self.observation_unit,
            "information_cutoff": self.information_cutoff,
            "train_rows": self._train_rows,
            "n_features": self._n_features,
            "n_components": 1,
            "occupancy_counts": [self._train_rows] if fitted else None,
            "occupancy_fractions": [1.0] if fitted else None,
            "fit_context": context_telemetry(self._fit_context),
            "causal_limitations": [
                "The constant state is a control and has no predictive density.",
                "State identifiers have no economic or semantic meaning.",
            ],
        }
