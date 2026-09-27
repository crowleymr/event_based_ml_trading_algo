"""Research-disabled Gaussian-mixture state representation.

The model performs an i.i.d. forward assignment using parameters and scaling fitted
only on the supplied training matrix. It is not a temporal state-space model: it does
not model persistence, transition probabilities, or causal filtering. Component IDs
are arbitrary and must never be given semantic regime labels based on future returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from trading_pipeline.experiments import ComponentSpec, FitContext, UnsupervisedModel

from .base import context_telemetry, numeric_matrix, require_fitted


GMM_REGIME_SPEC = ComponentSpec(
    component_id="gmm_regime_v1",
    interface="UnsupervisedModel",
    implementation=(
        "trading_pipeline.modelling.unsupervised_models.gmm_regime."
        "GaussianMixtureRegimeModel"
    ),
    research_enabled=False,
    capabilities={
        "causal_forward_transform": True,
        "probabilistic_states": True,
        "forward_density": True,
        "temporal_state_model": False,
    },
)

_COVARIANCE_TYPES = {"full", "tied", "diag", "spherical"}


class GaussianMixtureRegimeModel(UnsupervisedModel):
    """Train-only scaled GMM emitting posterior state probabilities.

    ``fixed_restarts`` is intentionally not part of the search space. Each fit uses
    that fixed number of deterministic sklearn restarts seeded by ``FitContext.seed``.
    Candidate selection and forward-density aggregation belong to the external
    temporal evaluator.
    """

    def __init__(
        self,
        *,
        n_components: int = 2,
        covariance_type: str = "diag",
        reg_covar: float = 1e-6,
        fixed_restarts: int = 5,
        max_iter: int = 200,
        observation_unit: str = "time_observation",
        information_cutoff: str | None = None,
    ) -> None:
        if n_components < 2:
            raise ValueError("GMM n_components must be at least 2")
        if covariance_type not in _COVARIANCE_TYPES:
            raise ValueError(f"Unsupported covariance_type: {covariance_type}")
        if not np.isfinite(reg_covar) or reg_covar <= 0:
            raise ValueError("reg_covar must be finite and positive")
        if fixed_restarts < 1:
            raise ValueError("fixed_restarts must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        if not observation_unit.strip():
            raise ValueError("observation_unit must be non-empty")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.fixed_restarts = fixed_restarts
        self.max_iter = max_iter
        self.observation_unit = observation_unit
        self.information_cutoff = information_cutoff
        self._scaler: StandardScaler | None = None
        self._model: GaussianMixture | None = None
        self._fit_context: FitContext | None = None
        self._train_rows: int | None = None
        self._n_features: int | None = None
        self._occupancy_counts: list[int] | None = None

    @property
    def spec(self) -> ComponentSpec:
        return GMM_REGIME_SPEC

    def search_space(self) -> Mapping[str, Any]:
        """Small, auditable space; restarts and iteration budget remain fixed."""

        return {
            "parameters": {
                "n_components": {"type": "categorical", "values": [2, 3, 4]},
                "covariance_type": {
                    "type": "categorical",
                    "values": ["diag", "full"],
                },
                "reg_covar": {
                    "type": "categorical",
                    "values": [1e-6, 1e-4],
                    "scale": "log",
                },
            },
            "fixed": {
                "fixed_restarts": self.fixed_restarts,
                "max_iter": self.max_iter,
                "preprocessing": "train_only_standard_scaler",
            },
            "objective": "forward_log_predictive_density",
        }

    def fit(
        self, x: ArrayLike, *, context: FitContext
    ) -> "GaussianMixtureRegimeModel":
        values = numeric_matrix(x, minimum_rows=self.n_components)
        if np.unique(values, axis=0).shape[0] < self.n_components:
            raise ValueError(
                "Training data must contain at least n_components distinct observations"
            )
        scaler = StandardScaler()
        scaled = scaler.fit_transform(values)
        model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            n_init=self.fixed_restarts,
            max_iter=self.max_iter,
            init_params="kmeans",
            random_state=context.seed,
        )
        model.fit(scaled)
        labels = model.predict(scaled)
        self._scaler = scaler
        self._model = model
        self._fit_context = context
        self._train_rows = values.shape[0]
        self._n_features = values.shape[1]
        self._occupancy_counts = np.bincount(
            labels, minlength=self.n_components
        ).astype(int).tolist()
        return self

    def _scaled(self, x: ArrayLike) -> NDArray[np.float64]:
        require_fitted(self._model, component_id=self.spec.component_id)
        require_fitted(self._scaler, component_id=self.spec.component_id)
        values = numeric_matrix(x)
        if values.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, received {values.shape[1]}"
            )
        return self._scaler.transform(values)

    def transform(self, x: ArrayLike) -> NDArray[np.float64]:
        """Return rows x components posterior probabilities under frozen parameters."""

        assert self._model is not None or self._n_features is None
        scaled = self._scaled(x)
        assert self._model is not None
        return self._model.predict_proba(scaled)

    def score_samples(self, x: ArrayLike) -> NDArray[np.float64]:
        """Return per-observation log density for external forward evaluation."""

        scaled = self._scaled(x)
        assert self._model is not None
        return self._model.score_samples(scaled)

    def save(self, path: str | Path) -> None:
        require_fitted(self._model, component_id=self.spec.component_id)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    def telemetry(self) -> Mapping[str, Any]:
        model = self._model
        fitted = model is not None
        counts = self._occupancy_counts
        fractions = None
        if counts is not None and self._train_rows:
            # Sorted values are deliberately permutation invariant. Raw component IDs
            # remain available only in transform output and carry no semantic labels.
            fractions = sorted(count / self._train_rows for count in counts)
        return {
            "component_id": self.spec.component_id,
            "research_enabled": self.spec.research_enabled,
            "fitted": fitted,
            "observation_unit": self.observation_unit,
            "information_cutoff": self.information_cutoff,
            "train_rows": self._train_rows,
            "n_features": self._n_features,
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "reg_covar": self.reg_covar,
            "fixed_restarts": self.fixed_restarts,
            "max_iter": self.max_iter,
            "converged": bool(model.converged_) if fitted else None,
            "n_iter": int(model.n_iter_) if fitted else None,
            "lower_bound": float(model.lower_bound_) if fitted else None,
            "occupancy_counts_by_component": counts,
            "occupancy_fractions_sorted": fractions,
            "empty_component_count": (
                sum(count == 0 for count in counts) if counts is not None else None
            ),
            "fit_context": context_telemetry(self._fit_context),
            "causal_limitations": [
                "Assignments are independent per row; temporal persistence is not modelled.",
                "Scaling and mixture parameters are frozen after train-only fitting.",
                "Component identifiers are arbitrary and have no semantic regime labels.",
                "Downstream trading value requires a separately authorised nested study.",
            ],
        }


COMPONENT_REGISTRATIONS = ((GMM_REGIME_SPEC, GaussianMixtureRegimeModel),)
