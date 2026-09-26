"""Small, task-specific contracts for registered learning components.

The contracts intentionally do not own data splitting, objective calculation or
candidate selection. Those are responsibilities of the temporal evaluator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


INTERFACES = {"SupervisedModel", "UnsupervisedModel", "RLPolicy"}


@dataclass(frozen=True)
class ComponentSpec:
    """Stable metadata used by configs, ledgers and reporting."""

    component_id: str
    interface: str
    implementation: str
    version: int = 1
    research_enabled: bool = False
    capabilities: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_id or any(ch.isspace() for ch in self.component_id):
            raise ValueError("component_id must be a non-empty whitespace-free identifier")
        if self.interface not in INTERFACES:
            raise ValueError(f"Unsupported component interface: {self.interface}")
        if self.version < 1:
            raise ValueError("component version must be positive")


@dataclass(frozen=True)
class FitContext:
    """Identity and resource facts for one trial/fold/seed fit."""

    study_id: str
    trial_id: str
    fold_id: str
    seed: int
    fidelity: Mapping[str, int | float] = field(default_factory=dict)


class RegisteredComponent(ABC):
    """Shared metadata boundary; subclasses deliberately have different APIs."""

    @property
    @abstractmethod
    def spec(self) -> ComponentSpec:
        raise NotImplementedError

    @abstractmethod
    def search_space(self) -> Mapping[str, Any]:
        """Return a declarative, serialisable conditional search space."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def telemetry(self) -> Mapping[str, Any]:
        raise NotImplementedError


class SupervisedModel(RegisteredComponent):
    """Predictive model fitted without ownership of folds or objectives."""

    @abstractmethod
    def fit(self, x, y, *, context: FitContext, stopping_data=None) -> "SupervisedModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError


class UnsupervisedModel(RegisteredComponent):
    """Train-only representation or state model with causal future transform."""

    @abstractmethod
    def fit(self, x, *, context: FitContext) -> "UnsupervisedModel":
        raise NotImplementedError

    @abstractmethod
    def transform(self, x):
        raise NotImplementedError


class RLPolicy(RegisteredComponent):
    """Policy learning boundary for a frozen external environment."""

    @abstractmethod
    def learn(self, environment, *, context: FitContext) -> "RLPolicy":
        raise NotImplementedError

    @abstractmethod
    def act(self, observation, *, deterministic: bool = True) -> int:
        raise NotImplementedError
