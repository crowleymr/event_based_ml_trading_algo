"""Shared lazy adapter machinery for Stable-Baselines3 policies.

The adapters in this package deliberately do not own evaluation, rewards, folds or
candidate selection.  They receive an already-constructed environment from the
temporal evaluator and use the seed and fidelity assigned by ``FitContext``.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
import math
import time
from typing import Any, Callable, Mapping

from trading_pipeline.experiments import ComponentSpec, FitContext, RLPolicy


BackendFactory = Callable[..., Any]


def _finite_float(name: str, value: Any, *, minimum: float | None = None,
                  maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _widths(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list of positive integers")
    widths = tuple(_positive_int(name, item) for item in value)
    if len(widths) > 4:
        raise ValueError(f"{name} supports at most four hidden layers")
    return widths


def _json_value(value: Any) -> Any:
    """Convert adapter-owned state to JSON-safe values without guessing."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


class StableBaselinesPolicyAdapter(RLPolicy):
    """Base for lazily constructed Stable-Baselines3 discrete policies."""

    algorithm_name: str
    component_spec: ComponentSpec

    def __init__(
        self,
        *,
        parameters: Mapping[str, Any] | None = None,
        total_timesteps: int,
        device: str = "cpu",
        verbose: int = 0,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda or auto")
        if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer")
        self.total_timesteps = _positive_int("total_timesteps", total_timesteps)
        self.device = device
        self.verbose = verbose
        self._parameters = self._validate_parameters(dict(parameters or {}))
        self._backend_factory = backend_factory
        self._model: Any | None = None
        self._telemetry: dict[str, Any] = {
            "schema_version": 1,
            "component_id": self.component_spec.component_id,
            "algorithm": self.algorithm_name,
            "research_enabled": False,
            "status": "not_fitted",
            "requested_device": device,
            "parameters": _json_value(self._parameters),
            "resource_request": {"environment_steps": self.total_timesteps},
        }

    @property
    def spec(self) -> ComponentSpec:
        return self.component_spec

    @abstractmethod
    def _validate_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _load_backend_factory(self) -> BackendFactory:
        """Import and return the optional backend only when learning starts."""
        raise NotImplementedError

    @abstractmethod
    def _backend_parameters(self) -> dict[str, Any]:
        raise NotImplementedError

    def _steps_for(self, context: FitContext) -> int:
        unknown = set(context.fidelity) - {"environment_steps"}
        if unknown:
            raise ValueError(f"Unsupported RL fidelity fields: {sorted(unknown)}")
        value = context.fidelity.get("environment_steps", self.total_timesteps)
        return _positive_int("fidelity.environment_steps", value)

    @staticmethod
    def _validate_discrete_environment(environment: Any) -> None:
        action_space = getattr(environment, "action_space", None)
        count = getattr(action_space, "n", None)
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("RL policy requires a discrete action space with at least two actions")

    def learn(self, environment: Any, *, context: FitContext) -> "StableBaselinesPolicyAdapter":
        self._validate_discrete_environment(environment)
        steps = self._steps_for(context)
        factory = self._backend_factory or self._load_backend_factory()
        started = time.perf_counter()
        try:
            self._model = factory(
                self._parameters["policy"],
                environment,
                seed=context.seed,
                device=self.device,
                verbose=self.verbose,
                **self._backend_parameters(),
            )
            self._model.learn(total_timesteps=steps, progress_bar=False)
        except Exception:
            self._model = None
            self._telemetry.update({
                "status": "failed",
                "study_id": context.study_id,
                "trial_id": context.trial_id,
                "fold_id": context.fold_id,
                "seed": context.seed,
                "environment_steps_requested": steps,
                "duration_seconds": time.perf_counter() - started,
            })
            raise
        self._telemetry.update({
            "status": "fitted",
            "study_id": context.study_id,
            "trial_id": context.trial_id,
            "fold_id": context.fold_id,
            "seed": context.seed,
            "environment_steps_requested": steps,
            "environment_steps_completed": int(getattr(self._model, "num_timesteps", steps)),
            "gradient_updates": self._gradient_updates(),
            "actual_device": str(getattr(self._model, "device", "unknown")),
            "duration_seconds": time.perf_counter() - started,
            "backend_metrics": self._backend_metrics(),
        })
        return self

    def _gradient_updates(self) -> int | None:
        if self._model is None:
            return None
        value = getattr(self._model, "_n_updates", None)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    def _backend_metrics(self) -> dict[str, float]:
        if self._model is None:
            return {}
        logger = getattr(self._model, "logger", None)
        values = getattr(logger, "name_to_value", {})
        if not isinstance(values, Mapping):
            return {}
        result: dict[str, float] = {}
        for name, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                result[str(name)] = number
        return result

    def act(self, observation: Any, *, deterministic: bool = True) -> int:
        if self._model is None:
            raise RuntimeError("Policy must be learned before act")
        action, _state = self._model.predict(observation, deterministic=deterministic)
        try:
            value = int(action.item())
        except AttributeError:
            value = int(action)
        return value

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("Policy must be learned before save")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(destination))

    def telemetry(self) -> Mapping[str, Any]:
        return _json_value(self._telemetry)

