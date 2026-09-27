"""Research-disabled DQN adapter for the frozen discrete selector environment."""

from __future__ import annotations

from typing import Any, Mapping

from trading_pipeline.experiments import ComponentSpec

from .base import (
    BackendFactory,
    StableBaselinesPolicyAdapter,
    _finite_float,
    _positive_int,
    _widths,
)


DQN_SPEC = ComponentSpec(
    component_id="rl_dqn_sb3_v1",
    interface="RLPolicy",
    implementation="trading_pipeline.rl.policy_models.DQNPolicy",
    version=1,
    research_enabled=False,
    capabilities={
        "study_adapter": True,
        "discrete_actions": True,
        "continuation": False,
        "staged_metrics": True,
        "gpu": True,
    },
)


class DQNPolicy(StableBaselinesPolicyAdapter):
    algorithm_name = "DQN"
    component_spec = DQN_SPEC

    _defaults: Mapping[str, Any] = {
        "policy": "MlpPolicy",
        "network_widths": [64, 64],
        "activation": "relu",
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "learning_starts": 1_000,
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "max_grad_norm": 10.0,
    }

    def search_space(self) -> Mapping[str, Any]:
        return {
            "schema_version": 1,
            "resource_axis": "environment_steps",
            "parameters": {
                "network_widths": {
                    "type": "categorical", "values": [[64], [64, 64], [128, 128]]
                },
                "activation": {"type": "categorical", "values": ["relu", "tanh"]},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
                "batch_size": {"type": "categorical", "values": [32, 64, 128]},
                "buffer_size": {"type": "categorical", "values": [10_000, 50_000, 100_000]},
                "learning_starts": {"type": "categorical", "values": [500, 1_000, 2_000]},
                "train_freq": {"type": "categorical", "values": [1, 4, 8]},
                "gradient_steps": {"type": "categorical", "values": [1, 2, 4]},
                "target_update_interval": {"type": "categorical", "values": [250, 1_000, 2_000]},
                "exploration_fraction": {"type": "float", "low": 0.05, "high": 0.4},
                "exploration_final_eps": {"type": "float", "low": 0.01, "high": 0.1},
                "gamma": {"type": "float", "low": 0.9, "high": 0.999},
            },
            "conditions": [
                {"if": {"batch_size": {"set": True}}, "require": "batch_size <= buffer_size"},
                {"if": {"learning_starts": {"set": True}}, "require": "learning_starts < buffer_size"},
                {"if": {"exploration_final_eps": {"set": True}},
                 "require": "exploration_final_eps <= exploration_initial_eps"},
            ],
        }

    def _validate_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        unknown = set(parameters) - set(self._defaults)
        if unknown:
            raise ValueError(f"Unsupported DQN parameters: {sorted(unknown)}")
        values = {**self._defaults, **parameters}
        if values["policy"] != "MlpPolicy":
            raise ValueError("DQN policy must be MlpPolicy")
        values["network_widths"] = list(_widths("network_widths", values["network_widths"]))
        if values["activation"] not in {"relu", "tanh"}:
            raise ValueError("activation must be relu or tanh")
        values["learning_rate"] = _finite_float("learning_rate", values["learning_rate"], minimum=0.0)
        for name in ("buffer_size", "learning_starts", "batch_size", "train_freq",
                     "gradient_steps", "target_update_interval"):
            values[name] = _positive_int(name, values[name], minimum=0 if name == "learning_starts" else 1)
        if values["batch_size"] > values["buffer_size"]:
            raise ValueError("batch_size must not exceed buffer_size")
        if values["learning_starts"] >= values["buffer_size"]:
            raise ValueError("learning_starts must be smaller than buffer_size")
        for name in ("tau", "gamma", "exploration_fraction", "exploration_initial_eps",
                     "exploration_final_eps"):
            values[name] = _finite_float(name, values[name], minimum=0.0, maximum=1.0)
        if values["exploration_final_eps"] > values["exploration_initial_eps"]:
            raise ValueError("exploration_final_eps must not exceed exploration_initial_eps")
        values["max_grad_norm"] = _finite_float("max_grad_norm", values["max_grad_norm"], minimum=0.0)
        return values

    def _load_backend_factory(self) -> BackendFactory:
        try:
            from stable_baselines3 import DQN
        except ImportError as exc:
            raise RuntimeError(
                "DQN requires the optional 'rl' dependencies; install the project RL extra"
            ) from exc
        return DQN

    def _backend_parameters(self) -> dict[str, Any]:
        values = dict(self._parameters)
        values.pop("policy")
        widths = values.pop("network_widths")
        activation = values.pop("activation")
        try:
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(
                "DQN requires the optional 'rl' dependencies; install the project RL extra"
            ) from exc
        activation_fn = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        values["policy_kwargs"] = {"net_arch": widths, "activation_fn": activation_fn}
        return values


COMPONENT_REGISTRATIONS = ((DQN_SPEC, DQNPolicy),)
