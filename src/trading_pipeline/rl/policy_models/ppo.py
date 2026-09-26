"""Research-disabled categorical PPO adapter for the discrete selector MDP."""

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


PPO_SPEC = ComponentSpec(
    component_id="rl_ppo_categorical_sb3_v1",
    interface="RLPolicy",
    implementation="trading_pipeline.rl.policy_models.CategoricalPPOPolicy",
    version=1,
    research_enabled=False,
    capabilities={
        "discrete_actions": True,
        "categorical_policy": True,
        "continuation": False,
        "staged_metrics": True,
        "gpu": True,
    },
)


class CategoricalPPOPolicy(StableBaselinesPolicyAdapter):
    algorithm_name = "PPO"
    component_spec = PPO_SPEC

    _defaults: Mapping[str, Any] = {
        "policy": "MlpPolicy",
        "architecture_mode": "matched",
        "policy_widths": [64, 64],
        "value_widths": [64, 64],
        "activation": "tanh",
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
    }

    def search_space(self) -> Mapping[str, Any]:
        return {
            "schema_version": 1,
            "resource_axis": "environment_steps",
            "parameters": {
                "architecture_mode": {"type": "categorical", "values": ["matched", "separate"]},
                "policy_widths": {
                    "type": "categorical", "values": [[64], [64, 64], [128, 128]]
                },
                "value_widths": {
                    "type": "categorical", "values": [[64], [64, 64], [128, 128]],
                    "active_if": {"architecture_mode": "separate"},
                },
                "activation": {"type": "categorical", "values": ["tanh", "relu"]},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
                "n_steps": {"type": "categorical", "values": [128, 256, 512, 1024]},
                "batch_size": {"type": "categorical", "values": [32, 64, 128, 256]},
                "n_epochs": {"type": "categorical", "values": [3, 5, 10, 20]},
                "gamma": {"type": "float", "low": 0.9, "high": 0.999},
                "gae_lambda": {"type": "float", "low": 0.8, "high": 1.0},
                "clip_range": {"type": "float", "low": 0.1, "high": 0.3},
                "ent_coef": {"type": "float", "low": 1e-6, "high": 0.05, "log": True},
                "vf_coef": {"type": "float", "low": 0.1, "high": 1.0},
            },
            "conditions": [
                {"if": {"architecture_mode": "matched"},
                 "require": "value_widths == policy_widths"},
                {"if": {"batch_size": {"set": True}},
                 "require": "n_steps % batch_size == 0"},
            ],
        }

    def _validate_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        unknown = set(parameters) - set(self._defaults)
        if unknown:
            raise ValueError(f"Unsupported PPO parameters: {sorted(unknown)}")
        explicit_value_widths = "value_widths" in parameters
        values = {**self._defaults, **parameters}
        if values["policy"] != "MlpPolicy":
            raise ValueError("PPO policy must be MlpPolicy")
        if values["architecture_mode"] not in {"matched", "separate"}:
            raise ValueError("architecture_mode must be matched or separate")
        policy_widths = _widths("policy_widths", values["policy_widths"])
        value_widths = _widths("value_widths", values["value_widths"])
        if values["architecture_mode"] == "matched":
            if explicit_value_widths and value_widths != policy_widths:
                raise ValueError("matched architecture requires identical policy/value widths")
            value_widths = policy_widths
        values["policy_widths"] = list(policy_widths)
        values["value_widths"] = list(value_widths)
        if values["activation"] not in {"relu", "tanh"}:
            raise ValueError("activation must be relu or tanh")
        values["learning_rate"] = _finite_float("learning_rate", values["learning_rate"], minimum=0.0)
        for name in ("n_steps", "batch_size", "n_epochs"):
            values[name] = _positive_int(name, values[name], minimum=2 if name != "n_epochs" else 1)
        if values["n_steps"] % values["batch_size"]:
            raise ValueError("n_steps must be divisible by batch_size for the single selector environment")
        for name in ("gamma", "gae_lambda", "clip_range", "vf_coef"):
            values[name] = _finite_float(name, values[name], minimum=0.0, maximum=1.0)
        values["ent_coef"] = _finite_float("ent_coef", values["ent_coef"], minimum=0.0)
        values["max_grad_norm"] = _finite_float("max_grad_norm", values["max_grad_norm"], minimum=0.0)
        if not isinstance(values["normalize_advantage"], bool):
            raise ValueError("normalize_advantage must be boolean")
        return values

    def _load_backend_factory(self) -> BackendFactory:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "PPO requires the optional 'rl' dependencies; install the project RL extra"
            ) from exc
        return PPO

    def _backend_parameters(self) -> dict[str, Any]:
        values = dict(self._parameters)
        values.pop("policy")
        mode = values.pop("architecture_mode")
        policy_widths = values.pop("policy_widths")
        value_widths = values.pop("value_widths")
        activation = values.pop("activation")
        try:
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(
                "PPO requires the optional 'rl' dependencies; install the project RL extra"
            ) from exc
        activation_fn = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        net_arch: list[int] | dict[str, list[int]]
        if mode == "matched":
            net_arch = policy_widths
        else:
            net_arch = {"pi": policy_widths, "vf": value_widths}
        values["policy_kwargs"] = {"net_arch": net_arch, "activation_fn": activation_fn}
        return values

