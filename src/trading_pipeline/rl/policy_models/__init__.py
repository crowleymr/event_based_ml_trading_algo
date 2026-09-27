"""Disabled RL policy adapters for configuration-driven engineering studies."""

from .dqn import DQNPolicy, DQN_SPEC
from .ppo import CategoricalPPOPolicy, PPO_SPEC
from trading_pipeline.experiments.discovery import register_package
from trading_pipeline.experiments.registry import ComponentRegistry


def register_policy_models(registry: ComponentRegistry) -> None:
    import sys

    register_package(registry, sys.modules[__name__])

__all__ = ["CategoricalPPOPolicy", "DQNPolicy", "DQN_SPEC", "PPO_SPEC", "register_policy_models"]
