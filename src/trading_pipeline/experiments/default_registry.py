"""Composition root for built-in components; family packages own their fragments."""

from trading_pipeline.modelling.supervised_models import register_supervised_models
from trading_pipeline.modelling.unsupervised_models import register_unsupervised_models
from trading_pipeline.rl.policy_models import register_policy_models

from .registry import ComponentRegistry


def default_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    register_supervised_models(registry)
    register_unsupervised_models(registry)
    register_policy_models(registry)
    return registry
