"""Research-disabled unsupervised component implementations."""

from trading_pipeline.experiments import ComponentRegistry
from trading_pipeline.experiments.discovery import register_package

from .gmm_regime import GMM_REGIME_SPEC, GaussianMixtureRegimeModel
from .no_regime import NO_REGIME_SPEC, NoRegimeModel


def register_unsupervised_models(registry: ComponentRegistry) -> None:
    """Add this lane's allowlisted factories to a caller-owned registry."""

    import sys

    register_package(registry, sys.modules[__name__])

__all__ = [
    "GMM_REGIME_SPEC",
    "GaussianMixtureRegimeModel",
    "NO_REGIME_SPEC",
    "NoRegimeModel",
    "register_unsupervised_models",
]
