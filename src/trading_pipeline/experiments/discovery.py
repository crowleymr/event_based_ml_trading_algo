"""Discover component declarations only in trusted, installed model packages."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType

from .contracts import ComponentSpec, RegisteredComponent
from .registry import ComponentRegistry


TRUSTED_PACKAGES = frozenset({
    "trading_pipeline.modelling.supervised_models",
    "trading_pipeline.modelling.unsupervised_models",
    "trading_pipeline.rl.policy_models",
})


def register_package(registry: ComponentRegistry, package: ModuleType) -> None:
    """Import direct child modules of a fixed package and register their metadata.

    Configured component IDs never become import paths. A newly installed trusted
    implementation opts in by declaring ``COMPONENT_REGISTRATIONS`` in its module.
    """
    if package.__name__ not in TRUSTED_PACKAGES or not hasattr(package, "__path__"):
        raise ValueError("Component discovery requires a trusted model package")
    for found in sorted(iter_modules(package.__path__), key=lambda item: item.name):
        if found.ispkg or found.name.startswith("_"):
            continue
        module = import_module(f"{package.__name__}.{found.name}")
        entries = getattr(module, "COMPONENT_REGISTRATIONS", ())
        if not isinstance(entries, tuple):
            raise ValueError(f"Invalid component registrations in {module.__name__}")
        for entry in entries:
            if (not isinstance(entry, tuple) or len(entry) != 2
                    or not isinstance(entry[0], ComponentSpec)
                    or not isinstance(entry[1], type)
                    or not issubclass(entry[1], RegisteredComponent)
                    or entry[1].__module__ != module.__name__):
                raise ValueError(f"Invalid component registration in {module.__name__}")
            registry.register(*entry)
