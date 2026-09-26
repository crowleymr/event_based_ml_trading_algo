"""Allowlisted component registry; config files never import arbitrary Python."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .contracts import ComponentSpec, RegisteredComponent


@dataclass(frozen=True)
class _Registration:
    spec: ComponentSpec
    factory: Callable[..., RegisteredComponent]


class ComponentRegistry:
    def __init__(self) -> None:
        self._items: dict[str, _Registration] = {}

    def register(
        self, spec: ComponentSpec, factory: Callable[..., RegisteredComponent]
    ) -> None:
        if spec.component_id in self._items:
            raise ValueError(f"Duplicate component id: {spec.component_id}")
        self._items[spec.component_id] = _Registration(spec, factory)

    def spec(self, component_id: str) -> ComponentSpec:
        try:
            return self._items[component_id].spec
        except KeyError as exc:
            raise ValueError(f"Unknown component id: {component_id}") from exc

    def create(
        self, component_id: str, *, require_research_enabled: bool = False, **kwargs: Any
    ) -> RegisteredComponent:
        registration = self._items.get(component_id)
        if registration is None:
            raise ValueError(f"Unknown component id: {component_id}")
        if require_research_enabled and not registration.spec.research_enabled:
            raise ValueError(f"Component is not enabled for research: {component_id}")
        value = registration.factory(**kwargs)
        if value.spec != registration.spec:
            raise ValueError(f"Factory metadata mismatch for component: {component_id}")
        return value

    def ids(self, interface: str | None = None) -> tuple[str, ...]:
        values: Iterable[_Registration] = self._items.values()
        if interface is not None:
            values = (item for item in values if item.spec.interface == interface)
        return tuple(sorted(item.spec.component_id for item in values))
