"""Deterministic proposal generation, independent of evaluation data."""

from __future__ import annotations

from itertools import product
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _active(dimension: Mapping[str, Any], proposal: Mapping[str, Any]) -> bool:
    condition = dimension.get("when", dimension.get("active_if"))
    if condition is None:
        return True
    if not isinstance(condition, dict) or len(condition) != 1:
        raise ValueError("when must contain exactly one parent/value condition")
    parent, expected = next(iter(condition.items()))
    return proposal.get(parent) == expected


def _choices(name: str, dimension: Mapping[str, Any]) -> list[Any]:
    if dimension.get("kind", dimension.get("type")) != "categorical" or not dimension.get("values"):
        raise ValueError(f"Grid dimension {name} must provide categorical values")
    return list(dimension["values"])


def grid_proposals(space: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand a small declarative grid while respecting parent conditions."""
    names = list(space)
    candidates = [dict(zip(names, values)) for values in product(*[
        _choices(name, space[name]) for name in names
    ])]
    result = []
    for candidate in candidates:
        resolved = {
            name: value for name, value in candidate.items()
            if _active(space[name], candidate)
        }
        if resolved not in result:
            result.append(resolved)
    return result


def _draw(generator: np.random.Generator, name: str, dimension: Mapping[str, Any]):
    kind = dimension.get("kind", dimension.get("type"))
    if kind == "categorical":
        values: Sequence[Any] = dimension.get("values", ())
        if not values:
            raise ValueError(f"Categorical dimension {name} has no values")
        return values[int(generator.integers(0, len(values)))]
    low, high = dimension.get("low"), dimension.get("high")
    if low is None or high is None or not low < high:
        raise ValueError(f"Dimension {name} requires low < high")
    if kind in {"uniform", "float"} and not dimension.get("log", False):
        return float(generator.uniform(low, high))
    if kind == "loguniform" or (kind == "float" and dimension.get("log", False)):
        if low <= 0:
            raise ValueError(f"Log-uniform dimension {name} requires low > 0")
        return float(math.exp(generator.uniform(math.log(low), math.log(high))))
    if kind in {"int", "integer"}:
        return int(generator.integers(int(low), int(high) + 1))
    raise ValueError(f"Unsupported search dimension kind for {name}: {kind}")


def random_proposals(
    space: Mapping[str, Mapping[str, Any]], attempts: int, seed: int
) -> list[dict[str, Any]]:
    """Produce a reproducible mixed-space screen without observing scores."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    generator = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    max_draws = attempts * 100
    for _ in range(max_draws):
        proposal: dict[str, Any] = {}
        for name, dimension in space.items():
            if _active(dimension, proposal):
                proposal[name] = _draw(generator, name, dimension)
        if proposal not in results:
            results.append(proposal)
        if len(results) == attempts:
            return results
    raise ValueError("Search space could not produce the requested unique proposals")


def _reference_value(name: str, dimension: Mapping[str, Any]) -> Any:
    kind = dimension.get("kind", dimension.get("type"))
    if kind == "categorical":
        return _choices(name, dimension)[0]
    low, high = dimension.get("low"), dimension.get("high")
    if low is None or high is None or not low < high:
        raise ValueError(f"Dimension {name} requires low < high")
    if kind in {"loguniform", "float"} and (
        kind == "loguniform" or dimension.get("log", False)
    ):
        if low <= 0:
            raise ValueError(f"Log-scaled dimension {name} requires low > 0")
        return float(math.sqrt(low * high))
    if kind in {"uniform", "float"}:
        return float((low + high) / 2)
    if kind in {"int", "integer"}:
        return int((int(low) + int(high)) // 2)
    raise ValueError(f"Unsupported search dimension kind for {name}: {kind}")


def _contrast_values(name: str, dimension: Mapping[str, Any], reference: Any) -> list[Any]:
    kind = dimension.get("kind", dimension.get("type"))
    if kind == "categorical":
        return [value for value in _choices(name, dimension) if value != reference]
    low, high = dimension.get("low"), dimension.get("high")
    values = [high, low]
    if kind in {"int", "integer"}:
        values = [int(value) for value in values]
    else:
        values = [float(value) for value in values]
    return [value for value in values if value != reference]


def controlled_one_factor_design(
    space: Mapping[str, Mapping[str, Any]],
    validate: Callable[[Mapping[str, Any]], None],
) -> dict[str, Any]:
    """Build a deterministic local sensitivity design from declarative metadata.

    The design contains one common reference and one valid contrast per active
    dimension. Conditional dimensions receive a separate conditional reference.
    Each comparison changes only its named factor, apart from fields that become
    active or inactive as a direct consequence of changing a parent condition.
    """
    if not isinstance(space, Mapping) or not space:
        raise ValueError("Controlled sensitivity requires a nonempty search space")

    def resolved(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        proposal: dict[str, Any] = {}
        requested = dict(overrides or {})
        for name, dimension in space.items():
            if _active(dimension, proposal):
                proposal[name] = requested.get(name, _reference_value(name, dimension))
        unknown = set(requested) - set(proposal)
        if unknown:
            raise ValueError(f"Inactive controlled parameters: {sorted(unknown)}")
        validate(proposal)
        return proposal

    proposals: list[dict[str, Any]] = []

    def index(candidate: dict[str, Any]) -> int:
        if candidate not in proposals:
            proposals.append(candidate)
        return proposals.index(candidate)

    reference = resolved()
    reference_index = index(reference)
    contrasts: list[dict[str, Any]] = []
    for name, dimension in space.items():
        condition = dimension.get("when", dimension.get("active_if"))
        base = reference
        if condition is not None and not _active(dimension, reference):
            if not isinstance(condition, Mapping) or len(condition) != 1:
                raise ValueError(f"Conditional dimension {name} requires one parent condition")
            parent, expected = next(iter(condition.items()))
            base = resolved({parent: expected})
        base_index = index(base)
        if name not in base:
            raise ValueError(f"Controlled reference could not activate {name}")
        variant = None
        for value in _contrast_values(name, dimension, base[name]):
            try:
                candidate = resolved({**base, name: value})
            except (TypeError, ValueError):
                continue
            if candidate != base:
                variant = candidate
                break
        if variant is None:
            raise ValueError(f"No valid controlled contrast for {name}")
        variant_index = index(variant)
        directly_changed = sorted(
            key for key in set(base) | set(variant) if base.get(key) != variant.get(key)
        )
        allowed = {name}
        if condition is None:
            for child, child_dimension in space.items():
                child_condition = child_dimension.get("when", child_dimension.get("active_if"))
                if isinstance(child_condition, Mapping) and name in child_condition:
                    allowed.add(child)
        if not set(directly_changed) <= allowed:
            raise ValueError(f"Contrast for {name} changes unrelated fields: {directly_changed}")
        contrasts.append({
            "comparison_group_id": f"factor::{name}",
            "factor_name": name,
            "reference_proposal_index": base_index,
            "contrast_proposal_index": variant_index,
            "reference_level": base[name],
            "contrast_level": variant[name],
            "directly_changed_fields": directly_changed,
            "held_constant_parameters": {
                key: value for key, value in base.items()
                if key in variant and key not in directly_changed
            },
        })
    return {
        "schema_version": 1,
        "design": "deterministic_controlled_one_factor_v1",
        "global_reference_proposal_index": reference_index,
        "proposals": proposals,
        "contrasts": contrasts,
    }
