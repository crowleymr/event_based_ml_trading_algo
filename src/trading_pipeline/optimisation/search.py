"""Deterministic proposal generation, independent of evaluation data."""

from __future__ import annotations

from itertools import product
import math
from typing import Any, Mapping, Sequence

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
