"""Complete-fold aggregation and an immutable pre-holdout selection lock."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .evaluator import FitEvidence


@dataclass(frozen=True)
class CandidateScore:
    trial_id: str
    mean_score: float
    complexity: float
    secondary_score: float
    resource_use: float


def aggregate_complete_trials(
    evidence: Sequence[FitEvidence], *, expected_folds: Sequence[str],
    expected_seeds: Sequence[int], complexity: Mapping[str, float],
    secondary: Mapping[str, float], resource_use: Mapping[str, float],
) -> tuple[CandidateScore, ...]:
    expected = {(fold, seed) for fold in expected_folds for seed in expected_seeds}
    if not expected or len(expected) != len(expected_folds) * len(expected_seeds):
        raise ValueError("Expected folds and seeds must be nonempty and unique")
    groups: dict[str, list[FitEvidence]] = {}
    for item in evidence:
        groups.setdefault(item.trial_id, []).append(item)
    result = []
    for trial_id, records in groups.items():
        keys = [(item.inner_fold_id, item.seed) for item in records]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError(f"Incomplete or duplicate fold/seed matrix for {trial_id}")
        if len({(item.study_id, item.outer_fold_id) for item in records}) != 1:
            raise ValueError(f"Mixed study or outer fold for {trial_id}")
        if any(not math.isfinite(item.score) for item in records):
            raise ValueError(f"Nonfinite score for {trial_id}")
        extras = (complexity[trial_id], secondary[trial_id], resource_use[trial_id])
        if any(not math.isfinite(value) for value in extras):
            raise ValueError(f"Nonfinite selection metadata for {trial_id}")
        # Equal fold weighting; each fold first averages its complete seed set.
        fold_means = [sum(item.score for item in records if item.inner_fold_id == fold)
                      / len(expected_seeds) for fold in expected_folds]
        result.append(CandidateScore(trial_id, sum(fold_means) / len(fold_means), *extras))
    return tuple(result)


def select_candidate(candidates: Sequence[CandidateScore], *, direction: str,
                     equivalence_tolerance: float,
                     secondary_direction: str) -> CandidateScore:
    if (not candidates or direction not in {"maximize", "minimize"}
            or secondary_direction not in {"maximize", "minimize"}):
        raise ValueError("Candidates and objective direction are required")
    if not math.isfinite(equivalence_tolerance) or equivalence_tolerance < 0:
        raise ValueError("Equivalence tolerance must be finite and nonnegative")
    signed = 1 if direction == "maximize" else -1
    secondary_signed = 1 if secondary_direction == "maximize" else -1
    best = max(signed * item.mean_score for item in candidates)
    equivalent = [item for item in candidates
                  if best - signed * item.mean_score <= equivalence_tolerance]
    return min(equivalent, key=lambda item:
               (item.complexity, -secondary_signed * item.secondary_score,
                item.resource_use, item.trial_id))


def write_selection_lock(path: str | Path, *, study_id: str, outer_fold_id: str,
                         selected: CandidateScore, objective: str,
                         direction: str, evidence_sha256: str) -> Path:
    """Seal the selected configuration before any outer/holdout evaluation."""
    if not all((study_id, outer_fold_id, objective, evidence_sha256)):
        raise ValueError("Selection lock requires complete identity and evidence hash")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"study_id": study_id, "outer_fold_id": outer_fold_id,
               "selected": asdict(selected), "objective": objective,
               "direction": direction, "evidence_sha256": evidence_sha256,
               "sealed_at": datetime.now(timezone.utc).isoformat()}
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    return destination
