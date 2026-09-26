"""Fail-closed engineering evaluator over predeclared nested temporal folds.

The only executable role here is synthetic verification.  A score-bearing study
needs separately audited authority, vintage and exposure manifests before this
gate can be widened by the authoritative pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Callable, Mapping, Sequence

from trading_pipeline.experiments import FitContext
from .splits import TemporalFold


@dataclass(frozen=True)
class FitEvidence:
    study_id: str
    trial_id: str
    outer_fold_id: str
    inner_fold_id: str
    seed: int
    score: float
    fit_sessions: int
    stopping_sessions: int
    score_sessions: int
    fidelity: Mapping[str, int | float]
    telemetry: Mapping[str, Any]


class NestedTemporalEvaluator:
    """Supplies only inner-fold rows to a caller-owned fit and score function."""

    def __init__(self, folds: Sequence[tuple[TemporalFold, Sequence[TemporalFold]]]):
        if not folds:
            raise ValueError("At least one outer fold is required")
        self._folds = {outer.fold_id: (outer, tuple(inner)) for outer, inner in folds}
        if len(self._folds) != len(folds):
            raise ValueError("Outer fold identifiers must be unique")
        for outer, inner in self._folds.values():
            if not inner or len({part.fold_id for part in inner}) != len(inner):
                raise ValueError("Each outer fold requires uniquely identified inner folds")
            outer_fit = set(outer.fit_dates)
            for part in inner:
                if not set(part.fit_dates + part.score_dates + part.embargo_dates) <= outer_fit:
                    raise ValueError("Inner fold accesses dates outside outer fit history")

    def evaluate_trial(
        self, *, study_id: str, trial_id: str, outer_fold_id: str,
        rows: Sequence[Mapping[str, Any]], seeds: Sequence[int],
        fit: Callable[[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...], FitContext], Any],
        score: Callable[[Any, tuple[Mapping[str, Any], ...]], float],
        data_role: str, research_enabled: bool = False,
        stopping_sessions: int = 0,
        fidelity: Mapping[str, int | float] | None = None,
    ) -> tuple[FitEvidence, ...]:
        if data_role != "synthetic" or research_enabled:
            raise PermissionError("Evaluator is restricted to synthetic engineering verification")
        if outer_fold_id not in self._folds:
            raise ValueError(f"Unknown outer fold: {outer_fold_id}")
        if not study_id or not trial_id or not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Study/trial identifiers and unique seeds are required")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise ValueError("Seeds must be integers")
        if isinstance(stopping_sessions, bool) or not isinstance(stopping_sessions, int) or stopping_sessions < 0:
            raise ValueError("stopping_sessions must be a nonnegative integer")
        _, inner = self._folds[outer_fold_id]
        allowed = set().union(*(set(part.fit_dates + part.score_dates) for part in inner))
        by_date: dict[date, list[Mapping[str, Any]]] = {}
        for row in rows:
            session, label_end = row.get("session_date"), row.get("label_end_date")
            if not isinstance(session, date):
                raise ValueError("Every row requires a valid session_date")
            if session not in allowed:
                continue
            if not isinstance(label_end, date) or label_end < session:
                raise ValueError("Inner rows require a valid label_end_date")
            by_date.setdefault(session, []).append(row)
        for part in inner:
            if any(day not in by_date for day in part.fit_dates + part.score_dates):
                raise ValueError(f"Missing required date in inner fold {part.fold_id}")
            if any(row["label_end_date"] >= part.score_dates[0]
                   for day in part.fit_dates for row in by_date[day]):
                raise ValueError("Fit label interval overlaps score dates")
        output: list[FitEvidence] = []
        for part in inner:
            fit_dates = part.fit_dates
            stop_dates: tuple[date, ...] = ()
            if stopping_sessions:
                if len(fit_dates) <= stopping_sessions:
                    raise ValueError("Stopping tail exhausts fit history")
                stop_dates = fit_dates[-stopping_sessions:]
                fit_dates = tuple(day for day in fit_dates[:-stopping_sessions]
                                  if all(row["label_end_date"] < stop_dates[0] for row in by_date[day]))
                if not fit_dates:
                    raise ValueError("Stopping purge exhausts fit history")
            fit_rows = tuple(row for day in fit_dates for row in by_date[day])
            stop_rows = tuple(row for day in stop_dates for row in by_date[day])
            score_rows = tuple(row for day in part.score_dates for row in by_date[day])
            for seed in seeds:
                context = FitContext(study_id, trial_id, part.fold_id, seed, fidelity or {})
                model = fit(fit_rows, stop_rows, context)
                value = float(score(model, score_rows))
                if not math.isfinite(value):
                    raise ValueError("Inner score must be finite")
                telemetry = model.telemetry() if callable(getattr(model, "telemetry", None)) else {}
                output.append(FitEvidence(study_id, trial_id, outer_fold_id, part.fold_id,
                                          seed, value, len(fit_dates), len(stop_dates),
                                          len(part.score_dates), dict(context.fidelity), telemetry))
        return tuple(output)
