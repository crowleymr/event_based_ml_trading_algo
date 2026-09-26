"""Date-grouped purged expanding walk-forward manifests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TemporalFold:
    fold_id: str
    fit_dates: tuple[date, ...]
    score_dates: tuple[date, ...]
    embargo_dates: tuple[date, ...]

    def __post_init__(self) -> None:
        if not self.fit_dates or not self.score_dates:
            raise ValueError("Every temporal fold requires fit and score dates")
        if max(self.fit_dates) >= min(self.score_dates):
            raise ValueError("Fit dates must precede score dates")
        if set(self.fit_dates) & set(self.score_dates):
            raise ValueError("Fit and score dates must be disjoint")


def _folds(
    dates: Sequence[date], label_end: Mapping[date, date], *, prefix: str,
    fold_count: int, score_sessions: int, min_fit_sessions: int,
    embargo_sessions: int,
) -> tuple[TemporalFold, ...]:
    calendar = tuple(sorted(set(dates)))
    if any(item not in label_end for item in calendar):
        raise ValueError("label_end must be available for every session date")
    required = min_fit_sessions + embargo_sessions + fold_count * score_sessions
    if len(calendar) < required:
        raise ValueError("Insufficient sessions for requested walk-forward folds")
    first_score = len(calendar) - fold_count * score_sessions
    folds = []
    for index in range(fold_count):
        score_start = first_score + index * score_sessions
        score = calendar[score_start:score_start + score_sessions]
        embargo_start = score_start - embargo_sessions
        embargo = calendar[embargo_start:score_start]
        candidate_fit = calendar[:embargo_start]
        # Information intervals reaching the score boundary are purged.
        fit = tuple(item for item in candidate_fit if label_end[item] < score[0])
        if len(fit) < min_fit_sessions:
            raise ValueError("Purge leaves too few fit sessions")
        folds.append(TemporalFold(f"{prefix}{index + 1}", fit, tuple(score), tuple(embargo)))
    return tuple(folds)


def nested_purged_walk_forward(
    dates: Sequence[date], label_end: Mapping[date, date], *,
    outer_folds: int = 5, inner_folds: int = 3,
    outer_score_sessions: int = 20, inner_score_sessions: int = 10,
    min_fit_sessions: int = 60, embargo_sessions: int = 5,
) -> tuple[tuple[TemporalFold, tuple[TemporalFold, ...]], ...]:
    """Return outer folds and inner folds built only from each outer fit history."""
    if min(outer_folds, inner_folds, outer_score_sessions, inner_score_sessions,
           min_fit_sessions) < 1 or embargo_sessions < 0:
        raise ValueError("Fold counts/window sizes must be positive and embargo nonnegative")
    outer = _folds(
        dates, label_end, prefix="outer_", fold_count=outer_folds,
        score_sessions=outer_score_sessions, min_fit_sessions=min_fit_sessions,
        embargo_sessions=embargo_sessions,
    )
    result = []
    for outer_fold in outer:
        inner = _folds(
            outer_fold.fit_dates, label_end, prefix=f"{outer_fold.fold_id}_inner_",
            fold_count=inner_folds, score_sessions=inner_score_sessions,
            min_fit_sessions=min_fit_sessions, embargo_sessions=embargo_sessions,
        )
        result.append((outer_fold, inner))
    return tuple(result)
