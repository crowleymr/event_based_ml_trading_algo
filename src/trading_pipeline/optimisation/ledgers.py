"""Append-only optimisation trial evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from .evaluator import FitEvidence


@dataclass(frozen=True)
class TrialRecord:
    study_id: str
    trial_id: str
    component_id: str
    proposal_index: int
    status: str
    parameters: Mapping[str, Any]
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"proposed", "complete", "failed", "pruned"}:
            raise ValueError(f"Unsupported trial status: {self.status}")
        if self.status in {"failed", "pruned"} and not self.failure_reason:
            raise ValueError("Failed/pruned trials require a reason")


def write_trial_ledger(path: str | Path, records: Sequence[TrialRecord]) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite trial ledger: {destination}")
    keys = [(item.study_id, item.trial_id) for item in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate study/trial key")
    rows = []
    for item in records:
        row = asdict(item)
        row["parameters_json"] = json.dumps(row.pop("parameters"), sort_keys=True)
        rows.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(destination)
    return destination


def append_fit_evidence(path: str | Path, records: Sequence[FitEvidence]) -> Path:
    """Append complete fit/seed facts without replacing prior evidence."""
    destination = Path(path)
    if not records:
        raise ValueError("At least one fit record is required")
    existing: set[tuple[str, str, str, str, int]] = set()
    if destination.exists():
        for line in destination.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            existing.add(tuple(item[key] for key in
                               ("study_id", "trial_id", "outer_fold_id", "inner_fold_id", "seed")))
    additions = []
    for record in records:
        key = (record.study_id, record.trial_id, record.outer_fold_id,
               record.inner_fold_id, record.seed)
        if key in existing:
            raise ValueError(f"Fit evidence already exists: {key}")
        existing.add(key)
        item = asdict(record)
        item["recorded_at"] = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(item, sort_keys=True, default=str, allow_nan=False)
        item["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        additions.append(json.dumps(item, sort_keys=True, default=str, allow_nan=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        for line in additions:
            stream.write(line + "\n")
    return destination
