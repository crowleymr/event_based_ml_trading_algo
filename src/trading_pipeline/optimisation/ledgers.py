"""Append-only optimisation trial evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl


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
