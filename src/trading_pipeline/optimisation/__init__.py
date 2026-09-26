"""Leakage-safe study scaffolding; real-data execution requires an approved protocol."""

from .ledgers import TrialRecord, write_trial_ledger
from .search import grid_proposals, random_proposals
from .splits import TemporalFold, nested_purged_walk_forward

__all__ = [
    "TemporalFold",
    "TrialRecord",
    "grid_proposals",
    "nested_purged_walk_forward",
    "random_proposals",
    "write_trial_ledger",
]
