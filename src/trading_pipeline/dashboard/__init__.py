"""Read-only dashboard loaders for generated Slice 1 and Phase 2 evidence."""

from .loader import load_catalog, load_report, load_rl_run

__all__ = ["load_catalog", "load_report", "load_rl_run"]
