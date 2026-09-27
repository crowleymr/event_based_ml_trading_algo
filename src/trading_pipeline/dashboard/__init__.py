"""Read-only dashboard loaders for generated Slice 1 and Phase 2 evidence."""

from .loader import discover_expanded_reports, load_catalog, load_expanded_report, load_report, load_rl_run

__all__ = ["discover_expanded_reports", "load_catalog", "load_expanded_report", "load_report", "load_rl_run"]
