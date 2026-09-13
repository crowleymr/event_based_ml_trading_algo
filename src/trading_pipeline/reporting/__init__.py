"""Deterministic read-only reporting from one immutable run contract."""

from .generate import REPORT_SCHEMA_VERSION, build_report, generate_report

__all__ = ["REPORT_SCHEMA_VERSION", "build_report", "generate_report"]
