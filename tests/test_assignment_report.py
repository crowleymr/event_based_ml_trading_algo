from __future__ import annotations

import json
from pathlib import Path

from trading_pipeline.reporting.assignment_report import (
    ARMS,
    PUBLIC_NOTEBOOK_PLACEHOLDER,
    REQUIRED_CITATIONS,
    _bib_keys,
    _table,
    _word_count,
)


RUN_ID = "20260928T005423Z-0634efaf"


def test_report_helpers_are_deterministic_and_escape_tables() -> None:
    assert _word_count("one two-three `four`") == 3
    assert _table(["A", "B"], [["x|y", 2]]) == (
        "| A | B |\n| --- | --- |\n| x\\|y | 2 |"
    )


def test_verified_bibliography_is_exact_and_has_no_placeholders() -> None:
    root = Path(__file__).parents[1]
    text = (root / "docs" / "assignment" / "references.bib").read_text(encoding="utf-8")
    assert _bib_keys(text) == REQUIRED_CITATIONS
    assert "KJU4AWFC" not in text
    assert "[cite:" not in text


def test_generated_report_contract_when_present() -> None:
    root = Path(__file__).parents[1]
    report_dir = root / "reports" / "assignment" / RUN_ID / "v1" / "report"
    if not report_dir.exists():
        return
    report = (report_dir / "PROJECT_REPORT_DRAFT.md").read_text(encoding="utf-8")
    manifest = json.loads((report_dir / "report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_run_id"] == RUN_ID
    assert manifest["holdout_role"] == "descriptive_only"
    assert manifest["word_count"] == _word_count(report)
    assert PUBLIC_NOTEBOOK_PLACEHOLDER in report
    assert "STUDENT PROMPT — Personal reflection" in report
    assert "STUDENT PROMPT — Personal financial implications" in report
    assert "STUDENT PROMPT — Knowledge gaps" in report
    assert all(arm in report for arm in ARMS)
    assert (report_dir / "references.bib").is_file()
    for name in ("descriptive_risk_return.png", "predictive_diagnostics.png",
                 "descriptive_equity_paths.png"):
        assert (report_dir / "figures" / name).stat().st_size > 1_000
