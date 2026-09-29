"""Contract checks for the assignment release packager."""

from pathlib import Path

from scripts.build_assignment_release import _zip_info


def test_release_zip_entries_are_deterministic_and_plain_files():
    info = _zip_info("reports/example.parquet")
    assert info.filename == "reports/example.parquet"
    assert info.date_time == (2026, 9, 28, 0, 0, 0)
    assert info.external_attr == 0o100644 << 16


def test_release_builder_contains_no_raw_run_input():
    source = Path("scripts/build_assignment_release.py").read_text(encoding="utf-8")
    assert 'root / "runs"' not in source
    assert "reports" in source
    assert "bundle_manifest.json" in source
