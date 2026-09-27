from datetime import date, timedelta
import json

import polars as pl
import pytest

from trading_pipeline.data.benchmark_vintage import prepare_benchmark_vintage


def test_benchmark_vintage_is_manifest_driven_and_complete(tmp_path):
    days = [date(2026, 1, 5) + timedelta(days=index) for index in range(3)]
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps({"sealed": True,
        "session_dates": [str(day) for day in days]}), encoding="utf-8")

    def download(security, cfg, root):
        assert cfg["end"] > str(days[-1])
        return pl.DataFrame({"security_id": [security["security_id"]] * 3,
            "ticker": [security["ticker"]] * 3, "session_date": days,
            "adjusted_close": [100.0, 101.0, 102.0], "source": ["test"] * 3,
            "ingested_at": ["fixture"] * 3})

    result = prepare_benchmark_vintage(repository_root=tmp_path,
        holdout_manifest=holdout, output_dir=tmp_path / "benchmark",
        cache_root=tmp_path, downloader=download)
    assert result["coverage"] == "complete_sealed_holdout_calendar"
    assert len(result["bars_sha256"]) == 64


def test_benchmark_vintage_rejects_missing_holdout_session(tmp_path):
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps({"sealed": True,
        "session_dates": ["2026-01-05", "2026-01-06"]}), encoding="utf-8")

    def incomplete(security, cfg, root):
        return pl.DataFrame({"security_id": [security["security_id"]],
            "ticker": [security["ticker"]], "session_date": [date(2026, 1, 5)],
            "adjusted_close": [100.0], "source": ["test"], "ingested_at": ["fixture"]})

    with pytest.raises(ValueError, match="does not cover"):
        prepare_benchmark_vintage(repository_root=tmp_path,
            holdout_manifest=holdout, output_dir=tmp_path / "benchmark",
            cache_root=tmp_path, downloader=incomplete)
