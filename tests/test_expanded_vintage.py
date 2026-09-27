"""Focused offline expanded-vintage contract checks."""

from datetime import datetime, timezone

import polars as pl
import pytest

from trading_pipeline.data.expanded_vintage import _market, _read_verified
from trading_pipeline.data.schemas import digest


def test_market_normalization_preserves_pinned_identity_and_dates(tmp_path):
    path = tmp_path / "market.parquet"
    pl.DataFrame({
        "Date": [datetime(2026, 9, 24, tzinfo=timezone.utc)],
        "Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5],
        "Adj Close": [10.25], "Volume": [100],
    }).write_parquet(path)
    _read_verified(path, digest(path))
    result = _market(path, {"security_id": "US-1-ABC", "ticker": "ABC"}, "sealed")
    assert result["security_id"].to_list() == ["US-1-ABC"]
    assert result["ticker"].to_list() == ["ABC"]
    assert str(result["session_date"][0]) == "2026-09-24"
    assert result["adjusted_close"].to_list() == [10.25]
    assert result["ingested_at"].to_list() == ["sealed"]


def test_pinned_input_hash_failure_is_closed(tmp_path):
    path = tmp_path / "market.parquet"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed sealed input"):
        _read_verified(path, "0" * 64)


def test_market_normalization_rejects_missing_adjusted_price(tmp_path):
    path = tmp_path / "market.parquet"
    pl.DataFrame({"Date": [datetime(2026, 9, 24)], "Close": [10.0]}).write_parquet(path)
    with pytest.raises(ValueError, match="Invalid pinned Yahoo schema"):
        _market(path, {"security_id": "US-1-ABC", "ticker": "ABC"}, "sealed")
