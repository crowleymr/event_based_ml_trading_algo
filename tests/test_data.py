from datetime import date
import polars as pl
import pytest
from trading_pipeline.config import load_config
from trading_pipeline.data import ingest, security_master, validate, cached_json


def test_mapping():
    frame = security_master(["BRK-B"], {"0": {"ticker": "BRK.B", "cik_str": 1067983}})
    assert frame["cik"][0] == "0001067983"
    with pytest.raises(ValueError, match="missing"):
        security_master(["MISSING"], {})


def test_ingestion_idempotency(tmp_path):
    cfg = load_config("configs/smoke.yaml") | {"data_dir": str(tmp_path), "synthetic_sessions": 280}
    a = ingest(cfg)
    b = ingest(cfg)
    assert all(x.equals(y) for x, y in zip(a, b))
    master, bars, facts = a
    assert set(facts["fact_name"]) == {"EarningsPerShareBasic", "NetIncomeLoss"}
    assert bars.schema["session_date"] == pl.Date
    with pytest.raises(ValueError, match="Duplicate market"):
        validate(master, pl.concat([bars, bars.head(1)]), facts)
    with pytest.raises(ValueError, match="Invalid market values"):
        validate(master, bars.with_columns(pl.lit(-1.).alias("volume")), facts)


def test_raw_cache_does_not_request_network(tmp_path, monkeypatch):
    path = tmp_path / "facts.json"
    path.write_text('{"original": true}')
    monkeypatch.setattr("requests.get", lambda *a, **kw: pytest.fail("cache must avoid request"))
    assert cached_json("https://data.sec.gov", path, {}) == {"original": True}
