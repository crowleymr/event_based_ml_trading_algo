"""Data-pipeline orchestration and deterministic synthetic fixtures."""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
import logging
import duckdb
import numpy as np
import polars as pl

from .schemas import FACTS, validate, write_parquet
from .sec_client import cached_json, events_from_facts, normalize_facts
from .security_master import build_security_master
from .universe import load_universe
from .yahoo_client import yahoo_bars


def synthetic(cfg: dict, root: Path):
    """Build deterministic test data isolated from research caches."""
    rng = np.random.default_rng(cfg["seed"])
    tickers = load_universe(cfg["universe"])[:cfg.get("synthetic_securities", 12)]
    mapping = {str(i): {"ticker": ticker, "cik_str": i + 1, "exchange": "SYNTHETIC"}
               for i, ticker in enumerate(tickers)}
    raw_directory = Path(root) / "raw" / "synthetic"
    raw_directory.mkdir(parents=True, exist_ok=True)
    (raw_directory / "mapping.json").write_text(json.dumps(mapping))
    master = build_security_master(tickers, mapping)
    days, current = [], date(2015, 1, 1)
    while len(days) < cfg.get("synthetic_sessions", 650):
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    markets, facts = [], []
    for index, security in enumerate(master.to_dicts()):
        close = (50 + index) * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(days))))
        opening = close * np.exp(rng.normal(0, 0.003, len(days)))
        frame = pl.DataFrame({
            "security_id": [security["security_id"]] * len(days), "ticker": [security["ticker"]] * len(days),
            "session_date": days, "open": opening, "high": np.maximum(opening, close) * 1.005,
            "low": np.minimum(opening, close) * 0.995, "close": close, "adjusted_close": close,
            "volume": rng.integers(100000, 1000000, len(days)).astype(float),
            "source": ["synthetic"] * len(days), "ingested_at": ["synthetic-v1"] * len(days),
        })
        write_parquet(frame, raw_directory / f"{security['ticker']}.parquet")
        markets.append(frame)
        concepts = {}
        for name, unit in FACTS.items():
            scale = 1e8 if name == "NetIncomeLoss" else 1
            concepts[name] = {"units": {unit: [{
                "val": float(rng.normal(2, 0.4) * scale),
                "start": (days[offset] - timedelta(days=90)).isoformat(),
                "end": days[offset].isoformat(), "filed": (days[offset] + timedelta(days=30)).isoformat(),
                "form": "10-Q", "accn": f"{index}-{offset}",
            } for offset in range(0, len(days) - 30, 63)]}}
        payload = {"facts": {"us-gaap": concepts}}
        (raw_directory / f"{security['cik']}.json").write_text(json.dumps(payload))
        facts.append(normalize_facts(payload, security, "synthetic-v1"))
    return master, pl.concat(markets), pl.concat(facts)


def _live(cfg: dict, root: Path):
    mapping = cached_json("https://www.sec.gov/files/company_tickers_exchange.json",
                          root / "raw/sec/company_tickers_exchange.json", cfg)
    master = build_security_master(load_universe(cfg["universe"]), mapping)
    markets, fundamentals, errors = [], [], []
    for security in master.to_dicts():
        logging.info("Ingesting %s", security["ticker"])
        try:
            markets.append(yahoo_bars(security, cfg, root))
            path = root / "raw/sec" / f"CIK{security['cik']}.json"
            payload = cached_json(
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{security['cik']}.json", path, cfg)
            timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            fundamentals.append(normalize_facts(payload, security, timestamp))
        except Exception as exc:
            logging.exception("Ingestion failed for %s", security["ticker"])
            errors.append({"ticker": security["ticker"], "error": str(exc)})
    if errors:
        (root / "ingestion_errors.json").write_text(json.dumps(errors, indent=2))
        raise RuntimeError("Ingestion incomplete; caches preserved; see ingestion_errors.json")
    return master, pl.concat(markets), pl.concat(fundamentals)


def ingest(cfg: dict):
    root = Path(cfg["data_dir"])
    master, bars, facts = synthetic(cfg, root) if cfg["mode"] == "synthetic" else _live(cfg, root)
    bars = bars.sort(["security_id", "session_date"])
    validate(master, bars, facts)
    tables = {
        "security_master": master,
        "identifier_map": master.select("security_id", "company_id", "ticker", "cik"),
        "market_bars": bars, "fundamental_facts": facts,
        "corporate_events": events_from_facts(facts),
    }
    curated = root / "curated"
    with duckdb.connect(str(root / "catalog.duckdb")) as connection:
        for name, frame in tables.items():
            path = curated / name / "part.parquet"
            write_parquet(frame, path)
            escaped = path.resolve().as_posix().replace("'", "''")
            connection.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{escaped}')")
            assert connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0] == frame.height
    return master, bars, facts
