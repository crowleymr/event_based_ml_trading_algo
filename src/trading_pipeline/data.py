"""Immutable source caches and canonical daily tables. No fundamentals from Yahoo."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import logging
import os
import time
import numpy as np
import polars as pl
import requests
import duckdb

FACTS = {"EarningsPerShareBasic": "USD/shares", "NetIncomeLoss": "USD"}
FACT_SCHEMA = {
    "company_id": pl.String, "security_id": pl.String, "cik": pl.String,
    "fact_name": pl.String, "fact_value": pl.Float64, "unit": pl.String,
    "fiscal_period_start": pl.Date, "fiscal_period_end": pl.Date, "filed_date": pl.Date,
    "form": pl.String, "accession_number": pl.String, "source": pl.String,
    "ingested_at": pl.String,
}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.write_parquet(tmp)
    tmp.replace(path)


def cached_json(url, path, cfg):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    ua = os.environ.get("SEC_USER_AGENT") or cfg.get("sec_user_agent")
    if not ua or "@" not in ua:
        raise ValueError("Set SEC_USER_AGENT to a real name/application and contact email")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(cfg.get("retries", 4)):
        try:
            time.sleep(cfg.get("throttle_seconds", .3))
            response = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=60)
            response.raise_for_status()
            payload = response.json()
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(response.content)
            tmp.replace(path)
            return payload
        except (requests.RequestException, ValueError):
            if attempt == cfg.get("retries", 4) - 1:
                raise
            time.sleep(2 ** attempt)


def security_master(tickers, mapping):
    entries = mapping["data"] if "data" in mapping else list(mapping.values())
    if "fields" in mapping:
        entries = [dict(zip(mapping["fields"], row)) for row in entries]
    lookup = {r["ticker"].replace(".", "-"): r for r in entries}
    missing = sorted(set(tickers) - lookup.keys())
    if missing:
        raise ValueError(f"SEC mapping missing tickers: {missing}; do not silently change universe")
    rows = []
    for ticker in tickers:
        item = lookup[ticker]
        cik = str(item.get("cik", item.get("cik_str"))).zfill(10)
        rows.append(dict(security_id=f"US-{cik}-{ticker}", company_id=f"CIK-{cik}",
                         ticker=ticker, exchange=item.get("exchange", "unknown"), currency="USD",
                         cik=cik, is_active=True))
    return pl.DataFrame(rows)


def normalize_facts(payload, security, ingested_at):
    rows = []
    concepts = payload.get("facts", {}).get("us-gaap", {})
    for name in FACTS:
        for unit, items in concepts.get(name, {}).get("units", {}).items():
            for item in items:
                if not all(k in item for k in ("val", "end", "filed")):
                    continue
                rows.append({**{k: security[k] for k in ("security_id", "company_id", "cik")},
                             "fact_name": name, "fact_value": float(item["val"]), "unit": unit,
                             "fiscal_period_start": date.fromisoformat(item["start"]) if item.get("start") else None,
                             "fiscal_period_end": date.fromisoformat(item["end"]),
                             "filed_date": date.fromisoformat(item["filed"]), "form": item.get("form"),
                             "accession_number": item.get("accn"), "source": "sec",
                             "ingested_at": ingested_at})
    return pl.DataFrame(rows, schema=FACT_SCHEMA).unique().sort(["security_id", "filed_date", "fact_name"])


def events_from_facts(facts):
    return facts.select("company_id", "security_id", "cik", pl.col("filed_date").alias("event_date"),
                        pl.lit(None, dtype=pl.Datetime).alias("event_datetime"),
                        pl.lit("sec_filing").alias("event_type"), "form", "accession_number", "source", "ingested_at").unique()


def yahoo_bars(security, cfg, root):
    import yfinance as yf
    yf.set_tz_cache_location(str(root / "raw" / "yahoo" / "library_cache"))
    ticker = security["ticker"]
    path = root / "raw" / "yahoo" / f"{ticker}_{cfg['start']}_{cfg['end']}.parquet"
    stamp = path.with_suffix(".json")
    if not path.exists():
        for attempt in range(cfg.get("retries", 4)):
            try:
                time.sleep(cfg.get("throttle_seconds", .3))
                raw = yf.Ticker(ticker).history(start=cfg["start"], end=cfg["end"], interval="1d",
                                               auto_adjust=False, actions=True, raise_errors=True)
                if raw.empty:
                    raise ValueError(f"No market history for {ticker}")
                # Preserve the complete library response before canonical transformation.
                write_parquet(pl.from_pandas(raw.reset_index()), path)
                stamp.write_text(json.dumps({"ingested_at": utcnow(), "source": "yfinance", "ticker": ticker}))
                break
            except Exception:
                if attempt == cfg.get("retries", 4) - 1:
                    raise
                time.sleep(2 ** attempt)
    raw = pl.read_parquet(path)
    ingested = json.loads(stamp.read_text())["ingested_at"] if stamp.exists() else datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    return raw.select(pl.lit(security["security_id"]).alias("security_id"), pl.lit(ticker).alias("ticker"),
                      pl.col("Date").dt.date().alias("session_date"),
                      *[pl.col(a).cast(pl.Float64).alias(b) for a, b in
                        [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"),
                         ("Adj Close", "adjusted_close"), ("Volume", "volume")]],
                      pl.lit("yahoo").alias("source"), pl.lit(ingested).alias("ingested_at"))


def validate(master, bars, facts):
    required = {"security_id", "ticker", "session_date", "open", "high", "low", "close", "adjusted_close", "volume", "source", "ingested_at"}
    if not required <= set(bars.columns):
        raise ValueError("Invalid market schema")
    if not set(FACT_SCHEMA) <= set(facts.columns):
        raise ValueError("Invalid SEC schema")
    if master["security_id"].n_unique() != master.height or master["ticker"].n_unique() != master.height:
        raise ValueError("Duplicate security mapping")
    if bars.select("security_id", "session_date").unique().height != bars.height:
        raise ValueError("Duplicate market key")
    if bars.filter(pl.any_horizontal(pl.col("open", "high", "low", "close", "adjusted_close").is_null() |
                                     ~pl.col("open", "high", "low", "close", "adjusted_close").is_finite() |
                                     (pl.col("open", "high", "low", "close", "adjusted_close") <= 0)) |
                   pl.col("volume").is_null() | ~pl.col("volume").is_finite() | (pl.col("volume") < 0)).height:
        raise ValueError("Invalid market values")
    if bars.filter((pl.col("high") < pl.max_horizontal("open", "close", "low")) |
                   (pl.col("low") > pl.min_horizontal("open", "close", "high"))).height:
        raise ValueError("Invalid OHLC range")
    if set(bars["security_id"]) != set(master["security_id"]):
        raise ValueError("Market mapping coverage incomplete")
    if not set(facts["security_id"]) <= set(master["security_id"]):
        raise ValueError("Unmapped SEC security")
    if facts.filter(pl.col("fact_value").is_null() | ~pl.col("fact_value").is_finite() |
                    pl.col("filed_date").is_null() | pl.col("fiscal_period_end").is_null()).height:
        raise ValueError("Invalid SEC values")
    for group in bars.partition_by("security_id"):
        if not group["session_date"].is_sorted():
            raise ValueError("Market dates not monotonic")


def synthetic(cfg, root):
    """Deterministic test data; explicitly isolated from research caches."""
    rng = np.random.default_rng(cfg["seed"])
    tickers = Path(cfg["universe"]).read_text().split()[:cfg.get("synthetic_securities", 12)]
    mapping = {str(i): dict(ticker=t, cik_str=i + 1, exchange="SYNTHETIC") for i, t in enumerate(tickers)}
    rawdir = root / "raw" / "synthetic"
    rawdir.mkdir(parents=True, exist_ok=True)
    (rawdir / "mapping.json").write_text(json.dumps(mapping))
    master = security_master(tickers, mapping)
    days, current = [], date(2015, 1, 1)
    while len(days) < cfg.get("synthetic_sessions", 650):
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    markets, facts = [], []
    for i, s in enumerate(master.to_dicts()):
        close = (50 + i) * np.exp(np.cumsum(rng.normal(.0002, .012, len(days))))
        opening = close * np.exp(rng.normal(0, .003, len(days)))
        frame = pl.DataFrame(dict(security_id=[s["security_id"]] * len(days), ticker=[s["ticker"]] * len(days),
                                 session_date=days, open=opening, high=np.maximum(opening, close) * 1.005,
                                 low=np.minimum(opening, close) * .995, close=close, adjusted_close=close,
                                 volume=rng.integers(100000, 1000000, len(days)).astype(float),
                                 source=["synthetic"] * len(days), ingested_at=["synthetic-v1"] * len(days)))
        write_parquet(frame, rawdir / f"{s['ticker']}.parquet")
        markets.append(frame)
        concepts = {}
        for name, unit in FACTS.items():
            concepts[name] = {"units": {unit: [dict(val=float(rng.normal(2, .4) * (1e8 if name == "NetIncomeLoss" else 1)),
                start=(days[j] - timedelta(days=90)).isoformat(), end=days[j].isoformat(),
                filed=(days[j] + timedelta(days=30)).isoformat(), form="10-Q", accn=f"{i}-{j}") for j in range(0, len(days) - 30, 63)]}}
        payload = {"facts": {"us-gaap": concepts}}
        (rawdir / f"{s['cik']}.json").write_text(json.dumps(payload))
        facts.append(normalize_facts(payload, s, "synthetic-v1"))
    return master, pl.concat(markets), pl.concat(facts)


def ingest(cfg):
    root = Path(cfg["data_dir"])
    if cfg["mode"] == "synthetic":
        master, bars, facts = synthetic(cfg, root)
    else:
        mapping = cached_json("https://www.sec.gov/files/company_tickers_exchange.json", root / "raw/sec/company_tickers_exchange.json", cfg)
        master = security_master(Path(cfg["universe"]).read_text().split(), mapping)
        markets, fundamentals, errors = [], [], []
        for s in master.to_dicts():
            logging.info("Ingesting %s", s["ticker"])
            try:
                markets.append(yahoo_bars(s, cfg, root))
                path = root / "raw/sec" / f"CIK{s['cik']}.json"
                payload = cached_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{s['cik']}.json", path, cfg)
                stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                fundamentals.append(normalize_facts(payload, s, stamp))
            except Exception as exc:
                logging.exception("Ingestion failed for %s", s["ticker"])
                errors.append({"ticker": s["ticker"], "error": str(exc)})
        if errors:
            (root / "ingestion_errors.json").write_text(json.dumps(errors, indent=2))
            raise RuntimeError("Ingestion incomplete; caches preserved; see ingestion_errors.json")
        bars, facts = pl.concat(markets), pl.concat(fundamentals)
    bars = bars.sort(["security_id", "session_date"])
    validate(master, bars, facts)
    tables = dict(security_master=master, identifier_map=master.select("security_id", "company_id", "ticker", "cik"),
                  market_bars=bars, fundamental_facts=facts, corporate_events=events_from_facts(facts))
    curated = root / "curated"
    with duckdb.connect(str(root / "catalog.duckdb")) as con:
        for name, frame in tables.items():
            path = curated / name / "part.parquet"
            write_parquet(frame, path)
            escaped = path.resolve().as_posix().replace("'", "''")
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{escaped}')")
            assert con.execute(f"SELECT count(*) FROM {name}").fetchone()[0] == frame.height
    return master, bars, facts
