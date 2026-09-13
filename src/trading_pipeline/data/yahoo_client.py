"""Retrying Yahoo daily-bar download with immutable raw caches."""

from datetime import datetime, timezone
from pathlib import Path
import json
import time
import polars as pl

from .schemas import write_parquet


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def yahoo_bars(security: dict, cfg: dict, root: Path) -> pl.DataFrame:
    import yfinance as yf

    yf.set_tz_cache_location(str(root / "raw" / "yahoo" / "library_cache"))
    ticker = security["ticker"]
    path = root / "raw" / "yahoo" / f"{ticker}_{cfg['start']}_{cfg['end']}.parquet"
    stamp = path.with_suffix(".json")
    if not path.exists():
        retries = cfg.get("retries", 4)
        for attempt in range(retries):
            try:
                time.sleep(cfg.get("throttle_seconds", 0.3))
                raw = yf.Ticker(ticker).history(
                    start=cfg["start"], end=cfg["end"], interval="1d",
                    auto_adjust=False, actions=True, raise_errors=True,
                )
                if raw.empty:
                    raise ValueError(f"No market history for {ticker}")
                write_parquet(pl.from_pandas(raw.reset_index()), path)
                stamp.write_text(json.dumps({"ingested_at": utcnow(), "source": "yfinance", "ticker": ticker}))
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
    raw = pl.read_parquet(path)
    ingested = (
        json.loads(stamp.read_text())["ingested_at"] if stamp.exists()
        else datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    )
    return raw.select(
        pl.lit(security["security_id"]).alias("security_id"), pl.lit(ticker).alias("ticker"),
        pl.col("Date").dt.date().alias("session_date"),
        *[pl.col(source).cast(pl.Float64).alias(target) for source, target in (
            ("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"),
            ("Adj Close", "adjusted_close"), ("Volume", "volume"),
        )],
        pl.lit("yahoo").alias("source"), pl.lit(ingested).alias("ingested_at"),
    )
