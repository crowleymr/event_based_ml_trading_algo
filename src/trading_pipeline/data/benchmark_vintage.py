"""Prepare a hash-declared benchmark vintage through the study holdout."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import polars as pl

from .schemas import write_parquet
from .yahoo_client import yahoo_bars


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_benchmark_vintage(
    *, repository_root: str | Path, holdout_manifest: str | Path,
    output_dir: str | Path, ticker: str = "SPY", start: str = "2015-01-01",
    cache_root: str | Path = "data", downloader=yahoo_bars,
) -> dict:
    root = Path(repository_root).resolve()
    holdout_path = Path(holdout_manifest).resolve()
    destination = Path(output_dir).resolve()
    cache = (root / cache_root).resolve() if not Path(cache_root).is_absolute() else Path(cache_root).resolve()
    for path in (holdout_path, destination, cache):
        if not path.is_relative_to(root):
            raise ValueError(f"Benchmark path escapes repository: {path}")
    if destination.exists():
        raise FileExistsError(destination)
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    dates = holdout.get("session_dates")
    if not holdout.get("sealed") or not isinstance(dates, list) or not dates:
        raise ValueError("Benchmark requires a sealed holdout calendar")
    last = datetime.fromisoformat(dates[-1]).date()
    end = (last + timedelta(days=4)).isoformat()
    frame = downloader({"security_id": f"BENCHMARK-{ticker}", "ticker": ticker},
        {"start": start, "end": end, "retries": 4, "throttle_seconds": 0.3}, cache)
    required = {"security_id", "ticker", "session_date", "adjusted_close", "source", "ingested_at"}
    if (not required <= set(frame.columns) or frame.is_empty()
            or frame.select("security_id", "session_date").is_duplicated().any()
            or frame.filter(pl.col("adjusted_close").is_null()
                            | ~pl.col("adjusted_close").is_finite()
                            | (pl.col("adjusted_close") <= 0)).height):
        raise ValueError("Benchmark bars violate the canonical daily-bar contract")
    missing = sorted(set(datetime.fromisoformat(value).date() for value in dates)
                     - set(frame["session_date"].to_list()))
    if missing:
        raise ValueError(f"Benchmark does not cover {len(missing)} holdout sessions")
    destination.mkdir(parents=True)
    table = destination / "bars.parquet"
    write_parquet(frame.sort("session_date"), table)
    manifest = {
        "schema_version": 1,
        "benchmark_id": f"B0-{ticker}",
        "ticker": ticker,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_sha256": _hash(Path(__file__)),
        "holdout_manifest_path": holdout_path.relative_to(root).as_posix(),
        "holdout_manifest_sha256": _hash(holdout_path),
        "bars_path": table.relative_to(root).as_posix(),
        "bars_sha256": _hash(table),
        "rows": frame.height,
        "start_date": str(frame["session_date"].min()),
        "end_date": str(frame["session_date"].max()),
        "coverage": "complete_sealed_holdout_calendar",
        "source": "yfinance_yahoo_adjusted_close",
    }
    manifest_path = destination / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--holdout-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--cache-root", default="data")
    args = parser.parse_args()
    result = prepare_benchmark_vintage(repository_root=args.repository_root,
        holdout_manifest=args.holdout_manifest, output_dir=args.output,
        ticker=args.ticker, start=args.start, cache_root=args.cache_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
