"""Bounded, resumable preflight for a dated candidate-security snapshot.

This module is deliberately separate from Slice 1 ingestion: it never edits the
checked-in 100-name universe or canonical data/run artefacts.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Callable

from .security_master import build_security_master

SCHEMA_VERSION = 1
CANDIDATE_FIELDS = ("ticker", "source", "source_as_of", "source_rank", "security_type", "source_cik")
MAX_CANDIDATES = 500


def build_sp500_candidate_snapshot(
    constituent_path: str | Path, baseline_path: str | Path,
    output_path: str | Path, *, as_of: str, limit: int = MAX_CANDIDATES,
) -> list[str]:
    """Freeze baseline-first candidates from a separately frozen constituent table.

    The table is used for candidate enumeration only; SEC and Yahoo remain the
    identifier, fundamental and market-data authorities during admission.
    """
    date.fromisoformat(as_of)
    if not 1 <= limit <= MAX_CANDIDATES:
        raise ValueError(f"limit must be between 1 and {MAX_CANDIDATES}")
    baseline = Path(baseline_path).read_text(encoding="utf-8").split()
    if not baseline or len(baseline) != len(set(baseline)) or len(baseline) > limit:
        raise ValueError("Baseline must be unique and fit within the candidate cap")
    with Path(constituent_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ("symbol", "cik", "source_rank"):
            raise ValueError("Constituent snapshot requires symbol,cik,source_rank")
        rows = list(reader)
    source = []
    ciks = {}
    for index, row in enumerate(rows, start=1):
        symbol = row["symbol"].strip().upper().replace(".", "-")
        if not symbol or not row["cik"].isdigit() or int(row["source_rank"]) != index:
            raise ValueError(f"Invalid constituent row {index}")
        source.append(symbol)
        ciks[symbol] = row["cik"].zfill(10)
    if len(source) != len(set(source)) or not set(baseline).issubset(source):
        raise ValueError("Constituent symbols must be unique and contain the baseline")
    ordered = (baseline + [symbol for symbol in source if symbol not in set(baseline)])[:limit]
    source_label = f"Wikipedia S&P 500 component stocks; frozen {as_of}"
    lines = [",".join(CANDIDATE_FIELDS)]
    for rank, ticker in enumerate(ordered, start=1):
        lines.append(f"{ticker},{source_label},{as_of},{rank},common_stock,{ciks[ticker]}")
    content = "\n".join(lines) + "\n"
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise ValueError(f"Immutable candidate snapshot differs: {destination}")
    else:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    return ordered


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_candidates(path: str | Path, limit: int = MAX_CANDIDATES) -> list[dict]:
    """Read a stable-order CSV candidate snapshot with explicit provenance."""
    if not 1 <= limit <= MAX_CANDIDATES:
        raise ValueError(f"limit must be between 1 and {MAX_CANDIDATES}")
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CANDIDATE_FIELDS:
            raise ValueError(f"Candidate CSV columns must be exactly {','.join(CANDIDATE_FIELDS)}")
        rows = list(reader)
    if not rows or len(rows) > MAX_CANDIDATES:
        raise ValueError(f"Candidate snapshot must contain 1..{MAX_CANDIDATES} rows")
    candidates = []
    tickers = set()
    source_identity = None
    ranks = set()
    for row in rows:
        ticker = row["ticker"].strip().upper()
        source = row["source"].strip()
        as_of = row["source_as_of"].strip()
        security_type = row["security_type"].strip().lower()
        source_cik = row["source_cik"].strip()
        try:
            rank = int(row["source_rank"])
            date.fromisoformat(as_of)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid source rank or source_as_of for {ticker or '<blank>'}") from exc
        if not ticker or not source or security_type != "common_stock" or not source_cik.isdigit() or len(source_cik) != 10:
            raise ValueError(f"Invalid candidate identity/type for {ticker or '<blank>'}")
        if ticker in tickers or rank in ranks:
            raise ValueError(f"Duplicate ticker or source rank: {ticker}/{rank}")
        tickers.add(ticker); ranks.add(rank)
        identity = (source, as_of)
        if source_identity is not None and identity != source_identity:
            raise ValueError("Candidate rows must share one source and as-of date")
        source_identity = identity
        candidates.append({"ticker": ticker, "source": source, "source_as_of": as_of,
                           "source_rank": rank, "security_type": security_type,
                           "source_cik": source_cik})
    candidates.sort(key=lambda item: item["source_rank"])
    return candidates[:limit]


def sec_mapping_status(ticker: str, mapping: dict) -> dict:
    try:
        entries = mapping["data"] if "data" in mapping else list(mapping.values())
        if "fields" in mapping:
            entries = [dict(zip(mapping["fields"], row)) for row in entries]
        matches = [row for row in entries if row["ticker"].replace(".", "-") == ticker]
        if len(matches) > 1:
            return {"status": "excluded", "reason": "sec_ticker_ambiguous",
                    "detail": f"{len(matches)} SEC rows for {ticker}", "cik": None}
        master = build_security_master([ticker], mapping)
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": "excluded", "reason": "sec_ticker_unmapped", "detail": str(exc), "cik": None}
    return {"status": "mapped", "reason": None, "detail": None, "cik": master["cik"][0]}


def sec_fact_coverage(payload: dict) -> dict:
    concepts = payload.get("facts", {}).get("us-gaap", {})
    return {tag: bool(concepts.get(tag, {}).get("units")) for tag in
            ("EarningsPerShareBasic", "NetIncomeLoss")}


def yahoo_history_status(frame, start: str, end: str, min_sessions: int) -> dict:
    """Validate an adjusted-price history frame; frame is yfinance's returned DataFrame."""
    import pandas as pd

    if frame is None or frame.empty:
        return {"status": "excluded", "reason": "yahoo_history_empty", "rows": 0,
                "first_date": None, "last_date": None}
    column = "Adj Close" if "Adj Close" in frame.columns else None
    if column is None or frame[column].isna().any() or (frame[column] <= 0).any():
        return {"status": "excluded", "reason": "yahoo_adjusted_close_invalid", "rows": int(len(frame)),
                "first_date": None, "last_date": None}
    dates = pd.DatetimeIndex(frame.index).tz_localize(None).date
    first, last = dates.min().isoformat(), dates.max().isoformat()
    count = int(len(frame))
    if count < min_sessions:
        return {"status": "excluded", "reason": "yahoo_history_too_short", "rows": count,
                "first_date": first, "last_date": last}
    if last < end:
        return {"status": "excluded", "reason": "yahoo_history_ends_early", "rows": count,
                "first_date": first, "last_date": last}
    return {"status": "available", "reason": None, "rows": count,
            "first_date": first, "last_date": last}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_new(path: Path, payload: dict) -> None:
    """Create immutable JSON atomically; an existing different object is an error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Immutable output already exists with different content: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(encoded, encoding="utf-8")
    try:
        tmp.replace(path)
    except OSError:
        if not path.exists():
            raise
        tmp.unlink(missing_ok=True)
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Concurrent immutable output differs: {path}")


def run_admission(
    candidate_path: str | Path,
    output_dir: str | Path,
    *,
    snapshot_id: str,
    start: str,
    end: str,
    min_sessions: int,
    sec_mapping_loader: Callable[[], dict],
    sec_facts_loader: Callable[[str], dict],
    yahoo_loader: Callable[[str], object],
    limit: int = MAX_CANDIDATES,
    baseline_universe: str | Path | None = None,
    sec_mapping_sha256: str | None = None,
    source_snapshot_sha256: str | None = None,
    raw_artifact_loader: Callable[[str, str | None], dict] | None = None,
) -> dict:
    """Preflight every candidate independently, checkpointing each result.

    A matching run can resume after interruption. Definitive outcomes are reused;
    transient source failures are retried and remain in the append-only log.
    """
    if not snapshot_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                              for char in snapshot_id):
        raise ValueError("snapshot_id must be a non-empty filesystem-safe identifier")
    if min_sessions < 26:
        raise ValueError("min_sessions must allow 20-session feature warm-up and five-day target")
    if date.fromisoformat(end) < date.fromisoformat(start):
        raise ValueError("history_end precedes history_start")
    candidates = load_candidates(candidate_path, limit)
    baseline_hash = None
    if baseline_universe is not None:
        baseline_path = Path(baseline_universe)
        baseline = [ticker.upper() for ticker in baseline_path.read_text(encoding="utf-8").split()]
        candidate_tickers = {row["ticker"] for row in candidates}
        missing_baseline = sorted(set(baseline) - candidate_tickers)
        if missing_baseline:
            raise ValueError(f"Candidate snapshot omits configured baseline tickers: {missing_baseline}")
        if [row["ticker"] for row in candidates[:len(baseline)]] != baseline:
            raise ValueError("Candidate snapshot must preserve baseline ticker order as its prefix")
        baseline_hash = sha256_file(baseline_path)
    root = Path(output_dir) / snapshot_id
    source_hash = sha256_file(candidate_path)
    manifest = {"schema_version": SCHEMA_VERSION, "snapshot_id": snapshot_id,
                "candidate_path": str(Path(candidate_path).resolve()), "candidate_sha256": source_hash,
                "candidate_count": len(candidates), "source": candidates[0]["source"],
                "source_as_of": candidates[0]["source_as_of"], "history_start": start,
                "history_end": end, "min_market_sessions": min_sessions,
                "admission_code_sha256": sha256_file(__file__),
                "baseline_universe_sha256": baseline_hash,
                "sec_mapping_sha256": sec_mapping_sha256,
                "source_snapshot_sha256": source_snapshot_sha256}
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(prior_manifest.get(key) != value for key, value in manifest.items()):
            raise ValueError("Resume parameters differ from immutable manifest")
    else:
        _write_new(manifest_path, manifest | {"created_at_utc": _utc_now()})
    completed_path = root / "admission.json"
    if completed_path.exists():
        return json.loads(completed_path.read_text(encoding="utf-8"))
    event_path = root / "events.jsonl"
    events = []
    if event_path.exists():
        lines = event_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError(f"Corrupt non-terminal admission event at line {index + 1}")
    latest = {}
    for event in events:
        latest[event["ticker"]] = event
    mapping = sec_mapping_loader()
    for candidate in candidates:
        ticker = candidate["ticker"]
        previous = latest.get(ticker)
        if previous and previous.get("outcome") in ("admitted", "excluded"):
            continue
        result = dict(candidate)
        sec = sec_mapping_status(ticker, mapping)
        result["sec_mapping"] = sec
        result["sec_facts"] = {"status": "not_checked", "coverage": None}
        result["yahoo"] = {"status": "not_checked", "reason": None}
        reasons = []
        if sec["status"] != "mapped":
            reasons.append(sec["reason"])
        elif sec["cik"] != candidate["source_cik"]:
            reasons.append("sec_cik_mismatch")
            result["sec_mapping"] = {**sec, "status": "excluded", "reason": "sec_cik_mismatch",
                                     "detail": f"source CIK {candidate['source_cik']} differs from SEC CIK {sec['cik']}"}
        else:
            try:
                payload = sec_facts_loader(sec["cik"])
                result["sec_facts"] = {"status": "available", "coverage": sec_fact_coverage(payload)}
            except Exception as exc:  # status evidence is retained; no silent substitution
                result["sec_facts"] = {"status": "unavailable", "coverage": None,
                                       "detail": f"{type(exc).__name__}: {exc}"}
            try:
                frame = yahoo_loader(ticker)
                yahoo = yahoo_history_status(frame, start, end, min_sessions)
                result["yahoo"] = yahoo
                if yahoo["status"] != "available":
                    reasons.append(yahoo["reason"])
            except Exception as exc:
                result["yahoo"] = {"status": "unavailable", "reason": "yahoo_request_failed",
                                   "detail": f"{type(exc).__name__}: {exc}"}
        result["outcome"] = ("pending" if result["yahoo"]["status"] == "unavailable" else
                             "excluded" if reasons else "admitted")
        result["admitted"] = result["outcome"] == "admitted"
        coverage = result["sec_facts"]["coverage"] or {}
        result["f0_preflight_ready"] = result["admitted"]
        result["f1_preflight_ready"] = result["admitted"] and all(coverage.get(tag, False) for tag in
                                      ("EarningsPerShareBasic", "NetIncomeLoss"))
        result["exclusion_reasons"] = reasons
        result["raw_artifacts"] = raw_artifact_loader(ticker, sec["cik"]) if raw_artifact_loader else {}
        result["checked_at_utc"] = _utc_now()
        line = json.dumps(result, sort_keys=True)
        with event_path.open("a", encoding="utf-8", newline="\n") as stream:
            if event_path.stat().st_size and not event_path.read_bytes().endswith(b"\n"):
                stream.write("\n")
            stream.write(line + "\n")
            stream.flush()
        latest[ticker] = result
        ordered = [latest[c["ticker"]] for c in candidates if c["ticker"] in latest]
        checkpoint_no = 1 + sum(1 for p in root.glob("checkpoint-*.json"))
        checkpoint = {"schema_version": SCHEMA_VERSION, "snapshot_id": snapshot_id,
                      "completed_count": len(ordered), "candidate_count": len(candidates),
                      "ledger": ordered, "updated_at_utc": _utc_now()}
        _write_new(root / f"checkpoint-{checkpoint_no:04d}.json", checkpoint)
    ledger = [latest[c["ticker"]] for c in candidates if c["ticker"] in latest]
    if len(ledger) == len(candidates) and all(row["outcome"] != "pending" for row in ledger):
        final_path = root / "admission.json"
        if final_path.exists():
            final = json.loads(final_path.read_text(encoding="utf-8"))
            if final.get("candidate_sha256") != source_hash or final.get("ledger") != ledger:
                raise ValueError("Completed immutable admission ledger does not match resumed state")
            return final
        result = {"schema_version": SCHEMA_VERSION, "snapshot_id": snapshot_id,
                  "candidate_sha256": source_hash, "candidate_count": len(candidates),
                  "admitted_count": sum(bool(row["admitted"]) for row in ledger),
                  "excluded_count": sum(row["outcome"] == "excluded" for row in ledger),
                  "f0_preflight_ready_count": sum(row["f0_preflight_ready"] for row in ledger),
                  "f1_preflight_ready_count": sum(row["f1_preflight_ready"] for row in ledger),
                  "ledger": ledger, "completed_at_utc": _utc_now()}
        _write_new(final_path, result)
        return result
    return {"schema_version": SCHEMA_VERSION, "snapshot_id": snapshot_id,
            "candidate_count": len(candidates),
            "completed_count": sum(row["outcome"] != "pending" for row in ledger),
            "pending_count": sum(row["outcome"] == "pending" for row in ledger), "ledger": ledger}


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", default="data/universe_admissions")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--start", required=True, help="Required adjusted-price history start, YYYY-MM-DD")
    parser.add_argument(
        "--end", required=True,
        help="Required last observed market session, inclusive, YYYY-MM-DD",
    )
    parser.add_argument("--min-sessions", type=int, default=260)
    parser.add_argument("--limit", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--sec-mapping-file", default="data/raw/sec/company_tickers_exchange.json",
                        help="Frozen SEC mapping JSON; its SHA256 is recorded in the admission manifest")
    parser.add_argument("--source-snapshot", default="configs/sp500_constituents_2026-09-26.csv",
                        help="Frozen candidate-enumeration table; its SHA256 is recorded")
    args = parser.parse_args()

    from .sec_client import cached_json
    cache = Path(args.output) / args.snapshot_id / "raw"
    request_cfg = {"retries": 4, "throttle_seconds": 0.3}
    mapping_path = Path(args.sec_mapping_file)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    def facts(cik: str) -> dict:
        return cached_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                           cache / f"CIK{cik}.json", request_cfg)
    def yahoo(ticker: str):
        import pandas as pd
        import yfinance as yf
        yf.set_tz_cache_location(str(cache / "yahoo_library"))
        raw_path = cache / "yahoo" / f"{ticker}_{args.start}_{args.end}.parquet"
        if raw_path.exists():
            return pd.read_parquet(raw_path)
        # yfinance treats end as exclusive; the admission contract treats it as
        # the required last observed session.
        request_end = (date.fromisoformat(args.end) + timedelta(days=1)).isoformat()
        for attempt in range(request_cfg["retries"]):
            try:
                time.sleep(request_cfg["throttle_seconds"])
                frame = yf.Ticker(ticker).history(start=args.start, end=request_end, interval="1d",
                                                  auto_adjust=False, actions=True, raise_errors=True)
                if not frame.empty:
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    data = io.BytesIO()
                    frame.to_parquet(data)
                    try:
                        with raw_path.open("xb") as stream:
                            stream.write(data.getvalue())
                    except FileExistsError:
                        return pd.read_parquet(raw_path)
                return frame
            except Exception:
                if attempt == request_cfg["retries"] - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Yahoo retry loop terminated unexpectedly")
    def raw_artifacts(ticker: str, cik: str | None) -> dict:
        paths = {
            "sec_company_facts": cache / f"CIK{cik}.json" if cik else None,
            "yahoo_daily_history": cache / "yahoo" / f"{ticker}_{args.start}_{args.end}.parquet",
        }
        return {kind: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for kind, path in paths.items() if path is not None and path.exists()}
    run_admission(args.candidates, args.output, snapshot_id=args.snapshot_id, start=args.start,
                  end=args.end, min_sessions=args.min_sessions, sec_mapping_loader=lambda: mapping,
                  sec_facts_loader=facts, yahoo_loader=yahoo, limit=args.limit,
                  baseline_universe="configs/universe.txt",
                  sec_mapping_sha256=sha256_file(mapping_path),
                  source_snapshot_sha256=sha256_file(args.source_snapshot),
                  raw_artifact_loader=raw_artifacts)


if __name__ == "__main__":
    _cli()
