import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from trading_pipeline.data.universe_admission import (
    CANDIDATE_FIELDS,
    build_sp500_candidate_snapshot,
    load_candidates,
    run_admission,
)


def _candidates(path, tickers=("AAA", "BBB")):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        for rank, ticker in enumerate(tickers, start=1):
            writer.writerow({"ticker": ticker, "source": "test-fixture", "source_as_of": "2026-09-25",
                             "source_rank": rank, "security_type": "common_stock",
                             "source_cik": f"{rank:010d}"})
    return path


def _frame():
    dates = pd.bdate_range("2025-01-01", "2026-09-25")
    return pd.DataFrame({"Adj Close": [100.0] * len(dates)}, index=dates)


def test_candidate_schema_order_and_bounds(tmp_path):
    path = _candidates(tmp_path / "candidates.csv", ("bbb", "aaa"))
    assert [item["ticker"] for item in load_candidates(path)] == ["BBB", "AAA"]
    with pytest.raises(ValueError, match="columns"):
        (tmp_path / "bad.csv").write_text("ticker\nAAA\n", encoding="utf-8")
        load_candidates(tmp_path / "bad.csv")


def test_admission_records_mapping_yahoo_and_optional_fundamental_status(tmp_path):
    source = _candidates(tmp_path / "candidates.csv")
    mapping = {"fields": ["cik", "name", "ticker", "exchange"],
               "data": [[1, "A", "AAA", "NYSE"], [2, "B", "BBB", "NASDAQ"]]}
    calls = []
    result = run_admission(
        source, tmp_path / "out", snapshot_id="test-v1", start="2025-01-01", end="2026-09-25",
        min_sessions=200, sec_mapping_loader=lambda: mapping,
        sec_facts_loader=lambda cik: {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": []}}}}},
        yahoo_loader=lambda ticker: calls.append(ticker) or _frame(),
    )
    assert result["candidate_count"] == 2
    assert result["admitted_count"] == 2
    assert calls == ["AAA", "BBB"]
    assert result["ledger"][0]["sec_mapping"]["cik"] == "0000000001"
    assert result["ledger"][0]["sec_facts"]["coverage"] == {
        "EarningsPerShareBasic": False, "NetIncomeLoss": True,
    }
    assert (tmp_path / "out/test-v1/manifest.json").exists()
    assert (tmp_path / "out/test-v1/events.jsonl").exists()
    assert (tmp_path / "out/test-v1/admission.json").exists()


def test_resume_reuses_successes_and_final_ledger_is_immutable(tmp_path):
    source = _candidates(tmp_path / "candidates.csv", ("AAA",))
    mapping = {"0": {"ticker": "AAA", "cik_str": 1}}
    args = dict(snapshot_id="resume-v1", start="2025-01-01", end="2026-09-25", min_sessions=200,
                sec_mapping_loader=lambda: mapping, sec_facts_loader=lambda cik: {},
                yahoo_loader=lambda ticker: _frame())
    first = run_admission(source, tmp_path / "out", **args)
    second = run_admission(source, tmp_path / "out", **args)
    assert first == second
    events = (tmp_path / "out/resume-v1/events.jsonl").read_text().splitlines()
    assert len(events) == 1


def test_missing_sec_mapping_is_explicit_exclusion(tmp_path):
    source = _candidates(tmp_path / "candidates.csv", ("ZZZ",))
    result = run_admission(
        source, tmp_path / "out", snapshot_id="missing-v1", start="2025-01-01", end="2026-09-25",
        min_sessions=200, sec_mapping_loader=lambda: {}, sec_facts_loader=lambda cik: {},
        yahoo_loader=lambda ticker: pytest.fail("Yahoo must not be queried without a SEC mapping"),
    )
    row = result["ledger"][0]
    assert not row["admitted"]
    assert row["exclusion_reasons"] == ["sec_ticker_unmapped"]
    assert row["yahoo"]["status"] == "not_checked"


def test_source_cik_mismatch_excludes_without_yahoo_request(tmp_path):
    source = _candidates(tmp_path / "candidates.csv", ("AAA",))
    result = run_admission(
        source, tmp_path / "out", snapshot_id="mismatch-v1", start="2025-01-01",
        end="2026-09-25", min_sessions=200,
        sec_mapping_loader=lambda: {"0": {"ticker": "AAA", "cik_str": 2}},
        sec_facts_loader=lambda _: pytest.fail("Facts must not be queried"),
        yahoo_loader=lambda _: pytest.fail("Yahoo must not be queried"),
    )
    assert result["ledger"][0]["exclusion_reasons"] == ["sec_cik_mismatch"]


def test_transient_yahoo_failure_is_pending_and_retried(tmp_path):
    source = _candidates(tmp_path / "candidates.csv", ("AAA",))
    calls = []
    def yahoo(_):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("offline")
        return _frame()
    args = dict(snapshot_id="retry-v1", start="2025-01-01", end="2026-09-25",
                min_sessions=200, sec_mapping_loader=lambda: {"0": {"ticker": "AAA", "cik_str": 1}},
                sec_facts_loader=lambda _: {}, yahoo_loader=yahoo)
    first = run_admission(source, tmp_path / "out", **args)
    assert first["pending_count"] == 1
    assert not (tmp_path / "out/retry-v1/admission.json").exists()
    second = run_admission(source, tmp_path / "out", **args)
    assert second["admitted_count"] == 1
    assert len(calls) == 2


def test_later_listing_with_sufficient_history_is_admitted(tmp_path):
    source = _candidates(tmp_path / "candidates.csv", ("AAA",))
    result = run_admission(
        source, tmp_path / "out", snapshot_id="later-listing-v1", start="2015-01-02",
        end="2026-09-25", min_sessions=260,
        sec_mapping_loader=lambda: {"0": {"ticker": "AAA", "cik_str": 1}},
        sec_facts_loader=lambda _: {}, yahoo_loader=lambda _: _frame(),
    )
    row = result["ledger"][0]
    assert row["admitted"]
    assert row["yahoo"]["first_date"] == "2025-01-01"
    assert row["f0_preflight_ready"]


def test_candidate_builder_preserves_baseline_and_source_order(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("symbol,cik,source_rank\nCCC,3,1\nAAA,1,2\nBBB,2,3\n", encoding="utf-8")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("BBB\nAAA\n", encoding="utf-8")
    path = tmp_path / "candidates.csv"
    assert build_sp500_candidate_snapshot(source, baseline, path, as_of="2026-09-26") == ["BBB", "AAA", "CCC"]
    assert [item["source_cik"] for item in load_candidates(path)] == ["0000000002", "0000000001", "0000000003"]
    assert build_sp500_candidate_snapshot(source, baseline, path, as_of="2026-09-26") == ["BBB", "AAA", "CCC"]


def test_resume_rejects_candidate_snapshot_change(tmp_path):
    source = _candidates(tmp_path / "candidates.csv", ("AAA",))
    mapping = {"0": {"ticker": "AAA", "cik_str": 1}}
    options = dict(snapshot_id="changed-v1", start="2025-01-01", end="2026-09-25", min_sessions=200,
                   sec_mapping_loader=lambda: mapping, sec_facts_loader=lambda cik: {}, yahoo_loader=lambda _: _frame())
    run_admission(source, tmp_path / "out", **options)
    _candidates(source, ("BBB",))
    with pytest.raises(ValueError, match="immutable manifest"):
        run_admission(source, tmp_path / "out", **options)


def test_existing_slice1_universe_remains_100_names():
    names = open("configs/universe.txt", encoding="utf-8").read().split()
    assert len(names) == len(set(names)) == 100


def test_frozen_candidate_mirror_and_sec_identity():
    source = Path("configs/sp500_constituents_2026-09-26.csv")
    candidates = load_candidates("configs/universe_candidates_500.csv")
    tickers = [row["ticker"] for row in candidates]
    baseline = Path("configs/universe.txt").read_text(encoding="utf-8").split()
    assert len(candidates) == 500
    assert tickers[:len(baseline)] == baseline
    assert tickers == Path("configs/universe_candidates_500.txt").read_text(encoding="utf-8").split()
    assert build_sp500_candidate_snapshot(source, "configs/universe.txt",
                                          "configs/universe_candidates_500.csv", as_of="2026-09-26") == tickers
    mapping = json.loads(Path("data/raw/sec/company_tickers_exchange.json").read_text(encoding="utf-8"))
    from trading_pipeline.data.universe_admission import sec_mapping_status
    assert all(sec_mapping_status(row["ticker"], mapping)["cik"] == row["source_cik"]
               for row in candidates)
