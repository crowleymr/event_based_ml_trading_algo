import csv
import json

import pandas as pd
import pytest

from trading_pipeline.data.universe_admission import (
    CANDIDATE_FIELDS,
    load_candidates,
    run_admission,
)


def _candidates(path, tickers=("AAA", "BBB")):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        for rank, ticker in enumerate(tickers, start=1):
            writer.writerow({"ticker": ticker, "source": "test-fixture", "source_as_of": "2026-09-25",
                             "source_rank": rank, "security_type": "common_stock"})
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
