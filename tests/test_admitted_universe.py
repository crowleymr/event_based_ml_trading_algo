import json

import pandas as pd
import polars as pl
import pytest

from trading_pipeline.data.admitted_universe import freeze_admitted_universe
from trading_pipeline.data.universe_admission import run_admission, sha256_file


def test_freeze_from_sealed_ledger_preserves_order_and_hashes(tmp_path):
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "ticker,source,source_as_of,source_rank,security_type,source_cik\n"
        "AAA,test,2026-09-26,1,common_stock,0000000001\n"
        "BBB,test,2026-09-26,2,common_stock,0000000002\n", encoding="utf-8",
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"fields": ["cik", "ticker", "exchange"],
                                   "data": [[1, "AAA", "NYSE"], [2, "BBB", "NASDAQ"]]}),
                       encoding="utf-8")
    frame = pd.DataFrame({"Adj Close": [100.0] * 300},
                         index=pd.bdate_range("2025-08-01", periods=300))
    admission_root = tmp_path / "admissions"
    result = run_admission(
        candidates, admission_root, snapshot_id="fixture", start="2015-01-02",
        end=frame.index[-1].date().isoformat(), min_sessions=260,
        sec_mapping_loader=lambda: json.loads(mapping.read_text(encoding="utf-8")),
        sec_facts_loader=lambda _: {},
        yahoo_loader=lambda ticker: frame if ticker == "AAA" else pd.DataFrame(),
        sec_mapping_sha256=sha256_file(mapping),
    )
    assert result["admitted_count"] == 1
    frozen = freeze_admitted_universe(admission_root / "fixture", candidates, mapping,
                                      tmp_path / "derived")
    assert frozen["admitted_count"] == 1
    assert (tmp_path / "derived/admitted_universe.txt").read_text(encoding="utf-8") == "AAA\n"
    assert pl.read_parquet(tmp_path / "derived/security_master.parquet")["ticker"].to_list() == ["AAA"]
    with pytest.raises(ValueError, match="already exists"):
        freeze_admitted_universe(admission_root / "fixture", candidates, mapping,
                                 tmp_path / "derived")
