import json
import hashlib
from pathlib import Path

import polars as pl
import pytest

from trading_pipeline.dashboard import load_report, load_rl_run
from trading_pipeline.dashboard.app import _metric_label, _model_kind, _overview_frame


def _write_tables(root, names):
    for name in names:
        pl.DataFrame({"value": [1]}).write_parquet(root / f"{name}.parquet")


def test_report_loader_is_read_only_and_schema_checked(tmp_path):
    from trading_pipeline.dashboard.loader import REPORT_TABLES_V1
    _write_tables(tmp_path, REPORT_TABLES_V1)
    (tmp_path / "provenance.json").write_text(json.dumps({"report_schema_version": 1}), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    loaded = load_report(tmp_path)
    assert set(loaded["tables"]) == set(REPORT_TABLES_V1)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    (tmp_path / "provenance.json").write_text(json.dumps({"report_schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_report(tmp_path)


def test_report_v2_loader_requires_expanded_contract(tmp_path):
    from trading_pipeline.dashboard.loader import OPTIONAL_REPORT_TABLES, REPORT_TABLES
    _write_tables(tmp_path, REPORT_TABLES)
    pl.DataFrame({"model_id": ["lstm"]}).write_parquet(
        tmp_path / "realised_risk_return_curve.parquet"
    )
    optional = tmp_path / "realised_risk_return_curve.parquet"
    (tmp_path / "provenance.json").write_text(
        json.dumps({
            "report_schema_version": 2,
            "outputs": [{
                "relative_path": optional.name,
                "sha256": hashlib.sha256(optional.read_bytes()).hexdigest(),
            }],
        }), encoding="utf-8"
    )
    loaded = load_report(tmp_path)["tables"]
    assert set(REPORT_TABLES) <= set(loaded)
    assert "realised_risk_return_curve" in loaded
    assert "realised_risk_return_curve" in OPTIONAL_REPORT_TABLES


def test_report_loader_rejects_undeclared_optional_table(tmp_path):
    from trading_pipeline.dashboard.loader import REPORT_TABLES
    _write_tables(tmp_path, REPORT_TABLES)
    pl.DataFrame({"model_id": ["lstm"]}).write_parquet(
        tmp_path / "realised_risk_return_curve.parquet"
    )
    (tmp_path / "provenance.json").write_text(
        json.dumps({"report_schema_version": 2}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not hash-declared"):
        load_report(tmp_path)


def test_rl_loader_requires_complete_audited_exploratory_run(tmp_path):
    from trading_pipeline.dashboard.loader import RL_TABLES
    _write_tables(tmp_path, RL_TABLES)
    metadata = {"status": "complete", "research_status": "exploratory_not_confirmatory"}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "audit.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    assert set(load_rl_run(tmp_path)["tables"]) == set(RL_TABLES)
    (tmp_path / "audit.json").write_text(json.dumps({"passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="completed, audited"):
        load_rl_run(tmp_path)


def test_overview_story_classifies_models_and_keeps_both_evidence_splits():
    comparison = pl.DataFrame({
        "experiment_id": ["E1", "E1", "R1"],
        "display_label": ["Elastic Net", "Elastic Net", "PPO"],
        "estimator_label": ["Elastic Net", "Elastic Net", "Categorical PPO"],
        "portfolio_label": ["Top 10", "Top 10", "Sleeve selector"],
        "split": ["validation", "test", "test"],
        "sharpe": [0.5, 0.4, 0.3],
    })

    overview = _overview_frame(comparison, ["E1", "R1"])

    assert set(overview["split"]) == {"validation", "test"}
    assert set(overview["model_kind"]) == {
        "Classical supervised", "Reinforcement learning",
    }
    assert _model_kind("causal Transformer") == "Deep supervised"
    assert _model_kind("Broad-market price baseline") == "Fixed baseline"
    assert _metric_label("sharpe") == "Sharpe ratio"
