import json
import hashlib
from pathlib import Path

import polars as pl
import pytest

from trading_pipeline.dashboard import discover_expanded_reports, load_expanded_report, load_report, load_rl_run
from trading_pipeline.dashboard.app import _metric_label, _model_kind, _overview_frame


def _write_tables(root, names):
    for name in names:
        pl.DataFrame({"value": [1]}).write_parquet(root / f"{name}.parquet")


def _expanded_fixture(root, *, protocol_hash="protocol-a", completed_at="2026-09-27T01:00:00+00:00",
                      day="2026-01-02"):
    from trading_pipeline.dashboard.loader import EXPANDED_REPORT_TABLES

    root.mkdir()
    frames = {
        "pipeline_stage_summary": {"stage": ["predicted"], "count": [1]},
        "pipeline_security_summary": {"security_id": ["A"], "ticker": ["AAA"]},
        "architecture_trial_summary": {"model_id": ["M"], "trial_id": ["A1"], "status": ["complete"]},
        "hpo_trial_summary": {"model_id": ["M"], "trial_id": ["H1"], "status": ["complete"]},
        "risk_scenario_summary": {"model_id": ["M"], "risk_scenario": ["balanced"]},
        "model_conditioned_frontier_points": {"rebalance_date": [day], "model_id": ["M|seed=1|scenario=balanced"], "status": ["complete"], "expected_volatility": [0.1], "expected_return": [0.2]},
        "model_conditioned_frontier_weights": {"rebalance_date": [day], "model_id": ["M|seed=1|scenario=balanced"], "security_id": ["A"], "weight": [0.5]},
        "realised_risk_return_curve": {"model_id": ["M"], "seed": [1], "risk_order": [1], "risk_scenario": ["balanced"], "annualised_volatility": [0.1], "annualised_return": [0.2], "risk_control_monotonic": [True]},
        "final_testbench_metrics": {"experiment_id": ["M"], "model_id": ["M"], "seed": [1], "split": ["descriptive_holdout"], "risk_scenario": ["balanced"]},
        "final_testbench_equity_curve": {"experiment_id": ["M"], "model_id": ["M"], "seed": [1], "display_label": ["Model M"], "split": ["descriptive_holdout"], "session_date": [day], "equity": [1.0], "drawdown": [0.0], "risk_scenario": ["balanced"]},
        "question_and_assumption_register": {"item_id": ["holdout_role"], "status": ["descriptive_only"]},
    }
    outputs = []
    for name in EXPANDED_REPORT_TABLES:
        target = root / f"{name}.parquet"
        pl.DataFrame(frames[name]).write_parquet(target)
        outputs.append({"relative_path": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "rows": 1})
    (root / "research_evidence_manifest.json").write_text(json.dumps({
        "schema_version": 1, "source_run_id": root.name,
        "source_run_path": f"runs/expanded_closeout/{root.name}",
        "claim_status": "exploratory_closeout_not_confirmatory", "outputs": outputs,
        "protocol_sha256": protocol_hash, "source_completed_at_utc": completed_at,
        "generated_at_utc": completed_at,
        "protocol_summary": {
            "study_id": "expanded_closeout_test",
            "budget_tier": "synthetic",
            "arms": [{"id": "M", "component_id": "supervised.test.v1",
                      "requested_device": "cpu"}],
        },
        "comparison_contract": {
            "metric_definition_id": "expanded_closeout_weekly_after_cost_v1",
            "calendar_sha256": hashlib.sha256(json.dumps([day], separators=(",", ":")).encode()).hexdigest(),
            "execution": "action_at_T_filled_at_T_plus_1_close",
            "cost_bps_one_way": 10,
            "annualisation_periods_per_year": 52,
            "benchmark_symbol": "SPY",
        },
    }), encoding="utf-8")


def _bind_completed_source(project, report):
    source = project / "runs" / "expanded_closeout" / report.name
    source.mkdir(parents=True)
    manifest_path = report / "research_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    (source / "metadata.json").write_text(json.dumps({
        "status": "complete", "run_id": report.name,
        "protocol_sha256": manifest["protocol_sha256"],
    }), encoding="utf-8")
    (source / "protocol.json").write_text(json.dumps({
        "authority": {"protocol_sha256": manifest["protocol_sha256"]},
        "portfolio": {"execution": "action_at_T_filled_at_T_plus_1_close", "cost_bps_one_way": 10},
        "inference": {"benchmark": "SPY"},
    }), encoding="utf-8")
    (source / "audit.json").write_text(json.dumps({
        "status": "passed", "artefact_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("metadata.json", "protocol.json")
        },
    }), encoding="utf-8")
    audit_hash = hashlib.sha256((source / "audit.json").read_bytes()).hexdigest()
    (source / "completion.json").write_text(json.dumps({
        "status": "complete", "run_id": report.name,
        "completed_at": manifest["source_completed_at_utc"], "audit_sha256": audit_hash,
    }), encoding="utf-8")
    manifest["source_run_audit_sha256"] = audit_hash
    manifest["source_run_completion_sha256"] = hashlib.sha256(
        (source / "completion.json").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_expanded_discovery_selects_latest_compatible_and_skips_bad_reports(tmp_path):
    reports = tmp_path / "reports" / "expanded_closeout"
    reports.mkdir(parents=True)
    older = reports / "older"
    latest = reports / "latest"
    unrelated = reports / "different-protocol"
    different_window = reports / "different-window"
    damaged = reports / "damaged"
    for path, stamp, protocol in (
        (older, "2026-09-27T01:00:00+00:00", "protocol-a"),
        (latest, "2026-09-27T02:00:00+00:00", "protocol-a"),
        (unrelated, "2026-09-27T03:00:00+00:00", "protocol-b"),
        (different_window, "2026-09-27T03:30:00+00:00", "protocol-c"),
        (damaged, "2026-09-27T04:00:00+00:00", "protocol-a"),
    ):
        _expanded_fixture(path, protocol_hash=protocol, completed_at=stamp,
                          day="2026-01-09" if path == different_window else "2026-01-02")
        _bind_completed_source(tmp_path, path)
    (damaged / "final_testbench_metrics.parquet").write_bytes(b"tampered")
    found = discover_expanded_reports(tmp_path)
    assert [item["root"].name for item in found] == ["different-window", "different-protocol", "latest", "older"]
    matching = [item for item in found if item["provenance"]["protocol_sha256"] == "protocol-a"]
    assert [item["root"].name for item in matching] == ["latest", "older"]
    compatible = [item for item in found if item["provenance"]["comparison_contract"] == matching[0]["provenance"]["comparison_contract"]]
    assert {item["root"].name for item in compatible} == {"older", "latest", "different-protocol"}


def test_expanded_loader_requires_complete_untampered_evidence(tmp_path):
    root = tmp_path / "synthetic-run"
    _expanded_fixture(root)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    loaded = load_expanded_report(root)
    assert len(loaded["tables"]) == 11
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}
    (root / "final_testbench_metrics.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_expanded_report(root)


def test_expanded_dashboard_controls_render_verified_tables(tmp_path):
    from streamlit.testing.v1 import AppTest
    root = tmp_path / "synthetic-run"
    _expanded_fixture(root)
    script = "from trading_pipeline.dashboard.app import _expanded_view\n_expanded_view(" + repr(str(root)) + ")"
    at = AppTest.from_string(script, default_timeout=10).run()
    assert not at.exception
    assert at.selectbox(key="expanded_ticker").value == "All"
    assert any("descriptive_holdout" in frame.value["split"].tolist()
               for frame in at.dataframe if "split" in frame.value.columns)
    at.selectbox(key="expanded_ticker").select("AAA").run()
    assert not at.exception


def test_zero_argument_dashboard_uses_latest_completed_report(tmp_path):
    from streamlit.testing.v1 import AppTest

    reports = tmp_path / "reports" / "expanded_closeout"
    reports.mkdir(parents=True)
    for name, stamp, protocol in (("older", "2026-09-27T01:00:00+00:00", "protocol-a"),
                                  ("latest", "2026-09-27T02:00:00+00:00", "protocol-b")):
        report = reports / name
        _expanded_fixture(report, completed_at=stamp, protocol_hash=protocol)
        _bind_completed_source(tmp_path, report)
    script = ("from pathlib import Path\nimport trading_pipeline.dashboard.app as app\n"
              + "app.repository_root = lambda: Path(" + repr(str(tmp_path)) + ")\napp.main()")
    at = AppTest.from_string(script, default_timeout=10).run()
    assert not at.exception
    assert any("latest" in item.value for item in at.caption)
    assert any("older" in item.value["source_run_id"].tolist()
               for item in at.dataframe if "source_run_id" in item.value.columns)
    monitoring = next(item.value for item in at.dataframe
                      if "implementation_scope" in item.value.columns)
    assert set(monitoring["run_label"]) == {"expanded_closeout_test · synthetic"}
    assert set(monitoring["implementation_scope"]) == {"M=supervised.test.v1[cpu]"}
    assert any("different research protocols" in item.value for item in at.warning)


def test_zero_argument_dashboard_launcher_passes_latest_report(tmp_path, monkeypatch):
    import sys
    import trading_pipeline.dashboard.__main__ as launcher

    chosen = tmp_path / "reports" / "expanded_closeout" / "latest"
    calls = []
    monkeypatch.setattr(sys, "argv", ["trading_pipeline.dashboard"])
    monkeypatch.setattr(launcher, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(launcher, "discover_expanded_reports",
                        lambda root: [{"root": chosen}])
    monkeypatch.setattr(launcher.subprocess, "call", lambda command: calls.append(command) or 0)
    with pytest.raises(SystemExit) as exit_state:
        launcher.main()
    assert exit_state.value.code == 0
    assert calls[0][-2:] == ["--expanded-report", str(chosen)]


def test_final_notebook_reads_standalone_expanded_report(tmp_path, monkeypatch):
    root = tmp_path / "synthetic-run"
    _expanded_fixture(root)
    monkeypatch.setenv("TRADING_REPORT_DIR", str(root))
    monkeypatch.delenv("LAUNCH_TRADING_DASHBOARD", raising=False)
    notebook = json.loads((Path(__file__).parents[1] / "notebooks" / "final_evidence.ipynb").read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "final_evidence.ipynb", "exec"), namespace)
    assert namespace["expanded_report"] is True
    assert len(namespace["tables"]) == 11


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
    pl.DataFrame({
        "model_id": ["lstm"], "risk_order": [1], "risk_scenario": ["balanced"],
        "annualised_volatility": [0.1], "annualised_return": [0.05],
        "risk_control_monotonic": [True],
    }).write_parquet(
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
    pl.DataFrame({
        "model_id": ["lstm"], "risk_order": [1], "risk_scenario": ["balanced"],
        "annualised_volatility": [0.1], "annualised_return": [0.05],
        "risk_control_monotonic": [True],
    }).write_parquet(
        tmp_path / "realised_risk_return_curve.parquet"
    )
    (tmp_path / "provenance.json").write_text(
        json.dumps({"report_schema_version": 2}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not hash-declared"):
        load_report(tmp_path)


def test_report_loader_rejects_hash_declared_optional_schema_gap(tmp_path):
    from trading_pipeline.dashboard.loader import REPORT_TABLES
    _write_tables(tmp_path, REPORT_TABLES)
    optional = tmp_path / "realised_risk_return_curve.parquet"
    pl.DataFrame({"model_id": ["lstm"]}).write_parquet(optional)
    (tmp_path / "provenance.json").write_text(json.dumps({
        "report_schema_version": 2,
        "outputs": [{"relative_path": optional.name,
                     "sha256": hashlib.sha256(optional.read_bytes()).hexdigest()}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="schema invalid"):
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
