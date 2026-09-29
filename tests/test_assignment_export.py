from __future__ import annotations

from datetime import date
import hashlib
import json

import polars as pl
import pytest

from trading_pipeline.modelling.evaluate import metrics as prediction_metrics
from trading_pipeline.reporting.assignment_export import (
    _controlled_sensitivity,
    _feature_statistics,
    _predictive_diagnostics,
    _verify_wp7,
)
from trading_pipeline.reporting.expanded_closeout import TABLES as WP7_TABLES


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_wp7_verification_rejects_a_changed_output(tmp_path):
    run_id = "run-1"
    run = tmp_path / "runs" / "expanded_closeout" / run_id
    run.mkdir(parents=True)
    _write_json(run / "audit.json", {"status": "passed"})
    _write_json(run / "completion.json", {"status": "complete"})
    report = tmp_path / "reports" / "expanded_closeout" / run_id
    report.mkdir(parents=True)
    outputs = []
    for name in WP7_TABLES:
        path = report / f"{name}.parquet"
        pl.DataFrame({"value": [1]}).write_parquet(path)
        outputs.append({"relative_path": path.name, "sha256": _hash(path), "rows": 1})
    metadata = {"protocol_sha256": "protocol"}
    _write_json(report / "research_evidence_manifest.json", {
        "schema_version": 1,
        "source_run_id": run_id,
        "source_run_audit_sha256": _hash(run / "audit.json"),
        "source_run_completion_sha256": _hash(run / "completion.json"),
        "protocol_sha256": "protocol",
        "claim_status": "exploratory_closeout_not_confirmatory",
        "outputs": outputs,
    })
    _, declared = _verify_wp7(tmp_path, report, run_id, metadata)
    assert len(declared) == len(WP7_TABLES)

    pl.DataFrame({"value": [2]}).write_parquet(report / f"{WP7_TABLES[0]}.parquet")
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_wp7(tmp_path, report, run_id, metadata)


def test_controlled_sensitivity_emits_paired_delta(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "sensitivity_design.json", {
        "arms": {"MODEL": {"component_id": "component", "contrasts": [{
            "factor_name": "depth", "directly_changed_fields": ["depth"],
            "reference_proposal_index": 0, "contrast_proposal_index": 1,
            "reference_level": 1, "contrast_level": 2,
        }]}}
    })
    _write_jsonl(run / "sensitivity_trial_ledger.jsonl", [
        {"trial_id": "ref", "status": "proposed", "arm_id": "MODEL",
         "outer_fold_id": "outer_1", "risk_scenario": None,
         "proposal_index": 0, "parameters": {"depth": 1}},
        {"trial_id": "ref", "status": "complete", "proposal_index": 0,
         "completed_cells": 1, "expected_cells": 1, "mean_ic": 0.1,
         "mean_rmse": 0.2, "resource_seconds": 1.0, "selection_eligible": False},
        {"trial_id": "alt", "status": "proposed", "arm_id": "MODEL",
         "outer_fold_id": "outer_1", "risk_scenario": None,
         "proposal_index": 1, "parameters": {"depth": 2}},
        {"trial_id": "alt", "status": "complete", "proposal_index": 1,
         "completed_cells": 1, "expected_cells": 1, "mean_ic": 0.15,
         "mean_rmse": 0.19, "resource_seconds": 2.0, "selection_eligible": False},
    ])
    table = _controlled_sensitivity(run)
    assert table.height == 1
    assert table["contrast_minus_reference"][0] == pytest.approx(0.05)
    assert table["selection_eligible"][0] is False


def test_predictive_diagnostics_recompute_and_reconcile(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    sessions = [date(2026, 1, 2), date(2026, 1, 5)]
    rows = []
    for partition, fold in (("outer", "outer_1"), ("descriptive_holdout", "holdout")):
        for session in sessions:
            rows.extend([
                {"arm_id": "MODEL", "partition": partition, "fold_id": fold,
                 "seed": 41, "session_date": session, "security_id": "A",
                 "actual_forward_return_5d": 0.02, "predicted_return_5d": 0.01},
                {"arm_id": "MODEL", "partition": partition, "fold_id": fold,
                 "seed": 41, "session_date": session, "security_id": "B",
                 "actual_forward_return_5d": -0.01, "predicted_return_5d": -0.02},
            ])
    predictions = pl.DataFrame(rows)
    predictions.write_parquet(run / "predictions.parquet")
    values, _ = prediction_metrics(predictions.filter(pl.col("partition") == "outer"))
    _write_jsonl(run / "outer_fit_ledger.jsonl", [{
        "arm_id": "MODEL", "outer_fold_id": "outer_1", "seed": 41,
        "ic": values["mean_ic"], "rmse": values["rmse"],
    }])
    _write_jsonl(run / "holdout_fit_ledger.jsonl", [{
        "arm_id": "MODEL", "seed": 41,
        "ic": values["mean_ic"], "rmse": values["rmse"],
    }])
    protocol = {"experiment_arms": [{"id": "MODEL", "interface": "SupervisedModel"}]}
    table = _predictive_diagnostics(run, protocol)
    assert table.height == 2
    assert set(table["partition"]) == {"outer", "descriptive_holdout"}
    assert table["mae"].is_not_null().all()


def test_feature_statistics_preserve_units_and_unavailable_dispersion():
    frame = pl.DataFrame({
        "session_date": [date(2026, 1, 2), date(2026, 1, 2), date(2026, 1, 5)],
        "feature": [1.0, None, 100.0],
    })
    table = _feature_statistics(frame, ["feature"], [
        ("outer_1::training", "training", [date(2026, 1, 2)]),
        ("descriptive_holdout", "descriptive_holdout", [date(2026, 1, 5)]),
    ])
    training = table.filter(pl.col("partition_role") == "training").row(0, named=True)
    assert training["row_count"] == 2
    assert training["missing_count"] == 1
    assert training["count_unit"] == "security_sessions"

