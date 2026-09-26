from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json

import pytest

from trading_pipeline.optimisation.evaluator import NestedTemporalEvaluator
from trading_pipeline.optimisation.ledgers import append_fit_evidence
from trading_pipeline.optimisation.objectives import (
    aggregate_complete_trials, select_candidate, write_selection_lock,
)
from trading_pipeline.optimisation.splits import TemporalFold


def _fixture():
    days = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(10))
    outer = TemporalFold("outer_1", days[:8], days[8:], ())
    inner = (
        TemporalFold("inner_1", days[:4], (days[4],), ()),
        TemporalFold("inner_2", days[:5], (days[5],), ()),
    )
    rows = [{"session_date": day, "label_end_date": day, "target": index}
            for index, day in enumerate(days)]
    return NestedTemporalEvaluator(((outer, inner),)), rows


class _Model:
    def __init__(self, fit_rows):
        self.mean = sum(row["target"] for row in fit_rows) / len(fit_rows)

    def telemetry(self):
        return {"fit_mean": self.mean}


def _run(evaluator, rows):
    return evaluator.evaluate_trial(
        study_id="synthetic", trial_id="trial_a", outer_fold_id="outer_1",
        rows=rows, seeds=[1, 2], data_role="synthetic",
        fit=lambda fit_rows, stop_rows, context: _Model(fit_rows),
        score=lambda model, score_rows: model.mean,
    )


def test_outer_rows_cannot_change_inner_scores_and_real_role_is_gated():
    evaluator, rows = _fixture()
    original = _run(evaluator, rows)
    changed = [dict(row) for row in rows]
    changed[-1]["target"] = 1_000_000
    changed[-1]["label_end_date"] = None
    assert [item.score for item in _run(evaluator, changed)] == [item.score for item in original]
    with pytest.raises(PermissionError, match="synthetic"):
        evaluator.evaluate_trial(
            study_id="x", trial_id="y", outer_fold_id="outer_1", rows=rows,
            seeds=[1], data_role="real", fit=lambda *args: None,
            score=lambda *args: 0.0,
        )


def test_missing_dates_and_overlap_fail_before_fit():
    evaluator, rows = _fixture()
    with pytest.raises(ValueError, match="Missing required date"):
        _run(evaluator, rows[:4] + rows[5:])
    changed = [dict(row) for row in rows]
    changed[0]["label_end_date"] = rows[4]["session_date"]
    with pytest.raises(ValueError, match="overlaps"):
        _run(evaluator, changed)


def test_complete_fit_ledger_aggregation_and_selection_lock(tmp_path):
    evaluator, rows = _fixture()
    evidence = _run(evaluator, rows)
    ledger = tmp_path / "fits.jsonl"
    append_fit_evidence(ledger, evidence[:2])
    append_fit_evidence(ledger, evidence[2:])
    assert len(ledger.read_text().splitlines()) == 4
    with pytest.raises(ValueError, match="already exists"):
        append_fit_evidence(ledger, evidence[:1])
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate_complete_trials(evidence[:3], expected_folds=["inner_1", "inner_2"],
                                  expected_seeds=[1, 2], complexity={"trial_a": 1},
                                  secondary={"trial_a": 0}, resource_use={"trial_a": 1})
    candidates = aggregate_complete_trials(
        evidence, expected_folds=["inner_1", "inner_2"], expected_seeds=[1, 2],
        complexity={"trial_a": 1}, secondary={"trial_a": 0}, resource_use={"trial_a": 1},
    )
    selected = select_candidate(candidates, direction="maximize", equivalence_tolerance=0,
                                secondary_direction="minimize")
    evidence_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
    lock = tmp_path / "selection.json"
    write_selection_lock(lock, study_id="synthetic", outer_fold_id="outer_1",
                         selected=selected, objective="test_mean", direction="maximize",
                         evidence_sha256=evidence_hash)
    assert json.loads(lock.read_text())["selected"]["trial_id"] == "trial_a"
    with pytest.raises(FileExistsError):
        write_selection_lock(lock, study_id="synthetic", outer_fold_id="outer_1",
                             selected=selected, objective="test_mean", direction="maximize",
                             evidence_sha256=evidence_hash)
