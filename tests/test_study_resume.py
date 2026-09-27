from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from trading_pipeline.experiments import study_runner as runner
from trading_pipeline.experiments.study_resume import CheckpointStore
from trading_pipeline.features import F1


def _fixture(tmp_path: Path, monkeypatch):
    days = [date(2026, 1, n) for n in range(1, 19)]
    rows = [{"security_id": security, "ticker": security, "session_date": day,
             "label_end_date": day, "forward_return_5d": float(index) / 100,
             **{column: 1.0 for column in F1}}
            for day in days for index, security in enumerate(("A", "B"))]
    feature = tmp_path / "features.parquet"
    pl.DataFrame(rows).write_parquet(feature)
    arms = [{"id": "EN", "interface": "SupervisedModel", "component_id": "synthetic",
             "feature_set_id": "F1"}]
    inner = [{"fold_id": "i1", "fit_dates": [d.isoformat() for d in days[:4]],
              "stopping_dates": [d.isoformat() for d in days[4:6]],
              "score_dates": [d.isoformat() for d in days[6:8]]},
             {"fold_id": "i2", "fit_dates": [d.isoformat() for d in days[:6]],
              "stopping_dates": [d.isoformat() for d in days[6:8]],
              "score_dates": [d.isoformat() for d in days[8:10]]}]
    outer = [{"fold_id": "o1", "fit_dates": [d.isoformat() for d in days[:12]],
              "score_dates": [d.isoformat() for d in days[13:15]]}]
    holdout = {"session_dates": [d.isoformat() for d in days[16:18]]}
    config = {"study_id": "resume-test", "outputs": {"runs_root": "runs/expanded"},
              "reproducibility": {"seeds": [41]},
              "selection": {"practical_equivalence_tolerance": 0.0},
              "validation": {"inner_windows_manifest": "inner.json",
                             "final_holdout": {"manifest": "holdout.json"}}}
    authority = SimpleNamespace(study_id="resume-test", protocol_sha256="a" * 64,
        input_sha256={"inner.json": "b" * 64, "holdout.json": "c" * 64},
        vintage_id="test", claim_status="exploratory_closeout_not_confirmatory")
    study = SimpleNamespace(config=config)
    monkeypatch.setattr(runner, "_preflight", lambda *_args: (
        None, arms, outer, {"o1": inner}, holdout, feature,
        {"feature_sha256": hashlib.sha256(feature.read_bytes()).hexdigest()}))
    monkeypatch.setattr(runner, "_feature_rows", lambda *_args:
                        [{key: value for key, value in row.items()
                          if key != "forward_return_5d"} for row in rows])
    monkeypatch.setattr(runner, "_proposals", lambda *_args: [{"alpha": 1}])
    monkeypatch.setattr(runner, "code_state_sha256", lambda *_args: "d" * 64)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_args, **_kwargs: "commit")
    return study, authority


def _run_dirs(tmp_path):
    return sorted((tmp_path / "runs" / "expanded").iterdir())


def test_interrupted_study_resumes_only_complete_cells_and_preserves_source(tmp_path, monkeypatch):
    study, authority = _fixture(tmp_path, monkeypatch)
    first_calls = []
    holdout_opened = []
    actual_holdout = runner._holdout_labelled

    def guarded_holdout(feature_path, rows_by_day, dates, family_lock_path):
        assert family_lock_path.is_file()
        assert json.loads(family_lock_path.read_text())["holdout_selection"] == "forbidden"
        holdout_opened.append(True)
        return actual_holdout(feature_path, rows_by_day, dates, family_lock_path)

    monkeypatch.setattr(runner, "_holdout_labelled", guarded_holdout)

    def interrupted_fit(_registry, _arm, _params, _seed, _trial, fold, _fit,
                        _stop, score, **_kwargs):
        first_calls.append(fold)
        if len(first_calls) == 2:
            raise KeyboardInterrupt("operator stopped")
        return None, 0.2, 0.1, {"wall_seconds": 1.0}, np.array([0.1] * len(score))

    monkeypatch.setattr(runner, "_fit_score", interrupted_fit)
    with pytest.raises(KeyboardInterrupt):
        runner.run_study(study, authority, repository_root=tmp_path)
    assert not holdout_opened
    source = _run_dirs(tmp_path)[0]
    assert (source / "failure.json").is_file()
    original = {p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob("*") if p.is_file()}
    resumed_calls = []

    def resumed_fit(_registry, _arm, _params, _seed, _trial, fold, _fit,
                    _stop, score, **_kwargs):
        resumed_calls.append(fold)
        return None, 0.2, 0.1, {"wall_seconds": 1.0}, np.array([0.1] * len(score))

    monkeypatch.setattr(runner, "_fit_score", resumed_fit)
    completed = runner.run_study(study, authority, repository_root=tmp_path,
                                 resume_from=source)
    assert resumed_calls == ["i2", "o1", "holdout"]
    assert (completed / "completion.json").is_file()
    assert json.loads((completed / "started.json").read_text())["resume_lineage"][
        "source_run_id"] == source.name
    fits = [json.loads(line) for line in (completed / "fit_ledger.jsonl").read_text().splitlines()]
    assert [row["inner_fold_id"] for row in fits] == ["i1", "i2"]
    assert all(hashlib.sha256((source / relative).read_bytes()).hexdigest() == digest
               for relative, digest in original.items())
    assert (completed / "family_locks.json").is_file()
    assert holdout_opened == [True]


def test_resume_rejects_protocol_and_checkpoint_tampering_before_new_run(tmp_path, monkeypatch):
    study, authority = _fixture(tmp_path, monkeypatch)
    calls = 0

    def fail_after_one(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return None, 0.2, 0.1, {"wall_seconds": 1.0}, np.array([0.1] * len(args[8]))

    monkeypatch.setattr(runner, "_fit_score", fail_after_one)
    with pytest.raises(RuntimeError, match="No complete candidates"):
        runner.run_study(study, authority, repository_root=tmp_path)
    source = _run_dirs(tmp_path)[0]
    changed = SimpleNamespace(**vars(authority) | {"protocol_sha256": "e" * 64})
    with pytest.raises(ValueError, match="protocol, input or code state mismatch"):
        runner.run_study(study, changed, repository_root=tmp_path, resume_from=source)
    assert len(_run_dirs(tmp_path)) == 1
    checkpoint = next((source / "checkpoints").glob("*.json"))
    checkpoint.write_text(checkpoint.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        runner.run_study(study, authority, repository_root=tmp_path, resume_from=source)
    assert len(_run_dirs(tmp_path)) == 1


def test_precheckpoint_failure_cannot_claim_reusable_cells(tmp_path, monkeypatch):
    study, authority = _fixture(tmp_path, monkeypatch)
    runs_root = tmp_path / "runs" / "expanded"
    source = runs_root / "old-failed-run"
    source.mkdir(parents=True)
    runner._write_new(source / "protocol.json", study.config)
    runner._write_new(source / "started.json", {
        "run_id": source.name, "study_id": authority.study_id,
        "protocol_sha256": authority.protocol_sha256,
        "input_sha256": authority.input_sha256,
        "git_commit": "dirty-checkout-without-code-state-binding"})
    runner._write_new(source / "failure.json", {"status": "failed"})
    with pytest.raises(ValueError, match="protocol, input or code state mismatch"):
        runner.run_study(study, authority, repository_root=tmp_path, resume_from=source)
    assert list(runs_root.iterdir()) == [source]


def test_rl_inner_checkpoint_reuses_complete_cell_without_retraining(tmp_path, monkeypatch):
    source = tmp_path / "source"
    derived = tmp_path / "derived"
    source.mkdir()
    derived.mkdir()
    protocol = "a" * 64
    code = "b" * 64
    study = SimpleNamespace(config={})
    authority = SimpleNamespace(study_id="test", protocol_sha256=protocol)
    arm = {"id": "DQN", "component_id": "test"}
    outer = {"fold_id": "o1", "fit_dates": ["2026-01-01", "2026-01-02"],
             "score_dates": ["2026-01-03", "2026-01-04"]}
    inner = [{"fold_id": "i1", "fit_dates": ["2026-01-01", "2026-01-02"],
              "score_dates": ["2026-01-03", "2026-01-04"]}]
    monkeypatch.setattr(runner, "_rl_scenarios", lambda *_args: ("balanced",))
    executed = []
    fail_outer = True

    def fake_rl_cell(*, trial_id, fold_id, seed, scenario, output, **_kwargs):
        executed.append(fold_id)
        if fail_outer and fold_id == "o1":
            raise RuntimeError("operator interruption")
        cell_id = hashlib.sha256(f"{trial_id}/{fold_id}/{seed}".encode()).hexdigest()[:20]
        directory = output / "rl_cells" / cell_id
        directory.mkdir(parents=True)
        artefacts = {}
        for name in ("actions.parquet", "equity_curve.parquet", "metrics.json", "telemetry.json"):
            path = directory / name
            path.write_text(name, encoding="utf-8")
            artefacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        identity = {"trial_id": trial_id, "fold_id": fold_id, "seed": seed,
                    "risk_scenario": scenario}
        runner._append(output / "rl_trial_ledger.jsonl", identity | {
            "status": "complete", "cell_id": cell_id, "artefact_sha256": artefacts})
        runner._append(output / "rl_resource_ledger.jsonl", identity | {
            "duration_seconds": 1.0})
        return {"certainty_equivalent_mean": 0.1,
                "telemetry": {"duration_seconds": 1.0}}

    monkeypatch.setattr(runner, "_rl_trial_cell", fake_rl_cell)
    first = CheckpointStore(source, protocol_sha256=protocol, code_sha256=code)
    kwargs = dict(study=study, authority=authority, arm=arm, outer_fold=outer,
                  inner=inner, proposals=[{"alpha": 1}], seeds=[41], rows=[], bars=None,
                  outputs=None, feature_sha256="c" * 64, outputs_sha256="d" * 64,
                  fold_sha256="e" * 64, tolerance=0.0)
    with pytest.raises(RuntimeError, match="operator interruption"):
        runner._run_rl_outer(**kwargs, output=source, checkpoints=first, source=None)
    assert executed == ["i1", "o1"]
    runner._write_new(source / "failure.json", {"status": "failed",
        "checkpoint_index_sha256": hashlib.sha256(
            (source / "checkpoint_index.jsonl").read_bytes()).hexdigest()})
    fail_outer = False
    resumed = CheckpointStore(derived, protocol_sha256=protocol, code_sha256=code,
                              source=source)
    runner._run_rl_outer(**kwargs, output=derived, checkpoints=resumed, source=source)
    assert executed == ["i1", "o1", "o1"]
    assert len((derived / "fit_ledger.jsonl").read_text().splitlines()) == 1
    assert len((derived / "outer_fit_ledger.jsonl").read_text().splitlines()) == 1
    assert len(list((derived / "rl_cells").iterdir())) == 2
