from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from trading_pipeline.experiments.authority import verify_study_authority
from trading_pipeline.experiments import FitContext
from trading_pipeline.experiments.schema import load_study, protocol_content_sha256
from trading_pipeline.rl.runner import run_registered_policy_trial
from trading_pipeline.run import main


def _approved(tmp_path: Path) -> Path:
    study_id = "fresh_exploratory_v1"
    files = {
        "approval.md": "Approved exploratory protocol.\n",
        "snapshot.json": {"study_id": study_id, "vintage_id": "fresh_v1",
                          "data_role": "expanded_fresh_vintage"},
        "exposure.json": {"study_id": study_id, "forbidden_selection_roles": ["legacy_observed_final"]},
        "outer.json": {"study_id": study_id, "folds": [{
            "fold_id": "o1", "fit_dates": [f"2026-01-{day:02d}" for day in range(1, 9)],
            "embargo_dates": ["2026-01-09"], "score_dates": ["2026-01-10", "2026-01-11"],
        }]},
        "inner.json": {"study_id": study_id, "folds": [{
            "fold_id": "i1", "outer_fold_id": "o1",
            "fit_dates": [f"2026-01-{day:02d}" for day in range(1, 5)],
            "embargo_dates": ["2026-01-05"], "score_dates": ["2026-01-06"],
        }]},
        "holdout.json": {"study_id": study_id, "sealed": True,
                         "session_dates": ["2026-01-13", "2026-01-14"]},
    }
    for name, payload in files.items():
        (tmp_path / name).write_text(
            payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
        )
    config = {
        "schema_version": 3, "kind": "optimisation_study", "study_id": study_id,
        "status": "approved",
        "authority": {"approval_record": "approval.md", "protocol_sha256": None,
                      "input_sha256": {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                                       for name in files}},
        "data": {"vintage_id": "fresh_v1", "snapshot_manifest": "snapshot.json",
                 "exposure_ledger": "exposure.json",
                 "forbidden_selection_role": "legacy_observed_final"},
        "validation": {"outer_windows_manifest": "outer.json", "inner_windows_manifest": "inner.json",
                       "final_holdout": {"sealed": True, "manifest": "holdout.json"}},
        "experiment_arms": [{"id": "EN_F1", "interface": "SupervisedModel",
                             "component_id": "supervised.elastic_net.v1"}],
        "objectives": {}, "selection": {"final_test_selection": "forbidden"},
        "search": {"real_data_execution": "enabled"},
        "reproducibility": {"seeds": [41, 42, 43]}, "outputs": {},
        "inference": {"claim_status": "exploratory_closeout_not_confirmatory"},
    }
    config["authority"]["protocol_sha256"] = protocol_content_sha256(config)
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_approved_study_requires_exact_protocol_and_pinned_inputs(tmp_path):
    path = _approved(tmp_path)
    study = load_study(path, allow_engineering_draft=False)
    authority = verify_study_authority(study, repository_root=tmp_path)
    assert len(authority.input_sha256) == 6
    assert authority.data_role == "expanded_fresh_vintage"
    (tmp_path / "snapshot.json").write_text('{"study_id":"other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_study_authority(study, repository_root=tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["selection"]["practical_equivalence_tolerance"] = 0.5
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol_sha256"):
        load_study(path, allow_engineering_draft=False)


def test_holdout_overlap_fails_before_research_execution(tmp_path):
    path = _approved(tmp_path)
    holdout = tmp_path / "holdout.json"
    payload = json.loads(holdout.read_text(encoding="utf-8"))
    payload["session_dates"] = ["2026-01-11", "2026-01-14"]
    holdout.write_text(json.dumps(payload), encoding="utf-8")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["authority"]["input_sha256"]["holdout.json"] = hashlib.sha256(holdout.read_bytes()).hexdigest()
    config["authority"]["protocol_sha256"] = protocol_content_sha256(config)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps outer"):
        verify_study_authority(load_study(path), repository_root=tmp_path)


def test_draft_study_cli_fails_before_creating_run(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["trading_pipeline.run", "--study",
                                  "configs/studies/expanded_closeout_draft.yaml"])
    with pytest.raises(ValueError, match="engineering verification"):
        main()


def test_valid_authority_cannot_directly_execute_real_rl(tmp_path):
    path = _approved(tmp_path)
    study = load_study(path, allow_engineering_draft=False)
    authority = verify_study_authority(study, repository_root=tmp_path)

    def forbidden_environment():
        pytest.fail("Direct real-data trial must fail before opening an environment")

    with pytest.raises(PermissionError, match="central study orchestration"):
        run_registered_policy_trial(
            component_id="rl_dqn_sb3_v1",
            context=FitContext(study.config["study_id"], "trial", "i1", 41),
            train_environment=forbidden_environment,
            evaluation_environment=forbidden_environment,
            parameters={}, total_timesteps=16,
            data_role=authority.data_role, study=study, authority=authority,
            repository_root=tmp_path,
        )
