"""Negative controls for the score-blind study freeze."""

from datetime import date, timedelta
from pathlib import Path
import json

import polars as pl
import pytest

from trading_pipeline.experiments.manifest_resolution import (
    _partition, _session_exposure, finalise_approved_study, freeze_study_manifests,
)
from trading_pipeline.optimisation.splits import TemporalFold
from trading_pipeline.experiments.budget_tiers import TIERS


def test_exposure_rejects_duplicate_security_session_and_missing_label(tmp_path: Path):
    path = tmp_path / "features.parquet"
    pl.DataFrame({
        "session_date": [date(2026, 1, 1), date(2026, 1, 1)],
        "security_id": ["A", "A"],
        "label_end_date": [date(2026, 1, 8), date(2026, 1, 8)],
        "forward_return_5d": [0.1, 0.2],
    }).write_parquet(path)
    with pytest.raises(ValueError, match="duplicated"):
        _session_exposure(path)
    pl.DataFrame({
        "session_date": [date(2026, 1, 1)], "security_id": ["A"],
        "label_end_date": [None], "forward_return_5d": [0.1],
    }).write_parquet(path)
    with pytest.raises(ValueError, match="null"):
        _session_exposure(path)


def test_partition_purges_label_exposure_before_stopping_and_scoring():
    start = date(2026, 1, 1)
    fit = tuple(start + timedelta(days=index) for index in range(8))
    stop = fit[-2:]
    score = (start + timedelta(days=10),)
    fold = TemporalFold("f", fit, score, (start + timedelta(days=9),))
    ends = {day: day + timedelta(days=1) for day in fit}
    ends[fit[-3]] = stop[0]
    result = _partition(fold, ends, stopping_sessions=2)
    assert fit[-3].isoformat() in result["purged_dates"]
    assert fit[-3].isoformat() not in result["fit_dates"]
    ends[stop[0]] = score[0]
    with pytest.raises(ValueError, match="crosses"):
        _partition(fold, ends, stopping_sessions=2)


def test_missing_authority_input_creates_no_protocol(tmp_path: Path):
    output = tmp_path / "protocol"
    approved = tmp_path / "approved.yaml"
    with pytest.raises(ValueError, match="missing"):
        freeze_study_manifests(repository_root=tmp_path,
            source_manifest=tmp_path / "source.json",
            vintage_manifest=tmp_path / "vintage.json",
            feature_manifest=tmp_path / "feature.json",
            feature_path=tmp_path / "features.parquet",
            output_dir=output, prepared_study=approved)
    assert not output.exists() and not approved.exists()


def test_partial_capability_cannot_approve(tmp_path: Path):
    import json
    import yaml
    from trading_pipeline.experiments.schema import protocol_content_sha256

    prepared = tmp_path / "prepared.yaml"
    gate = tmp_path / "gate.json"
    approved = tmp_path / "approved.yaml"
    config = yaml.safe_load((Path(__file__).parents[1] /
        "configs/studies/expanded_closeout_draft.yaml").read_text(encoding="utf-8"))
    config["authority"]["protocol_sha256"] = protocol_content_sha256(config)
    prepared.write_text(yaml.safe_dump(config), encoding="utf-8")
    arm_ids = [arm["id"] for arm in config["experiment_arms"]]
    gate.write_text(json.dumps({"study_id": config["study_id"],
        "arm_capabilities": {arm_ids[0]: {"real_data_bridge_passed": True,
                                           "synthetic_verification_passed": True}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="every declared arm"):
        finalise_approved_study(repository_root=tmp_path, prepared_study=prepared,
                               capability_gate=gate, approved_study=approved)
    assert not approved.exists()


def test_manifest_freeze_cli_passes_deadline_window_policy(monkeypatch, capsys):
    from trading_pipeline.experiments import manifest_resolution

    captured = {}
    def fake_freeze(**kwargs):
        captured.update(kwargs)
        return {"status": "captured"}

    monkeypatch.setattr(manifest_resolution, "freeze_study_manifests", fake_freeze)
    monkeypatch.setattr("sys.argv", ["manifest_resolution", "--source-manifest", "source.json",
        "--vintage-manifest", "vintage.json", "--feature-manifest", "feature.json",
        "--features", "features.parquet", "--output", "manifests",
        "--outer-folds", "2", "--inner-folds", "2", "--holdout-sessions", "50",
        "--outer-score-sessions", "15", "--inner-score-sessions", "8",
        "--min-fit-sessions", "100", "--stopping-sessions", "8",
        "--embargo-sessions", "5"])
    manifest_resolution._cli()
    assert json.loads(capsys.readouterr().out) == {"status": "captured"}
    assert {name: captured[name] for name in ("outer_folds", "inner_folds",
        "holdout_sessions", "outer_score_sessions", "inner_score_sessions",
        "min_fit_sessions", "stopping_sessions", "embargo_sessions")} == {
            "outer_folds": 2, "inner_folds": 2, "holdout_sessions": 50,
            "outer_score_sessions": 15, "inner_score_sessions": 8,
            "min_fit_sessions": 100, "stopping_sessions": 8, "embargo_sessions": 5,
        }
    assert TIERS["deadline_complete"] == (2, 2, 2)
