from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trading_pipeline.experiments.calibration import (
    write_capability_gate, write_score_blind_calibration,
)
from trading_pipeline.experiments.budget_tiers import TIERS


ROOT = Path(__file__).parents[1]
PREPARED = ROOT / "configs/studies/expanded_closeout_prepared.yaml"


def _records():
    import yaml
    config = yaml.safe_load(PREPARED.read_text(encoding="utf-8"))
    return [{"arm_id": arm["id"], "component_id": arm["component_id"],
             "wall_seconds": 0.001, "fit_rows": 20, "evaluation_rows": 10,
             "device": "cpu", "status": "complete", "evidence_sha256": "a" * 64}
            for arm in config["experiment_arms"]]


def test_score_blind_calibration_selects_complete_tier(tmp_path):
    output = tmp_path / "calibration.json"
    result = write_score_blind_calibration(prepared_study=PREPARED,
        records=_records(), deadline_seconds=1000, output_path=output)
    assert result["budget_tier"] == "full"
    assert result["outcome_rankings_opened"] is False
    assert output.is_file()


def test_deadline_tier_keeps_two_proposals_for_every_family(tmp_path):
    result = write_score_blind_calibration(prepared_study=PREPARED,
        records=_records(), deadline_seconds=1, output_path=tmp_path / "deadline.json")
    assert TIERS["deadline_complete"] == (2, 2, 2)
    assert result["budget_tier"] == "deadline_complete"
    assert result["estimated_seconds_by_tier"]["minimum_defensible"] > 1
    assert result["estimated_seconds_by_tier"]["deadline_complete"] <= 1
    assert result["estimation_policy"]["outer_folds"] == 5
    assert result["estimation_policy"]["inner_folds_per_outer"] == 3
    assert result["estimation_policy"]["rl_scenarios"] == 3
    assert "smoke" in result["estimation_policy"]["shape_extrapolation"]


def test_deadline_tier_generates_distinct_hpo_candidates_for_all_seven_arms():
    import yaml
    from trading_pipeline.experiments.default_registry import default_registry
    from trading_pipeline.experiments.study_runner import _proposals, _rl_proposals

    config = yaml.safe_load((ROOT / "configs/studies/expanded_closeout_draft.yaml")
                            .read_text(encoding="utf-8"))
    config["search"]["budgets"] = dict(zip(
        ("classical_per_family", "deep_supervised_per_family",
         "rl_per_family_per_risk_scenario"), TIERS["deadline_complete"]))
    assert config["reproducibility"]["seeds"] == [41]
    registry = default_registry()
    proposals = {}
    for arm in config["experiment_arms"]:
        producer = _rl_proposals if arm["interface"] == "RLPolicy" else _proposals
        first = producer(registry, arm, config)
        assert len(first) == 2 and first[0] != first[1]
        assert first == producer(registry, arm, config)
        proposals[arm["id"]] = first
    assert len(proposals) == 7
    for arm_id, architecture_fields in {
        "LSTM_F1_STACK": ("lookback", "depth", "width"),
        "TRANSFORMER_F1_STACK": ("heads", "feed_forward", "width"),
        "DQN_SELECTOR": ("network_widths", "activation"),
        "PPO_SELECTOR": ("architecture_mode", "policy_widths", "activation"),
    }.items():
        first, second = proposals[arm_id]
        assert any(first[field] != second[field] for field in architecture_fields)


def test_calibration_rejects_outcomes_and_partial_matrix(tmp_path):
    records = _records()
    records[0]["rmse"] = 0.1
    with pytest.raises(ValueError, match="cannot contain outcomes"):
        write_score_blind_calibration(prepared_study=PREPARED, records=records,
            deadline_seconds=1000, output_path=tmp_path / "bad.json")
    with pytest.raises(ValueError, match="every declared arm"):
        write_score_blind_calibration(prepared_study=PREPARED, records=_records()[:-1],
            deadline_seconds=1000, output_path=tmp_path / "partial.json")


def test_capability_gate_requires_hashable_exact_evidence(tmp_path):
    prepared = tmp_path / "prepared.yaml"
    prepared.write_bytes(PREPARED.read_bytes())
    calibration = tmp_path / "calibration.json"
    write_score_blind_calibration(prepared_study=prepared, records=_records(),
        deadline_seconds=1000, output_path=calibration)
    study_id = "expanded_closeout_exploratory_v1"
    synthetic, bridge = {}, {}
    for record in _records():
        arm_id = record["arm_id"]
        for kind, mapping in (("synthetic_verification", synthetic),
                              ("real_data_bridge_smoke", bridge)):
            path = tmp_path / f"{arm_id}-{kind}.json"
            path.write_text(json.dumps({"study_id": study_id, "arm_id": arm_id,
                "status": "passed", "evidence_kind": kind}), encoding="utf-8")
            mapping[arm_id] = path
    gate = tmp_path / "gate.json"
    result = write_capability_gate(repository_root=tmp_path, prepared_study=prepared,
        calibration_path=calibration, synthetic_evidence=synthetic,
        real_bridge_evidence=bridge, output_path=gate)
    assert len(result["arm_capabilities"]) == 7
    assert all(len(item["real_bridge_evidence"]["sha256"]) == 64
               for item in result["arm_capabilities"].values())
    broken = dict(bridge)
    broken.pop(next(iter(broken)))
    with pytest.raises(ValueError, match="all declared arms"):
        write_capability_gate(repository_root=tmp_path, prepared_study=prepared,
            calibration_path=calibration, synthetic_evidence=synthetic,
            real_bridge_evidence=broken, output_path=tmp_path / "bad-gate.json")
