from __future__ import annotations

from datetime import date
import json
from types import SimpleNamespace

from trading_pipeline.experiments.default_registry import default_registry
from trading_pipeline.experiments import study_runner as runner
from trading_pipeline.optimisation.search import controlled_one_factor_design


def _design(component_id: str, *, rl: bool = False):
    registry = default_registry()
    if rl:
        component = registry.create(component_id, total_timesteps=100)
        validate = lambda parameters: registry.create(
            component_id, parameters=parameters, total_timesteps=100)
    else:
        component = registry.create(component_id)
        validate = lambda parameters: registry.create(component_id, params=parameters)
    return controlled_one_factor_design(component.search_space()["parameters"], validate)


def test_every_supervised_search_dimension_has_a_controlled_contrast():
    registry = default_registry()
    for component_id in (
        "supervised.elastic_net.v1",
        "supervised.hist_gbt.v1",
        "supervised.xgboost.v1",
        "supervised.lstm.v1",
        "supervised.causal_transformer.v1",
    ):
        design = _design(component_id)
        expected = set(registry.create(component_id).search_space()["parameters"])
        assert {item["factor_name"] for item in design["contrasts"]} == expected
        for contrast in design["contrasts"]:
            reference = design["proposals"][contrast["reference_proposal_index"]]
            candidate = design["proposals"][contrast["contrast_proposal_index"]]
            assert reference != candidate
            assert set(contrast["directly_changed_fields"]) <= {
                contrast["factor_name"], "value_widths"
            }


def test_conditional_ppo_dimension_receives_its_own_matched_reference():
    design = _design("rl_ppo_categorical_sb3_v1", rl=True)
    by_factor = {item["factor_name"]: item for item in design["contrasts"]}
    width = by_factor["value_widths"]
    reference = design["proposals"][width["reference_proposal_index"]]
    candidate = design["proposals"][width["contrast_proposal_index"]]
    assert reference["architecture_mode"] == candidate["architecture_mode"] == "separate"
    assert reference["value_widths"] != candidate["value_widths"]
    assert width["directly_changed_fields"] == ["value_widths"]


def test_dqn_design_is_deterministic_and_covers_every_parameter():
    first = _design("rl_dqn_sb3_v1", rl=True)
    second = _design("rl_dqn_sb3_v1", rl=True)
    assert first == second
    assert len(first["contrasts"]) == 12
    assert len(first["proposals"]) == 13


def test_runner_collects_complete_and_failed_controlled_cells(monkeypatch, tmp_path):
    registry = default_registry()
    arm = {"id": "EN", "component_id": "supervised.elastic_net.v1",
           "interface": "SupervisedModel", "feature_set_id": "F1"}
    config = {
        "search": {
            "deep_training": {"epochs": 8, "patience": 2},
            "rl_training": {"total_timesteps": 5000},
            "controlled_sensitivity": {
                "enabled": True,
                "design": "deterministic_controlled_one_factor_v1",
                "inner_folds_per_outer": 1,
                "selection_eligible": False,
                "deep_training": {"epochs": 2, "patience": 1},
                "rl_training": {"total_timesteps": 1000},
            },
        },
        "objectives": {},
    }
    designs = runner._controlled_sensitivity_designs(registry, [arm], config)
    fold = {"fold_id": "i1", "fit_dates": ["2026-01-01"],
            "stopping_dates": ["2026-01-02"], "score_dates": ["2026-01-03"]}
    rows = [{"session_date": date(2026, 1, 1), "label_end_date": date(2026, 1, 1)}]

    class Labels:
        def partitions(self, *_args):
            return rows, rows, rows

    class Checkpoints:
        def get(self, *_args):
            return None

        def save(self, *_args, **_kwargs):
            return None

    calls = 0

    def fit_score(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic candidate failure")
        return None, 0.1, 0.2, {"wall_seconds": 0.01}, None

    monkeypatch.setattr(runner, "_fit_score", fit_score)
    runner._run_controlled_sensitivity(
        study=SimpleNamespace(config=config),
        authority=SimpleNamespace(study_id="study", protocol_sha256="a" * 64),
        registry=registry, arms=[arm], outer=[{"fold_id": "o1"}],
        inner={"o1": [fold]}, designs=designs, seeds=[41],
        development_labels=Labels(), rows=rows, bars=None, outputs=None,
        feature_sha256="b" * 64, outputs_sha256=None, fold_sha256="c" * 64,
        output=tmp_path, checkpoints=Checkpoints(), source=None,
    )
    ledger = [json.loads(line) for line in
              (tmp_path / "sensitivity_trial_ledger.jsonl").read_text().splitlines()]
    terminals = [row for row in ledger if row["status"] != "proposed"]
    assert len(terminals) == len(designs["EN"]["proposals"])
    assert {row["status"] for row in terminals} == {"complete", "failed"}
    failure = next(row for row in terminals if row["status"] == "failed")
    assert failure["completed_cells"] == 0
    assert failure["expected_cells"] == 1
    assert failure["exception_type"] == "RuntimeError"
    assert "synthetic candidate failure" in failure["diagnostic_traceback"]
