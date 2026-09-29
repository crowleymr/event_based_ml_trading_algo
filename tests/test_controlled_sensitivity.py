from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import polars as pl

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


def test_runner_collects_complete_and_failed_controlled_cells(monkeypatch, tmp_path, caplog):
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
    caplog.set_level(logging.INFO, logger=runner.__name__)
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
    messages = [record.getMessage() for record in caplog.records]
    assert sum("candidate started" in message for message in messages) == 3
    assert sum("candidate completed" in message for message in messages) == 2
    assert sum("candidate failed" in message for message in messages) == 1
    assert any("candidate=1/3 cells=0/3" in message for message in messages)
    assert any("completed candidates=3 cells=2/3" in message for message in messages)


def test_rl_trial_returns_current_price_rows_without_rescanning_history(monkeypatch, tmp_path):
    days = tuple(date(2026, 1, day) for day in range(1, 5))
    outputs = pl.DataFrame({"session_date": days})
    episode_calls = []

    def episode(*, split, **_kwargs):
        episode_calls.append(split)
        return SimpleNamespace(
            dataset=SimpleNamespace(split=split),
            eligibility=[{"signal_date": f"{split}-signal", "eligible_count": 2}],
            exclusions=[{"signal_date": f"{split}-signal", "security_id": "X",
                         "reason": "synthetic"}],
            source_hashes={"prices": "a" * 64},
        )

    execution = SimpleNamespace(
        metrics={"certainty_equivalent_mean": 0.25},
        telemetry={"duration_seconds": 0.01}, actions=[], equity_curve=[],
    )
    monkeypatch.setattr(runner, "_rl_episode", episode)
    monkeypatch.setattr(runner, "execute_study_rl_trial", lambda **_kwargs: execution)
    actual_open = Path.open
    evidence_append_opens = []

    def counted_open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path.name in {"rl_price_eligibility.jsonl", "rl_price_exclusions.jsonl"} \
                and mode == "a":
            evidence_append_opens.append(path.name)
        return actual_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    common = dict(
        study=SimpleNamespace(config={"search": {"rl_training": {"total_timesteps": 10}}}),
        authority=SimpleNamespace(study_id="study", protocol_sha256="b" * 64),
        arm={"component_id": "rl-test"}, params={}, seed=41,
        fold_id="inner-1", scenario="balanced", fit_dates=days[:2],
        score_dates=days[2:], rows=[], bars=None, outputs=outputs,
        feature_sha256="c" * 64, outputs_sha256="d" * 64,
        fold_sha256="e" * 64, output=tmp_path,
    )
    episode_cache = {}
    result = runner._rl_trial_cell(
        **common, trial_id="trial-1", episode_cache=episode_cache)
    second = runner._rl_trial_cell(
        **common, trial_id="trial-2", episode_cache=episode_cache)

    assert episode_calls == ["fit", "score"]
    assert len(episode_cache) == 1
    assert len(result["price_rows"]["rl_price_eligibility.jsonl"]) == 2
    assert len(result["price_rows"]["rl_price_exclusions.jsonl"]) == 2
    assert len(second["price_rows"]["rl_price_eligibility.jsonl"]) == 2
    assert evidence_append_opens == [
        "rl_price_eligibility.jsonl", "rl_price_exclusions.jsonl",
        "rl_price_eligibility.jsonl", "rl_price_exclusions.jsonl",
    ]
    assert len((tmp_path / "rl_price_eligibility.jsonl").read_text().splitlines()) == 4
    assert len((tmp_path / "rl_price_exclusions.jsonl").read_text().splitlines()) == 4


def test_rl_episode_cache_interns_scenario_invariant_price_storage(monkeypatch, tmp_path):
    @dataclass(frozen=True)
    class Dataset:
        split: str
        prices: dict
        calendar: tuple

    @dataclass(frozen=True)
    class Episode:
        dataset: Dataset
        eligibility: tuple = ()
        exclusions: tuple = ()
        source_hashes: dict | None = None

    days = tuple(date(2026, 1, day) for day in range(1, 5))
    outputs = pl.DataFrame({"session_date": days})

    def episode(*, split, **_kwargs):
        return Episode(Dataset(split, {(days[0], "A"): 1.0}, days),
                       source_hashes={"prices": "a" * 64})

    execution = SimpleNamespace(
        metrics={"certainty_equivalent_mean": 0.25},
        telemetry={"duration_seconds": 0.01}, actions=[], equity_curve=[],
    )
    monkeypatch.setattr(runner, "_rl_episode", episode)
    monkeypatch.setattr(runner, "execute_study_rl_trial", lambda **_kwargs: execution)
    common = dict(
        study=SimpleNamespace(config={"search": {"rl_training": {"total_timesteps": 10}}}),
        authority=SimpleNamespace(study_id="study", protocol_sha256="b" * 64),
        arm={"component_id": "rl-test"}, params={}, seed=41,
        fold_id="inner-1", fit_dates=days[:2], score_dates=days[2:],
        rows=[], bars=None, outputs=outputs, feature_sha256="c" * 64,
        outputs_sha256="d" * 64, fold_sha256="e" * 64, output=tmp_path,
    )
    cache = {}
    runner._rl_trial_cell(
        **common, scenario="balanced", trial_id="trial-balanced", episode_cache=cache)
    runner._rl_trial_cell(
        **common, scenario="aggressive", trial_id="trial-aggressive", episode_cache=cache)

    balanced = cache[(days[:2], days[2:], "balanced")]
    aggressive = cache[(days[:2], days[2:], "aggressive")]
    assert aggressive[0].dataset.prices is balanced[0].dataset.prices
    assert aggressive[1].dataset.prices is balanced[1].dataset.prices
