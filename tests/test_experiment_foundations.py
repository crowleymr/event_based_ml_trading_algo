from datetime import date, timedelta

import pytest
import yaml

from trading_pipeline.experiments import ComponentRegistry, ComponentSpec, load_study
from trading_pipeline.experiments.default_registry import default_registry
from trading_pipeline.optimisation import (
    TrialRecord, grid_proposals, nested_purged_walk_forward,
    random_proposals, write_trial_ledger,
)


def test_component_registry_is_allowlisted_and_research_gated():
    registry = ComponentRegistry()
    spec = ComponentSpec("gmm_regime_v1", "UnsupervisedModel", "example.GMM")

    class Example:
        def __init__(self):
            self.spec = spec

    registry.register(spec, Example)
    assert registry.ids("UnsupervisedModel") == ("gmm_regime_v1",)
    assert registry.create("gmm_regime_v1").spec == spec
    with pytest.raises(ValueError, match="not enabled"):
        registry.create("gmm_regime_v1", require_research_enabled=True)
    with pytest.raises(ValueError, match="Unknown"):
        registry.create("missing")
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(spec, Example)


def test_draft_study_loads_for_engineering_but_not_research(tmp_path):
    config = {
        "schema_version": 3, "kind": "optimisation_study", "study_id": "draft_v1",
        "status": "draft", "authority": {"approval_record": None},
        "data": {"vintage_id": None},
        "validation": {"final_holdout": {"sealed": True, "manifest": None}},
        "experiment_arms": [{
            "id": "EN_F0", "interface": "SupervisedModel",
            "component_id": "supervised.elastic_net.v1",
        }], "objectives": {},
        "selection": {}, "search": {}, "reproducibility": {}, "outputs": {},
    }
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    first = load_study(path)
    second = load_study(path)
    assert first.engineering_only and first.sha256 == second.sha256
    with pytest.raises(ValueError, match="engineering verification"):
        load_study(path, allow_engineering_draft=False)


def test_approved_study_rejects_unresolved_authority_and_data(tmp_path):
    value = yaml.safe_load(open("configs/studies/phase3_engineering.yaml", encoding="utf-8"))
    value["status"] = "approved"
    path = tmp_path / "unresolved.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unresolved fields"):
        load_study(path)


def test_checked_in_engineering_study_and_default_registry_are_research_gated():
    study = load_study("configs/studies/phase3_engineering.yaml")
    assert study.engineering_only
    registry = default_registry()
    assert set(registry.ids()) >= {
        "supervised.elastic_net.v1", "supervised.hist_gbt.v1",
        "gmm_regime_v1", "rl_dqn_sb3_v1", "rl_ppo_categorical_sb3_v1",
    }
    for component_id in registry.ids():
        assert not registry.spec(component_id).research_enabled


def test_search_proposals_are_deterministic_and_conditional():
    grid = {
        "family": {"kind": "categorical", "values": ["linear", "tree"]},
        "depth": {"kind": "categorical", "values": [2, 4], "when": {"family": "tree"}},
    }
    proposals = grid_proposals(grid)
    assert {tuple(sorted(item.items())) for item in proposals} == {
        (("family", "linear"),),
        (("depth", 2), ("family", "tree")),
        (("depth", 4), ("family", "tree")),
    }
    space = {
        "alpha": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
        "leaves": {"kind": "categorical", "values": [3, 7, 15]},
    }
    assert random_proposals(space, 5, 42) == random_proposals(space, 5, 42)


def test_nested_walk_forward_is_chronological_purged_and_inner_bounded():
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(220)]
    label_end = {item: item + timedelta(days=5) for item in dates}
    nested = nested_purged_walk_forward(
        dates, label_end, outer_folds=3, inner_folds=2,
        outer_score_sessions=20, inner_score_sessions=10,
        min_fit_sessions=60, embargo_sessions=5,
    )
    assert len(nested) == 3
    for outer, inner in nested:
        assert max(outer.fit_dates) < min(outer.score_dates)
        assert all(label_end[item] < min(outer.score_dates) for item in outer.fit_dates)
        assert len(inner) == 2
        assert all(set(fold.fit_dates) <= set(outer.fit_dates) for fold in inner)


def test_trial_ledger_is_immutable_and_retains_failures(tmp_path):
    path = tmp_path / "trial_ledger.parquet"
    rows = [
        TrialRecord("study", "t1", "elastic_net_v2", 0, "complete", {"alpha": 0.1}),
        TrialRecord("study", "t2", "elastic_net_v2", 1, "failed", {"alpha": 1.0}, "fit failed"),
    ]
    write_trial_ledger(path, rows)
    with pytest.raises(FileExistsError):
        write_trial_ledger(path, rows)
