from __future__ import annotations

from datetime import date
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from trading_pipeline.experiments.default_registry import default_registry
from trading_pipeline.experiments import ComponentSpec
from trading_pipeline.experiments import study_runner as runner
from trading_pipeline.features import F1


def _config():
    return {"study_id": "synthetic", "data": {},
            "search": {"method": "deterministic_random_with_multifidelity_for_deep_models",
                       "budgets": {"classical_per_family": 4,
                                   "deep_supervised_per_family": 3},
                       "deep_training": {"epochs": 1, "patience": 1}},
            "reproducibility": {"seeds": [41], "requested_device": "cpu"}}


def test_supervised_proposals_use_family_budget_and_deterministic_deep_search():
    registry = default_registry()
    config = _config()
    classical = {"id": "en", "component_id": "supervised.elastic_net.v1"}
    deep = {"id": "lstm", "component_id": "supervised.lstm.v1"}
    assert len(runner._proposals(registry, classical, config)) == 4
    first = runner._proposals(registry, deep, config)
    assert len(first) == 3
    assert first == runner._proposals(registry, deep, config)
    assert "lookback" in first[0] and "optimizer" in first[0]
    config["search"]["budgets"]["classical_per_family"] = 8
    expanded = runner._proposals(registry, classical, config)
    assert len(expanded) == 8
    assert expanded == runner._proposals(registry, classical, config)


def test_study_bridge_uses_registered_capabilities_instead_of_component_ids():
    synthetic_sequence = ComponentSpec(
        "supervised.new_sequence.v1", "SupervisedModel", "trusted.NewSequence",
        capabilities={"study_adapter": True, "sequence_view": True})
    synthetic_tabular = ComponentSpec(
        "supervised.new_tabular.v1", "SupervisedModel", "trusted.NewTabular",
        capabilities={"study_adapter": True, "tabular_view": True})
    synthetic_policy = ComponentSpec(
        "rl_new.v1", "RLPolicy", "trusted.NewPolicy",
        capabilities={"study_adapter": True, "discrete_actions": True})
    assert runner._study_bridge(synthetic_sequence) == "deep"
    assert runner._study_bridge(synthetic_tabular) == "tabular"
    assert runner._study_bridge(synthetic_policy) == "rl"
    with pytest.raises(ValueError, match="No audited study bridge"):
        runner._study_bridge(ComponentSpec(
            "supervised.unknown.v1", "SupervisedModel", "trusted.Unknown"))
    with pytest.raises(ValueError, match="No audited study bridge"):
        runner._study_bridge(ComponentSpec(
            "supervised.ambiguous.v1", "SupervisedModel", "trusted.Ambiguous",
            capabilities={"study_adapter": True, "tabular_view": True,
                          "sequence_view": True}))


def test_rl_proposals_use_discovered_policy_factory():
    registry = default_registry()
    arm = {"id": "dqn", "component_id": "rl_dqn_sb3_v1"}
    config = {"search": {"budgets": {"rl_per_family_per_risk_scenario": 1},
                         "rl_training": {"total_timesteps": 16}},
              "reproducibility": {"seeds": [41]}}
    proposals = runner._rl_proposals(registry, arm, config)
    assert len(proposals) == 1
    policy = registry.create(arm["component_id"], parameters=proposals[0],
                             total_timesteps=16)
    assert policy.spec == registry.spec(arm["component_id"])


def test_deep_fit_dispatch_records_common_metrics(monkeypatch):
    config = _config()
    arm = {"component_id": "supervised.lstm.v1", "feature_set_id": "F1"}
    rows = [{"security_id": key, "ticker": key, "session_date": date(2026, 1, 3),
             "forward_return_5d": outcome} for key, outcome in (("A", 0.1), ("B", -0.1))]
    seen = {}

    def fake_fit(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(predictions=np.array([0.2, -0.2]),
                               telemetry={"wall_seconds": 0.01})

    monkeypatch.setattr(runner, "fit_score_deep_study", fake_fit)
    _, ic, rmse, telemetry, predictions = runner._fit_score(
        default_registry(), arm, {"lookback": 2}, 41, "trial", "fold",
        rows, rows, rows, config=config, feature_pool=rows, predict=True)
    assert ic == pytest.approx(1.0)
    assert rmse == pytest.approx(0.1)
    assert telemetry["wall_seconds"] == 0.01
    assert seen["feature_columns"] == tuple(F1)
    assert seen["score_rows"][0]["split"] == "fold"
    np.testing.assert_array_equal(predictions, [0.2, -0.2])


def test_augmented_contract_requires_pinned_table_and_unchanged_f1(tmp_path):
    base = {"security_id": ["A", "B"], "session_date": [date(2026, 1, 1)] * 2,
            **{column: [1.0, 2.0] for column in F1},
            "forward_return_5d": [0.1, 0.2]}
    canonical = tmp_path / "canonical.parquet"
    pl.DataFrame(base).write_parquet(canonical)
    augmented = tmp_path / "augmented.parquet"
    pl.DataFrame({key: value for key, value in base.items()
                  if key != "forward_return_5d"} | {"upstream_oof": [0.3, 0.4]}).write_parquet(augmented)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"study_id": "synthetic", "feature_sha256": digest(canonical),
                                  "table_sha256": digest(augmented)}), encoding="utf-8")
    config = _config()
    config["experiment_arms"] = [{"feature_set_id": "F1_CAUSAL_STACK_V1"}]
    config["data"]["augmented_feature_contract"] = {
        "feature_path": "augmented.parquet", "feature_sha256": digest(augmented),
        "feature_columns": [*F1, "upstream_oof"], "source_manifest": "source.json",
        "source_manifest_sha256": digest(source)}
    authority = SimpleNamespace(input_sha256={"augmented.parquet": digest(augmented),
                                                   "source.json": digest(source)})
    rows = runner._feature_rows(tmp_path, config, authority, canonical, digest(canonical))
    assert [row["upstream_oof"] for row in rows] == [0.3, 0.4]
    pl.DataFrame({key: value for key, value in base.items()
                  if key != "forward_return_5d"} | {"upstream_oof": [0.3, 0.4],
                                                       "latest_eps": [99.0, 99.0]}).write_parquet(augmented)
    config["data"]["augmented_feature_contract"]["feature_sha256"] = digest(augmented)
    authority.input_sha256["augmented.parquet"] = digest(augmented)
    source.write_text(json.dumps({"study_id": "synthetic", "feature_sha256": digest(canonical),
                                  "table_sha256": digest(augmented)}), encoding="utf-8")
    with pytest.raises(ValueError, match="changes canonical F1"):
        runner._feature_rows(tmp_path, config, authority, canonical, digest(canonical))


def test_development_label_cache_reuses_only_declared_non_holdout_partitions(tmp_path,
                                                                              monkeypatch):
    days = [date(2026, 1, day) for day in range(1, 6)]
    feature_path = tmp_path / "features.parquet"
    pl.DataFrame({"security_id": ["A"] * 5, "session_date": days,
                  "forward_return_5d": [0.01, 0.02, 0.03, 0.04, 9.99]}).write_parquet(feature_path)
    by_day = {day: [{"security_id": "A", "session_date": day}] for day in days}
    fold = {"fit_dates": [days[0].isoformat()],
            "stopping_dates": [days[1].isoformat()],
            "score_dates": [days[2].isoformat()]}
    outer = [{"fit_dates": [days[0].isoformat(), days[1].isoformat()],
              "score_dates": [days[3].isoformat()]}]
    scanned = []
    original_scan = runner.pl.scan_parquet

    def counted_scan(*args, **kwargs):
        scanned.append(args[0])
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(runner.pl, "scan_parquet", counted_scan)
    cache = runner._DevelopmentLabelCache(feature_path, by_day, outer, {"o1": [fold]},
                                          (days[4],))
    assert cache.allowed_dates == frozenset(days[:4])
    first = cache.partitions((days[0],), (days[1],), (days[2],))
    assert [part[0]["forward_return_5d"] for part in first] == [0.01, 0.02, 0.03]
    assert len(scanned) == 1
    cache.partitions((days[0],), (days[1],), (days[2],))
    assert len(scanned) == 1
    assert days[4] not in cache._loaded_dates
    with pytest.raises(PermissionError, match="outside declared development"):
        cache.partitions((days[4],))
    assert len(scanned) == 1
    assert cache.partitions((days[3],))[0][0]["forward_return_5d"] == 0.04
    assert len(scanned) == 2


def test_holdout_labels_require_completed_family_lock_before_parquet_read(tmp_path,
                                                                         monkeypatch):
    holdout_day = date(2026, 1, 5)
    feature_path = tmp_path / "features.parquet"
    pl.DataFrame({"security_id": ["A"], "session_date": [holdout_day],
                  "forward_return_5d": [0.5]}).write_parquet(feature_path)
    by_day = {holdout_day: [{"security_id": "A", "session_date": holdout_day}]}
    lock_path = tmp_path / "family_locks.json"
    scanned = []
    original_scan = runner.pl.scan_parquet

    def counted_scan(*args, **kwargs):
        scanned.append(args[0])
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(runner.pl, "scan_parquet", counted_scan)
    with pytest.raises(PermissionError, match="Family lock required"):
        runner._holdout_labelled(feature_path, by_day, (holdout_day,), lock_path)
    assert not scanned
    lock_path.write_text(json.dumps({"holdout_selection": "forbidden",
                                     "final_choices": {}}), encoding="utf-8")
    with pytest.raises(PermissionError, match="Complete family lock"):
        runner._holdout_labelled(feature_path, by_day, (holdout_day,), lock_path)
    assert not scanned
    lock_path.write_text(json.dumps({"holdout_selection": "forbidden",
                                     "final_choices": {"arm": {"parameters": {}}}}),
                         encoding="utf-8")
    assert runner._holdout_labelled(feature_path, by_day, (holdout_day,), lock_path)[0][
        "forward_return_5d"] == 0.5
    assert len(scanned) == 1
