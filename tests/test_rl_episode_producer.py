from datetime import date, timedelta

import polars as pl
import pytest

from trading_pipeline.experiments.rl_episode_producer import (
    produce_selector_episode, selector_feature_frame,
)
from trading_pipeline.rl.environment import StrategySelectorEnv


def _inputs():
    days = tuple(date(2026, 1, 5) + timedelta(days=i) for i in range(15))
    signals = (days[7], days[14])
    features = pl.DataFrame([
        {"session_date": day, "security_id": security, "return_5d": 0.01 + j / 100,
         "return_20d": 0.03 + j / 100, "vol_20d": 0.1 + j / 100}
        for day in signals for j, security in enumerate(("A", "B"))
    ])
    outputs = pl.DataFrame([
        {"session_date": day, "security_id": security, "model_id": model,
         "predicted_return_5d": 0.02 + j / 100 + k / 1000,
         "fit_cutoff_date": days[6], "fit_max_label_end_date": days[5],
         "prediction_role": "inner_oof"}
        for day in signals for k, model in enumerate(("E1", "E2", "E3", "E4"))
        for j, security in enumerate(("A", "B"))
    ])
    bars = pl.DataFrame([
        {"session_date": day, "security_id": security,
         "adjusted_close": 100 + i * (1 + j / 2) + (i % 3) * 0.2}
        for i, day in enumerate(days) for j, security in enumerate(("A", "B"))
    ])
    kwargs = dict(split="fit", signal_sessions=signals, calendar=days,
                  features=features, model_outputs=outputs, bars=bars,
                  source_hashes={name: letter * 64 for name, letter in zip(
                      ("canonical_features", "model_outputs", "adjusted_close",
                       "fold_manifest", "protocol"), "abcde")},
                  e5_source="E3", top_k=2, gross_exposure_cap=0.6,
                  max_position=0.2, annualised_volatility_target=0.08,
                  volatility_lookback_sessions=5,
                  allowed_prediction_roles=("inner_oof",))
    return kwargs


def test_producer_builds_causal_weekly_sleeves_with_three_constraints():
    result = produce_selector_episode(**_inputs())
    assert result.dataset.signal_dates == (date(2026, 1, 12),)
    assert result.dataset.execution_dates == (date(2026, 1, 13),)
    assert result.dataset.end_dates == (date(2026, 1, 19),)
    assert len(result.scaling) == 6
    assert all(row["resulting_gross_exposure"] <= 0.6 for row in result.scaling)
    assert all(row["estimated_annualised_volatility"] * row["applied_scale"] <= 0.08 + 1e-12
               for row in result.scaling)
    assert all(max(weights.values()) <= 0.2 for weights in result.dataset.targets.values() if weights)
    assert result.dataset.targets[(0, 0)] == {}
    assert result.dataset.targets[(0, 6)] != {}
    assert set(result.source_hashes) == {"canonical_features", "model_outputs",
                                          "adjusted_close", "fold_manifest", "protocol"}


def test_producer_rejects_future_model_fit_and_missing_held_bar():
    kwargs = _inputs()
    kwargs["model_outputs"] = kwargs["model_outputs"].with_columns(
        pl.when(pl.col("model_id") == "E2").then(pl.col("session_date"))
        .otherwise(pl.col("fit_cutoff_date")).alias("fit_cutoff_date"))
    with pytest.raises(ValueError, match="point-in-time"):
        produce_selector_episode(**kwargs)
    kwargs = _inputs()
    kwargs["bars"] = kwargs["bars"].filter(~((pl.col("session_date") == date(2026, 1, 17))
                                             & (pl.col("security_id") == "B")))
    with pytest.raises(ValueError, match="Missing held-security"):
        produce_selector_episode(**kwargs)


def test_declared_endpoint_policy_uses_exact_weekly_prices_without_interior_fill():
    kwargs = _inputs()
    daily = produce_selector_episode(**kwargs)
    kwargs["price_eligibility_policy"] = "observed_history_and_endpoint_valuation_v1"
    complete = produce_selector_episode(**kwargs)
    daily_env, endpoint_env = (StrategySelectorEnv(item.dataset) for item in (daily, complete))
    daily_env.reset(seed=41)
    endpoint_env.reset(seed=41)
    assert daily_env.step(2)[1] == pytest.approx(endpoint_env.step(2)[1])
    missing_day = date(2026, 1, 17)
    kwargs["bars"] = kwargs["bars"].filter(~((pl.col("session_date") == missing_day)
                                             & (pl.col("security_id") == "B")))
    episode = produce_selector_episode(**kwargs)
    assert episode.eligibility[0]["eligible_count"] == 2
    assert episode.eligibility[0]["excluded_count"] == 0
    environment = StrategySelectorEnv(episode.dataset)
    environment.reset(seed=41)
    _, reward, done, _, _ = environment.step(2)
    assert done and isinstance(reward, float)
    assert (missing_day, "B") not in episode.dataset.prices

    kwargs["bars"] = kwargs["bars"].filter(~((pl.col("session_date") == date(2026, 1, 13))
                                             & (pl.col("security_id") == "B")))
    with pytest.raises(ValueError, match="required endpoint"):
        produce_selector_episode(**kwargs)


def test_observed_gap_excludes_security_from_later_signal_only():
    kwargs = _inputs()
    days = tuple(date(2026, 1, 5) + timedelta(days=i) for i in range(22))
    kwargs["calendar"] = days
    kwargs["signal_sessions"] = (days[7], days[14], days[21])
    kwargs["price_eligibility_policy"] = "observed_history_and_endpoint_valuation_v1"
    kwargs["features"] = pl.concat([kwargs["features"], kwargs["features"].filter(
        pl.col("session_date") == days[14]).with_columns(pl.lit(days[21]).alias("session_date"))])
    kwargs["model_outputs"] = pl.concat([kwargs["model_outputs"], kwargs["model_outputs"].filter(
        pl.col("session_date") == days[14]).with_columns(pl.lit(days[21]).alias("session_date"))])
    extra = [{"session_date": day, "security_id": security,
              "adjusted_close": 100 + i * (1 + j / 2) + (i % 3) * 0.2}
             for i, day in enumerate(days[15:], 15) for j, security in enumerate(("A", "B"))]
    kwargs["bars"] = pl.concat([kwargs["bars"], pl.DataFrame(extra)]).filter(
        ~((pl.col("session_date") == days[10]) & (pl.col("security_id") == "B")))
    episode = produce_selector_episode(**kwargs)
    assert episode.eligibility[0]["eligible_count"] == 2
    assert episode.eligibility[1]["eligible_count"] == 1
    assert episode.eligibility[1]["reason_counts"] == {"missing_observed_volatility_history": 1}
    assert episode.exclusions[0]["security_id"] == "B"
    assert episode.exclusions[0]["missing_sessions"] == [days[10].isoformat()]


def test_producer_requires_declared_volatility_history_and_complete_outputs():
    kwargs = _inputs()
    kwargs["volatility_lookback_sessions"] = 20
    with pytest.raises(ValueError, match="Insufficient pre-signal"):
        produce_selector_episode(**kwargs)
    kwargs = _inputs()
    kwargs["model_outputs"] = kwargs["model_outputs"].filter(~((pl.col("model_id") == "E4")
                                                                & (pl.col("security_id") == "B")))
    with pytest.raises(ValueError, match="Missing causal model outputs"):
        produce_selector_episode(**kwargs)
def test_selector_feature_frame_scans_sparse_expanded_schema():
    rows = [{"security_id": "A", "session_date": date(2020, 1, 2),
             "return_5d": None, "return_20d": 0.0, "vol_20d": 0.1,
             "unselected_sparse_column": None} for _ in range(100)]
    rows.append({**rows[-1], "return_5d": -0.002504,
                 "unselected_sparse_column": 1.0})
    frame = selector_feature_frame(rows)
    assert frame["return_5d"].dtype == pl.Float64
    assert frame["return_5d"][-1] == pytest.approx(-0.002504)
