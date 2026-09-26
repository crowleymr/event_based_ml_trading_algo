from datetime import date

import numpy as np
import polars as pl
import pytest

from trading_pipeline.portfolio import model_conditioned_frontier, realised_risk_return_curve


def test_model_conditioned_frontier_is_complete_and_reconciled():
    points, weights = model_conditioned_frontier(
        model_id="supervised.lstm.v1",
        rebalance_date=date(2026, 9, 25),
        security_ids=["A", "B", "C"],
        expected_returns=[0.08, 0.12, 0.05],
        covariance=np.array([
            [0.040, 0.008, 0.004],
            [0.008, 0.090, 0.006],
            [0.004, 0.006, 0.025],
        ]),
        risk_aversions=[8.0, 4.0, 2.0],
        gross_exposure_cap=0.8,
        max_position=0.5,
        covariance_id="sample_cov_60d_v1",
        input_sha256="abc123",
    )
    assert points.height == 3
    assert weights.height == 9
    assert set(points["status"]) == {"complete"}
    assert points["gross_exposure"].max() <= 0.8 + 1e-8
    reconciled = weights.group_by("point_id").agg(pl.col("weight").sum().alias("weight_sum"))
    assert reconciled["weight_sum"].max() <= 0.8 + 1e-8
    assert points.sort("risk_aversion")["expected_volatility"].to_list() == sorted(
        points["expected_volatility"].to_list(), reverse=True
    )


def test_model_conditioned_frontier_rejects_invalid_covariance():
    with pytest.raises(ValueError, match="symmetric"):
        model_conditioned_frontier(
            model_id="m", rebalance_date=date(2026, 9, 25), security_ids=["A", "B"],
            expected_returns=[0.1, 0.2], covariance=np.array([[1.0, 0.2], [0.0, 1.0]]),
            risk_aversions=[1.0], covariance_id="c", input_sha256="h",
        )


def _metrics():
    rows = []
    for model, offset in (("lstm", 0.00), ("ppo", 0.01)):
        for scenario, volatility, annual_return in (
            ("conservative", 0.08, 0.06),
            ("balanced", 0.12, 0.09),
            ("aggressive", 0.16, 0.10),
        ):
            rows.append({
                "model_id": model, "risk_scenario": scenario,
                "annualised_return": annual_return + offset,
                "annualised_volatility": volatility,
                "sharpe": 0.8, "maximum_drawdown": -0.1,
                "total_turnover": 2.0, "cumulative_transaction_cost": 0.01,
                "start_date": date(2025, 1, 2), "end_date": date(2026, 9, 25),
                "sessions": 420, "calendar_sha256": "calendar-a",
                "risk_control_value": {"conservative": 0.6, "balanced": 0.8, "aggressive": 1.0}[scenario],
            })
    return pl.DataFrame(rows)


def test_realised_curve_keeps_every_model_scenario_and_marks_dominance():
    result = realised_risk_return_curve(_metrics())
    assert result.height == 6
    assert sorted(result.group_by("model_id").len()["len"].to_list()) == [3, 3]
    assert result["risk_control_monotonic"].all()
    assert result.filter(pl.col("model_id") == "lstm")["dominated"].all()


def test_realised_curve_requires_complete_aligned_scenarios():
    with pytest.raises(ValueError, match="each risk scenario"):
        realised_risk_return_curve(_metrics().filter(pl.col("risk_scenario") != "balanced"))
    misaligned = _metrics().with_columns(
        pl.when(pl.col("model_id") == "ppo")
        .then(pl.lit(date(2025, 2, 1)))
        .otherwise(pl.col("start_date"))
        .alias("start_date")
    )
    with pytest.raises(ValueError, match="identical evaluation window"):
        realised_risk_return_curve(misaligned)
    different_calendar = _metrics().with_columns(
        pl.when(pl.col("model_id") == "ppo")
        .then(pl.lit("calendar-b"))
        .otherwise(pl.col("calendar_sha256"))
        .alias("calendar_sha256")
    )
    with pytest.raises(ValueError, match="identical evaluation window"):
        realised_risk_return_curve(different_calendar)
