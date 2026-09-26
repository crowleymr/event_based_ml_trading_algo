"""Deterministic risk-return evidence for model-conditioned portfolios.

The classical frontier functions are only applicable when a model emits a
security-level expected-return vector.  Realised risk-return profiles use
completed backtest metrics and therefore also support selector policies such as
DQN and PPO.  The two evidence types intentionally remain separate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.optimize import minimize


RISK_SCENARIOS = ("conservative", "balanced", "aggressive")


def _validated_inputs(
    security_ids: Sequence[str],
    expected_returns: Sequence[float],
    covariance: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(security_ids)
    means = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if not names or len(set(names)) != len(names):
        raise ValueError("security_ids must be non-empty and unique")
    if means.shape != (len(names),) or cov.shape != (len(names), len(names)):
        raise ValueError("Expected returns and covariance dimensions must match securities")
    if not np.isfinite(means).all() or not np.isfinite(cov).all():
        raise ValueError("Frontier inputs must be finite")
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError("Covariance matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues.min() < -1e-9:
        raise ValueError("Covariance matrix must be positive semidefinite")
    return names, means, cov


def model_conditioned_frontier(
    *,
    model_id: str,
    rebalance_date,
    security_ids: Sequence[str],
    expected_returns: Sequence[float],
    covariance: np.ndarray,
    risk_aversions: Sequence[float],
    gross_exposure_cap: float = 1.0,
    max_position: float = 1.0,
    covariance_id: str,
    input_sha256: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Solve a long-only model-conditioned utility frontier.

    Expected returns and covariance must use the same annualised convention.
    Cash has zero expected return and variance, so ``sum(weights)`` may remain
    below the gross cap.  Every requested risk-aversion value produces an
    explicit completed or infeasible point.
    """

    names, means, cov = _validated_inputs(security_ids, expected_returns, covariance)
    if not model_id or not covariance_id or not input_sha256:
        raise ValueError("Model, covariance and input lineage identifiers are required")
    if not 0 < gross_exposure_cap <= 1:
        raise ValueError("gross_exposure_cap must be in (0, 1]")
    if not 0 < max_position <= gross_exposure_cap:
        raise ValueError("max_position must be in (0, gross_exposure_cap]")
    lambdas = [float(value) for value in risk_aversions]
    if not lambdas or any(not np.isfinite(value) or value <= 0 for value in lambdas):
        raise ValueError("risk_aversions must contain positive finite values")
    if len(set(lambdas)) != len(lambdas):
        raise ValueError("risk_aversions must be unique")

    point_rows: list[dict] = []
    weight_rows: list[dict] = []
    initial = np.full(len(names), min(gross_exposure_cap / len(names), max_position))
    for frontier_index, risk_aversion in enumerate(sorted(lambdas, reverse=True)):
        result = minimize(
            lambda weights: -(means @ weights - 0.5 * risk_aversion * (weights @ cov @ weights)),
            initial,
            method="SLSQP",
            bounds=[(0.0, max_position)] * len(names),
            constraints=[{"type": "ineq", "fun": lambda weights: gross_exposure_cap - weights.sum()}],
            options={"maxiter": 500, "ftol": 1e-12, "disp": False},
        )
        status = "complete" if result.success else "solver_failed"
        weights = np.asarray(result.x, dtype=float) if result.success else np.full(len(names), np.nan)
        expected_return = float(means @ weights) if result.success else None
        variance = float(weights @ cov @ weights) if result.success else None
        expected_volatility = float(np.sqrt(max(variance, 0.0))) if variance is not None else None
        point_id = f"{model_id}:{rebalance_date}:{frontier_index}"
        point_rows.append({
            "point_id": point_id,
            "model_id": model_id,
            "rebalance_date": rebalance_date,
            "frontier_index": frontier_index,
            "risk_aversion": risk_aversion,
            "expected_return": expected_return,
            "expected_volatility": expected_volatility,
            "gross_exposure": float(weights.sum()) if result.success else None,
            "status": status,
            "solver_message": str(result.message),
            "covariance_id": covariance_id,
            "input_sha256": input_sha256,
        })
        for security_id, weight in zip(names, weights, strict=True):
            weight_rows.append({
                "point_id": point_id,
                "model_id": model_id,
                "rebalance_date": rebalance_date,
                "security_id": security_id,
                "weight": float(weight) if result.success else None,
                "status": status,
            })
        if result.success:
            initial = weights
    return pl.DataFrame(point_rows), pl.DataFrame(weight_rows)


def realised_risk_return_curve(metrics: pl.DataFrame) -> pl.DataFrame:
    """Validate and annotate complete three-scenario realised profiles.

    The input is backtest evidence, not a classical efficient frontier.  No row,
    including a dominated or non-monotonic outcome, is removed.
    """

    required = {
        "model_id", "risk_scenario", "annualised_return", "annualised_volatility",
        "sharpe", "maximum_drawdown", "total_turnover",
        "cumulative_transaction_cost", "start_date", "end_date", "sessions",
        "calendar_sha256", "risk_control_value",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing realised risk-return fields: {sorted(missing)}")
    if metrics.is_empty():
        raise ValueError("Realised risk-return metrics cannot be empty")
    if metrics.select(
        pl.any_horizontal(
            pl.col("annualised_return").is_null() | ~pl.col("annualised_return").is_finite(),
            pl.col("annualised_volatility").is_null() | ~pl.col("annualised_volatility").is_finite(),
        ).any()
    ).item():
        raise ValueError("Realised return and volatility must be finite")

    expected_scenarios = set(RISK_SCENARIOS)
    rows: list[dict] = []
    shared_window = None
    for model in metrics["model_id"].unique().sort().to_list():
        group = metrics.filter(pl.col("model_id") == model)
        scenarios = group["risk_scenario"].to_list()
        if set(scenarios) != expected_scenarios or len(scenarios) != len(expected_scenarios):
            raise ValueError(f"Model {model} must contain each risk scenario exactly once")
        windows = group.select(
            "start_date", "end_date", "sessions", "calendar_sha256"
        ).unique()
        if windows.height != 1:
            raise ValueError(f"Model {model} scenarios do not share one evaluation window")
        window = tuple(windows.row(0))
        if shared_window is None:
            shared_window = window
        elif window != shared_window:
            raise ValueError("All models must use an identical evaluation window")
        ordered = group.with_columns(
            pl.col("risk_scenario").replace_strict({
                "conservative": 0, "balanced": 1, "aggressive": 2,
            }).alias("risk_order")
        ).sort("risk_order")
        control = ordered["risk_control_value"].to_list()
        monotonic = (
            all(value is not None and np.isfinite(value) for value in control)
            and all(right > left for left, right in zip(control, control[1:]))
        )
        group_rows = ordered.to_dicts()
        for row in group_rows:
            dominated = any(
                other["annualised_return"] >= row["annualised_return"]
                and other["annualised_volatility"] <= row["annualised_volatility"]
                and (
                    other["annualised_return"] > row["annualised_return"]
                    or other["annualised_volatility"] < row["annualised_volatility"]
                )
                for other in metrics.to_dicts()
                if other["model_id"] != row["model_id"]
                or other["risk_scenario"] != row["risk_scenario"]
            )
            rows.append({**row, "risk_control_monotonic": monotonic, "dominated": dominated})
    return pl.DataFrame(rows).sort(["model_id", "risk_order"])
