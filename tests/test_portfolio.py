from datetime import date, timedelta
import numpy as np
import polars as pl
import pytest
from trading_pipeline.portfolio import select_weights, backtest, solve_rebalance


def fixture():
    days = [date(2020, 1, 6) + timedelta(days=i) for i in range(5)]
    pred = pl.DataFrame({"session_date": days * 2, "security_id": ["a"] * 5 + ["b"] * 5,
                         "predicted_return_5d": [2.] * 5 + [1.] * 5, "vol_20d": [.1] * 5 + [.2] * 5})
    bars = pl.DataFrame({"session_date": days * 2, "security_id": ["a"] * 5 + ["b"] * 5,
                         "adjusted_close": [1., 10., 20., 20., 20.] + [1.] * 5})
    return pred, bars


def test_weight_engines_same_selection():
    p, _ = fixture()
    one = p.filter(pl.col("session_date") == p["session_date"][0])
    ew, iv = select_weights(one, 2), select_weights(one, 2, True)
    assert ew.keys() == iv.keys()
    assert sum(iv.values()) == pytest.approx(1)
    assert min(iv.values()) >= 0
    assert iv["a"] == pytest.approx(2 / 3)


def test_t_plus_one_and_cost_accounting():
    p, bars = fixture()
    free, positions, trades = backtest(p, bars, "E1", "test", top_k=1, cost_bps=0)
    paid, pos, trading = backtest(p, bars, "E1", "test", top_k=1, cost_bps=10)
    assert free["equity"][0] == free["equity"][1] == 1.
    assert free["equity"][2] == 2.
    assert paid["equity"][1] == pytest.approx(1 / 1.001)
    assert paid["equity"][-1] < free["equity"][-1]
    assert all(t > s for t, s in zip(trading["execution_date"], trading["signal_date"]))
    assert trading["cost"].sum() == pytest.approx(paid["cost"].sum())
    assert pos.group_by("session_date").agg(pl.col("weight").sum())["weight"].max() <= 1 + 1e-12


def test_holdings_drift_without_daily_rebalance():
    p, bars = fixture()
    curve, pos, _ = backtest(p, bars, "E1", "test", top_k=2, cost_bps=0)
    assert curve["equity"][-1] == pytest.approx(1.5)
    assert pos.filter((pl.col("session_date") == date(2020, 1, 8)) & (pl.col("security_id") == "a"))["weight"][0] == pytest.approx(2/3)


def test_missing_held_bar_fails():
    p, bars = fixture()
    with pytest.raises(ValueError, match="Missing held"):
        backtest(p, bars.filter(~((pl.col("security_id") == "a") & (pl.col("session_date") == date(2020, 1, 8)))), "E1", "test", top_k=1)


def test_buy_hold_trades_once():
    p, bars = fixture()
    _, _, trades = backtest(p, bars, "B0", "test", top_k=1, buy_hold=True)
    assert trades.height == 1


def test_shared_rebalance_solver_is_long_only_and_reconciled():
    holdings, cash, turnover, cost, dollars = solve_rebalance(
        1.0, {}, {"a": 0.6, "b": 0.4}, 10
    )
    assert sum(holdings.values()) + cash == pytest.approx(1 - cost)
    assert turnover == pytest.approx(sum(abs(value) for value in dollars.values()))
    with pytest.raises(ValueError, match="leverage"):
        solve_rebalance(1.0, {}, {"a": 1.01}, 10)
