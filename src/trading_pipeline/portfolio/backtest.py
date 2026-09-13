"""Weekly T+1 execution, drifted holdings and transaction costs.

Trading at T+1 close is a conservative daily-bar convention. Old holdings earn
the T -> T+1 return; new holdings start earning at T+1 -> T+2. No open-price
assumption, no same-close fill and no daily rebalancing between weekly trades.
"""
import numpy as np
import polars as pl
from .signals import select_weights


def financial_metrics(curve):
    r = curve["daily_return"].to_numpy()
    equity = curve["equity"].to_numpy()
    vol = float(np.std(r, ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.
    drawdown = equity / np.maximum.accumulate(np.r_[1., equity])[1:] - 1
    return {"total_return": float(equity[-1] - 1), "annualised_return": float(equity[-1] ** (252 / len(r)) - 1),
            "annualised_volatility": vol, "sharpe": float(np.mean(r) * 252 / vol) if vol else None,
            "maximum_drawdown": float(drawdown.min()), "average_turnover": float(curve["turnover"].mean()),
            "total_turnover": float(curve["turnover"].sum()),
            "cumulative_transaction_cost": float(curve["cost"].sum()), "sessions": len(r)}


def backtest(predictions, bars, experiment, split, top_k=10, cost_bps=10, inverse=False, buy_hold=False):
    if predictions.is_empty():
        raise ValueError("Empty predictions")
    calendar = bars["session_date"].unique().sort().to_list()
    start, end = predictions["session_date"].min(), predictions["session_date"].max()
    dates = [d for d in calendar if start <= d <= end]
    # First available session in each ISO week: known at T, does not look ahead to week-end.
    signal_dates, seen = [], set()
    for day in dates:
        week = day.isocalendar()[:2]
        if week not in seen:
            signal_dates.append(day)
            seen.add(week)
    if buy_hold:
        signal_dates = signal_dates[:1]
    groups = {g["session_date"][0]: g for g in predictions.partition_by("session_date")}
    executions = {}
    for day in signal_dates:
        index = calendar.index(day)
        if day in groups and index + 1 < len(calendar) and calendar[index + 1] <= end:
            executions[calendar[index + 1]] = (day, select_weights(groups[day], top_k, inverse))
    prices = {(r["session_date"], r["security_id"]): r["adjusted_close"]
              for r in bars.filter(pl.col("session_date").is_between(start, end)).select("session_date", "security_id", "adjusted_close").to_dicts()}
    holdings, last_price, cash, previous_equity = {}, {}, 1., 1.
    curves, positions, trades = [], [], []
    rate = cost_bps / 10000
    for day in dates:
        for security, value in list(holdings.items()):
            if value == 0:
                continue
            price = prices.get((day, security))
            if price is None:
                raise ValueError(f"Missing held-security valuation: {security} {day}; no silent forward fill")
            holdings[security] *= price / last_price[security]
            last_price[security] = price
        before = cash + sum(holdings.values())
        turnover, cost = 0., 0.
        signal_day = None
        target = {}
        if day in executions:
            signal_day, target = executions[day]
            for security in target:
                if (day, security) not in prices:
                    raise ValueError(f"Selected security has no T+1 execution bar: {security} {day}")
            names = sorted(set(holdings) | set(target))
            # Solve post-cost NAV = pre-cost NAV - bps * actual dollars traded.
            after = before
            for _ in range(50):
                cost = rate * sum(abs(target.get(s, 0.) * after - holdings.get(s, 0.)) for s in names)
                updated = before - cost
                if abs(updated - after) < 1e-14:
                    after = updated
                    break
                after = updated
            dollars = {s: target.get(s, 0.) * after - holdings.get(s, 0.) for s in names}
            cost = rate * sum(abs(v) for v in dollars.values())
            turnover = sum(abs(v) for v in dollars.values()) / before
            for s, amount in dollars.items():
                trades.append(dict(experiment=experiment, split=split, signal_date=signal_day, execution_date=day,
                                   security_id=s, trade_value=amount, turnover=abs(amount) / before,
                                   cost=abs(amount) * rate))
            holdings = {s: w * after for s, w in target.items()}
            cash = before - cost - sum(holdings.values())
            last_price = {s: prices[(day, s)] for s in holdings}
        equity = cash + sum(holdings.values())
        for s, value in holdings.items():
            positions.append(dict(experiment=experiment, split=split, session_date=day, security_id=s,
                                  weight=value / equity, position_value=value, signal_date=signal_day,
                                  target_weight=target.get(s) if signal_day else None))
        curves.append(dict(experiment=experiment, split=split, session_date=day, equity=equity,
                           daily_return=equity / previous_equity - 1, turnover=turnover, cost=cost,
                           cash=cash, gross_exposure=sum(holdings.values()) / equity))
        previous_equity = equity
    return pl.DataFrame(curves), pl.DataFrame(positions), pl.DataFrame(trades)
