"""Fixed, random and trained policy evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np
import polars as pl

from trading_pipeline.portfolio import financial_metrics
from .environment import StrategySelectorEnv
from .protocol import ACTION_IDS


def fixed_action(action: int) -> Callable:
    return lambda observation: action


def random_action(seed: int) -> Callable:
    generator = np.random.default_rng(seed)
    return lambda observation: int(generator.integers(0, len(ACTION_IDS)))


def evaluate(env: StrategySelectorEnv, policy: Callable, policy_id: str, seed: int):
    observation, _ = env.reset(seed=seed)
    rows, equity_rows = [], []
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(policy(observation))
        observation, reward, terminated, truncated, info = env.step(action)
        rows.append({
            "policy_id": policy_id,
            "seed": seed,
            "reward": reward,
            **info,
        })
        equity_rows.append({
            "policy_id": policy_id,
            "seed": seed,
            "session_date": info["end_date"],
            "equity": info["equity"],
            "daily_return": float(np.expm1(reward)),
            "turnover": info["turnover"],
            "cost": info["cost"],
        })
    actions, curve = pl.DataFrame(rows), pl.DataFrame(equity_rows)
    metrics = financial_metrics(curve, periods_per_year=52)
    metrics["periods_per_year"] = 52
    metrics["reward_sum"] = float(actions["reward"].sum())
    metrics["policy_id"] = policy_id
    metrics["seed"] = seed
    return actions, curve, metrics


def model_action(model) -> Callable:
    def choose(observation):
        action, _ = model.predict(observation, deterministic=True)
        return int(action)
    return choose
