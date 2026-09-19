"""Gymnasium environment for weekly selection among frozen strategy sleeves."""

from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from trading_pipeline.portfolio import solve_rebalance
from .dataset import PilotDataset
from .protocol import ACTION_IDS


class StrategySelectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset: PilotDataset, cost_bps: float = 10.0):
        super().__init__()
        if cost_bps < 0 or not np.isfinite(cost_bps):
            raise ValueError("cost_bps must be finite and non-negative")
        self.dataset = dataset
        self.cost_bps = float(cost_bps)
        self.action_space = spaces.Discrete(len(ACTION_IDS))
        size = dataset.base_observations.shape[1] + len(ACTION_IDS) + 3
        self.observation_space = spaces.Box(-1e6, 1e6, shape=(size,), dtype=np.float32)
        self._calendar_index = {day: index for index, day in enumerate(dataset.calendar)}
        self._step = 0
        self._holdings: dict[str, float] = {}
        self._last_prices: dict[str, float] = {}
        self._cash = 1.0
        self._equity = 1.0
        self._action = 0
        self._turnover = 0.0

    @property
    def observation_fields(self) -> tuple[str, ...]:
        return self.dataset.observation_fields + tuple(
            f"current_action_{name}" for name in ACTION_IDS.values()
        ) + ("current_cash_weight", "current_gross_exposure", "previous_turnover")

    def _observation(self) -> np.ndarray:
        action = np.zeros(len(ACTION_IDS), dtype=np.float32)
        action[self._action] = 1.0
        equity = self._cash + sum(self._holdings.values())
        state = np.asarray([
            self._cash / equity,
            sum(self._holdings.values()) / equity,
            self._turnover,
        ], dtype=np.float32)
        value = np.concatenate([self.dataset.base_observations[self._step], action, state])
        if not np.isfinite(value).all() or not self.observation_space.contains(value):
            raise ValueError("Environment produced a non-finite observation or exceeded frozen bounds")
        return value

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._holdings = {}
        self._last_prices = {}
        self._cash = 1.0
        self._equity = 1.0
        self._action = 0
        self._turnover = 0.0
        return self._observation(), {"split": self.dataset.split, "signal_date": self.dataset.signal_dates[0]}

    def _mark_through(self, end_date):
        if not self._holdings:
            return
        for security, value in list(self._holdings.items()):
            price = self.dataset.prices.get((end_date, security))
            if price is None:
                raise ValueError(f"Missing held-security valuation: {security} {end_date}")
            self._holdings[security] = value * price / self._last_prices[security]
            self._last_prices[security] = price

    def _days_after(self, start, end):
        left, right = self._calendar_index[start], self._calendar_index[end]
        if right <= left:
            raise ValueError("Transition dates are not strictly chronological")
        return self.dataset.calendar[left + 1:right + 1]

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        signal = self.dataset.signal_dates[self._step]
        execution = self.dataset.execution_dates[self._step]
        end = self.dataset.end_dates[self._step]
        if self._calendar_index[execution] != self._calendar_index[signal] + 1:
            raise ValueError("Frozen transition violates T+1 execution")
        start_equity = self._cash + sum(self._holdings.values())
        for day in self._days_after(signal, execution):
            self._mark_through(day)
        before = self._cash + sum(self._holdings.values())
        target = self.dataset.targets[(self._step, action)]
        self._holdings, self._cash, self._turnover, cost, dollars = solve_rebalance(
            before, self._holdings, target, self.cost_bps
        )
        self._last_prices = {}
        for security in self._holdings:
            price = self.dataset.prices.get((execution, security))
            if price is None:
                raise ValueError(f"Selected security has no T+1 execution bar: {security} {execution}")
            self._last_prices[security] = price
        for day in self._days_after(execution, end):
            self._mark_through(day)
        self._equity = self._cash + sum(self._holdings.values())
        if self._equity <= 0 or not np.isfinite(self._equity):
            raise ValueError("Environment equity became invalid")
        reward = math.log(self._equity / start_equity)
        self._action = action
        info = {
            "split": self.dataset.split,
            "signal_date": signal,
            "execution_date": execution,
            "end_date": end,
            "action_id": ACTION_IDS[action],
            "equity": self._equity,
            "turnover": self._turnover,
            "cost": cost,
            "traded_value": sum(abs(value) for value in dollars.values()),
            "cash_weight": self._cash / self._equity,
            "gross_exposure": sum(self._holdings.values()) / self._equity,
        }
        terminated = self._step == len(self.dataset.signal_dates) - 1
        truncated = False
        if not terminated:
            self._step += 1
        observation = self._observation()
        return observation, float(reward), terminated, truncated, info
