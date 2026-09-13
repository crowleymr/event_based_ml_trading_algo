"""Assemble the F0 market and F1 market-plus-SEC feature sets."""

import polars as pl
from .market import F0, market_features
from .fundamentals import join_fundamentals
F1 = F0 + ["latest_eps", "latest_net_income"]


def build(bars: pl.DataFrame, facts: pl.DataFrame) -> pl.DataFrame:
    return join_fundamentals(market_features(bars), facts)
