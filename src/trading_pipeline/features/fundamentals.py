"""Strict point-in-time SEC fundamental features."""

import polars as pl
from trading_pipeline.data.schemas import FACTS


def join_fundamentals(frame: pl.DataFrame, facts: pl.DataFrame) -> pl.DataFrame:
    """Join only facts filed before the market session; exact filing dates are excluded."""
    frame = frame.sort(["security_id", "session_date"])
    for name, output in (("EarningsPerShareBasic", "latest_eps"), ("NetIncomeLoss", "latest_net_income")):
        available = facts.filter(
            (pl.col("fact_name") == name) & (pl.col("unit") == FACTS[name]) &
            (pl.col("fiscal_period_end") <= pl.col("filed_date"))
        )
        available = available.with_columns(
            (pl.col("fiscal_period_end") - pl.col("fiscal_period_start")).dt.total_days().alias("duration")
        ).sort(
            ["security_id", "filed_date", "fiscal_period_end", "duration", "accession_number", "fact_value"],
            descending=[False, False, False, True, False, False], nulls_last=False,
        ).unique(["security_id", "filed_date"], keep="last", maintain_order=True)
        available = available.select(
            "security_id", pl.col("filed_date").alias(f"{output}_filed_date"),
            pl.col("fact_value").alias(output),
        )
        frame = frame.join_asof(
            available, left_on="session_date", right_on=f"{output}_filed_date",
            by="security_id", strategy="backward", allow_exact_matches=False,
            check_sortedness=False,
        )
        if frame.filter(pl.col(f"{output}_filed_date") >= pl.col("session_date")).height:
            raise ValueError("PIT violation")
    return frame
