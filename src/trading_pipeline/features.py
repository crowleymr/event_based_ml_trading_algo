"""Backward-looking features and explicit label end dates."""
import polars as pl
from trading_pipeline.data import FACTS

F0 = ["return_5d", "return_10d", "return_20d", "vol_5d", "vol_20d", "avg_dollar_volume_20d",
      "volume_ratio_20d", "ma_distance_20d", "price_to_52w_high"]
F1 = F0 + ["latest_eps", "latest_net_income"]


def market_features(bars):
    frame = bars.sort(["security_id", "session_date"])
    frame = frame.with_columns((pl.col("adjusted_close") / pl.col("adjusted_close").shift(1) - 1)
                               .over("security_id").alias("daily_return"))
    frame = frame.with_columns(
        *[(pl.col("adjusted_close") / pl.col("adjusted_close").shift(n) - 1).over("security_id").alias(f"return_{n}d") for n in (5, 10, 20)],
        *[pl.col("daily_return").rolling_std(n).over("security_id").alias(f"vol_{n}d") for n in (5, 20)],
        (pl.col("close") * pl.col("volume")).rolling_mean(20).over("security_id").alias("avg_dollar_volume_20d"),
        (pl.col("volume") / pl.col("volume").rolling_mean(20)).over("security_id").alias("volume_ratio_20d"),
        (pl.col("adjusted_close") / pl.col("adjusted_close").rolling_mean(20) - 1).over("security_id").alias("ma_distance_20d"),
        (pl.col("adjusted_close") / pl.col("adjusted_close").rolling_max(252) - 1).over("security_id").alias("price_to_52w_high"),
    )
    return frame.with_columns(*[pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in F0])


def join_fundamentals(frame, facts):
    """Latest public filing, latest period end, shortest duration; no growth inference.

    Keep original filings/restatements separate. Deterministic same-day ties select
    accession then value; no later filing can affect an earlier join.
    """
    frame = frame.sort(["security_id", "session_date"])
    for name, output in [("EarningsPerShareBasic", "latest_eps"), ("NetIncomeLoss", "latest_net_income")]:
        available = facts.filter((pl.col("fact_name") == name) & (pl.col("unit") == FACTS[name]) &
                                 (pl.col("fiscal_period_end") <= pl.col("filed_date")))
        available = available.with_columns((pl.col("fiscal_period_end") - pl.col("fiscal_period_start")).dt.total_days().alias("duration"))
        available = available.sort(["security_id", "filed_date", "fiscal_period_end", "duration", "accession_number", "fact_value"],
                                    descending=[False, False, False, True, False, False], nulls_last=False)
        available = available.unique(["security_id", "filed_date"], keep="last", maintain_order=True)
        available = available.select("security_id", pl.col("filed_date").alias(f"{output}_filed_date"),
                                     pl.col("fact_value").alias(output))
        frame = frame.join_asof(available, left_on="session_date", right_on=f"{output}_filed_date",
                                by="security_id", strategy="backward", allow_exact_matches=False,
                                check_sortedness=False)
        if frame.filter(pl.col(f"{output}_filed_date") >= pl.col("session_date")).height:
            raise ValueError("PIT violation")
    return frame


def add_target(frame):
    return frame.sort(["security_id", "session_date"]).with_columns(
        (pl.col("adjusted_close").shift(-5) / pl.col("adjusted_close") - 1).over("security_id").alias("forward_return_5d"),
        pl.col("session_date").shift(-5).over("security_id").alias("label_end_date"))


def build(bars, facts):
    return add_target(join_fundamentals(market_features(bars), facts))


def temporal_split(frame):
    # Calendar shared across all securities, after the mandatory 20-session warmup.
    eligible = frame.filter(pl.col("return_20d").is_not_null())
    dates = eligible["session_date"].unique().sort().to_list()
    if len(dates) < 100:
        raise ValueError("Insufficient dates for 60/20/20 split and embargo")
    a, b = int(len(dates) * .6), int(len(dates) * .8)
    valid_start, test_start = dates[a], dates[b]
    split = eligible.with_columns(
        pl.when((pl.col("session_date") < valid_start) & (pl.col("label_end_date") < valid_start)).then(pl.lit("train"))
        .when((pl.col("session_date") >= dates[a + 5]) & (pl.col("session_date") < test_start) &
              (pl.col("label_end_date") < test_start)).then(pl.lit("validation"))
        .when(pl.col("session_date") >= dates[b + 5]).then(pl.lit("test"))
        .otherwise(pl.lit("excluded")).alias("split"))
    # Test tail remains for portfolio valuation; labels may be null and are excluded only from ML metrics.
    manifest = {"policy": "chronological 60/20/20; purge label_end >= next boundary; five-session post-boundary embargo",
                "raw_validation_boundary": str(valid_start), "raw_test_boundary": str(test_start),
                "horizon": 5, "embargo_sessions": 5, "splits": {}}
    for name in ("train", "validation", "test", "excluded"):
        part = split.filter(pl.col("split") == name)
        manifest["splits"][name] = {"rows": part.height, "start": str(part["session_date"].min()), "end": str(part["session_date"].max())}
    return split, manifest
