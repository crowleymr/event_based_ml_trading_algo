from datetime import date, timedelta
import polars as pl
import pytest
from trading_pipeline.config import load_config
from trading_pipeline.data import synthetic, FACT_SCHEMA
from trading_pipeline.features import market_features, join_fundamentals, add_target, temporal_split, F0


def sample(tmp_path):
    return synthetic(load_config("configs/smoke.yaml"), tmp_path)


def test_market_does_not_read_future(tmp_path):
    _, bars, _ = sample(tmp_path)
    cutoff = bars["session_date"].unique().sort()[400]
    full = market_features(bars).filter(pl.col("session_date") <= cutoff)
    truncated = market_features(bars.filter(pl.col("session_date") <= cutoff))
    assert full.select(F0).equals(truncated.select(F0))


def test_target_exact(tmp_path):
    _, bars, _ = sample(tmp_path)
    group = bars.filter(pl.col("security_id") == bars["security_id"][0])
    target = add_target(group)
    assert target["forward_return_5d"][0] == pytest.approx(group["adjusted_close"][5] / group["adjusted_close"][0] - 1)
    assert target["label_end_date"][0] == group["session_date"][5]
    assert target["forward_return_5d"].tail(5).null_count() == 5


def test_pit_strict_and_future_restatement(tmp_path):
    _, bars, facts = sample(tmp_path)
    joined = join_fundamentals(bars, facts)
    for col in ("latest_eps", "latest_net_income"):
        assert joined.filter(pl.col(col + "_filed_date") >= pl.col("session_date")).is_empty()
    cutoff = bars["session_date"].unique().sort()[350]
    original = join_fundamentals(bars.filter(pl.col("session_date") <= cutoff), facts.filter(pl.col("filed_date") <= cutoff))
    assert joined.filter(pl.col("session_date") <= cutoff).equals(original)
    s = bars["security_id"][0]
    filed = date(2015, 2, 2)
    one = facts.head(1).with_columns(pl.lit(s).alias("security_id"), pl.lit(filed).alias("filed_date"),
                                    pl.lit(date(2015, 1, 1)).alias("fiscal_period_end"),
                                    pl.lit("EarningsPerShareBasic").alias("fact_name"), pl.lit("USD/shares").alias("unit"))
    exact = join_fundamentals(bars, one).filter(pl.col("security_id") == s)
    assert exact.filter(pl.col("session_date") == filed)["latest_eps"][0] is None
    assert exact.filter(pl.col("session_date") == filed + timedelta(days=1))["latest_eps"][0] is not None


def test_split_purge_embargo(tmp_path):
    _, bars, _ = sample(tmp_path)
    frame, manifest = temporal_split(add_target(market_features(bars)))
    train = frame.filter(pl.col("split") == "train")
    valid = frame.filter(pl.col("split") == "validation")
    test = frame.filter(pl.col("split") == "test")
    assert train["label_end_date"].max() < date.fromisoformat(manifest["raw_validation_boundary"]) < valid["session_date"].min()
    assert valid["label_end_date"].max() < date.fromisoformat(manifest["raw_test_boundary"]) < test["session_date"].min()
    assert train["session_date"].max() < valid["session_date"].min() < test["session_date"].min()
