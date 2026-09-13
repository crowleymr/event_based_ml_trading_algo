"""Backward-looking daily market features using data through session T."""

import polars as pl

F0 = ["return_5d", "return_10d", "return_20d", "vol_5d", "vol_20d",
      "avg_dollar_volume_20d", "volume_ratio_20d", "ma_distance_20d",
      "price_to_52w_high"]


def market_features(bars: pl.DataFrame) -> pl.DataFrame:
    frame = bars.sort(["security_id", "session_date"])
    frame = frame.with_columns(
        (pl.col("adjusted_close") / pl.col("adjusted_close").shift(1) - 1)
        .over("security_id").alias("daily_return")
    )
    frame = frame.with_columns(
        *[(pl.col("adjusted_close") / pl.col("adjusted_close").shift(window) - 1)
          .over("security_id").alias(f"return_{window}d") for window in (5, 10, 20)],
        *[pl.col("daily_return").rolling_std(window).over("security_id").alias(f"vol_{window}d")
          for window in (5, 20)],
        (pl.col("close") * pl.col("volume")).rolling_mean(20).over("security_id").alias("avg_dollar_volume_20d"),
        (pl.col("volume") / pl.col("volume").rolling_mean(20)).over("security_id").alias("volume_ratio_20d"),
        (pl.col("adjusted_close") / pl.col("adjusted_close").rolling_mean(20) - 1).over("security_id").alias("ma_distance_20d"),
        (pl.col("adjusted_close") / pl.col("adjusted_close").rolling_max(252) - 1).over("security_id").alias("price_to_52w_high"),
    )
    return frame.with_columns(
        *[pl.when(pl.col(column).is_finite()).then(pl.col(column)).otherwise(None).alias(column)
          for column in F0]
    )
