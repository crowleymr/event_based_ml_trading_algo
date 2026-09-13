"""Five-session adjusted-close regression target."""

import polars as pl


def add_target(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(["security_id", "session_date"]).with_columns(
        (pl.col("adjusted_close").shift(-5) / pl.col("adjusted_close") - 1)
        .over("security_id").alias("forward_return_5d"),
        pl.col("session_date").shift(-5).over("security_id").alias("label_end_date"),
    )
