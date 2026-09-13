"""Chronological 60/20/20 split with five-session purge and embargo."""

import polars as pl


def temporal_split(frame: pl.DataFrame):
    eligible = frame.filter(pl.col("return_20d").is_not_null())
    dates = eligible["session_date"].unique().sort().to_list()
    if len(dates) < 100:
        raise ValueError("Insufficient dates for 60/20/20 split and embargo")
    train_end, validation_end = int(len(dates) * 0.6), int(len(dates) * 0.8)
    validation_boundary, test_boundary = dates[train_end], dates[validation_end]
    split = eligible.with_columns(
        pl.when((pl.col("session_date") < validation_boundary) &
                (pl.col("label_end_date") < validation_boundary)).then(pl.lit("train"))
        .when((pl.col("session_date") >= dates[train_end + 5]) &
              (pl.col("session_date") < test_boundary) &
              (pl.col("label_end_date") < test_boundary)).then(pl.lit("validation"))
        .when(pl.col("session_date") >= dates[validation_end + 5]).then(pl.lit("test"))
        .otherwise(pl.lit("excluded")).alias("split")
    )
    manifest = {
        "policy": "chronological 60/20/20; purge label_end >= next boundary; five-session post-boundary embargo",
        "raw_validation_boundary": str(validation_boundary), "raw_test_boundary": str(test_boundary),
        "horizon": 5, "embargo_sessions": 5, "splits": {},
    }
    for name in ("train", "validation", "test", "excluded"):
        part = split.filter(pl.col("split") == name)
        manifest["splits"][name] = {
            "rows": part.height, "start": str(part["session_date"].min()),
            "end": str(part["session_date"].max()),
        }
    return split, manifest
