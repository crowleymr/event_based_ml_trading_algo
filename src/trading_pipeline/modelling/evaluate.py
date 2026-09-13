"""Regression and cross-sectional ranking evaluation."""

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def metrics(predictions: pl.DataFrame):
    frame = predictions.filter(pl.col("actual_forward_return_5d").is_not_null())
    daily = frame.group_by("session_date").agg(
        pl.corr("actual_forward_return_5d", "predicted_return_5d", method="spearman").alias("ic")
    ).sort("session_date")
    values = daily["ic"].to_numpy()
    finite = values[np.isfinite(values)]
    return {
        "mae": float(mean_absolute_error(frame["actual_forward_return_5d"], frame["predicted_return_5d"])),
        "rmse": float(root_mean_squared_error(frame["actual_forward_return_5d"], frame["predicted_return_5d"])),
        "mean_ic": float(finite.mean()) if len(finite) else None, "ic_dates": len(finite),
    }, daily
