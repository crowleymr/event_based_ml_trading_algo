"""Cross-sectional top-K selection shared by both portfolio engines."""

import polars as pl
from .equal_weight import equal_weights
from .inverse_vol import inverse_vol_weights


def select_weights(predictions: pl.DataFrame, top_k: int = 10, inverse: bool = False) -> dict[str, float]:
    eligible = predictions.filter(
        pl.col("predicted_return_5d").is_finite() &
        pl.col("vol_20d").is_finite() & (pl.col("vol_20d") > 0)
    )
    top = eligible.sort(
        ["predicted_return_5d", "security_id"], descending=[True, False]
    ).head(top_k)
    security_ids = top["security_id"].to_list()
    if inverse:
        return inverse_vol_weights(security_ids, top["vol_20d"].to_list())
    return equal_weights(security_ids)
