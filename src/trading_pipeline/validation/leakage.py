"""Verify persisted run integrity without fitting, tuning, or selecting models."""
import argparse
import json
from pathlib import Path
import numpy as np
import polars as pl
from trading_pipeline.data.schemas import digest


def audit(root):
    root = Path(root)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    split = json.loads((root / "split_manifest.json").read_text())
    frame = pl.read_parquet(manifest["feature_path"])
    predictions = pl.read_parquet(root / "predictions.parquet")
    positions = pl.read_parquet(root / "positions.parquet")
    trades = pl.read_parquet(root / "trades.parquet")
    curves = pl.read_parquet(root / "equity_curve.parquet")
    bars = pl.read_parquet(root / "datasets/market_bars.parquet")
    checks = {}
    checks["raw_hashes"] = all(
        digest(item.get("cache_path", item.get("path"))) == item["sha256"]
        for item in manifest["raw_files"]
    )
    checks["snapshot_hashes"] = all(digest(item["path"]) == item["sha256"] for item in manifest["curated_snapshot"])
    checks["feature_hash"] = digest(manifest["feature_path"]) == manifest["feature_sha256"]
    checks["strict_filing_availability"] = all(
        frame.filter(
            pl.col(value).is_not_null() &
            (pl.col(filed).is_null() | (pl.col(filed) >= pl.col("session_date")))
        ).is_empty()
        for value, filed in (
            ("latest_eps", "latest_eps_filed_date"),
            ("latest_net_income", "latest_net_income_filed_date"),
        )
    )
    checks["prediction_unique_keys"] = predictions.select("model_id", "split", "session_date", "security_id").unique().height == predictions.height
    checks["prediction_finite"] = predictions.filter(
        pl.col("predicted_return_5d").is_null() | ~pl.col("predicted_return_5d").is_finite()
    ).is_empty()
    checks["long_only"] = positions.filter(
        pl.col("weight").is_null() | ~pl.col("weight").is_finite() | (pl.col("weight") < -1e-12)
    ).is_empty()
    sums = positions.group_by("experiment", "split", "session_date").agg(pl.col("weight").sum())
    checks["weights_at_most_one"] = sums.filter(pl.col("weight") > 1 + 1e-10).is_empty()
    checks["cash_nonnegative"] = curves.filter(pl.col("cash") < -1e-10).is_empty()
    calendar = bars["session_date"].unique().sort().to_list()
    next_session = dict(zip(calendar[:-1], calendar[1:]))
    checks["exact_next_session_execution"] = all(next_session[t["signal_date"]] == t["execution_date"] for t in trades.to_dicts())
    for name, boundary in (("train", "raw_validation_boundary"), ("validation", "raw_test_boundary")):
        part = frame.filter(pl.col("split") == name)
        checks[f"{name}_purge"] = str(part["label_end_date"].max()) < split[boundary]
    cols = ["session_date", "security_id", "split", "predicted_return_5d"]
    checks["e5_identical_predictions"] = predictions.filter(pl.col("model_id") == selection["e5_source"]).select(cols).equals(
        predictions.filter(pl.col("model_id") == "E5").select(cols))
    rebalance = positions.filter(pl.col("signal_date").is_not_null())
    holding_cols = ["split", "session_date", "security_id"]
    checks["e5_same_top_k"] = rebalance.filter(pl.col("experiment") == selection["e5_source"]).select(holding_cols).sort(holding_cols).equals(
        rebalance.filter(pl.col("experiment") == "E5").select(holding_cols).sort(holding_cols))
    checks["trade_costs"] = bool(np.allclose(trades["cost"].to_numpy(), np.abs(trades["trade_value"].to_numpy()) * .001))
    cost_ok, returns_ok = True, True
    for group in curves.partition_by("experiment", "split"):
        e, s = group["experiment"][0], group["split"][0]
        costs = trades.filter((pl.col("experiment") == e) & (pl.col("split") == s))["cost"].sum()
        cost_ok &= bool(np.isclose(costs, group["cost"].sum()))
        returns_ok &= bool(np.allclose(np.cumprod(1 + group["daily_return"].to_numpy()), group["equity"].to_numpy()))
    checks["cost_reconciliation"] = cost_ok
    checks["equity_return_reconciliation"] = returns_ok
    output = {"checks": checks, "passed": all(checks.values()), "check_count": len(checks),
              "note": "Integrity audit only; no performance-driven decisions."}
    (root / "audit.json").write_text(json.dumps(output, indent=2))
    if not output["passed"]:
        raise ValueError(f"Audit failed: {[k for k, v in checks.items() if not v]}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    print(json.dumps(audit(parser.parse_args().run), indent=2))
