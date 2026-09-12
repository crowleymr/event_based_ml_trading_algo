import json
from pathlib import Path
import polars as pl
from trading_pipeline.config import load_config
from trading_pipeline.run import run
from trading_pipeline.data import synthetic
from trading_pipeline.features import build, temporal_split
from trading_pipeline.models import choose_models
from trading_pipeline.audit import audit


def test_selection_ignores_final_test(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    _, bars, facts = synthetic(cfg, tmp_path / "raw")
    frame, _ = temporal_split(build(bars, facts))
    changed = frame.with_columns(pl.when(pl.col("split") == "test").then(1e8)
                                 .otherwise(pl.col("forward_return_5d")).alias("forward_return_5d"))
    _, first, predictions = choose_models(frame, 42)
    _, second, predictions2 = choose_models(changed, 42)
    assert first == second
    assert predictions.equals(predictions2)


def test_smoke_reproducibility_and_contract(tmp_path):
    cfg = load_config("configs/smoke.yaml") | {"data_dir": str(tmp_path / "data"), "runs_dir": str(tmp_path / "runs")}
    first, second = run(cfg), run(cfg)
    assert audit(first)["passed"]
    for name in ("config.yaml", "metadata.json", "dataset_manifest.json", "split_manifest.json", "metrics.json",
                 "predictions.parquet", "positions.parquet", "trades.parquet", "equity_curve.parquet",
                 "feature_importance.parquet", "selection.json", "summary.md", "plots/equity_curve.png",
                 "plots/drawdown.png", "plots/model_comparison.png"):
        assert (first / name).is_file()
    assert json.loads((first / "metrics.json").read_text()) == json.loads((second / "metrics.json").read_text())
    a, b = pl.read_parquet(first / "predictions.parquet"), pl.read_parquet(second / "predictions.parquet")
    assert a.equals(b)
    assert set(a["model_id"]) == {f"E{i}" for i in range(6)}
    winner = json.loads((first / "selection.json").read_text())["e5_source"]
    cols = ["session_date", "security_id", "split", "predicted_return_5d"]
    assert a.filter(pl.col("model_id") == winner).select(cols).equals(a.filter(pl.col("model_id") == "E5").select(cols))
    trades = pl.read_parquet(first / "trades.parquet")
    assert trades.filter(pl.col("execution_date") <= pl.col("signal_date")).is_empty()
