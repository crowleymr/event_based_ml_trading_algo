import json
from pathlib import Path
import polars as pl
import pytest
from trading_pipeline.config import load_config
from trading_pipeline.run import run
from trading_pipeline.data import synthetic
from trading_pipeline.features import build
from trading_pipeline.modelling.targets import add_target
from trading_pipeline.modelling.splits import temporal_split
from trading_pipeline.modelling.training import choose_models
from trading_pipeline.audit import audit


def test_selection_ignores_final_test(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    _, bars, facts = synthetic(cfg, tmp_path / "raw")
    frame, _ = temporal_split(add_target(build(bars, facts)))
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
                 "training_trace.parquet", "training_summary.parquet",
                 "device_benchmark.parquet",
                 "feature_importance.parquet", "selection.json", "summary.md", "plots/equity_curve.png",
                 "plots/drawdown.png", "plots/model_comparison.png"):
        assert (first / name).is_file()
    assert json.loads((first / "metrics.json").read_text()) == json.loads((second / "metrics.json").read_text())
    metadata = json.loads((first / "metadata.json").read_text())
    assert any(path.endswith("data/schemas.py") for path in metadata["source_sha256"])
    assert any(path.endswith("portfolio/backtest.py") for path in metadata["source_sha256"])
    manifest = json.loads((first / "dataset_manifest.json").read_text())
    assert Path(manifest["feature_path"]).parent == first / "datasets"
    assert all("relative_cache_path" in item for item in manifest["raw_files"])
    metadata = json.loads((first / "metadata.json").read_text())
    assert "src\\trading_pipeline\\data\\schemas.py" in metadata["source_sha256"] or (
        "src/trading_pipeline/data/schemas.py" in metadata["source_sha256"]
    )
    a, b = pl.read_parquet(first / "predictions.parquet"), pl.read_parquet(second / "predictions.parquet")
    assert a.equals(b)
    assert set(a["model_id"]) == {f"E{i}" for i in range(8)}
    trace = pl.read_parquet(first / "training_trace.parquet")
    training_summary = pl.read_parquet(first / "training_summary.parquet")
    assert set(training_summary["experiment_id"]) == {"E1", "E2", "E3", "E4", "E6", "E7"}
    assert set(trace.filter(pl.col("model_family") == "Elastic Net")["metric_name"]) >= {
        "final/n_iter", "final/dual_gap", "final/objective", "train/rmse", "validation/rmse"
    }
    assert trace.filter(pl.col("model_family") == "Histogram GBT")["step"].n_unique() == 100
    assert set(trace.filter(pl.col("model_family") == "XGBoost")["metric_name"]) == {
        "train/rmse", "validation/rmse"
    }
    assert set(training_summary.filter(pl.col("model_family") == "XGBoost")["actual_device"]) <= {
        "cpu", "cuda"
    }
    importance = pl.read_parquet(first / "feature_importance.parquet")
    assert set(importance["model_id"]) == {"E1", "E2", "E3", "E4", "E6", "E7"}
    assert set(importance.filter(pl.col("model_id").is_in(["E2", "E4"]))["method"]) == {
        "validation_permutation_delta_neg_rmse_3_repeats"
    }
    assert set(importance.filter(pl.col("model_id").is_in(["E6", "E7"]))["method"]) == {
        "xgboost_gain", "validation_permutation_delta_neg_rmse_3_repeats"
    }
    device_benchmark = pl.read_parquet(first / "device_benchmark.parquet")
    assert set(device_benchmark["experiment_id"]) == {"E6", "E7"}
    assert set(device_benchmark["requested_device"]) == {"cpu", "cuda"}
    assert set(device_benchmark["research_status"]) == {"diagnostic_reproduction_not_model_selection"}
    winner = json.loads((first / "selection.json").read_text())["e5_source"]
    cols = ["session_date", "security_id", "split", "predicted_return_5d"]
    assert a.filter(pl.col("model_id") == winner).select(cols).equals(a.filter(pl.col("model_id") == "E5").select(cols))
    trades = pl.read_parquet(first / "trades.parquet")
    assert trades.filter(pl.col("execution_date") <= pl.col("signal_date")).is_empty()
    corrupted = a.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None)
        .otherwise(pl.col("predicted_return_5d")).alias("predicted_return_5d")
    )
    corrupted.write_parquet(first / "predictions.parquet")
    with pytest.raises(ValueError, match="prediction_finite"):
        audit(first)
