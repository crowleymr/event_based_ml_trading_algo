"""Single-command ingestion, selection, frozen test evaluation and artefacts."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import subprocess
import uuid
import joblib
import polars as pl
import yaml
from trading_pipeline.config import load_config
from trading_pipeline.data import ingest, digest, write_parquet, yahoo_bars
from trading_pipeline.features import build, temporal_split, F0, F1
from trading_pipeline.models import choose_models, predict_test, metrics, MATRIX
from trading_pipeline.portfolio import backtest, financial_metrics


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def plots(curves, comparison, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root.mkdir(exist_ok=True)
    for metric, filename in [("equity", "equity_curve.png"), ("drawdown", "drawdown.png")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for group in curves.filter(pl.col("split") == "test").partition_by("experiment"):
            values = group["equity"].to_numpy()
            if metric == "drawdown":
                import numpy as np
                values = values / np.maximum.accumulate(np.r_[1., values])[1:] - 1
            ax.plot(group["session_date"].to_list(), values, label=group["experiment"][0])
        ax.set(title=f"Held-out test {metric}", ylabel=metric)
        ax.legend(ncol=3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(root / filename, dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    test = comparison.filter(pl.col("split") == "test")
    ax.bar(test["experiment"].to_list(), test["total_return"].to_list())
    ax.set(title="Held-out test return after costs (not a selection criterion)", ylabel="Total return")
    fig.tight_layout()
    fig.savefig(root / "model_comparison.png", dpi=140)
    plt.close(fig)


def run(cfg):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    root = Path(cfg["runs_dir"]) / run_id
    root.mkdir(parents=True, exist_ok=False)
    clean_cfg = {k: v for k, v in cfg.items() if k != "sec_user_agent"}
    (root / "config.yaml").write_text(yaml.safe_dump(clean_cfg))
    handler = logging.FileHandler(root / "pipeline.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    metadata = {"run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(), "mode": cfg["mode"],
                "seed": cfg["seed"], "label_horizon": 5, "cost_bps_one_way": cfg["cost_bps"],
                "execution": "first session of ISO week signal after close T, fill T+1 close",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
                "source_sha256": {str(p): digest(p) for p in sorted(Path("src/trading_pipeline").glob("*.py"))},
                "dependencies": {p: importlib.metadata.version(p) for p in
                                 ("polars", "duckdb", "pyarrow", "scikit-learn", "yfinance", "numpy")}}
    json_write(root / "metadata.json", metadata | {"status": "running"})
    try:
        logging.info("Run %s (%s)", run_id, cfg["mode"])
        master, bars, facts = ingest(cfg)
        benchmark = None
        benchmark_status = "not applicable to synthetic verification"
        if cfg["mode"] == "live":
            try:
                benchmark = yahoo_bars({"security_id": "BENCHMARK-SPY", "ticker": "SPY"}, cfg, Path(cfg["data_dir"]))
                write_parquet(benchmark, root / "datasets" / "benchmark.parquet")
                benchmark_status = "SPY Yahoo adjusted buy-and-hold, T+1 close and same costs"
            except Exception as exc:
                benchmark_status = f"Optional B0 unavailable: {exc}"
                logging.warning(benchmark_status)
        logging.info("Building features and temporal split")
        frame, split_manifest = temporal_split(build(bars, facts))
        feature_path = Path(cfg["data_dir"]) / "features" / run_id / "features.parquet"
        write_parquet(frame, feature_path)
        # Per-run immutable curated snapshot: later ingestion cannot invalidate an experiment.
        for name, data in [("security_master", master), ("market_bars", bars), ("fundamental_facts", facts)]:
            write_parquet(data, root / "datasets" / f"{name}.parquet")
        json_write(root / "split_manifest.json", split_manifest)
        raw_root = Path(cfg["data_dir"]) / "raw"
        manifest = {"benchmark": benchmark_status, "security_count": master.height, "mapping_coverage": 1., "market_rows": bars.height,
                    "market_start": str(bars["session_date"].min()), "market_end": str(bars["session_date"].max()),
                    "fact_rows": facts.height, "feature_path": str(feature_path.resolve()), "feature_sha256": digest(feature_path),
                    "market_coverage": bars.group_by("ticker").agg(pl.len().alias("sessions"),
                         pl.col("session_date").min().alias("start"), pl.col("session_date").max().alias("end")).sort("ticker").to_dicts(),
                    "sec_coverage": {name: facts.filter(pl.col("fact_name") == name)["security_id"].n_unique() / master.height
                                     for name in ("EarningsPerShareBasic", "NetIncomeLoss")},
                    "feature_null_fraction": {c: frame[c].null_count() / frame.height for c in F1},
                    "raw_files": [{"path": str(p.resolve()), "sha256": digest(p)} for p in sorted(raw_root.rglob("*"))
                                  if p.is_file() and "library_cache" not in p.parts],
                    "curated_snapshot": [{"path": str(p.resolve()), "sha256": digest(p)} for p in sorted((root / "datasets").glob("*.parquet"))]}
        json_write(root / "dataset_manifest.json", manifest)
        logging.info("Training four model/feature combinations; validation selection only")
        models, selection, validation_predictions = choose_models(frame, cfg["seed"])
        # This persisted lock precedes every final-test prediction and metric.
        json_write(root / "selection.json", selection)
        (root / "models").mkdir()
        for e, model in models.items():
            joblib.dump(model, root / "models" / f"{e}.joblib")
        logging.info("Selection frozen; E5 source %s. Evaluating final test once.", selection["e5_source"])
        predictions = pl.concat([validation_predictions, predict_test(models, frame)])
        momentum = frame.filter(pl.col("split").is_in(["validation", "test"])).select(
            "session_date", "security_id", "ticker", "vol_20d", "split",
            pl.col("forward_return_5d").alias("actual_forward_return_5d"),
            pl.col("return_20d").alias("predicted_return_5d"), pl.lit("E0").alias("model_id"),
            pl.lit("momentum").alias("feature_set")).sort(["session_date", "predicted_return_5d", "security_id"], descending=[False, True, False]).with_columns(
                pl.col("predicted_return_5d").rank("ordinal", descending=True).over("session_date").alias("predicted_rank"))
        predictions = pl.concat([predictions, momentum], how="diagonal_relaxed")
        e5 = predictions.filter(pl.col("model_id") == selection["e5_source"]).with_columns(pl.lit("E5").alias("model_id"))
        predictions = pl.concat([predictions, e5])
        write_parquet(predictions, root / "predictions.parquet")
        curves, positions, trades, rows, ic_frames = [], [], [], [], []
        all_metrics = {}
        for split in ("validation", "test"):
            for experiment in ("E0", "E1", "E2", "E3", "E4", "E5"):
                p = predictions.filter((pl.col("split") == split) & (pl.col("model_id") == experiment))
                curve, pos, trade = backtest(p, bars, experiment, split, cfg["top_k"], cfg["cost_bps"], experiment == "E5")
                finance = financial_metrics(curve)
                ml, ic = metrics(p)
                all_metrics[f"{experiment}/{split}"] = {"ml": ml, "financial": finance}
                rows.append({"experiment": experiment, "split": split, **finance, **ml})
                curves.append(curve)
                positions.append(pos)
                trades.append(trade)
                ic_frames.append(ic.with_columns(pl.lit(experiment).alias("experiment"), pl.lit(split).alias("split")))
            if benchmark is not None:
                period = frame.filter(pl.col("split") == split)
                bp = benchmark.filter(pl.col("session_date").is_between(period["session_date"].min(), period["session_date"].max())).with_columns(
                    pl.lit(1.).alias("predicted_return_5d"), pl.lit(1.).alias("vol_20d"))
                bc, bpos, bt = backtest(bp, benchmark, "B0", split, 1, cfg["cost_bps"], buy_hold=True)
                bm = financial_metrics(bc)
                all_metrics[f"B0/{split}"] = {"financial": bm}
                rows.append({"experiment": "B0", "split": split, **bm})
                curves.append(bc)
                positions.append(bpos)
                trades.append(bt)
        curve = pl.concat(curves)
        comparison = pl.DataFrame(rows)
        for name, data in [("equity_curve", curve), ("positions", pl.concat(positions)), ("trades", pl.concat(trades)),
                           ("daily_ic", pl.concat(ic_frames)), ("experiment_comparison", comparison)]:
            write_parquet(data, root / f"{name}.parquet")
        comparison.write_csv(root / "experiment_comparison.csv")
        json_write(root / "metrics.json", all_metrics)
        importance = []
        for experiment in ("E1", "E3"):
            cols = F0 if experiment == "E1" else F1
            importance.extend(dict(model_id=experiment, feature=c, importance=float(v), method="standardized_elastic_net_coefficient")
                              for c, v in zip(cols, models[experiment]["model"].coef_))
        write_parquet(pl.DataFrame(importance), root / "feature_importance.parquet")
        plots(curve, comparison, root / "plots")
        summary = ["# Slice 1 run", f"Mode: **{cfg['mode']}**. Synthetic runs are software verification only.",
                   f"Universe: {master.height} securities. E5 validation-selected source: {selection['e5_source']}.",
                   "Selection frozen in selection.json before final-test prediction. Models fit on train only.",
                   "## Test results (descriptive; do not select configurations from this table)",
                   "| Experiment | Return | Sharpe | Max drawdown |", "|---|---:|---:|---:|"]
        for r in rows:
            if r["split"] == "test":
                summary.append(f"| {r['experiment']} | {r['total_return']:.4f} | {r['sharpe']} | {r['maximum_drawdown']:.4f} |")
        summary += ["## Limitations", "Survivorship-biased fixed universe; retrospective Yahoo adjustments; no delisting model.",
                    "SEC latest values can mix fiscal durations; only USD and USD/share enter features. No growth ratios.",
                    "Company Facts event table covers filings containing the two selected facts, not every SEC filing.",
                    "T+1-close fills; close-to-close target differs from executable return. No slippage model beyond 10 bps.",
                    f"B0: {benchmark_status}. No tree importance extras. Terminal holdings marked, not liquidated.",
                    "Stage-gate success requires human research judgement; no alpha claim is made."]
        (root / "summary.md").write_text("\n\n".join(summary), encoding="utf-8")
        json_write(root / "metadata.json", metadata | {"status": "complete", "split_dates": split_manifest,
                   "experiments": MATRIX | {"E0": ["momentum", "equal_weight"], "E5": [selection["e5_source"], "inverse_vol"]}})
        logging.info("Completed: %s", root)
        return root
    except Exception as exc:
        logging.exception("Run failed; artefacts and raw caches preserved")
        json_write(root / "metadata.json", metadata | {"status": "failed", "error": str(exc)})
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ingest-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.ingest_only:
        ingest(cfg)
    else:
        print(run(cfg))


if __name__ == "__main__":
    main()
