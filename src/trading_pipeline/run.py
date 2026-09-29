"""Single-command ingestion, selection, frozen test evaluation and artefacts."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import logging
from pathlib import Path
import subprocess
import uuid
import joblib
import polars as pl
import yaml
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
from trading_pipeline.config import load_config
from trading_pipeline.data import ingest, digest, write_parquet, yahoo_bars
from trading_pipeline.features import build, F0, F1
from trading_pipeline.modelling.targets import add_target
from trading_pipeline.modelling.splits import temporal_split
from trading_pipeline.modelling.training import (
    choose_models, predict_test, supervised_training_telemetry, MATRIX,
)
from trading_pipeline.modelling.xgboost_model import device_benchmark as xgboost_device_benchmark
from trading_pipeline.modelling.evaluate import metrics
from trading_pipeline.portfolio import backtest, financial_metrics
from trading_pipeline.environment import detect_environment, model_devices
from trading_pipeline.audit import audit
from trading_pipeline.tracking.artefacts import input_vintage, json_write, plots
from trading_pipeline.tracking.telemetry import write_training_telemetry
from trading_pipeline.experiments.schema import load_study
from trading_pipeline.experiments.authority import verify_study_authority
from trading_pipeline.experiments.study_runner import run_study


def run(cfg):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    root = Path(cfg["runs_dir"]) / run_id
    root.mkdir(parents=True, exist_ok=False)
    clean_cfg = {k: v for k, v in cfg.items() if k != "sec_user_agent"}
    (root / "config.yaml").write_text(yaml.safe_dump(clean_cfg))
    handler = logging.FileHandler(root / "pipeline.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    dependency_names = ["polars", "duckdb", "pyarrow", "scikit-learn", "yfinance", "numpy"]
    if cfg.get("xgboost", {}).get("enabled"):
        dependency_names.append("xgboost")
    metadata = {"run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(), "mode": cfg["mode"],
                "environment": detect_environment(), **model_devices(cfg["seed"]),
                "research_status": cfg.get("research_status", "frozen_protocol"),
                "seed": cfg["seed"], "label_horizon": 5, "cost_bps_one_way": cfg["cost_bps"],
                "execution": "first session of ISO week signal after close T, fill T+1 close",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
                "source_sha256": {p.as_posix(): digest(p) for p in sorted(Path("src/trading_pipeline").rglob("*.py"))},
                "dependencies": {p: importlib.metadata.version(p) for p in dependency_names}}
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
        frame, split_manifest = temporal_split(add_target(build(bars, facts)))
        feature_path = Path(cfg["data_dir"]) / "features" / run_id / "features.parquet"
        write_parquet(frame, feature_path)
        run_feature_path = root / "datasets" / "features.parquet"
        write_parquet(frame, run_feature_path)
        # Per-run immutable curated snapshot: later ingestion cannot invalidate an experiment.
        for name, data in [("security_master", master), ("market_bars", bars), ("fundamental_facts", facts)]:
            write_parquet(data, root / "datasets" / f"{name}.parquet")
        json_write(root / "split_manifest.json", split_manifest)
        manifest = {"benchmark": benchmark_status, "security_count": master.height, "mapping_coverage": 1., "market_rows": bars.height,
                    "market_start": str(bars["session_date"].min()), "market_end": str(bars["session_date"].max()),
                    "fact_rows": facts.height, "feature_path": str(run_feature_path.resolve()),
                    "canonical_feature_path": str(feature_path.resolve()),
                    "feature_sha256": digest(run_feature_path),
                    "market_coverage": bars.group_by("ticker").agg(pl.len().alias("sessions"),
                         pl.col("session_date").min().alias("start"), pl.col("session_date").max().alias("end")).sort("ticker").to_dicts(),
                    "sec_coverage": {name: facts.filter(pl.col("fact_name") == name)["security_id"].n_unique() / master.height
                                     for name in ("EarningsPerShareBasic", "NetIncomeLoss")},
                    "feature_null_fraction": {c: frame[c].null_count() / frame.height for c in F1},
                    "raw_files": input_vintage(cfg, master, benchmark is not None),
                    "curated_snapshot": [{"path": str(p.resolve()), "sha256": digest(p)} for p in sorted((root / "datasets").glob("*.parquet"))]}
        json_write(root / "dataset_manifest.json", manifest)
        logging.info("Training registered model/feature combinations; validation selection only")
        xgboost_config = cfg.get("xgboost", {})
        models, selection, validation_predictions = choose_models(
            frame, cfg["seed"], xgboost_config
        )
        trace_rows, summary_rows = supervised_training_telemetry(
            run_id, models, selection, frame, cfg["seed"]
        )
        write_training_telemetry(root, trace_rows, summary_rows)
        metadata["actual_device_per_model"] = {
            experiment: getattr(model, "_telemetry_actual_device", "cpu")
            for experiment, model in models.items()
        }
        metadata["gpu_fallback_reason"] = {
            experiment: getattr(model, "_telemetry_fallback_reason", None)
            for experiment, model in models.items()
            if getattr(model, "_telemetry_fallback_reason", None)
        } or None
        if xgboost_config.get("enabled") and xgboost_config.get("device_benchmark", False):
            train = frame.filter(
                (pl.col("split") == "train") & pl.col("forward_return_5d").is_not_null()
            )
            validation = frame.filter(
                (pl.col("split") == "validation") & pl.col("forward_return_5d").is_not_null()
            )
            benchmark_rows = []
            for experiment in ("E6", "E7"):
                columns = F0 if MATRIX[experiment][0] == "F0" else F1
                benchmark_rows.extend(xgboost_device_benchmark(
                    experiment, selection["models"][experiment]["parameters"], cfg["seed"],
                    xgboost_config,
                    train.select(columns).to_numpy(), train["forward_return_5d"].to_numpy(),
                    validation.select(columns).to_numpy(), validation["forward_return_5d"].to_numpy(),
                ))
            write_parquet(pl.DataFrame(benchmark_rows), root / "device_benchmark.parquet")
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
        experiment_ids = ["E0", *models.keys(), "E5"]
        for split in ("validation", "test"):
            for experiment in experiment_ids:
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
        validation = frame.filter(
            (pl.col("split") == "validation") & pl.col("forward_return_5d").is_not_null()
        )
        for experiment in ("E2", "E4"):
            cols = F0 if experiment == "E2" else F1
            with threadpool_limits(limits=1):
                result = permutation_importance(
                    models[experiment], validation.select(cols).to_numpy(),
                    validation["forward_return_5d"].to_numpy(),
                    scoring="neg_root_mean_squared_error", n_repeats=3,
                    random_state=cfg["seed"], n_jobs=1,
                )
            importance.extend(
                dict(model_id=experiment, feature=column, importance=float(value),
                     method="validation_permutation_delta_neg_rmse_3_repeats")
                for column, value in zip(cols, result.importances_mean)
            )
        for experiment in ("E6", "E7"):
            if experiment not in models:
                continue
            cols = F0 if experiment == "E6" else F1
            booster_scores = models[experiment].named_steps["model"].get_booster().get_score(
                importance_type="gain"
            )
            importance.extend(
                dict(model_id=experiment, feature=column,
                     importance=float(booster_scores.get(f"f{index}", 0.0)),
                     method="xgboost_gain")
                for index, column in enumerate(cols)
            )
            with threadpool_limits(limits=1):
                result = permutation_importance(
                    models[experiment], validation.select(cols).to_numpy(),
                    validation["forward_return_5d"].to_numpy(),
                    scoring="neg_root_mean_squared_error", n_repeats=3,
                    random_state=cfg["seed"], n_jobs=1,
                )
            importance.extend(
                dict(model_id=experiment, feature=column, importance=float(value),
                     method="validation_permutation_delta_neg_rmse_3_repeats")
                for column, value in zip(cols, result.importances_mean)
            )
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
                    f"B0: {benchmark_status}. Tree importance is predictive, not causal. Terminal holdings marked, not liquidated.",
                    f"Research status: {cfg.get('research_status', 'frozen_protocol')}.",
                    "Stage-gate success requires human research judgement; no alpha claim is made."]
        (root / "summary.md").write_text("\n\n".join(summary).replace("|\n\n|", "|\n|"), encoding="utf-8")
        audit(root)
        json_write(root / "metadata.json", metadata | {"status": "complete", "split_dates": split_manifest,
                   "experiments": {e: MATRIX[e] for e in models} | {
                       "E0": ["momentum", "equal_weight"],
                       "E5": [selection["e5_source"], "inverse_vol"],
                   }})
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Locked Slice 1 pipeline config")
    source.add_argument("--study", help="Expanded optimisation study protocol")
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--resume-from", type=Path,
                        help="Failed expanded run to continue in a new lineage-linked run")
    parser.add_argument("--allow-code-drift", action="store_true",
                        help="Explicitly permit verified cell reuse after an engineering code change")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.study:
        if args.ingest_only:
            parser.error("--ingest-only applies only to --config")
        study = load_study(args.study, allow_engineering_draft=False)
        repository_root = Path(__file__).resolve().parents[2]
        authority = verify_study_authority(study, repository_root=repository_root)
        print(run_study(study, authority, repository_root=repository_root,
                        resume_from=args.resume_from,
                        allow_code_drift=args.allow_code_drift))
        return
    if args.resume_from:
        parser.error("--resume-from applies only to --study")
    if args.allow_code_drift:
        parser.error("--allow-code-drift applies only to --study resume")
    cfg = load_config(args.config)
    if args.ingest_only:
        ingest(cfg)
    else:
        print(run(cfg))


if __name__ == "__main__":
    main()
