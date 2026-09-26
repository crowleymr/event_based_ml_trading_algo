"""Validation-only model selection and frozen test prediction."""
from datetime import timedelta
import importlib.metadata
import json
import time
import warnings

import numpy as np
import polars as pl
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from trading_pipeline.features import F0, F1
from .elastic_net import GRID as ELASTIC_NET_GRID, build_elastic_net
from .gbt import GRID as GBT_GRID, build_gbt
from .xgboost_model import (
    GRID as XGBOOST_GRID,
    build_info as xgboost_build_info,
    fit_xgboost,
    selected_details as xgboost_selected_details,
)
from .evaluate import metrics
from trading_pipeline.tracking.telemetry import TELEMETRY_SCHEMA_VERSION, utc_now

CORE_MATRIX = {"E1": ("F0", "elastic_net"), "E2": ("F0", "gbt"),
               "E3": ("F1", "elastic_net"), "E4": ("F1", "gbt")}
XGBOOST_MATRIX = {"E6": ("F0", "xgboost"), "E7": ("F1", "xgboost")}
MATRIX = CORE_MATRIX | XGBOOST_MATRIX
GRIDS = {"elastic_net": ELASTIC_NET_GRID, "gbt": GBT_GRID, "xgboost": XGBOOST_GRID}


def active_matrix(xgboost_config=None):
    enabled = bool((xgboost_config or {}).get("enabled", False))
    return MATRIX if enabled else CORE_MATRIX


def estimator(family, params, seed):
    if family == "elastic_net":
        return build_elastic_net(params, seed)
    if family == "gbt":
        return build_gbt(params, seed)
    raise ValueError(f"Unsupported model family: {family}")


def predict(model, frame, columns, experiment, feature_set):
    with threadpool_limits(limits=1):
        predictions = model.predict(frame.select(columns).to_numpy())
    return frame.select("session_date", "security_id", "ticker", "vol_20d", "split",
                        pl.col("forward_return_5d").alias("actual_forward_return_5d")).with_columns(
        pl.Series("predicted_return_5d", np.asarray(predictions, dtype=np.float64)),
        pl.lit(experiment).alias("model_id"),
        pl.lit(feature_set).alias("feature_set")).sort(["session_date", "predicted_return_5d", "security_id"], descending=[False, True, False]).with_columns(
        pl.col("predicted_return_5d").rank("ordinal", descending=True).over("session_date").alias("predicted_rank"))


def choose_models(frame, seed, xgboost_config=None):
    train = frame.filter((pl.col("split") == "train") & pl.col("forward_return_5d").is_not_null())
    validation = frame.filter((pl.col("split") == "validation") & pl.col("forward_return_5d").is_not_null())
    if min(train.height, validation.height) == 0:
        raise ValueError("Empty train or validation split")
    models, selection, outputs = {}, {}, []
    matrix = active_matrix(xgboost_config)
    for experiment, (feature_set, family) in matrix.items():
        cols = F0 if feature_set == "F0" else F1
        candidates = []
        best_key = None
        for index, params in enumerate(GRIDS[family]):
            if family == "xgboost":
                model = fit_xgboost(
                    params, seed, (xgboost_config or {}).get("device", "auto"),
                    xgboost_config or {},
                    train.select(cols).to_numpy(), train["forward_return_5d"].to_numpy(),
                    validation.select(cols).to_numpy(), validation["forward_return_5d"].to_numpy(),
                )
                caught = []
                model._telemetry_convergence_warning = False
            else:
                model = estimator(family, params, seed)
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                    warnings.simplefilter("always")
                    model.fit(train.select(cols).to_numpy(), train["forward_return_5d"].to_numpy())
                model._telemetry_fit_duration_seconds = time.perf_counter() - started
                model._telemetry_warnings = [str(item.message) for item in caught]
                model._telemetry_convergence_warning = any(
                    issubclass(item.category, ConvergenceWarning) for item in caught
                )
            pred = predict(model, validation, cols, experiment, feature_set)
            score, _ = metrics(pred)
            # Stable fallback for undefined/constant IC: validation RMSE, then fixed grid order.
            key = (score["mean_ic"] if score["mean_ic"] is not None else -float("inf"), -score["rmse"], -index)
            candidates.append({"parameters": params, "validation_metrics": score})
            if best_key is None or key > best_key:
                best_key, best_model, best_pred, best_index = key, model, pred, index
        models[experiment] = best_model
        selection[experiment] = {"feature_set": feature_set, "family": family, "selected_index": best_index,
                                 "parameters": GRIDS[family][best_index], "candidates": candidates,
                                 "fit_rows": train.height, "fit_start": str(train["session_date"].min()),
                                 "fit_end": str(train["session_date"].max())}
        if family == "xgboost":
            details = xgboost_selected_details(best_model)
            details.pop("evals_result")
            selection[experiment]["training_details"] = details
        outputs.append(best_pred)
    def selection_key(experiment):
        item = selection[experiment]
        score = item["candidates"][item["selected_index"]]["validation_metrics"]
        return (score["mean_ic"] if score["mean_ic"] is not None else -float("inf"), -score["rmse"])
    # E5 remains locked to the original E1-E4 candidate set. E6/E7 are diagnostic
    # additions and cannot retrospectively change the observed-test protocol.
    winner = max(CORE_MATRIX, key=selection_key)
    return models, {"criterion": "validation mean daily IC; RMSE tie-break; stable grid order",
                    "e5_source": winner,
                    "research_status": (xgboost_config or {}).get("research_status", "frozen_protocol"),
                    "models": selection}, pl.concat(outputs)


def supervised_training_telemetry(run_id, models, selection, frame, seed):
    """Record fitted-estimator facts and supported staged metrics without refitting."""
    train = frame.filter((pl.col("split") == "train") & pl.col("forward_return_5d").is_not_null())
    validation = frame.filter(
        (pl.col("split") == "validation") & pl.col("forward_return_5d").is_not_null()
    )
    traces, summaries = [], []
    completed = utc_now()
    package_versions = json.dumps({
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "numpy": np.__version__, "polars": pl.__version__,
    }, sort_keys=True)
    matrix = {experiment: MATRIX[experiment] for experiment in models}
    for experiment, (feature_set, family) in matrix.items():
        columns = F0 if feature_set == "F0" else F1
        pipeline = models[experiment]
        fitted = pipeline.named_steps["model"]
        duration = float(getattr(pipeline, "_telemetry_fit_duration_seconds", 0.0))
        warning_messages = list(getattr(pipeline, "_telemetry_warnings", []))
        converged_warning = bool(getattr(pipeline, "_telemetry_convergence_warning", False))
        x_train = train.select(columns).to_numpy()
        y_train = train["forward_return_5d"].to_numpy()
        x_validation = validation.select(columns).to_numpy()
        y_validation = validation["forward_return_5d"].to_numpy()
        transformed_train = pipeline[:-1].transform(x_train)
        transformed_validation = pipeline[:-1].transform(x_validation)
        if family == "xgboost":
            iterations = int(fitted.best_iteration) + 1
        else:
            iterations = int(getattr(fitted, "n_iter_", getattr(fitted, "max_iter", 0)))
        learning_rate = float(getattr(fitted, "learning_rate", 0.0)) or None

        metric_rows = []
        if family == "elastic_net":
            train_prediction = fitted.predict(transformed_train)
            validation_prediction = fitted.predict(transformed_validation)
            residual = y_train - train_prediction
            alpha, ratio = float(fitted.alpha), float(fitted.l1_ratio)
            objective = (
                float(np.dot(residual, residual)) / (2 * len(y_train))
                + alpha * ratio * float(np.abs(fitted.coef_).sum())
                + 0.5 * alpha * (1 - ratio) * float(np.square(fitted.coef_).sum())
            )
            metric_rows = [
                ("final/n_iter", float(fitted.n_iter_), "training"),
                ("final/dual_gap", float(fitted.dual_gap_), "training"),
                ("final/objective", objective, "training"),
                ("train/rmse", float(np.sqrt(np.mean(np.square(y_train - train_prediction)))), "training"),
                ("validation/rmse", float(np.sqrt(np.mean(np.square(y_validation - validation_prediction)))), "validation"),
            ]
            stopping = "convergence_warning" if converged_warning else "solver_converged_or_tolerance_reached"
            model_family = "Elastic Net"
        elif family == "gbt":
            for stage, (train_prediction, validation_prediction) in enumerate(zip(
                fitted.staged_predict(transformed_train),
                fitted.staged_predict(transformed_validation),
            ), start=1):
                metric_rows.extend([
                    ("train/rmse", float(np.sqrt(np.mean(np.square(y_train - train_prediction)))), "training", stage),
                    ("validation/rmse", float(np.sqrt(np.mean(np.square(y_validation - validation_prediction)))), "validation", stage),
                ])
            stopping = "fixed_boosting_stage_budget_reached"
            model_family = "Histogram GBT"
        else:
            evaluation = fitted.evals_result()
            train_rmse = evaluation["validation_0"]["rmse"]
            validation_rmse = evaluation["validation_1"]["rmse"]
            for stage, (train_value, validation_value) in enumerate(
                zip(train_rmse, validation_rmse), start=1
            ):
                metric_rows.extend([
                    ("train/rmse", float(train_value), "training", stage),
                    ("validation/rmse", float(validation_value), "validation", stage),
                ])
            stopping = (
                "validation_early_stopping"
                if iterations < int(fitted.n_estimators)
                else "boosting_stage_budget_reached"
            )
            model_family = "XGBoost"

        for item in metric_rows:
            name, value, phase = item[:3]
            stage = item[3] if len(item) == 4 else iterations
            traces.append({
                "schema_version": TELEMETRY_SCHEMA_VERSION, "run_id": run_id,
                "experiment_id": experiment, "policy_id": None,
                "model_family": model_family, "seed": seed,
                "requested_device": getattr(pipeline, "_telemetry_requested_device", "cpu"),
                "actual_device": getattr(pipeline, "_telemetry_actual_device", "cpu"),
                "phase": phase, "step": int(stage), "epoch": None,
                "metric_name": name, "metric_value": value,
                "elapsed_seconds": duration, "learning_rate": learning_rate,
                "timestamp": completed,
            })
        summaries.append({
            "schema_version": TELEMETRY_SCHEMA_VERSION, "run_id": run_id,
            "experiment_id": experiment, "policy_id": None,
            "model_family": model_family, "seed": seed,
            "requested_device": getattr(pipeline, "_telemetry_requested_device", "cpu"),
            "actual_device": getattr(pipeline, "_telemetry_actual_device", "cpu"),
            "fallback_reason": getattr(pipeline, "_telemetry_fallback_reason", None),
            "duration_seconds": duration, "data_rows": train.height,
            "data_columns": len(columns), "iterations": iterations, "epochs": None,
            "stopping_reason": stopping, "peak_gpu_memory_bytes": None,
            "package_versions_json": package_versions,
            "cuda_versions_json": json.dumps(
                xgboost_build_info() if family == "xgboost"
                else {"not_applicable": "sklearn CPU estimator"}, sort_keys=True
            ),
            "determinism_json": json.dumps({
                "seed": seed, "thread_limit": 1, "warning_count": len(warning_messages),
                "warnings": warning_messages, "convergence_warning": converged_warning,
                "selected_parameters": selection["models"][experiment]["parameters"],
                "dual_gap": float(fitted.dual_gap_) if family == "elastic_net" else None,
                "selected_iteration": iterations,
            }, sort_keys=True),
            "started_at": completed - timedelta(seconds=duration), "completed_at": completed,
        })
    return traces, summaries


def predict_test(models, frame):
    test = frame.filter(pl.col("split") == "test")
    return pl.concat([
        predict(models[e], test, F0 if MATRIX[e][0] == "F0" else F1, e, MATRIX[e][0])
        for e in models
    ])
