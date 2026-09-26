"""Optional CPU/CUDA XGBoost regression support with explicit fallback evidence."""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


GRID = [
    {"max_depth": depth, "reg_lambda": regularization}
    for depth in (3, 5)
    for regularization in (1.0, 10.0)
]


def _xgb():
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - exercised in dependency-light CI
        raise RuntimeError(
            "XGBoost experiments require the optional 'xgboost' dependency"
        ) from exc
    return xgb


def validate_device(value: str) -> str:
    device = str(value).lower()
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("XGBoost device must be cpu, cuda or auto")
    return device


def _probe_cuda(seed: int) -> tuple[bool, str | None]:
    """Execute a tiny CUDA fit; driver inventory alone is not execution evidence."""
    xgb = _xgb()
    x = np.arange(32, dtype=np.float32).reshape(16, 2)
    y = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            xgb.XGBRegressor(
                n_estimators=2, tree_method="hist", device="cuda", n_jobs=1,
                random_state=seed, objective="reg:squarederror", verbosity=0,
            ).fit(x, y, verbose=False)
        return True, None
    except Exception as exc:  # XGBoost uses both errors and warnings for unavailable CUDA
        return False, f"CUDA execution preflight failed: {type(exc).__name__}: {exc}"


def resolve_device(requested: str, seed: int) -> tuple[str, str | None]:
    requested = validate_device(requested)
    if requested == "cpu":
        return "cpu", None
    available, reason = _probe_cuda(seed)
    if available:
        return "cuda", None
    return "cpu", reason


def _regressor(params: dict, seed: int, device: str, config: dict):
    xgb = _xgb()
    return xgb.XGBRegressor(
        **params,
        n_estimators=int(config.get("n_estimators", 200)),
        learning_rate=float(config.get("learning_rate", 0.05)),
        early_stopping_rounds=int(config.get("early_stopping_rounds", 20)),
        tree_method="hist",
        device=device,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def fit_xgboost(
    params: dict,
    seed: int,
    requested_device: str,
    config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Pipeline:
    """Fit preprocessing on train only and early-stop using validation only."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train = imputer.fit_transform(x_train)
    validation = imputer.transform(x_validation)
    actual_device, fallback_reason = resolve_device(requested_device, seed)

    def fit(device: str):
        model = _regressor(params, seed, device, config)
        started = time.perf_counter()
        model.fit(
            train, y_train,
            eval_set=[(train, y_train), (validation, y_validation)],
            verbose=False,
        )
        return model, time.perf_counter() - started

    try:
        model, duration = fit(actual_device)
    except Exception as exc:
        if actual_device != "cuda":
            raise
        actual_device = "cpu"
        fallback_reason = f"CUDA training failed: {type(exc).__name__}: {exc}"
        model, duration = fit("cpu")

    pipeline = Pipeline([("imputer", imputer), ("model", model)])
    pipeline._telemetry_fit_duration_seconds = duration
    pipeline._telemetry_requested_device = validate_device(requested_device)
    pipeline._telemetry_actual_device = actual_device
    pipeline._telemetry_fallback_reason = fallback_reason
    pipeline._telemetry_warnings = []
    return pipeline


def build_info() -> dict:
    xgb = _xgb()
    info = xgb.build_info()
    return {key: value for key, value in info.items() if key in {
        "USE_CUDA", "CUDA_VERSION", "NCCL_VERSION", "DEBUG", "USE_OPENMP"
    }}


def device_benchmark(
    experiment: str,
    params: dict,
    seed: int,
    config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> list[dict]:
    """Diagnostic timing/parity fit; its output is never used for selection."""
    rows, predictions = [], {}
    for requested in ("cpu", "cuda"):
        started = time.perf_counter()
        try:
            fitted = fit_xgboost(
                params, seed, requested, config,
                x_train, y_train, x_validation, y_validation,
            )
            elapsed = time.perf_counter() - started
            actual = fitted._telemetry_actual_device
            prediction = fitted.predict(x_validation)
            predictions[requested] = prediction
            rmse = float(np.sqrt(np.mean(np.square(y_validation - prediction))))
            rows.append({
                "experiment_id": experiment, "requested_device": requested,
                "actual_device": actual, "status": "completed",
                "fallback_reason": fitted._telemetry_fallback_reason,
                "duration_seconds": elapsed, "validation_rmse": rmse,
                "selected_iteration": int(fitted.named_steps["model"].best_iteration) + 1,
                "max_abs_prediction_delta_vs_cpu": None,
                "research_status": "diagnostic_reproduction_not_model_selection",
            })
        except Exception as exc:
            rows.append({
                "experiment_id": experiment, "requested_device": requested,
                "actual_device": None, "status": "unavailable",
                "fallback_reason": f"{type(exc).__name__}: {exc}",
                "duration_seconds": time.perf_counter() - started,
                "validation_rmse": None, "selected_iteration": None,
                "max_abs_prediction_delta_vs_cpu": None,
                "research_status": "diagnostic_reproduction_not_model_selection",
            })
    if "cpu" in predictions and "cuda" in predictions:
        delta = float(np.max(np.abs(predictions["cpu"] - predictions["cuda"])))
        for row in rows:
            row["max_abs_prediction_delta_vs_cpu"] = 0.0 if row["requested_device"] == "cpu" else delta
    return rows


def selected_details(pipeline: Pipeline) -> dict:
    model = pipeline.named_steps["model"]
    return {
        "selected_iteration": int(model.best_iteration) + 1,
        "best_score": float(model.best_score),
        "requested_device": pipeline._telemetry_requested_device,
        "actual_device": pipeline._telemetry_actual_device,
        "fallback_reason": pipeline._telemetry_fallback_reason,
        "build_info": build_info(),
        "evals_result": json.loads(json.dumps(model.evals_result())),
    }
