"""Small validation-only grids. Final test is scored only after selection is frozen."""
import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from threadpoolctl import threadpool_limits
from trading_pipeline.features import F0, F1

MATRIX = {"E1": ("F0", "elastic_net"), "E2": ("F0", "gbt"),
          "E3": ("F1", "elastic_net"), "E4": ("F1", "gbt")}
GRIDS = {"elastic_net": [{"alpha": a, "l1_ratio": r} for a in (.0001, .001) for r in (.1, .5)],
         "gbt": [{"max_leaf_nodes": leaves, "l2_regularization": l2} for leaves in (7, 15) for l2 in (1., 10.)]}


def estimator(family, params, seed):
    if family == "elastic_net":
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scaler", StandardScaler()),
                         ("model", ElasticNet(**params, max_iter=10000, tol=1e-5, random_state=seed))])
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("model", HistGradientBoostingRegressor(**params, max_iter=100, learning_rate=.05,
                                                             early_stopping=False, random_state=seed))])


def metrics(predictions):
    frame = predictions.filter(pl.col("actual_forward_return_5d").is_not_null())
    daily = frame.group_by("session_date").agg(
        pl.corr("actual_forward_return_5d", "predicted_return_5d", method="spearman").alias("ic")).sort("session_date")
    values = daily["ic"].to_numpy()
    finite = values[np.isfinite(values)]
    return {"mae": float(mean_absolute_error(frame["actual_forward_return_5d"], frame["predicted_return_5d"])),
            "rmse": float(root_mean_squared_error(frame["actual_forward_return_5d"], frame["predicted_return_5d"])),
            "mean_ic": float(finite.mean()) if len(finite) else None, "ic_dates": len(finite)}, daily


def predict(model, frame, columns, experiment, feature_set):
    with threadpool_limits(limits=1):
        predictions = model.predict(frame.select(columns).to_numpy())
    return frame.select("session_date", "security_id", "ticker", "vol_20d", "split",
                        pl.col("forward_return_5d").alias("actual_forward_return_5d")).with_columns(
        pl.Series("predicted_return_5d", predictions), pl.lit(experiment).alias("model_id"),
        pl.lit(feature_set).alias("feature_set")).sort(["session_date", "predicted_return_5d", "security_id"], descending=[False, True, False]).with_columns(
        pl.col("predicted_return_5d").rank("ordinal", descending=True).over("session_date").alias("predicted_rank"))


def choose_models(frame, seed):
    train = frame.filter((pl.col("split") == "train") & pl.col("forward_return_5d").is_not_null())
    validation = frame.filter((pl.col("split") == "validation") & pl.col("forward_return_5d").is_not_null())
    if min(train.height, validation.height) == 0:
        raise ValueError("Empty train or validation split")
    models, selection, outputs = {}, {}, []
    for experiment, (feature_set, family) in MATRIX.items():
        cols = F0 if feature_set == "F0" else F1
        candidates = []
        best_key = None
        for index, params in enumerate(GRIDS[family]):
            model = estimator(family, params, seed)
            with threadpool_limits(limits=1):
                model.fit(train.select(cols).to_numpy(), train["forward_return_5d"].to_numpy())
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
        outputs.append(best_pred)
    def selection_key(experiment):
        item = selection[experiment]
        score = item["candidates"][item["selected_index"]]["validation_metrics"]
        return (score["mean_ic"] if score["mean_ic"] is not None else -float("inf"), -score["rmse"])
    winner = max(MATRIX, key=selection_key)
    return models, {"criterion": "validation mean daily IC; RMSE tie-break; stable grid order", "e5_source": winner,
                    "models": selection}, pl.concat(outputs)


def predict_test(models, frame):
    test = frame.filter(pl.col("split") == "test")
    return pl.concat([predict(models[e], test, F0 if fs == "F0" else F1, e, fs) for e, (fs, _) in MATRIX.items()])
