"""Validation-only model selection and frozen test prediction."""
import polars as pl
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from trading_pipeline.features import F0, F1
from .elastic_net import GRID as ELASTIC_NET_GRID, build_elastic_net
from .gbt import GRID as GBT_GRID, build_gbt
from .evaluate import metrics

MATRIX = {"E1": ("F0", "elastic_net"), "E2": ("F0", "gbt"),
          "E3": ("F1", "elastic_net"), "E4": ("F1", "gbt")}
GRIDS = {"elastic_net": ELASTIC_NET_GRID, "gbt": GBT_GRID}


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
