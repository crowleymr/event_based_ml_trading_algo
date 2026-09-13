"""CPU-first histogram gradient-boosted tree model."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

GRID = [{"max_leaf_nodes": leaves, "l2_regularization": regularization}
        for leaves in (7, 15) for regularization in (1.0, 10.0)]


def build_gbt(params: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", HistGradientBoostingRegressor(
            **params, max_iter=100, learning_rate=0.05,
            early_stopping=False, random_state=seed,
        )),
    ])
