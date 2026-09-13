"""Elastic Net with train-fitted imputation and scaling."""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

GRID = [{"alpha": alpha, "l1_ratio": ratio}
        for alpha in (0.0001, 0.001) for ratio in (0.1, 0.5)]


def build_elastic_net(params: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(**params, max_iter=10000, tol=1e-5, random_state=seed)),
    ])
