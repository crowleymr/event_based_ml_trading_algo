import numpy as np
import pytest
from trading_pipeline.modelling.training import estimator
from trading_pipeline.modelling.xgboost_model import fit_xgboost, validate_device


def test_preprocessing_training_only():
    x = np.array([[1., 3.], [2., np.nan], [3., 5.], [4., 6.]])
    model = estimator("elastic_net", {"alpha": .001, "l1_ratio": .5}, 42).fit(x, np.arange(4.))
    impute = model["imputer"].statistics_.copy()
    means = model["scaler"].mean_.copy()
    model.predict(np.array([[1e12, np.nan]]))
    np.testing.assert_array_equal(impute, model["imputer"].statistics_)
    np.testing.assert_array_equal(means, model["scaler"].mean_)
    assert means[0] == 2.5


def test_xgboost_cpu_trace_and_train_only_imputation():
    x_train = np.array([[1., 3.], [2., np.nan], [3., 5.], [4., 6.], [5., 8.]])
    y_train = np.arange(5., dtype=float)
    x_validation = np.array([[6., np.nan], [7., 10.]])
    y_validation = np.array([5., 6.])
    model = fit_xgboost(
        {"max_depth": 2, "reg_lambda": 1.0}, 42, "cpu",
        {"n_estimators": 10, "learning_rate": 0.1, "early_stopping_rounds": 2},
        x_train, y_train, x_validation, y_validation,
    )
    assert model._telemetry_actual_device == "cpu"
    assert set(model.named_steps["model"].evals_result()) == {"validation_0", "validation_1"}
    np.testing.assert_array_equal(model.named_steps["imputer"].statistics_, [3., 5.5])


def test_xgboost_device_validation():
    with pytest.raises(ValueError, match="cpu, cuda or auto"):
        validate_device("quantum")
