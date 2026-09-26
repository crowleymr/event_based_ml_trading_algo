import json
import importlib.util

import joblib
import numpy as np
import pytest

from trading_pipeline.experiments import ComponentRegistry, FitContext, SupervisedModel
from trading_pipeline.modelling.elastic_net import build_elastic_net
from trading_pipeline.modelling.gbt import build_gbt
from trading_pipeline.modelling.supervised_models import register_supervised_models


CONTEXT = FitContext(
    study_id="engineering-contract-test",
    trial_id="trial-001",
    fold_id="fold-001",
    seed=42,
    fidelity={"fit_fraction": 1.0},
)


def _training_data():
    x = np.array(
        [[1.0, 3.0], [2.0, np.nan], [3.0, 5.0], [4.0, 6.0], [5.0, 8.0]],
        dtype=float,
    )
    y = np.arange(5.0, dtype=float)
    return x, y


@pytest.fixture()
def registry():
    value = ComponentRegistry()
    register_supervised_models(value)
    return value


@pytest.mark.parametrize(
    "component_id,params",
    [
        ("supervised.elastic_net.v1", {"alpha": 0.001, "l1_ratio": 0.5}),
        ("supervised.hist_gbt.v1", {"max_leaf_nodes": 7, "l2_regularization": 1.0}),
    ],
)
def test_registered_adapters_fit_predict_and_preserve_train_only_state(
    registry, component_id, params
):
    x, y = _training_data()
    model = registry.create(component_id, params=params)
    assert isinstance(model, SupervisedModel)
    assert model.spec == registry.spec(component_id)
    assert model.spec.interface == "SupervisedModel"
    assert model.spec.research_enabled is False

    returned = model.fit(x, y, context=CONTEXT)
    assert returned is model
    imputation = model.fitted_pipeline.named_steps["imputer"].statistics_.copy()
    predictions = model.predict(np.array([[1e12, np.nan], [6.0, 9.0]]))

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
    np.testing.assert_array_equal(
        imputation, model.fitted_pipeline.named_steps["imputer"].statistics_
    )
    np.testing.assert_array_equal(imputation, [3.0, 5.5])
    telemetry = model.telemetry()
    assert telemetry["status"] == "fitted"
    assert telemetry["fit_rows"] == 5
    assert telemetry["fit_columns"] == 2
    assert telemetry["seed"] == 42
    assert telemetry["parameters"] == params


def test_registry_metadata_and_search_spaces_are_serialisable(registry):
    assert registry.ids("SupervisedModel") == (
        "supervised.causal_transformer.v1",
        "supervised.elastic_net.v1",
        "supervised.hist_gbt.v1",
        "supervised.lstm.v1",
        "supervised.xgboost.v1",
    )
    for component_id in registry.ids("SupervisedModel"):
        model = registry.create(component_id)
        encoded = json.dumps(
            {"spec": model.spec.__dict__, "search_space": model.search_space()},
            sort_keys=True,
        )
        assert '"strategy":' in encoded
        assert model.spec.capabilities["train_only_preprocessing"] is True
        with pytest.raises(ValueError, match="not enabled for research"):
            registry.create(component_id, require_research_enabled=True)


@pytest.mark.parametrize(
    "component_id,params,legacy_builder",
    [
        (
            "supervised.elastic_net.v1",
            {"alpha": 0.001, "l1_ratio": 0.5},
            build_elastic_net,
        ),
        (
            "supervised.hist_gbt.v1",
            {"max_leaf_nodes": 15, "l2_regularization": 10.0},
            build_gbt,
        ),
    ],
)
def test_adapter_predictions_match_frozen_legacy_builder(
    registry, component_id, params, legacy_builder
):
    x, y = _training_data()
    prediction_x = np.array([[1.5, np.nan], [6.0, 9.0]])
    adapter = registry.create(component_id, params=params).fit(x, y, context=CONTEXT)
    legacy = legacy_builder(params, CONTEXT.seed).fit(x, y)
    np.testing.assert_allclose(adapter.predict(prediction_x), legacy.predict(prediction_x))


def test_unsupported_stopping_data_fails_closed(registry):
    x, y = _training_data()
    model = registry.create("supervised.elastic_net.v1")
    with pytest.raises(ValueError, match="does not support stopping_data"):
        model.fit(x, y, context=CONTEXT, stopping_data=(x[-1:], y[-1:]))


def test_xgboost_requires_explicit_stopping_data(registry):
    x, y = _training_data()
    model = registry.create("supervised.xgboost.v1")
    with pytest.raises(ValueError, match="requires chronological stopping_data"):
        model.fit(x, y, context=CONTEXT)


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="optional xgboost")
def test_xgboost_cpu_fit_uses_train_only_imputation_and_stopping_set(registry):
    x, y = _training_data()
    x_stopping = np.array([[6.0, np.nan], [7.0, 10.0]], dtype=float)
    y_stopping = np.array([5.0, 6.0], dtype=float)
    model = registry.create(
        "supervised.xgboost.v1",
        params={"max_depth": 3, "reg_lambda": 1.0},
        device="cpu",
        training_config={
            "n_estimators": 4,
            "learning_rate": 0.1,
            "early_stopping_rounds": 2,
        },
    ).fit(x, y, context=CONTEXT, stopping_data=(x_stopping, y_stopping))

    np.testing.assert_array_equal(
        model.fitted_pipeline.named_steps["imputer"].statistics_, [3.0, 5.5]
    )
    prediction = model.predict(x_stopping)
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    telemetry = model.telemetry()
    assert telemetry["actual_device"] == "cpu"
    assert telemetry["stopping_rows"] == 2
    assert telemetry["selected_iteration"] >= 1
    assert set(telemetry["evals_result"]) == {"validation_0", "validation_1"}


def test_fitted_adapter_round_trips_through_joblib(registry, tmp_path):
    x, y = _training_data()
    model = registry.create("supervised.elastic_net.v1").fit(x, y, context=CONTEXT)
    expected = model.predict(x)
    destination = tmp_path / "elastic-net.joblib"
    model.save(destination)

    restored = joblib.load(destination)
    np.testing.assert_allclose(restored.predict(x), expected)
    assert restored.spec == model.spec
