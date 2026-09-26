import json

import joblib
import numpy as np
import pytest

from trading_pipeline.experiments import ComponentRegistry, FitContext, UnsupervisedModel
from trading_pipeline.modelling.unsupervised_models import (
    GaussianMixtureRegimeModel,
    NoRegimeModel,
    register_unsupervised_models,
)


def context(seed: int = 41) -> FitContext:
    return FitContext("study", "trial", "fold", seed)


def clustered_values() -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.vstack(
        [rng.normal(-2.0, 0.25, (30, 2)), rng.normal(2.0, 0.25, (30, 2))]
    )


def test_components_implement_disabled_shared_contract_and_serialisable_spaces():
    baseline = NoRegimeModel()
    gmm = GaussianMixtureRegimeModel()
    for component in (baseline, gmm):
        assert isinstance(component, UnsupervisedModel)
        assert component.spec.interface == "UnsupervisedModel"
        assert component.spec.research_enabled is False
        json.dumps(component.search_space())
    assert "fixed_restarts" not in gmm.search_space()["parameters"]
    assert gmm.search_space()["objective"] == "forward_log_predictive_density"


def test_registration_fragment_is_allowlisted_and_research_gated():
    registry = ComponentRegistry()
    register_unsupervised_models(registry)
    assert registry.ids("UnsupervisedModel") == ("gmm_regime_v1", "no_regime_v1")
    assert isinstance(registry.create("gmm_regime_v1"), GaussianMixtureRegimeModel)
    with pytest.raises(ValueError, match="not enabled for research"):
        registry.create("gmm_regime_v1", require_research_enabled=True)


def test_no_regime_is_constant_and_independent_of_training_values():
    first = NoRegimeModel(information_cutoff="2026-01-31").fit(
        np.zeros((3, 2)), context=context()
    )
    second = NoRegimeModel().fit(np.full((3, 2), 1e9), context=context())
    future = np.array([[1.0, 2.0], [-9.0, 3.0]])
    np.testing.assert_array_equal(first.transform(future), np.ones((2, 1)))
    np.testing.assert_array_equal(first.transform(future), second.transform(future))
    assert first.telemetry()["occupancy_fractions"] == [1.0]


def test_gmm_fit_is_deterministic_and_emits_probabilities_and_telemetry():
    train = clustered_values()
    kwargs = dict(n_components=2, fixed_restarts=3, information_cutoff="2026-01-31")
    first = GaussianMixtureRegimeModel(**kwargs).fit(train, context=context())
    second = GaussianMixtureRegimeModel(**kwargs).fit(train, context=context())
    probabilities = first.transform(train[:5])
    np.testing.assert_allclose(probabilities, second.transform(train[:5]))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        first.score_samples(train[:5]), second.score_samples(train[:5])
    )
    telemetry = first.telemetry()
    assert telemetry["converged"] is True
    assert sum(telemetry["occupancy_counts_by_component"]) == len(train)
    assert telemetry["fixed_restarts"] == 3
    assert telemetry["information_cutoff"] == "2026-01-31"
    assert "semantic regime labels" in " ".join(telemetry["causal_limitations"])


def test_gmm_transform_is_forward_only_and_does_not_refit_scaler():
    train = clustered_values()
    model = GaussianMixtureRegimeModel(fixed_restarts=2).fit(train, context=context())
    scaler_mean = model._scaler.mean_.copy()
    prefix = np.array([[-2.0, -2.0], [2.0, 2.0]])
    prefix_probabilities = model.transform(prefix)
    extended = np.vstack([prefix, np.full((20, 2), 1e12)])
    np.testing.assert_allclose(model.transform(extended)[: len(prefix)], prefix_probabilities)
    np.testing.assert_array_equal(model._scaler.mean_, scaler_mean)


def test_gmm_rejects_empty_nonfinite_and_degenerate_training_data():
    model = GaussianMixtureRegimeModel(n_components=2)
    with pytest.raises(ValueError, match="At least 2 observations"):
        model.fit(np.empty((0, 2)), context=context())
    with pytest.raises(ValueError, match="finite"):
        model.fit(np.array([[0.0], [np.nan]]), context=context())
    with pytest.raises(ValueError, match="distinct observations"):
        model.fit(np.ones((5, 2)), context=context())


def test_models_require_fit_validate_width_and_can_be_saved(tmp_path):
    model = GaussianMixtureRegimeModel()
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.transform(np.ones((2, 2)))
    model.fit(clustered_values(), context=context())
    with pytest.raises(ValueError, match="Expected 2 features"):
        model.transform(np.ones((2, 3)))
    destination = tmp_path / "gmm.joblib"
    model.save(destination)
    restored = joblib.load(destination)
    np.testing.assert_allclose(
        restored.transform(clustered_values()[:3]), model.transform(clustered_values()[:3])
    )
