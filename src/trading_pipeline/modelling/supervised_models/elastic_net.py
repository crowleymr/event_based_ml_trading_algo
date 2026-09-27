"""Registered adapter for the frozen Slice 1 Elastic Net implementation."""

from __future__ import annotations

from typing import Any, Mapping

from trading_pipeline.experiments import ComponentSpec
from trading_pipeline.modelling.elastic_net import build_elastic_net

from ._base import LegacySklearnAdapter


ELASTIC_NET_SPEC = ComponentSpec(
    component_id="supervised.elastic_net.v1",
    interface="SupervisedModel",
    implementation=(
        "trading_pipeline.modelling.supervised_models.elastic_net."
        "ElasticNetModel"
    ),
    version=1,
    research_enabled=False,
    capabilities={
        "study_adapter": True,
        "tabular_view": True,
        "train_only_preprocessing": True,
        "stopping_data": False,
        "continuation": False,
        "staged_metrics": False,
        "feature_importance": True,
    },
)

ELASTIC_NET_SEARCH_SPACE: Mapping[str, Any] = {
    "schema_version": 1,
    "strategy": "explicit_grid",
    "parameters": {
        "alpha": {"type": "categorical", "values": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]},
        "l1_ratio": {"type": "categorical", "values": [0.0, 0.1, 0.5, 0.9]},
    },
}


class ElasticNetModel(LegacySklearnAdapter):
    """Thin lifecycle adapter; fitting remains the legacy sklearn pipeline."""

    SPEC = ELASTIC_NET_SPEC
    SEARCH_SPACE = ELASTIC_NET_SEARCH_SPACE

    @classmethod
    def default_params(cls) -> Mapping[str, Any]:
        return {"alpha": 0.0001, "l1_ratio": 0.1}

    def _build(self, *, seed: int):
        return build_elastic_net(dict(self._params), seed)


COMPONENT_REGISTRATIONS = ((ELASTIC_NET_SPEC, ElasticNetModel),)
