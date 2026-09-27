"""Registered adapter for the frozen Slice 1 histogram GBT implementation."""

from __future__ import annotations

from typing import Any, Mapping

from trading_pipeline.experiments import ComponentSpec
from trading_pipeline.modelling.gbt import build_gbt

from ._base import LegacySklearnAdapter


HIST_GBT_SPEC = ComponentSpec(
    component_id="supervised.hist_gbt.v1",
    interface="SupervisedModel",
    implementation=(
        "trading_pipeline.modelling.supervised_models.hist_gbt."
        "HistGradientBoostingModel"
    ),
    version=1,
    research_enabled=False,
    capabilities={
        "study_adapter": True,
        "tabular_view": True,
        "train_only_preprocessing": True,
        "stopping_data": False,
        "continuation": False,
        "staged_metrics": True,
        "feature_importance": True,
    },
)

HIST_GBT_SEARCH_SPACE: Mapping[str, Any] = {
    "schema_version": 1,
    "strategy": "explicit_grid",
    "parameters": {
        "max_leaf_nodes": {"type": "categorical", "values": [7, 15, 31, 63, 127, 255]},
        "l2_regularization": {"type": "categorical", "values": [0.0, 1.0, 10.0, 100.0]},
    },
}


class HistGradientBoostingModel(LegacySklearnAdapter):
    """Thin lifecycle adapter; fitting remains the legacy sklearn pipeline."""

    SPEC = HIST_GBT_SPEC
    SEARCH_SPACE = HIST_GBT_SEARCH_SPACE

    @classmethod
    def default_params(cls) -> Mapping[str, Any]:
        return {"max_leaf_nodes": 7, "l2_regularization": 1.0}

    def _build(self, *, seed: int):
        return build_gbt(dict(self._params), seed)


COMPONENT_REGISTRATIONS = ((HIST_GBT_SPEC, HistGradientBoostingModel),)
