"""Supervised component adapters and their isolated registry fragment."""

from __future__ import annotations

from trading_pipeline.experiments import ComponentRegistry

from .elastic_net import ELASTIC_NET_SPEC, ElasticNetModel
from .hist_gbt import HIST_GBT_SPEC, HistGradientBoostingModel
from .xgboost import XGBOOST_SPEC, XGBoostModel
from .deep_sequence import (
    LSTM_SPEC, TRANSFORMER_SPEC, LSTMModel, CausalTransformerModel,
)


SUPERVISED_REGISTRATIONS = (
    (ELASTIC_NET_SPEC, ElasticNetModel),
    (HIST_GBT_SPEC, HistGradientBoostingModel),
    (XGBOOST_SPEC, XGBoostModel),
    (LSTM_SPEC, LSTMModel),
    (TRANSFORMER_SPEC, CausalTransformerModel),
)


def register_supervised_models(registry: ComponentRegistry) -> None:
    """Add this lane's allowlisted factories to a caller-owned registry."""
    for spec, factory in SUPERVISED_REGISTRATIONS:
        registry.register(spec, factory)


__all__ = [
    "ELASTIC_NET_SPEC",
    "HIST_GBT_SPEC",
    "XGBOOST_SPEC",
    "LSTM_SPEC",
    "TRANSFORMER_SPEC",
    "ElasticNetModel",
    "HistGradientBoostingModel",
    "XGBoostModel",
    "LSTMModel",
    "CausalTransformerModel",
    "SUPERVISED_REGISTRATIONS",
    "register_supervised_models",
]
