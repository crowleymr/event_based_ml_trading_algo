from importlib import import_module
from pathlib import Path


PLANNED_MODULES = [
    "trading_pipeline.data.universe",
    "trading_pipeline.data.yahoo_client",
    "trading_pipeline.data.sec_client",
    "trading_pipeline.data.security_master",
    "trading_pipeline.data.schemas",
    "trading_pipeline.features.market",
    "trading_pipeline.features.fundamentals",
    "trading_pipeline.features.build",
    "trading_pipeline.modelling.targets",
    "trading_pipeline.modelling.splits",
    "trading_pipeline.modelling.elastic_net",
    "trading_pipeline.modelling.gbt",
    "trading_pipeline.modelling.evaluate",
    "trading_pipeline.portfolio.signals",
    "trading_pipeline.portfolio.equal_weight",
    "trading_pipeline.portfolio.inverse_vol",
    "trading_pipeline.portfolio.backtest",
    "trading_pipeline.tracking.artefacts",
    "trading_pipeline.validation.leakage",
    "trading_pipeline.run",
]


def test_planned_modules_are_importable():
    for module in PLANNED_MODULES:
        assert import_module(module)


def test_operator_entry_points_and_documentation_exist():
    for path in ("README.md", "docs/ARCHITECTURE.md", "notebooks/poc_results.ipynb"):
        assert Path(path).is_file()
