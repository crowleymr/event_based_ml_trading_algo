"""The operator download command delegates to the canonical ingestion service."""

import sys

import trading_pipeline.data.download as download


def test_download_cli_defaults_to_poc_config(monkeypatch):
    cfg = object()
    calls = []
    monkeypatch.setattr(sys, "argv", ["trading_pipeline.data.download"])
    monkeypatch.setattr(download, "load_config", lambda path: calls.append(("config", path)) or cfg)
    monkeypatch.setattr(download, "ingest", lambda value: calls.append(("ingest", value)))

    download.main()

    assert calls == [("config", "configs/poc.yaml"), ("ingest", cfg)]


def test_download_cli_uses_explicit_config(monkeypatch):
    cfg = object()
    calls = []
    monkeypatch.setattr(sys, "argv", ["trading_pipeline.data.download", "--config", "configs/custom.yaml"])
    monkeypatch.setattr(download, "load_config", lambda path: calls.append(("config", path)) or cfg)
    monkeypatch.setattr(download, "ingest", lambda value: calls.append(("ingest", value)))

    download.main()

    assert calls == [("config", "configs/custom.yaml"), ("ingest", cfg)]
