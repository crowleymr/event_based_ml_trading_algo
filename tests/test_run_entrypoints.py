"""CLI routing must preserve the locked runner while study execution is gated."""

from __future__ import annotations

import sys

import pytest

import trading_pipeline.run as runner


def test_config_cli_keeps_legacy_run_route(monkeypatch, capsys):
    config = object()
    calls = []
    monkeypatch.setattr(sys, "argv", ["trading_pipeline.run", "--config", "configs/smoke.yaml"])
    monkeypatch.setattr(runner, "load_config", lambda path: calls.append(("config", path)) or config)
    monkeypatch.setattr(runner, "run", lambda value: calls.append(("run", value)) or "run-id")
    monkeypatch.setattr(runner, "ingest", lambda value: pytest.fail("unexpected ingestion-only route"))

    runner.main()

    assert calls == [("config", "configs/smoke.yaml"), ("run", config)]
    assert capsys.readouterr().out.strip() == "run-id"


def test_config_ingest_only_keeps_legacy_ingestion_route(monkeypatch):
    config = object()
    calls = []
    monkeypatch.setattr(sys, "argv", ["trading_pipeline.run", "--config", "configs/smoke.yaml", "--ingest-only"])
    monkeypatch.setattr(runner, "load_config", lambda path: calls.append(("config", path)) or config)
    monkeypatch.setattr(runner, "ingest", lambda value: calls.append(("ingest", value)))
    monkeypatch.setattr(runner, "run", lambda value: pytest.fail("unexpected research run"))

    runner.main()

    assert calls == [("config", "configs/smoke.yaml"), ("ingest", config)]


def test_study_cli_rejects_ingest_only_before_loading(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["trading_pipeline.run", "--study", "draft.yaml", "--ingest-only"])
    monkeypatch.setattr(runner, "load_study", lambda *args, **kwargs: pytest.fail("study must not load"))

    with pytest.raises(SystemExit, match="2"):
        runner.main()
