"""Synthetic checks for read-only expanded evidence contracts."""

from datetime import date, timedelta
import hashlib
import json

import polars as pl
import pytest

from trading_pipeline.reporting.expanded_closeout import (
    _hash, _newest_completed_run_id, _profile, _protocol_inputs, _sample_weekly, _security_and_stage,
    _supervised_portfolio, _verified_run,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_completed_run_requires_exact_audit_inventory(tmp_path):
    root = tmp_path
    run = root / "runs" / "expanded_closeout" / "test-run"
    _write(run / "protocol.json", {"study_id": "synthetic"})
    _write(run / "metadata.json", {"run_id": "test-run", "status": "complete",
                                   "holdout_role": "descriptive_only"})
    artefacts = {p.relative_to(run).as_posix(): _hash(p) for p in run.iterdir()}
    _write(run / "audit.json", {"status": "passed", "artefact_sha256": artefacts})
    _write(run / "completion.json", {"run_id": "test-run", "status": "complete",
                                     "audit_sha256": _hash(run / "audit.json")})
    assert _verified_run(root, run)[0]["run_id"] == "test-run"
    _write(run / "protocol.json", {"study_id": "tampered"})
    with pytest.raises(ValueError, match="hash inventory"):
        _verified_run(root, run)


def test_newest_completed_run_skips_partial_and_tampered_runs(tmp_path):
    base = tmp_path / "runs" / "expanded_closeout"
    for name, stamp in (("older", "2026-09-27T01:00:00+00:00"),
                        ("latest", "2026-09-27T02:00:00+00:00"),
                        ("bad", "2026-09-27T03:00:00+00:00")):
        run = base / name
        _write(run / "protocol.json", {"study_id": "synthetic"})
        _write(run / "metadata.json", {"run_id": name, "status": "complete",
                                       "holdout_role": "descriptive_only"})
        inventory = {p.relative_to(run).as_posix(): _hash(p) for p in run.iterdir()}
        _write(run / "audit.json", {"status": "passed", "artefact_sha256": inventory})
        _write(run / "completion.json", {"run_id": name, "status": "complete",
                                         "audit_sha256": _hash(run / "audit.json"),
                                         "completed_at": stamp})
    _write(base / "bad" / "protocol.json", {"study_id": "tampered"})
    assert _newest_completed_run_id(tmp_path) == "latest"


def test_zero_argument_generator_cli_uses_auto_selected_run(tmp_path, monkeypatch, capsys):
    import sys
    import trading_pipeline.reporting.expanded_closeout as module

    selected = []
    monkeypatch.setattr(sys, "argv", ["expanded_closeout"])
    monkeypatch.setattr(module, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_newest_completed_run_id", lambda root: "audited-latest")
    monkeypatch.setattr(module, "generate_expanded_closeout", lambda **kwargs: selected.append(kwargs) or "generated")
    module.main()
    assert selected == [{"repository_root": tmp_path, "run_id": "audited-latest"}]
    assert capsys.readouterr().out.strip() == "generated"


def test_supervised_portfolio_uses_next_close_and_costs():
    days = tuple(date(2026, 1, 5) + timedelta(days=i) for i in range(5))
    history = tuple(date(2025, 12, 1) + timedelta(days=i) for i in range(35))
    prices = [100 + i for i in range(len(history))]
    rows = [{"security_id": "A", "session_date": day, "adjusted_close": price}
            for day, price in zip(history, prices, strict=True)]
    rows += [{"security_id": "A", "session_date": day, "adjusted_close": 140 + i}
             for i, day in enumerate(days)]
    features = pl.DataFrame(rows)
    preds = pl.DataFrame([{"security_id": "A", "session_date": day,
                           "predicted_return_5d": 0.1} for day in days])
    scenario = {"gross_exposure_cap": 0.6, "max_position": 0.6,
                "annualised_volatility_target": 0.08}
    config = {"volatility_lookback_sessions": 20, "top_k": 1,
              "cost_bps_one_way": 10}
    curve, positions, trades, contributions = _supervised_portfolio(
        preds, features, days, scenario, config)
    assert curve["equity"][0] == 1.0
    assert curve["cost"][0] == 0.0
    assert curve["cost"][1] > 0
    assert trades[0]["signal_date"] == days[0]
    assert trades[0]["session_date"] == days[1]
    assert all(row["weight"] <= 0.6 + 1e-12 for row in positions
               if row["session_date"] == days[1])
    assert sum(row["gross_contribution"] for row in contributions) - sum(
        row["cost"] for row in trades) == pytest.approx(curve["equity"][-1] - 1)
    weekly = _sample_weekly(curve, (days[2], days[4]))
    assert weekly.height == 2
    assert weekly["cost"].sum() == pytest.approx(curve["cost"].sum())


def test_weekly_profile_rejects_equity_mismatch():
    day = date(2026, 1, 1)
    curve = pl.DataFrame([{"session_date": day, "equity": 1.1,
                           "daily_return": 0.0, "turnover": 0.0, "cost": 0.0}])
    with pytest.raises(ValueError, match="reconcile"):
        _profile(curve, "M", 41, "balanced", 0.12, "descriptive_holdout", False)


def test_stage_counts_reconcile_to_security_rows():
    day = date(2026, 1, 1)
    features = pl.DataFrame([{"security_id": "A", "ticker": "AAA",
        "session_date": day, "adjusted_close": 100.0, "latest_eps": 1.0,
        "latest_net_income": 10.0, "return_5d": 0.01,
        "forward_return_5d": 0.02}])
    preds = pl.DataFrame([{"security_id": "A", "session_date": day}])
    stages, security = _security_and_stage(features, preds, [], [], [],
                                           {"requested_count": 1, "mapped_count": 1}, (day,))
    assert security[0]["predicted_rows"] == 1
    assert next(row["count"] for row in stages if row["stage"] == "predicted") == 1


def test_protocol_inputs_uses_separately_pinned_benchmark(tmp_path):
    day = date(2026, 1, 1)
    features = tmp_path / "features.parquet"
    benchmark = tmp_path / "benchmark.parquet"
    pl.DataFrame([{"security_id": "A", "ticker": "AAA", "session_date": day,
                   "adjusted_close": 10.0, "vol_20d": 0.2}]).write_parquet(features)
    pl.DataFrame([{"security_id": "BENCHMARK-SPY", "ticker": "SPY",
                   "session_date": day, "adjusted_close": 100.0}]).write_parquet(benchmark)
    source = tmp_path / "source.json"
    _write(source, {"requested_count": 500, "admitted_count": 496})
    snapshot = tmp_path / "snapshot.json"
    _write(snapshot, {"source_manifest_path": "source.json",
                      "source_manifest_sha256": _hash(source),
                      "feature_path": "features.parquet", "feature_sha256": _hash(features)})
    benchmark_manifest = tmp_path / "benchmark.json"
    _write(benchmark_manifest, {"benchmark_id": "B0-SPY", "ticker": "SPY",
        "bars_path": "benchmark.parquet", "bars_sha256": _hash(benchmark)})
    pinned = {"snapshot.json": _hash(snapshot)}
    protocol = {"authority": {"input_sha256": pinned, "protocol_sha256": "p"},
        "data": {"snapshot_manifest": "snapshot.json",
                 "rl_inputs": {"bars_path": "features.parquet",
                               "bars_sha256": _hash(features)},
                 "benchmark_contract": {"benchmark_id": "B0-SPY",
                    "manifest_path": "benchmark.json",
                    "manifest_sha256": _hash(benchmark_manifest),
                    "bars_path": "benchmark.parquet", "bars_sha256": _hash(benchmark)}}}
    metadata = {"input_sha256": pinned, "protocol_sha256": "p",
                "feature_path": "features.parquet", "feature_sha256": _hash(features)}
    canonical, spy, returned_snapshot, sources = _protocol_inputs(
        tmp_path, metadata, protocol)
    assert canonical["ticker"].to_list() == ["AAA"]
    assert spy["ticker"].to_list() == ["SPY"]
    assert returned_snapshot["requested_count"] == 500
    assert sources["benchmark.parquet"] == _hash(benchmark)
