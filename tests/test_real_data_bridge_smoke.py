"""Fail-closed checks for the score-blind prepared-draft smoke."""

from pathlib import Path
from datetime import date

import pytest

from trading_pipeline.experiments import real_data_bridge_smoke as smoke


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "configs/studies/expanded_closeout_prepared_v3.yaml"


def test_prepared_v3_rejects_unresolved_causal_stack_before_parquet_read(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Parquet must not be opened before protocol validation")

    monkeypatch.setattr(smoke.pl, "scan_parquet", forbidden)
    monkeypatch.setattr(smoke.pl, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="augmented causal-stack contract"):
        smoke.validate_inputs(ROOT, PREPARED)


def test_unresolved_draft_emits_no_evidence():
    output = ROOT / "never_written_bridge_smoke"
    records = ROOT / "never_written_bridge_records.json"
    with pytest.raises(ValueError, match="augmented causal-stack contract"):
        smoke.run_smoke(repository_root=ROOT, prepared_study=PREPARED,
                        output_dir=output, records_path=records)
    assert not output.exists()
    assert not records.exists()


def test_pinned_path_rejects_placeholder_and_escape():
    for relative in ("pending_generated_causal_stack", "../outside.parquet"):
        with pytest.raises(ValueError):
            smoke._pinned(ROOT, relative, "0" * 64)


def test_smoke_covers_required_families_without_score_bearing_runner():
    import yaml
    arms = yaml.safe_load(PREPARED.read_text(encoding="utf-8"))["experiment_arms"]
    assert len(arms) == 7  # The study YAML, rather than Python, owns this roster.
    assert not hasattr(smoke, "run_study")


def test_rl_smoke_signals_begin_at_causal_output_availability():
    days = [date(2020, 1, day) for day in range(2, 7)]
    fit = [{"session_date": day} for day in days]
    assert smoke._causal_rl_signal_dates(fit, set(days[2:])) == tuple(days[2:])
    with pytest.raises(ValueError, match="two causal-output signal sessions"):
        smoke._causal_rl_signal_dates(fit, {days[-1]})
