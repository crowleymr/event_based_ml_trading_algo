"""Operational monitoring must preserve evidence even when the child fails."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

from trading_pipeline.operations import study_supervisor as monitor


class FakeProcess:
    pid = 12345

    def __init__(self, returncode):
        self.stdout = io.StringIO("fit started\nWARNING: retry\n")
        self.stderr = io.StringIO("RuntimeError: failed fit\n")
        self.returncode = returncode

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


def test_supervisor_records_failed_child_and_resources(tmp_path, monkeypatch, capsys):
    study = tmp_path / "approved.yaml"
    study.write_text("study: mock", encoding="utf-8")
    calls = []

    def start(command, **kwargs):
        calls.append(command)
        return FakeProcess(7)

    monkeypatch.setattr(monitor.subprocess, "Popen", start)
    sample = {"system": {"ram_available_bytes": 10},
              "process_tree": {"rss_bytes": 20},
              "gpu": {"available": False, "devices": []}}
    output = tmp_path / "monitor"
    assert monitor.supervise(study, output, interval_seconds=0.01,
                             sampler=lambda pid: sample) == 7
    assert calls[0][2:] == ["-m", "trading_pipeline.run", "--study", str(study.resolve())]
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "starting"
    assert events[1]["event"] == "running"
    assert {entry["event"] for entry in events[2:-1]} == {"info", "warning", "error"}
    assert events[-1]["event"] == "failed"
    assert events[-1]["exit_code"] == 7
    assert metrics[0]["process_tree"]["rss_bytes"] == 20
    assert "WARNING: retry" in (output / "stdout.log").read_text()
    assert "RuntimeError" in (output / "stderr.log").read_text()
    console = capsys.readouterr().out
    assert "Study process running" in console
    assert "WARNING: retry" in console
    assert "RuntimeError: failed fit" in console
    assert "Study failed (exit code 7)" in console


def test_supervisor_forwards_resume_source(tmp_path, monkeypatch):
    study = tmp_path / "approved.yaml"
    study.write_text("study: mock", encoding="utf-8")
    source = tmp_path / "failed-run"
    source.mkdir()
    calls = []

    def start(command, **kwargs):
        calls.append(command)
        return FakeProcess(0)

    monkeypatch.setattr(monitor.subprocess, "Popen", start)
    output = tmp_path / "monitor"
    sample = {"system": {}, "process_tree": {}, "gpu": {"available": False}}
    assert monitor.supervise(study, output, interval_seconds=0.01,
                             resume_from=source,
                             sampler=lambda pid: sample) == 0
    assert calls[0][-2:] == ["--resume-from", str(source.resolve())]
    starting = json.loads((output / "events.jsonl").read_text().splitlines()[0])
    assert starting["resume_from"] == str(source.resolve())


def test_console_filter_prints_status_but_not_complete_log(capsys):
    events = io.StringIO()
    output = io.StringIO()
    source = io.StringIO(
        "ordinary dependency chatter\n"
        "2026-09-27 INFO Inner candidate started fold=outer_1 arm=E1 trial=T1\n"
    )
    monitor._drain(source, "stderr", output, events, monitor.threading.Lock())
    console = capsys.readouterr().out
    assert "ordinary dependency chatter" not in console
    assert "Inner candidate started" in console


def test_human_log_retains_one_local_timestamp_without_milliseconds_or_utc(capsys):
    events = io.StringIO()
    output = io.StringIO()
    source = io.StringIO(
        "2026-09-27 22:53:44,537 INFO Inner candidate started "
        "fold=outer_1 arm=LSTM_F1_STACK trial=LSTM_F1_STACK-outer_1-002\n"
    )

    monitor._drain(source, "stderr", output, events, monitor.threading.Lock())

    expected = (
        "2026-09-27 22:53:44 INFO Inner candidate started "
        "fold=outer_1 arm=LSTM_F1_STACK trial=LSTM_F1_STACK-outer_1-002\n"
    )
    assert output.getvalue() == expected
    assert capsys.readouterr().out == expected
    assert "+00:00" not in output.getvalue()
    structured = json.loads(events.getvalue())
    assert structured["timestamp_utc"].endswith("+00:00")
    assert structured["message"].startswith("2026-09-27 22:53:44,537 INFO")


def test_gpu_sampler_handles_missing_nvidia_smi(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(monitor.subprocess, "run", missing)
    assert monitor._gpu_sample() == {"available": False, "devices": [],
                                     "process_tree_vram_used_mib": None}


def test_process_cpu_uses_cumulative_time_delta(monkeypatch):
    elapsed = iter([100.0, 102.0])
    cpu_seconds = iter([4.0, 5.0])

    class Process:
        pid = 101

        def __init__(self, pid):
            assert pid == 101

        def children(self, recursive):
            return []

        def memory_info(self):
            return SimpleNamespace(rss=4096)

        def cpu_times(self):
            return SimpleNamespace(user=next(cpu_seconds), system=0.0)

        def create_time(self):
            return 10.0

    monkeypatch.setattr(monitor.time, "monotonic", lambda: next(elapsed))
    monkeypatch.setattr(monitor.psutil, "Process", Process)
    monkeypatch.setattr(monitor.psutil, "virtual_memory", lambda: SimpleNamespace(
        total=100, available=50, used=50))
    monkeypatch.setattr(monitor.psutil, "cpu_percent", lambda interval: 25.0)
    monkeypatch.setattr(monitor.psutil, "getloadavg", lambda: (1.0, 2.0, 3.0))
    monkeypatch.setattr(monitor, "_gpu_sample", lambda pids: {"available": False})
    sampler = monitor.ResourceSampler()
    first = sampler(101)
    second = sampler(101)
    assert first["process_tree"]["cpu_utilization_percent"] is None
    assert second["process_tree"]["cpu_utilization_percent"] == 50.0
    assert second["process_tree"]["cumulative_cpu_seconds"] == 5.0
    assert second["system"]["load_average_1m_5m_15m"] == [1.0, 2.0, 3.0]
