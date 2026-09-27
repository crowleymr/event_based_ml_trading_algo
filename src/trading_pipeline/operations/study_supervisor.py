"""Launch and observe the authoritative expanded study without reading results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import psutil


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_timestamp() -> str:
    """Return the operator-facing local timestamp at whole-second precision."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _record(handle, event: str, **fields: object) -> None:
    handle.write(json.dumps({"timestamp_utc": _utc(), "event": event, **fields},
                            sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def _gpu_sample(pids: set[int] | None = None) -> dict:
    """Return physical GPU figures; WDDM may hide per-process VRAM on Windows."""
    command = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        devices = []
        for row in result.stdout.splitlines():
            cells = [cell.strip() for cell in row.split(",")]
            if len(cells) == 4:
                devices.append({"index": int(cells[0]), "utilization_percent": int(cells[1]),
                                "vram_used_mib": int(cells[2]), "vram_total_mib": int(cells[3])})
        process_vram = None
        if pids:
            try:
                applications = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=True)
                used = []
                for row in applications.stdout.splitlines():
                    cells = [cell.strip() for cell in row.split(",")]
                    if len(cells) == 2 and int(cells[0]) in pids:
                        used.append(int(cells[1]))
                process_vram = sum(used)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        return {"available": bool(devices), "devices": devices,
                "process_tree_vram_used_mib": process_vram}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"available": False, "devices": [], "process_tree_vram_used_mib": None}


class ResourceSampler:
    """Sample process CPU from cumulative CPU-time differences across heartbeats."""

    def __init__(self) -> None:
        self._previous: dict[tuple[int, float], float] = {}
        self._previous_wall: float | None = None
        self._cumulative_cpu_seconds = 0.0

    def __call__(self, pid: int) -> dict:
        now = time.monotonic()
        ram = psutil.virtual_memory()
        try:
            load = list(psutil.getloadavg())
        except (AttributeError, OSError):
            load = None
        system = {"ram_total_bytes": ram.total, "ram_available_bytes": ram.available,
                  "ram_used_bytes": ram.used, "cpu_utilization_percent": psutil.cpu_percent(interval=None),
                  "load_average_1m_5m_15m": load}
        tree_rss = 0
        current: dict[tuple[int, float], float] = {}
        pids = set()
        try:
            parent = psutil.Process(pid)
            processes = [parent, *parent.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            processes = []
        for process in processes:
            try:
                tree_rss += process.memory_info().rss
                cpu = process.cpu_times()
                current[(process.pid, process.create_time())] = cpu.user + cpu.system
                pids.add(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        utilization = None
        if self._previous_wall is None:
            self._cumulative_cpu_seconds = sum(current.values())
        else:
            delta = sum(max(0.0, value - self._previous.get(key, 0.0))
                        for key, value in current.items())
            self._cumulative_cpu_seconds += delta
            elapsed = now - self._previous_wall
            utilization = 100.0 * delta / elapsed if elapsed > 0 else None
        self._previous = current
        self._previous_wall = now
        return {"system": system, "process_tree": {"pid": pid, "process_count": len(current),
                "rss_bytes": tree_rss, "cpu_utilization_percent": utilization,
                "cumulative_cpu_seconds": round(self._cumulative_cpu_seconds, 6)},
                "gpu": _gpu_sample(pids)}


_WARNING = re.compile(r"warning|\bwarn(?:ing)?\s*:", re.IGNORECASE)
_ERROR = re.compile(r"\berror\b|\bexception\b|\btraceback\b|\bfailed\b", re.IGNORECASE)
_STATUS = re.compile(
    r"\b(?:Expanded study|Outer fold|Supervised arm|Inner candidate|Outer evaluation|"
    r"RL arm|Family locks|Descriptive(?: RL)? holdout)\b",
    re.IGNORECASE,
)
_CHILD_TIMESTAMP = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"(?:[,.]\d+)?(?P<message>\s+.*)?$"
)


def _operator_log_line(message: str) -> str:
    """Use a child's local timestamp once, without sub-second noise or UTC prefix."""
    match = _CHILD_TIMESTAMP.match(message)
    if match:
        return f"{match.group('timestamp')}{match.group('message') or ''}"
    return f"{_local_timestamp()} {message}"


def _drain(stream, name: str, output, events, lock: threading.Lock) -> None:
    for line in iter(stream.readline, ""):
        message = line.rstrip()
        operator_line = _operator_log_line(message)
        output.write(f"{operator_line}\n")
        output.flush()
        severity = "error" if _ERROR.search(message) else "warning" if _WARNING.search(message) else "info"
        with lock:
            _record(events, severity, source=name, message=message[:4000])
            if severity in {"warning", "error"} or _STATUS.search(message):
                print(operator_line, flush=True)
    stream.close()


def supervise(study: Path, log_dir: Path, *, interval_seconds: float = 10.0,
              resume_from: Path | None = None, sampler=None) -> int:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if not study.is_file():
        raise FileNotFoundError(study)
    if resume_from is not None and not resume_from.is_dir():
        raise FileNotFoundError(resume_from)
    sampler = sampler if sampler is not None else ResourceSampler()
    log_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "-m", "trading_pipeline.run", "--study", str(study.resolve())]
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from.resolve())])
    start = time.monotonic()
    with (log_dir / "events.jsonl").open("x", encoding="utf-8") as events, \
         (log_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics, \
         (log_dir / "stdout.log").open("x", encoding="utf-8") as stdout, \
         (log_dir / "stderr.log").open("x", encoding="utf-8") as stderr:
        lock = threading.Lock()
        _record(events, "starting", command=command, study=str(study.resolve()),
                resume_from=str(resume_from.resolve()) if resume_from is not None else None,
                pid=os.getpid(), interval_seconds=interval_seconds)
        print(f"{_local_timestamp()} STATUS Starting supervised study process", flush=True)
        process = None
        interrupted = False
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
            _record(events, "running", child_pid=process.pid)
            print(f"{_local_timestamp()} STATUS Study process running (PID {process.pid})", flush=True)
            readers = [threading.Thread(target=_drain, args=(process.stdout, "stdout", stdout, events, lock),
                                        daemon=True),
                       threading.Thread(target=_drain, args=(process.stderr, "stderr", stderr, events, lock),
                                        daemon=True)]
            for reader in readers:
                reader.start()
            while True:
                try:
                    sample = sampler(process.pid)
                    _record(metrics, "heartbeat", elapsed_seconds=round(time.monotonic() - start, 3),
                            child_pid=process.pid, **sample)
                except Exception as exc:  # Monitoring failure must not stop the research run.
                    _record(events, "warning", source="sampler", message=repr(exc))
                try:
                    process.wait(timeout=interval_seconds)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except KeyboardInterrupt:
            interrupted = True
            if process is not None:
                _record(events, "interrupted", child_pid=process.pid,
                        elapsed_seconds=round(time.monotonic() - start, 3))
                try:
                    descendants = psutil.Process(process.pid).children(recursive=True)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    descendants = []
                for child in descendants:
                    try:
                        child.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        except Exception as exc:
            _record(events, "supervisor_failure", error=repr(exc),
                    elapsed_seconds=round(time.monotonic() - start, 3))
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
            raise
        finally:
            if process is not None:
                for reader in locals().get("readers", []):
                    reader.join(timeout=10)
        code = 130 if interrupted else process.returncode
        _record(events, "finished" if code == 0 else "failed", exit_code=code,
                elapsed_seconds=round(time.monotonic() - start, 3), child_pid=process.pid)
        state = "completed" if code == 0 else "failed"
        print(f"{_local_timestamp()} STATUS Study {state} (exit code {code})", flush=True)
        return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="Approved study protocol")
    parser.add_argument("--log-dir", type=Path, required=True, help="New monitoring directory")
    parser.add_argument("--resume-from", type=Path,
                        help="Terminally failed checkpointed run to continue immutably")
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    raise SystemExit(supervise(args.study, args.log_dir,
                               interval_seconds=args.interval_seconds,
                               resume_from=args.resume_from))


if __name__ == "__main__":
    main()
