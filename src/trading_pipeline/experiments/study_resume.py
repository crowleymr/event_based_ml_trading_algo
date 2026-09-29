"""Immutable cell checkpoints for authoritative expanded-study continuation.

Only the central study runner calls this module. A continuation creates a new run;
the source run and its terminal failure record are never modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Callable
import psutil


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_state_sha256(root: Path) -> str:
    """Bind dirty and committed Python/dependency bytes, independent of Git HEAD."""
    paths = sorted((root / "src" / "trading_pipeline").rglob("*.py"))
    paths += [root / name for name in ("pyproject.toml", "requirements-lock.txt")]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str, allow_nan=False).encode("utf-8")


def cell_key(kind: str, identity: dict[str, Any]) -> str:
    return f"{kind}-{hashlib.sha256(_canonical(identity)).hexdigest()[:24]}"


def process_still_running(started: dict[str, Any]) -> bool:
    """Match both PID and process creation time to avoid PID reuse."""
    pid = started.get("pid")
    created = started.get("process_create_time")
    if type(pid) is not int or not isinstance(created, (int, float)):
        raise ValueError("Unmarked source lacks a verifiable process identity")
    try:
        process = psutil.Process(pid)
        return process.is_running() and abs(process.create_time() - float(created)) < 0.01
    except psutil.NoSuchProcess:
        return False


class CheckpointStore:
    def __init__(self, output: Path, *, protocol_sha256: str, code_sha256: str,
                 source: Path | None = None, allow_unmarked: bool = False,
                 source_code_sha256: str | None = None,
                 code_guard: Callable[[], str] | None = None):
        self.output = output
        self.protocol_sha256 = protocol_sha256
        self.code_sha256 = code_sha256
        self.code_guard = code_guard
        self.source = source
        expected_source_code = source_code_sha256 or code_sha256
        self._source_cells: dict[str, dict[str, Any]] = {}
        if source is not None:
            index = source / "checkpoint_index.jsonl"
            failure = source / "failure.json"
            if not failure.is_file() and not allow_unmarked:
                raise ValueError("Only a terminally failed study may be resumed")
            terminal = json.loads(failure.read_text(encoding="utf-8")) if failure.is_file() else {}
            if failure.is_file() and terminal.get("status") != "failed":
                raise ValueError("Resume source lacks a failed terminal status")
            expected_index = terminal.get("checkpoint_index_sha256")
            if allow_unmarked and not failure.is_file() and not index.is_file():
                raise ValueError("Interrupted source has no verified cell checkpoints")
            if expected_index is None:
                if allow_unmarked and not failure.is_file():
                    expected_index = sha256(index)
                else:
                    if index.exists():
                        raise ValueError("Source checkpoint index has no terminal hash")
                    return
            if not index.is_file() or sha256(index) != expected_index:
                raise ValueError("Source terminal checkpoint index hash mismatch")
            for line in index.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                key = entry["key"]
                if not isinstance(key, str) or not re.fullmatch(
                    r"(?:supervised_sensitivity|rl_sensitivity|supervised_inner|"
                    r"supervised_outer|supervised_holdout|rl_inner|rl_outer|rl_holdout)-"
                    r"[0-9a-f]{24}",
                    key):
                    raise ValueError("Malformed source checkpoint key")
                if key in self._source_cells:
                    raise ValueError(f"Duplicate source checkpoint: {key}")
                path = source / "checkpoints" / f"{key}.json"
                if not path.is_file() or sha256(path) != entry["sha256"]:
                    raise ValueError(f"Source checkpoint hash mismatch: {key}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (payload.get("key") != key
                        or cell_key(payload.get("kind"), payload.get("identity")) != key
                        or payload.get("protocol_sha256") != protocol_sha256
                        or payload.get("code_sha256") != expected_source_code):
                    raise ValueError(f"Source checkpoint authority mismatch: {key}")
                for relative, expected in payload.get("artefact_sha256", {}).items():
                    artefact = (source / relative).resolve()
                    if (not artefact.is_relative_to(source) or not artefact.is_file()
                            or sha256(artefact) != expected):
                        raise ValueError(f"Source cell artefact hash mismatch: {relative}")
                self._source_cells[key] = payload

    def get(self, kind: str, identity: dict[str, Any]) -> dict[str, Any] | None:
        key = cell_key(kind, identity)
        payload = self._source_cells.get(key)
        if payload is not None and _canonical(payload.get("identity")) != _canonical(identity):
            raise ValueError(f"Source cell identity mismatch: {key}")
        return payload

    def save(self, kind: str, identity: dict[str, Any], record: dict[str, Any],
             artefacts: tuple[Path, ...] = (), *, reused_from: str | None = None) -> None:
        if self.code_guard is not None and self.code_guard() != self.code_sha256:
            raise ValueError("Study code changed during score-bearing execution")
        key = cell_key(kind, identity)
        payload = {"key": key, "kind": kind, "identity": identity,
                   "protocol_sha256": self.protocol_sha256,
                   "code_sha256": self.code_sha256, "record": record,
                   "artefact_sha256": {
                       path.relative_to(self.output).as_posix(): sha256(path)
                       for path in artefacts}, "reused_from": reused_from}
        path = self.output / "checkpoints" / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2,
                      default=str, allow_nan=False)
        with (self.output / "checkpoint_index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "sha256": sha256(path)},
                                    sort_keys=True) + "\n")

    def copy_artefact(self, payload: dict[str, Any], relative: str) -> Path:
        if self.source is None or relative not in payload["artefact_sha256"]:
            raise ValueError(f"Unverified resumed artefact: {relative}")
        source = (self.source / relative).resolve()
        target = (self.output / relative).resolve()
        if not source.is_relative_to(self.source) or not target.is_relative_to(self.output):
            raise ValueError("Cell artefact path escapes run")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(source, target)
        if sha256(target) != payload["artefact_sha256"][relative]:
            raise ValueError(f"Copied cell artefact hash mismatch: {relative}")
        return target
