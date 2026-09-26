"""Read-only authority gate for an expanded, score-bearing study."""

from __future__ import annotations

from datetime import date
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .schema import ResolvedStudy


INPUT_FIELDS = (
    ("authority", "approval_record"),
    ("data", "snapshot_manifest"),
    ("data", "exposure_ledger"),
    ("validation", "outer_windows_manifest"),
    ("validation", "inner_windows_manifest"),
    ("validation", "final_holdout", "manifest"),
)


@dataclass(frozen=True)
class VerifiedStudyAuthority:
    study_id: str
    protocol_sha256: str
    input_sha256: Mapping[str, str]
    vintage_id: str
    claim_status: str
    data_role: str = "expanded_fresh_vintage"


def _field(config: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping):
            raise ValueError(f"Invalid authority field: {'.'.join(keys)}")
        value = value.get(key)
    if not isinstance(value, str) or not value.strip() or value.startswith("pending_"):
        raise ValueError(f"Unresolved authority field: {'.'.join(keys)}")
    return value


def _dates(values: Any, label: str) -> tuple[date, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} requires nonempty session dates")
    try:
        result = tuple(date.fromisoformat(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has an invalid session date") from exc
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{label} session dates must be sorted and unique")
    return result


def _folds(payload: Mapping[str, Any], label: str) -> dict[str, dict[str, set[date]]]:
    folds = payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"{label} requires folds")
    output: dict[str, dict[str, set[date]]] = {}
    for fold in folds:
        if not isinstance(fold, dict) or not isinstance(fold.get("fold_id"), str):
            raise ValueError(f"{label} requires fold_id")
        key = fold["fold_id"]
        if key in output:
            raise ValueError(f"{label} has duplicate fold_id {key}")
        fit = set(_dates(fold.get("fit_dates"), f"{label}.{key}.fit_dates"))
        score = set(_dates(fold.get("score_dates"), f"{label}.{key}.score_dates"))
        embargo_values = fold.get("embargo_dates", [])
        embargo = set(_dates(embargo_values, f"{label}.{key}.embargo_dates")) if embargo_values else set()
        if fit & score or fit & embargo or score & embargo or max(fit) >= min(score):
            raise ValueError(f"{label}.{key} has overlapping or unordered windows")
        output[key] = {"fit": fit, "score": score, "embargo": embargo}
    return output


def verify_study_authority(study: ResolvedStudy, *, repository_root: str | Path) -> VerifiedStudyAuthority:
    """Verify pinned files and temporal separation before any research I/O.

    All paths are repository-relative. JSON manifests name the same study ID and
    use explicit sorted ISO session dates. This gate never writes to the study.
    """
    if study.engineering_only:
        raise PermissionError("Draft study cannot execute on real data")
    config = study.config
    _field(config, ("data", "vintage_id"))
    if config.get("inference", {}).get("claim_status") != "exploratory_closeout_not_confirmatory":
        raise ValueError("Expanded study must retain exploratory claim status")
    root = Path(repository_root).resolve()
    pinned = config["authority"].get("input_sha256")
    if not isinstance(pinned, dict):
        raise ValueError("Approved study requires authority.input_sha256")
    verified: dict[str, str] = {}
    parsed: dict[str, Mapping[str, Any]] = {}
    for keys in INPUT_FIELDS:
        relative = _field(config, keys)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Authority input missing or outside repository: {relative}")
        expected = pinned.get(relative)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Authority input lacks SHA-256 pin: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Authority input hash mismatch: {relative}")
        verified[relative] = actual
        if keys != ("authority", "approval_record") and path.suffix.lower() != ".json":
            raise ValueError(f"Authority manifest must be JSON: {relative}")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("study_id") != config["study_id"]:
                raise ValueError(f"Authority manifest study_id mismatch: {relative}")
            parsed[".".join(keys)] = payload
    if set(pinned) != set(verified):
        raise ValueError("Authority input pins must match required inputs exactly")
    snapshot = parsed.get("data.snapshot_manifest")
    exposure = parsed.get("data.exposure_ledger")
    if (not snapshot or snapshot.get("vintage_id") != config["data"]["vintage_id"]
            or snapshot.get("data_role") != "expanded_fresh_vintage"):
        raise ValueError("Snapshot vintage does not match study")
    if not exposure or "legacy_observed_final" not in exposure.get("forbidden_selection_roles", []):
        raise ValueError("Exposure ledger must exclude the legacy observed final")
    outer = _folds(parsed["validation.outer_windows_manifest"], "outer")
    inner_payload = parsed["validation.inner_windows_manifest"]
    inner = _folds(inner_payload, "inner")
    for fold in inner_payload["folds"]:
        parent = fold.get("outer_fold_id")
        if parent not in outer or not set().union(*inner[fold["fold_id"]].values()) <= outer[parent]["fit"]:
            raise ValueError("Inner fold escapes its outer fit history")
    holdout_payload = parsed["validation.final_holdout.manifest"]
    if holdout_payload.get("sealed") is not True:
        raise ValueError("Final holdout manifest must be sealed")
    holdout = set(_dates(holdout_payload.get("session_dates"), "final_holdout"))
    for fold in outer.values():
        if holdout & set().union(*fold.values()):
            raise ValueError("Final holdout overlaps outer evaluation")
        if max(fold["score"]) >= min(holdout):
            raise ValueError("Final holdout must follow outer evaluation")
    return VerifiedStudyAuthority(
        study_id=config["study_id"],
        protocol_sha256=config["authority"]["protocol_sha256"],
        input_sha256=verified,
        vintage_id=config["data"]["vintage_id"],
        claim_status="exploratory_closeout_not_confirmatory",
    )
