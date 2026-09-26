"""Strict draft-study loading and canonical hashing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import INTERFACES


TOP_LEVEL_FIELDS = {
    "schema_version", "kind", "study_id", "status", "authority", "data",
    "validation", "experiment_arms", "objectives", "selection", "search",
    "reproducibility", "outputs", "portfolio", "inference",
}
REQUIRED_FIELDS = {
    "schema_version", "kind", "study_id", "status", "authority", "data",
    "validation", "experiment_arms", "objectives", "selection", "search",
    "reproducibility", "outputs",
}


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class ResolvedStudy:
    config: Mapping[str, Any]
    sha256: str
    engineering_only: bool


def load_study(path: str | Path, *, allow_engineering_draft: bool = True) -> ResolvedStudy:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Study config must be a mapping")
    missing = REQUIRED_FIELDS - value.keys()
    unknown = value.keys() - TOP_LEVEL_FIELDS
    if missing:
        raise ValueError(f"Missing study fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown study fields: {sorted(unknown)}")
    if value["schema_version"] != 3 or value["kind"] != "optimisation_study":
        raise ValueError("Only optimisation_study schema version 3 is supported")
    if value["status"] not in {"draft", "approved"}:
        raise ValueError("Study status must be draft or approved")
    arms = value["experiment_arms"]
    if not isinstance(arms, list) or not arms:
        raise ValueError("experiment_arms must be a non-empty list")
    arm_ids = [arm.get("id") for arm in arms if isinstance(arm, dict)]
    if len(arm_ids) != len(arms) or any(not item for item in arm_ids):
        raise ValueError("Every experiment arm requires an id")
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("Experiment arm ids must be unique")
    for arm in arms:
        if arm.get("interface") not in INTERFACES:
            raise ValueError(f"Unsupported experiment arm interface: {arm.get('interface')}")
        if not arm.get("component_id"):
            raise ValueError(f"Experiment arm {arm['id']} requires component_id")
    engineering_only = value["status"] == "draft"
    if engineering_only and not allow_engineering_draft:
        raise ValueError("Draft studies are restricted to engineering verification")
    if not engineering_only:
        required_paths = {
            "authority.approval_record": value["authority"].get("approval_record"),
            "authority.protocol_sha256": value["authority"].get("protocol_sha256"),
            "data.vintage_id": value["data"].get("vintage_id"),
            "data.snapshot_manifest": value["data"].get("snapshot_manifest"),
            "data.exposure_ledger": value["data"].get("exposure_ledger"),
            "validation.outer_windows_manifest": value["validation"].get("outer_windows_manifest"),
            "validation.inner_windows_manifest": value["validation"].get("inner_windows_manifest"),
        }
        unresolved = sorted(name for name, resolved in required_paths.items() if not resolved)
        if unresolved:
            raise ValueError(f"Approved study has unresolved fields: {unresolved}")
        final = value["validation"].get("final_holdout", {})
        if not final.get("sealed") or not final.get("manifest"):
            raise ValueError("Approved studies require a sealed final-holdout manifest")
    encoded = _canonical(value).encode("utf-8")
    return ResolvedStudy(value, hashlib.sha256(encoded).hexdigest(), engineering_only)
