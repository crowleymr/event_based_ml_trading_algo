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


def protocol_content_sha256(value: Mapping[str, Any]) -> str:
    """Hash the complete protocol with its self-referential hash field blanked."""
    snapshot = json.loads(_canonical(value))
    snapshot["authority"]["protocol_sha256"] = None
    return hashlib.sha256(_canonical(snapshot).encode("utf-8")).hexdigest()


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
    if any(arm["interface"] == "RLPolicy" for arm in arms):
        inputs = value.get("data", {}).get("rl_inputs", {})
        if inputs.get("eligibility_policy") == "complete_causal_upstream_dates_plus_observed_price_history_v2":
            if inputs.get("price_eligibility_policy") != "observed_history_and_endpoint_valuation_v1":
                raise ValueError("Revised RL eligibility requires its explicit price_eligibility_policy")
        elif inputs.get("price_eligibility_policy") == "observed_history_and_endpoint_valuation_v1":
            raise ValueError("Revised RL price policy requires the paired eligibility_policy")
    engineering_only = value["status"] == "draft"
    if engineering_only and not allow_engineering_draft:
        raise ValueError("Draft studies are restricted to engineering verification")
    if not engineering_only:
        for name in ("authority", "data", "validation", "search", "selection", "reproducibility"):
            if not isinstance(value[name], dict):
                raise ValueError(f"Approved study {name} must be a mapping")
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
        if value["authority"]["protocol_sha256"] != protocol_content_sha256(value):
            raise ValueError("Approved study protocol_sha256 does not match frozen content")
        if value["search"].get("real_data_execution") != "enabled":
            raise ValueError("Approved study must explicitly enable real_data_execution")
        if value["selection"].get("final_test_selection") != "forbidden":
            raise ValueError("Final holdout cannot be used for selection")
        if value["data"].get("forbidden_selection_role") != "legacy_observed_final":
            raise ValueError("Legacy observed final must be excluded from selection")
        seeds = value["reproducibility"].get("seeds")
        if (not isinstance(seeds, list) or not seeds
                or any(type(seed) is not int for seed in seeds)
                or len(seeds) != len(set(seeds))):
            raise ValueError("Approved study requires unique integer seeds")
    encoded = _canonical(value).encode("utf-8")
    return ResolvedStudy(value, hashlib.sha256(encoded).hexdigest(), engineering_only)
