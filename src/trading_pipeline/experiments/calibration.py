"""Score-blind runtime calibration and metadata-declared capability evidence.

This module never reads predictions, labels, objective values or rankings.  It
turns timed bridge-smoke records and hash-pinned verification records into the
two immutable gates required before an expanded study can be approved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import load_study
from .default_registry import default_registry
from .budget_tiers import TIERS


FORBIDDEN_OUTCOME_KEYS = {
    "ic", "mean_ic", "rmse", "mae", "sharpe", "return", "reward",
    "objective", "loss", "prediction", "rank", "ranking", "metric",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _contains_outcome(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalised = str(key).lower().replace("-", "_")
            words = set(normalised.split("_"))
            if normalised in FORBIDDEN_OUTCOME_KEYS or words & FORBIDDEN_OUTCOME_KEYS:
                return True
            if _contains_outcome(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_outcome(item) for item in value)
    return False


def write_score_blind_calibration(
    *, prepared_study: str | Path, records: Sequence[Mapping[str, Any]],
    deadline_seconds: float, output_path: str | Path,
) -> dict[str, Any]:
    """Freeze the largest complete budget tier supported by timed bridge smokes.

    One record is required for every arm.  Records may contain resource and
    shape telemetry only; any outcome-like field is rejected recursively.
    Estimates deliberately assume the complete inner-fold/seed matrix and, for
    RL, all three risk scenarios.  No result value influences the chosen tier.
    """
    study = load_study(prepared_study, allow_engineering_draft=True)
    if not study.engineering_only:
        raise ValueError("Calibration must precede study approval")
    if not isinstance(deadline_seconds, (int, float)) or deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    arms = {arm["id"]: arm for arm in study.config["experiment_arms"]}
    by_arm = {record.get("arm_id"): dict(record) for record in records}
    if len(by_arm) != len(records) or set(by_arm) != set(arms):
        raise ValueError("Calibration must cover every declared arm exactly once")
    required = {"arm_id", "component_id", "wall_seconds", "fit_rows",
                "evaluation_rows", "device", "status", "evidence_sha256"}
    for arm_id, record in by_arm.items():
        if _contains_outcome(record):
            raise ValueError("Score-blind calibration cannot contain outcomes or rankings")
        if not required <= set(record) or record["component_id"] != arms[arm_id]["component_id"]:
            raise ValueError(f"Incomplete or mismatched calibration record: {arm_id}")
        if (record["status"] != "complete" or record["wall_seconds"] <= 0
                or record["fit_rows"] < 1 or record["evaluation_rows"] < 1
                or not isinstance(record["evidence_sha256"], str)
                or len(record["evidence_sha256"]) != 64):
            raise ValueError(f"Invalid calibration evidence: {arm_id}")
    outer = int(study.config["validation"]["window_policy"]["outer_folds"])
    inner = int(study.config["validation"]["window_policy"]["inner_folds"])
    seeds = len(study.config["reproducibility"]["seeds"])
    estimates: dict[str, float] = {}
    registry = default_registry()
    for tier, (classical_budget, deep_budget, rl_budget) in TIERS.items():
        total = 0.0
        for arm_id, arm in arms.items():
            spec = registry.spec(arm["component_id"])
            if spec.interface != arm["interface"]:
                raise ValueError(f"Calibration interface mismatch: {arm_id}")
            if spec.interface == "RLPolicy":
                cells = rl_budget * outer * inner * seeds * 3
            elif spec.interface == "SupervisedModel" and spec.capabilities.get("sequence_view"):
                cells = deep_budget * outer * inner * seeds
            elif spec.interface == "SupervisedModel":
                cells = classical_budget * outer * inner * seeds
            else:
                raise ValueError(f"No score-bearing calibration interface: {arm_id}")
            total += float(by_arm[arm_id]["wall_seconds"]) * cells
        estimates[tier] = total
    selected = next((tier for tier in TIERS if estimates[tier] <= deadline_seconds), None)
    if selected is None:
        raise RuntimeError("No complete declared budget tier fits the score-blind deadline")
    output = Path(output_path)
    payload = {
        "schema_version": 1,
        "study_id": study.config["study_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_blind": True,
        "outcome_rankings_opened": False,
        "deadline_seconds": float(deadline_seconds),
        "budget_tier": selected,
        "estimated_seconds_by_tier": estimates,
        "estimation_policy": {
            "formula": "sum_per_arm_timed_cell_seconds_x_tier_proposals_x_outer_folds_x_inner_folds_x_seeds_x_rl_scenarios",
            "timing_source": "caller_supplied_score_blind_per_arm_records",
            "timing_scope_requirement": "full_vintage_fit_shape_representative_of_declared_windows",
            "shape_extrapolation": "none; smaller-universe bridge-smoke timings are insufficient for a full-vintage deadline estimate",
            "outer_folds": outer,
            "inner_folds_per_outer": inner,
            "seeds": seeds,
            "rl_scenarios": 3,
            "excluded_work": "data preparation, final fits, holdout evaluation, reports, and retry overhead",
        },
        "records": [by_arm[arm_id] for arm_id in sorted(by_arm)],
    }
    _write_new(output, payload)
    return payload


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Capability evidence is outside repository: {resolved}")
    return resolved.relative_to(root.resolve()).as_posix()


def _evidence(path: str | Path, *, root: Path, study_id: str, arm_id: str,
              evidence_kind: str) -> dict[str, str]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"Missing capability evidence: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("study_id") != study_id
            or value.get("arm_id") != arm_id or value.get("status") != "passed"
            or value.get("evidence_kind") != evidence_kind):
        raise ValueError(f"Invalid {evidence_kind} evidence for {arm_id}")
    return {"path": _relative(source, root), "sha256": _hash(source)}


def write_capability_gate(
    *, repository_root: str | Path, prepared_study: str | Path,
    calibration_path: str | Path,
    synthetic_evidence: Mapping[str, str | Path],
    real_bridge_evidence: Mapping[str, str | Path], output_path: str | Path,
) -> dict[str, Any]:
    """Create an exact declared-arm gate from immutable per-arm evidence files."""
    root = Path(repository_root).resolve()
    study = load_study(prepared_study, allow_engineering_draft=True)
    arms = {arm["id"]: arm for arm in study.config["experiment_arms"]}
    if not arms or set(synthetic_evidence) != set(arms) or set(real_bridge_evidence) != set(arms):
        raise ValueError("Capability evidence must cover all declared arms exactly")
    calibration = Path(calibration_path).resolve()
    calibration_value = json.loads(calibration.read_text(encoding="utf-8"))
    if (calibration_value.get("study_id") != study.config["study_id"]
            or calibration_value.get("score_blind") is not True
            or calibration_value.get("outcome_rankings_opened") is not False
            or calibration_value.get("budget_tier") not in TIERS):
        raise ValueError("Invalid score-blind calibration evidence")
    capabilities = {}
    for arm_id in sorted(arms):
        capabilities[arm_id] = {
            "component_id": arms[arm_id]["component_id"],
            "interface": arms[arm_id]["interface"],
            "synthetic_verification_passed": True,
            "real_data_bridge_passed": True,
            "synthetic_evidence": _evidence(synthetic_evidence[arm_id], root=root,
                study_id=study.config["study_id"], arm_id=arm_id,
                evidence_kind="synthetic_verification"),
            "real_bridge_evidence": _evidence(real_bridge_evidence[arm_id], root=root,
                study_id=study.config["study_id"], arm_id=arm_id,
                evidence_kind="real_data_bridge_smoke"),
        }
    payload = {
        "schema_version": 1,
        "study_id": study.config["study_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_blind_calibration": True,
        "outcome_rankings_opened": False,
        "budget_tier": calibration_value["budget_tier"],
        "calibration_evidence": {"path": _relative(calibration, root), "sha256": _hash(calibration)},
        "arm_capabilities": capabilities,
    }
    _write_new(Path(output_path), payload)
    return payload


def _json_file(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--prepared-study", required=True)
    calibrate.add_argument("--records", required=True,
                           help="JSON array of resource-only per-arm smoke records")
    calibrate.add_argument("--deadline-seconds", required=True, type=float)
    calibrate.add_argument("--output", required=True)
    gate = commands.add_parser("gate")
    gate.add_argument("--repository-root", default=".")
    gate.add_argument("--prepared-study", required=True)
    gate.add_argument("--calibration", required=True)
    gate.add_argument("--synthetic-evidence", required=True,
                      help="JSON object mapping arm ID to evidence path")
    gate.add_argument("--real-bridge-evidence", required=True,
                      help="JSON object mapping arm ID to evidence path")
    gate.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        result = write_score_blind_calibration(prepared_study=args.prepared_study,
            records=_json_file(args.records), deadline_seconds=args.deadline_seconds,
            output_path=args.output)
    else:
        result = write_capability_gate(repository_root=args.repository_root,
            prepared_study=args.prepared_study, calibration_path=args.calibration,
            synthetic_evidence=_json_file(args.synthetic_evidence),
            real_bridge_evidence=_json_file(args.real_bridge_evidence),
            output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _cli()
