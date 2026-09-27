"""Freeze a fresh-vintage exploratory study without reading model outcomes.

This is a preparation command. Score-bearing execution remains in trading_pipeline.run.
The output directory must be new; source and completed run artefacts are read only.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from .schema import load_study, protocol_content_sha256
from .authority import verify_study_authority
from .budget_tiers import TIERS
from trading_pipeline.optimisation.splits import _folds, TemporalFold


STUDY_ID = "expanded_closeout_exploratory_v1"
VINTAGE_ID = "expanded_20260926_v1"
CLAIM_STATUS = "exploratory_closeout_not_confirmatory"
LEGACY_ROLE = "legacy_observed_final"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Study authority input must be inside repository: {path}")
    return resolved.relative_to(root.resolve()).as_posix()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _iso(days: tuple[date, ...]) -> list[str]:
    return [day.isoformat() for day in days]


def _partition(fold: TemporalFold, ends: dict[date, date], stopping_sessions: int) -> dict[str, Any]:
    if len(fold.fit_dates) <= stopping_sessions:
        raise ValueError(f"Stopping tail exhausts {fold.fold_id}")
    stopping = fold.fit_dates[-stopping_sessions:]
    fit = tuple(day for day in fold.fit_dates[:-stopping_sessions]
                if ends[day] < stopping[0])
    if not fit or any(ends[day] >= fold.score_dates[0] for day in stopping):
        raise ValueError(f"Information interval crosses a {fold.fold_id} boundary")
    return {
        "fold_id": fold.fold_id,
        "fit_dates": _iso(fit),
        "stopping_dates": _iso(stopping),
        "embargo_dates": _iso(fold.embargo_dates),
        "score_dates": _iso(fold.score_dates),
        "purged_dates": _iso(tuple(day for day in fold.fit_dates[:-stopping_sessions]
                                   if day not in fit)),
    }


def _session_exposure(features: Path) -> tuple[list[dict[str, Any]], dict[date, date]]:
    frame = pl.scan_parquet(features)
    required = {"session_date", "security_id", "label_end_date", "forward_return_5d"}
    columns = set(frame.collect_schema().names())
    if not required <= columns:
        raise ValueError(f"Feature snapshot lacks exposure columns: {sorted(required - columns)}")
    usable = frame.filter(pl.col("forward_return_5d").is_not_null()).select(
        "session_date", "security_id", "label_end_date").collect()
    if usable.is_empty() or usable.select(["session_date", "security_id"]).is_duplicated().any():
        raise ValueError("Feature exposure keys are empty or duplicated")
    if usable.filter(pl.any_horizontal(pl.col("session_date").is_null(),
                                      pl.col("security_id").is_null(),
                                      pl.col("label_end_date").is_null())).height:
        raise ValueError("Usable feature exposure has null keys or label end")
    if usable.filter(pl.col("label_end_date") <= pl.col("session_date")).height:
        raise ValueError("Label information interval must end after its session")
    grouped = usable.group_by("session_date").agg(
        pl.col("label_end_date").max().alias("max_label_end_date"),
        pl.col("security_id").n_unique().alias("security_count"),
        pl.len().alias("row_count"),
    ).sort("session_date")
    rows = [{"session_date": str(row["session_date"]),
             "max_label_end_date": str(row["max_label_end_date"]),
             "security_count": row["security_count"], "row_count": row["row_count"]}
            for row in grouped.to_dicts()]
    ends = {date.fromisoformat(row["session_date"]): date.fromisoformat(row["max_label_end_date"])
            for row in rows}
    return rows, ends


def freeze_study_manifests(
    *, repository_root: str | Path, source_manifest: str | Path,
    vintage_manifest: str | Path, feature_manifest: str | Path,
    feature_path: str | Path, output_dir: str | Path,
    draft_study: str | Path = "configs/studies/expanded_closeout_draft.yaml",
    prepared_study: str | Path = "configs/studies/expanded_closeout_prepared_v2.yaml",
    holdout_sessions: int = 60, outer_folds: int = 5,
    outer_score_sessions: int = 20, inner_folds: int = 3,
    inner_score_sessions: int = 10, min_fit_sessions: int = 120,
    stopping_sessions: int = 10, embargo_sessions: int = 5,
    causal_stack_manifest: str | Path | None = None,
    benchmark_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Produce pinned inputs and a non-executable self-hashed checkpoint."""
    root = Path(repository_root).resolve()
    source = Path(source_manifest).resolve()
    vintage = Path(vintage_manifest).resolve()
    feature = Path(feature_manifest).resolve()
    features = Path(feature_path).resolve()
    destination = Path(output_dir).resolve()
    prepared_path = Path(prepared_study)
    if not prepared_path.is_absolute():
        prepared_path = root / prepared_path
    draft_path = Path(draft_study)
    if not draft_path.is_absolute():
        draft_path = root / draft_path
    for path in (source, vintage, feature, features, draft_path):
        _relative(path, root)
        if not path.is_file():
            raise ValueError(f"Required authority input missing: {path}")
    _relative(destination, root)
    _relative(prepared_path, root)
    if destination.exists() or prepared_path.exists():
        raise ValueError("Protocol output already exists; use a fresh destination and study ID")
    source_data, vintage_data, feature_data = _json(source), _json(vintage), _json(feature)
    if source_data.get("admission_snapshot_id") != "sp500-20260926-v3":
        raise ValueError("Only sealed v3 admission may authorize this study")
    admission = Path(source_data["admission_ledger"])
    if not admission.is_file() or _hash(admission) != source_data["admission_ledger_sha256"]:
        raise ValueError("Sealed admission ledger hash mismatch")
    if not _json(admission).get("admitted_count") == source_data.get("admitted_count"):
        raise ValueError("Derived universe count differs from sealed admission")
    # The vintage producer must explicitly declare each consumed input. A missing
    # pin is a failure, even if an undeclared file happens to exist locally.
    if vintage_data.get("source_manifest_sha256") != _hash(source):
        raise ValueError("Derived source manifest hash missing or mismatched in data vintage")
    feature_artifact = feature_data.get("artifacts", {}).get("features", {})
    if (feature_artifact.get("sha256") != _hash(features)
            or Path(feature_artifact.get("path", "")).resolve() != features):
        raise ValueError("Feature artifact path/hash missing or mismatched in feature manifest")
    feature_columns = feature_data.get("feature_columns")
    if (not isinstance(feature_columns, dict) or not all(
        isinstance(feature_columns.get(key), list) and feature_columns[key]
        for key in ("F0", "F1"))):
        raise ValueError("Feature manifest must declare F0 and F1 columns")
    rows, ends = _session_exposure(features)
    calendar = tuple(ends)
    needed = holdout_sessions + outer_folds * outer_score_sessions + min_fit_sessions
    if (len(calendar) < needed or min(holdout_sessions, outer_score_sessions,
            inner_score_sessions, min_fit_sessions, stopping_sessions,
            embargo_sessions) < 1 or min(outer_folds, inner_folds) < 2):
        raise ValueError("Insufficient labelled sessions or invalid fixed window sizes")
    holdout = calendar[-holdout_sessions:]
    # A separate five-session gap seals the holdout. Historical labels that
    # extend into the gap are removed from the development calendar as well.
    pre_holdout = calendar[:-holdout_sessions]
    if len(pre_holdout) <= embargo_sessions:
        raise ValueError("Holdout embargo exhausts development sessions")
    holdout_embargo = pre_holdout[-embargo_sessions:]
    development = tuple(day for day in pre_holdout[:-embargo_sessions]
                        if ends[day] < holdout[0])
    outer_raw = _folds(development, ends, prefix="outer_", fold_count=outer_folds,
                       score_sessions=outer_score_sessions, min_fit_sessions=min_fit_sessions,
                       embargo_sessions=embargo_sessions)
    outer_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    for outer in outer_raw:
        partition = _partition(outer, ends, stopping_sessions)
        outer_rows.append(partition)
        inner_calendar = tuple(date.fromisoformat(item) for item in partition["fit_dates"])
        inner_raw = _folds(inner_calendar, ends, prefix=f"{outer.fold_id}_inner_",
                           fold_count=inner_folds, score_sessions=inner_score_sessions,
                           min_fit_sessions=min_fit_sessions, embargo_sessions=embargo_sessions)
        for inner in inner_raw:
            child = _partition(inner, ends, stopping_sessions)
            child["outer_fold_id"] = outer.fold_id
            inner_rows.append(child)
    draft = yaml.safe_load(draft_path.read_text(encoding="utf-8"))
    if (not isinstance(draft, dict)
            or draft.get("study_id") != STUDY_ID
            or draft.get("status") != "draft"):
        raise ValueError("Expected unchanged expanded closeout engineering draft")
    approval_relative = draft.get("authority", {}).get("approval_record")
    if not isinstance(approval_relative, str):
        raise ValueError("Draft requires an immutable human approval record")
    approval = (root / approval_relative).resolve()
    if not approval.is_relative_to(root) or not approval.is_file():
        raise ValueError("Human approval record missing or outside repository")
    stamp = datetime.now(timezone.utc).isoformat()
    common = {"schema_version": 1, "study_id": STUDY_ID, "vintage_id": VINTAGE_ID,
              "generated_at_utc": stamp, "generator_sha256": _hash(Path(__file__)),
              "source_manifest_path": _relative(source, root), "source_manifest_sha256": _hash(source),
              "data_vintage_manifest_path": _relative(vintage, root),
              "data_vintage_manifest_sha256": _hash(vintage),
              "feature_manifest_path": _relative(feature, root), "feature_manifest_sha256": _hash(feature),
              "features_path": _relative(features, root), "features_sha256": _hash(features)}
    destination.mkdir(parents=True)
    paths = {name: destination / f"{name}.json" for name in
             ("snapshot_manifest", "exposure_ledger", "outer_windows", "inner_windows", "final_holdout")}
    _write_json(paths["snapshot_manifest"], {**common, "data_role": "expanded_fresh_vintage",
             "admission_snapshot_id": source_data["admission_snapshot_id"],
             "admitted_count": source_data["admitted_count"],
             "feature_path": _relative(features, root), "feature_sha256": _hash(features),
             "source_as_of": _json(Path(source_data["admission_manifest"]))["source_as_of"],
             "legacy_comparison": "protocol_incompatible"})
    _write_json(paths["exposure_ledger"], {**common,
             "forbidden_selection_roles": [LEGACY_ROLE],
             "information_interval": "session_date_through_max_label_end_date_inclusive",
             "sessions": rows})
    _write_json(paths["outer_windows"], {**common, "embargo_sessions": embargo_sessions,
             "stopping_sessions": stopping_sessions, "folds": outer_rows})
    _write_json(paths["inner_windows"], {**common, "embargo_sessions": embargo_sessions,
             "stopping_sessions": stopping_sessions, "folds": inner_rows})
    _write_json(paths["final_holdout"], {**common, "sealed": True,
             "requires_selection_lock": True, "claim_status": CLAIM_STATUS,
             "session_dates": _iso(holdout),
             "pre_holdout_embargo_dates": _iso(holdout_embargo)})
    config = json.loads(json.dumps(draft))
    config["authority"]["approval_record"] = _relative(approval, root)
    config["data"].update({"vintage_id": VINTAGE_ID,
        "snapshot_manifest": _relative(paths["snapshot_manifest"], root),
        "exposure_ledger": _relative(paths["exposure_ledger"], root),
        "feature_columns": feature_columns})
    config["validation"].update({
        "outer_windows_manifest": _relative(paths["outer_windows"], root),
        "inner_windows_manifest": _relative(paths["inner_windows"], root),
        "embargo_sessions": embargo_sessions,
        "window_policy": {"holdout_sessions": holdout_sessions, "outer_folds": outer_folds,
            "outer_score_sessions": outer_score_sessions, "inner_folds": inner_folds,
            "inner_score_sessions": inner_score_sessions, "min_fit_sessions": min_fit_sessions,
            "stopping_sessions": stopping_sessions}})
    config["validation"]["final_holdout"]["manifest"] = _relative(paths["final_holdout"], root)
    if benchmark_manifest is not None:
        benchmark_path = Path(benchmark_manifest).resolve()
        _relative(benchmark_path, root)
        benchmark = _json(benchmark_path)
        benchmark_bars = (root / str(benchmark.get("bars_path", ""))).resolve()
        referenced_holdout = (root / str(benchmark.get("holdout_manifest_path", ""))).resolve()
        if (benchmark.get("benchmark_id") != "B0-SPY"
                or benchmark.get("ticker") != "SPY"
                or benchmark.get("coverage") != "complete_sealed_holdout_calendar"
                or not benchmark_bars.is_relative_to(root)
                or not benchmark_bars.is_file()
                or _hash(benchmark_bars) != benchmark.get("bars_sha256")
                or not referenced_holdout.is_relative_to(root)
                or not referenced_holdout.is_file()
                or _hash(referenced_holdout) != benchmark.get("holdout_manifest_sha256")):
            raise ValueError("Benchmark manifest is absent, incomplete or hash-mismatched")
        referenced = _json(referenced_holdout)
        if (referenced.get("sealed") is not True
                or referenced.get("session_dates") != _iso(holdout)):
            raise ValueError("Benchmark was not prepared for the frozen holdout calendar")
        benchmark_frame = pl.read_parquet(benchmark_bars)
        if (benchmark_frame.filter(pl.col("session_date").is_in(list(holdout)))
                .select("session_date").unique().height != len(holdout)):
            raise ValueError("Benchmark bars do not cover the frozen holdout calendar")
        config["data"]["benchmark_contract"] = {
            "benchmark_id": benchmark["benchmark_id"],
            "manifest_path": _relative(benchmark_path, root),
            "manifest_sha256": _hash(benchmark_path),
            "bars_path": _relative(benchmark_bars, root),
            "bars_sha256": _hash(benchmark_bars),
        }
    if causal_stack_manifest is not None:
        stack_path = Path(causal_stack_manifest).resolve()
        _relative(stack_path, root)
        stack = _json(stack_path)
        if (stack.get("study_id") != STUDY_ID
                or stack.get("feature_sha256") != _hash(features)
                or stack.get("protected_labels_used_for_fit") is not False):
            raise ValueError("Causal stack manifest is incompatible with the prepared study")
        required_stack_files = {
            "table": (stack.get("table_path"), stack.get("table_sha256")),
            "source": (stack.get("source_prediction_path"), stack.get("source_prediction_sha256")),
        }
        verified_stack = {}
        for name, (relative, expected) in required_stack_files.items():
            candidate = (root / str(relative)).resolve()
            if (not candidate.is_relative_to(root) or not candidate.is_file()
                    or _hash(candidate) != expected):
                raise ValueError(f"Causal stack {name} artifact is absent or hash-mismatched")
            verified_stack[name] = candidate
        columns = stack.get("feature_columns")
        if (not isinstance(columns, list) or not set(feature_columns["F1"]) < set(columns)
                or len(columns) != len(set(columns))):
            raise ValueError("Causal stack must explicitly extend F1")
        config["data"]["augmented_feature_contract"] = {
            "feature_path": _relative(verified_stack["table"], root),
            "feature_sha256": _hash(verified_stack["table"]),
            "feature_columns": columns,
            "source_manifest": _relative(stack_path, root),
            "source_manifest_sha256": _hash(stack_path),
        }
        config["data"]["rl_inputs"].update({
            "model_outputs_path": _relative(verified_stack["source"], root),
            "model_outputs_sha256": _hash(verified_stack["source"]),
            "bars_path": _relative(features, root),
            "bars_sha256": _hash(features),
        })
    config["search"]["real_data_execution"] = "forbidden_until_manifests_and_component_enablement"
    required_inputs = [approval, *paths.values()]
    config["authority"]["input_sha256"] = {_relative(path, root): _hash(path)
                                               for path in required_inputs}
    config["authority"]["protocol_sha256"] = protocol_content_sha256(config)
    # Exercise the same six-pin, study-ID and temporal gate that execution will
    # use, while keeping the written checkpoint in forbidden draft state.
    from .schema import ResolvedStudy
    verify_study_authority(ResolvedStudy(config, config["authority"]["protocol_sha256"], False),
                           repository_root=root)
    with prepared_path.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    load_study(prepared_path, allow_engineering_draft=True)
    return {"study_path": _relative(prepared_path, root), "protocol_sha256":
            config["authority"]["protocol_sha256"], "input_sha256": config["authority"]["input_sha256"]}


def finalise_approved_study(
    *, repository_root: str | Path, prepared_study: str | Path,
    capability_gate: str | Path, approved_study: str | Path,
) -> dict[str, Any]:
    """Approve only after every declared arm and score-blind budget gate passes.

    The capability gate is itself a generated, hash-pinned evidence record. It
    must name all arms and the exact calibration evidence; partial readiness
    cannot silently reduce the approved matrix or open the holdout.
    """
    root = Path(repository_root).resolve()
    prepared_path = Path(prepared_study).resolve()
    gate_path = Path(capability_gate).resolve()
    output_path = Path(approved_study).resolve()
    for path in (prepared_path, gate_path, output_path):
        _relative(path, root)
    if output_path.exists():
        raise ValueError("Approved protocol already exists")
    prepared = load_study(prepared_path)
    config = json.loads(json.dumps(prepared.config))
    if not prepared.engineering_only or config["study_id"] != STUDY_ID:
        raise ValueError("Finalisation requires the prepared expanded-study draft")
    if config["authority"]["protocol_sha256"] != protocol_content_sha256(config):
        raise ValueError("Prepared protocol self-hash mismatch")
    gate = _json(gate_path)
    arm_ids = {arm["id"] for arm in config["experiment_arms"]}
    if not arm_ids or gate.get("study_id") != STUDY_ID:
        raise ValueError("Capability gate does not cover the declared study")
    capabilities = gate.get("arm_capabilities")
    if not isinstance(capabilities, dict) or set(capabilities) != arm_ids:
        raise ValueError("Capability gate must cover every declared arm exactly")
    if any(not isinstance(item, dict) or item.get("real_data_bridge_passed") is not True
           or item.get("synthetic_verification_passed") is not True
           or item.get("component_id") != arm["component_id"]
           or item.get("interface") != arm["interface"]
           for arm in config["experiment_arms"] for item in [capabilities[arm["id"]]]):
        raise ValueError("Partial arm capability cannot approve expanded study")
    if gate.get("score_blind_calibration") is not True or gate.get("outcome_rankings_opened") is not False:
        raise ValueError("Budget tier requires score-blind calibration before outcome rankings")
    calibration = gate.get("calibration_evidence")
    if not isinstance(calibration, dict):
        raise ValueError("Calibration evidence path/hash required")
    calibration_path = root / calibration.get("path", "")
    if (not calibration_path.is_file() or _relative(calibration_path, root) != calibration.get("path")
            or _hash(calibration_path) != calibration.get("sha256")):
        raise ValueError("Calibration evidence hash mismatch")
    tier = gate.get("budget_tier")
    if tier not in TIERS:
        raise ValueError("Capability gate lacks a declared complete budget tier")
    classical, deep, rl = TIERS[tier]
    config["search"]["policy_id"] = f"{tier}_deadline_tier_v1"
    config["search"]["budgets"] = {
        "classical_per_family": classical, "deep_supervised_per_family": deep,
        "rl_per_family_per_risk_scenario": rl,
    }
    config["status"] = "approved"
    config["search"]["real_data_execution"] = "enabled"
    config["authority"]["capability_gate"] = _relative(gate_path, root)
    config["authority"]["capability_gate_sha256"] = _hash(gate_path)
    config["authority"]["prepared_protocol_sha256"] = _hash(prepared_path)
    config["authority"]["protocol_sha256"] = protocol_content_sha256(config)
    # Verify all six pinned authority inputs and temporal separation before
    # creating the approved file. No score-bearing data is read by this gate.
    from .schema import ResolvedStudy
    verify_study_authority(ResolvedStudy(config, config["authority"]["protocol_sha256"], False),
                           repository_root=root)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    load_study(output_path, allow_engineering_draft=False)
    return {"study_path": _relative(output_path, root),
            "protocol_sha256": config["authority"]["protocol_sha256"],
            "capability_gate_sha256": _hash(gate_path)}


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--vintage-manifest", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared-study", default="configs/studies/expanded_closeout_prepared_v2.yaml")
    parser.add_argument("--causal-stack-manifest")
    parser.add_argument("--benchmark-manifest")
    parser.add_argument("--holdout-sessions", type=int, default=60)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-score-sessions", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--inner-score-sessions", type=int, default=10)
    parser.add_argument("--min-fit-sessions", type=int, default=120)
    parser.add_argument("--stopping-sessions", type=int, default=10)
    parser.add_argument("--embargo-sessions", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(freeze_study_manifests(repository_root=args.repository_root,
        source_manifest=args.source_manifest, vintage_manifest=args.vintage_manifest,
        feature_manifest=args.feature_manifest, feature_path=args.features,
        output_dir=args.output, prepared_study=args.prepared_study,
        causal_stack_manifest=args.causal_stack_manifest,
        benchmark_manifest=args.benchmark_manifest,
        holdout_sessions=args.holdout_sessions,
        outer_folds=args.outer_folds,
        outer_score_sessions=args.outer_score_sessions,
        inner_folds=args.inner_folds,
        inner_score_sessions=args.inner_score_sessions,
        min_fit_sessions=args.min_fit_sessions,
        stopping_sessions=args.stopping_sessions,
        embargo_sessions=args.embargo_sessions), indent=2))


if __name__ == "__main__":
    _cli()
