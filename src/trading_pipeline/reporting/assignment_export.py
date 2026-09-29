"""Assignment-specific evidence derived from one audited expanded study.

The exporter is a read-only consumer: it verifies the immutable run and its WP7
report, derives presentation tables, and writes only to a new versioned directory
under ``reports/assignment``.  It never trains, selects, or mutates source evidence.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import polars as pl

from trading_pipeline.modelling.evaluate import metrics as prediction_metrics
from trading_pipeline.reporting.expanded_closeout import TABLES as WP7_TABLES
from trading_pipeline.reporting.expanded_closeout import _verified_run


FEATURE_DEFINITIONS = {
    "return_5d": ("market", 5, "Adjusted-close return over the prior five sessions", "simple return"),
    "return_10d": ("market", 10, "Adjusted-close return over the prior ten sessions", "simple return"),
    "return_20d": ("market", 20, "Adjusted-close return over the prior twenty sessions", "simple return"),
    "vol_5d": ("market", 5, "Realised volatility of daily returns over five sessions", "rolling standard deviation"),
    "vol_20d": ("market", 20, "Realised volatility of daily returns over twenty sessions", "rolling standard deviation"),
    "avg_dollar_volume_20d": ("market", 20, "Mean adjusted close multiplied by volume over twenty sessions", "rolling mean"),
    "volume_ratio_20d": ("market", 20, "Current volume relative to its twenty-session mean", "ratio"),
    "ma_distance_20d": ("market", 20, "Adjusted close relative to its twenty-session moving average", "relative distance"),
    "price_to_52w_high": ("market", 252, "Adjusted close relative to the trailing 252-session high", "ratio"),
    "latest_eps": ("fundamental", None, "Latest point-in-time eligible SEC earnings-per-share fact", "as-of join by filed date"),
    "latest_net_income": ("fundamental", None, "Latest point-in-time eligible SEC net-income fact", "as-of join by filed date"),
    "upstream_prediction_0": ("causal_upstream", None, "Causal out-of-fold equal-weight upstream prediction", "train-only median imputation plus mask"),
    "upstream_prediction_1": ("causal_upstream", None, "Causal out-of-fold Elastic Net upstream prediction", "train-only median imputation plus mask"),
    "upstream_prediction_2": ("causal_upstream", None, "Causal out-of-fold HistGBT upstream prediction", "train-only median imputation plus mask"),
    "upstream_prediction_3": ("causal_upstream", None, "Causal out-of-fold XGBoost upstream prediction", "train-only median imputation plus mask"),
    "upstream_prediction_0_is_available": ("availability_mask", None, "Availability indicator for upstream prediction 0", "boolean mask"),
    "upstream_prediction_1_is_available": ("availability_mask", None, "Availability indicator for upstream prediction 1", "boolean mask"),
    "upstream_prediction_2_is_available": ("availability_mask", None, "Availability indicator for upstream prediction 2", "boolean mask"),
    "upstream_prediction_3_is_available": ("availability_mask", None, "Availability indicator for upstream prediction 3", "boolean mask"),
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or out-of-root input: {relative}")
    return path


def _verify_wp7(root: Path, report: Path, run_id: str, metadata: dict) -> tuple[dict, dict[str, str]]:
    if not report.is_relative_to(root / "reports") or not report.is_dir():
        raise ValueError("WP7 report must be an existing directory under reports/")
    manifest_path = report / "research_evidence_manifest.json"
    manifest = _json(manifest_path)
    if (manifest.get("schema_version") != 1
            or manifest.get("source_run_id") != run_id
            or manifest.get("source_run_audit_sha256") != _hash(
                root / "runs" / "expanded_closeout" / run_id / "audit.json")
            or manifest.get("source_run_completion_sha256") != _hash(
                root / "runs" / "expanded_closeout" / run_id / "completion.json")
            or manifest.get("protocol_sha256") != metadata.get("protocol_sha256")
            or manifest.get("claim_status") != "exploratory_closeout_not_confirmatory"):
        raise ValueError("WP7 report does not match the completed expanded run")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("WP7 manifest lacks an output inventory")
    declared = {}
    for item in outputs:
        name, sha, rows = item.get("relative_path"), item.get("sha256"), item.get("rows")
        if not isinstance(name, str) or name in declared or not isinstance(sha, str):
            raise ValueError("Invalid or duplicate WP7 output declaration")
        path = report / name
        if not path.is_file() or _hash(path) != sha:
            raise ValueError(f"WP7 output hash mismatch: {name}")
        if path.suffix != ".parquet" or pl.scan_parquet(path).select(pl.len()).collect().item() != rows:
            raise ValueError(f"WP7 output row-count mismatch: {name}")
        declared[name] = sha
    expected = {f"{name}.parquet" for name in WP7_TABLES}
    actual = {path.name for path in report.glob("*.parquet")}
    if set(declared) != expected or actual != expected:
        raise ValueError("WP7 output inventory is incomplete or contains undeclared tables")
    return manifest, declared


def _feature_inputs(root: Path, protocol: dict) -> tuple[pl.DataFrame, list[str], dict[str, str]]:
    contract = protocol.get("data", {}).get("augmented_feature_contract")
    if not isinstance(contract, dict):
        raise ValueError("Approved protocol lacks the augmented feature contract")
    relative = contract.get("feature_path")
    source_relative = contract.get("source_manifest")
    feature = _inside(root, relative)
    source = _inside(root, source_relative)
    if (_hash(feature) != contract.get("feature_sha256")
            or _hash(source) != contract.get("source_manifest_sha256")):
        raise ValueError("Augmented feature contract hash mismatch")
    source_manifest = _json(source)
    if (source_manifest.get("table_path") != relative
            or source_manifest.get("table_sha256") != contract.get("feature_sha256")
            or source_manifest.get("feature_columns") != contract.get("feature_columns")):
        raise ValueError("Augmented feature manifest differs from the approved protocol")
    columns = contract["feature_columns"]
    if set(columns) != set(FEATURE_DEFINITIONS):
        raise ValueError(f"Feature dictionary is incomplete: {sorted(set(columns) ^ set(FEATURE_DEFINITIONS))}")
    frame = pl.read_parquet(feature)
    required = {"security_id", "session_date", *columns}
    if not required <= set(frame.columns):
        raise ValueError(f"Augmented feature table lacks {sorted(required - set(frame.columns))}")
    if frame.select("security_id", "session_date").is_duplicated().any():
        raise ValueError("Augmented feature table has duplicate security/session keys")
    return frame, columns, {relative: contract["feature_sha256"],
                            source_relative: contract["source_manifest_sha256"]}


def _feature_dictionary(protocol: dict, columns: list[str]) -> pl.DataFrame:
    arms = sorted(arm["id"] for arm in protocol["experiment_arms"])
    feature_sets = sorted({arm["feature_set_id"] for arm in protocol["experiment_arms"]})
    rows = []
    for name in columns:
        group, lookback, definition, transform = FEATURE_DEFINITIONS[name]
        rows.append({"feature": name, "feature_group": group,
                     "definition": definition, "lookback_sessions": lookback,
                     "transform_or_availability_rule": transform,
                     "eligible_feature_sets_json": json.dumps(feature_sets),
                     "eligible_arms_json": json.dumps(arms),
                     "point_in_time_required": True})
    return pl.DataFrame(rows).sort("feature")


def _window_partitions(root: Path, protocol: dict) -> tuple[list[tuple[str, str, list[date]]], dict[str, str]]:
    outer_relative = protocol["validation"]["outer_windows_manifest"]
    holdout_relative = protocol["validation"]["final_holdout"]["manifest"]
    outer_path, holdout_path = _inside(root, outer_relative), _inside(root, holdout_relative)
    outer, holdout = _json(outer_path), _json(holdout_path)
    partitions = []
    for fold in outer.get("folds", []):
        for key, role in (("fit_dates", "training"), ("stopping_dates", "early_stopping"),
                          ("score_dates", "outer_validation")):
            dates = [date.fromisoformat(value) for value in fold.get(key, [])]
            if not dates:
                raise ValueError(f"Empty {key} in {fold.get('fold_id')}")
            partitions.append((f"{fold['fold_id']}::{role}", role, dates))
    holdout_dates = [date.fromisoformat(value) for value in holdout.get("session_dates", [])]
    if not holdout.get("sealed") or not holdout_dates:
        raise ValueError("Final holdout calendar is not sealed")
    partitions.append(("descriptive_holdout", "descriptive_holdout", holdout_dates))
    return partitions, {outer_relative: _hash(outer_path), holdout_relative: _hash(holdout_path)}


def _feature_statistics(frame: pl.DataFrame, columns: list[str],
                        partitions: list[tuple[str, str, list[date]]]) -> pl.DataFrame:
    rows = []
    for partition, role, dates in partitions:
        part = frame.filter(pl.col("session_date").is_in(dates))
        if part.is_empty():
            raise ValueError(f"No feature rows for {partition}")
        row_count = part.height
        for feature in columns:
            values = part.select(pl.col(feature).cast(pl.Float64, strict=False).alias("value"))
            finite = values.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
            finite_count = finite.height
            missing_count = row_count - finite_count
            stats = finite.select(
                pl.col("value").mean().alias("mean"),
                pl.col("value").std().alias("standard_deviation"),
                pl.col("value").quantile(0.01).alias("p01"),
                pl.col("value").quantile(0.25).alias("p25"),
                pl.col("value").median().alias("p50"),
                pl.col("value").quantile(0.75).alias("p75"),
                pl.col("value").quantile(0.99).alias("p99"),
                pl.col("value").min().alias("minimum"),
                pl.col("value").max().alias("maximum"),
            ).to_dicts()[0]
            p25, p50, p75 = stats["p25"], stats["p50"], stats["p75"]
            iqr = None if p25 is None or p75 is None else p75 - p25
            extreme_iqr = None
            if iqr is not None and iqr > 0 and p50 is not None:
                extreme_iqr = max(abs(stats["minimum"] - p50),
                                  abs(stats["maximum"] - p50)) / iqr
            rows.append({"partition": partition, "partition_role": role,
                         "feature": feature, "row_count": row_count,
                         "finite_count": finite_count, "missing_count": missing_count,
                         "missing_fraction": missing_count / row_count,
                         **stats, "extreme_distance_iqr": extreme_iqr,
                         "heavy_tail_flag": extreme_iqr is not None and extreme_iqr >= 10,
                         "heavy_tail_rule": "max absolute distance from median >= 10 IQR",
                         "count_unit": "security_sessions"})
    return pl.DataFrame(rows).sort("partition", "feature")


def _predictive_diagnostics(run: Path, protocol: dict) -> pl.DataFrame:
    predictions = pl.read_parquet(run / "predictions.parquet")
    required = {"arm_id", "partition", "fold_id", "seed", "session_date",
                "security_id", "actual_forward_return_5d", "predicted_return_5d"}
    if not required <= set(predictions.columns):
        raise ValueError(f"Prediction table lacks {sorted(required - set(predictions.columns))}")
    supervised = {arm["id"] for arm in protocol["experiment_arms"]
                  if arm["interface"] == "SupervisedModel"}
    ledger = {}
    for name, partition in (("outer_fit_ledger.jsonl", "outer"),
                            ("holdout_fit_ledger.jsonl", "descriptive_holdout")):
        for item in _jsonl(run / name):
            if item["arm_id"] not in supervised:
                continue
            fold = item.get("outer_fold_id", "holdout")
            key = (partition, fold, item["arm_id"], item["seed"])
            if key in ledger:
                raise ValueError(f"Duplicate predictive ledger row: {key}")
            ledger[key] = item
    rows = []
    cells = predictions.filter(pl.col("arm_id").is_in(supervised)).partition_by(
        ["partition", "fold_id", "arm_id", "seed"], as_dict=True)
    for key, cell in cells.items():
        raw_partition, fold, arm, seed = key
        partition = "descriptive_holdout" if raw_partition == "descriptive_holdout" else "outer"
        ledger_key = (partition, fold, arm, seed)
        if ledger_key not in ledger:
            raise ValueError(f"Prediction cell lacks ledger evidence: {ledger_key}")
        values, _ = prediction_metrics(cell)
        recorded = ledger[ledger_key]
        recorded_ic = recorded.get("ic")
        ic_matches = (math.isclose(values["mean_ic"], recorded_ic, abs_tol=1e-12)
                      if values["mean_ic"] is not None and recorded_ic is not None
                      else values["mean_ic"] is None and recorded_ic in {None, 0.0})
        if (not math.isclose(values["rmse"], recorded["rmse"], abs_tol=1e-12)
                or not ic_matches):
            raise ValueError(f"Recomputed predictive metrics disagree with ledger: {ledger_key}")
        rows.append({"partition": partition, "partition_role": partition,
                     "fold_id": fold, "arm_id": arm, "seed": seed,
                     "row_count": cell.height,
                     "labelled_rows": cell.filter(pl.col("actual_forward_return_5d").is_not_null()).height,
                     "mae": values["mae"], "rmse": values["rmse"],
                     "mean_daily_spearman_ic": recorded_ic,
                     "recomputed_finite_mean_daily_spearman_ic": values["mean_ic"],
                     "finite_ic_dates": values["ic_dates"],
                     "ic_availability_status": ("available" if values["mean_ic"] is not None
                                                else "unavailable_no_finite_daily_cross_sections"),
                     "metric_source": "recomputed_from_hash_verified_predictions",
                     "selection_eligible": partition == "outer",
                     "claim_status": "descriptive_only" if partition == "descriptive_holdout"
                                     else "out_of_sample_outer_evaluation"})
    if set(ledger) != {(r["partition"], r["fold_id"], r["arm_id"], r["seed"]) for r in rows}:
        raise ValueError("Predictive diagnostics matrix is incomplete")
    return pl.DataFrame(rows).sort("partition", "fold_id", "arm_id", "seed")


def _telemetry_row(item: dict, stage: str) -> dict:
    telemetry = item.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = {}
    return {
        "evaluation_stage": stage, "arm_id": item.get("arm_id"),
        "trial_id": item.get("trial_id") or telemetry.get("trial_id"),
        "fold_id": (item.get("inner_fold_id") or item.get("outer_fold_id")
                    or telemetry.get("fold_id")),
        "seed": item.get("seed", telemetry.get("seed")),
        "risk_scenario": item.get("risk_scenario", telemetry.get("risk_scenario")),
        "status": item.get("status", telemetry.get("status")),
        "requested_device": telemetry.get("requested_device", telemetry.get("device")),
        "actual_device": telemetry.get("actual_device", telemetry.get("device")),
        "duration_seconds": telemetry.get("duration_seconds"),
        "wall_seconds": telemetry.get("wall_seconds"),
        "fit_duration_seconds": telemetry.get("fit_duration_seconds"),
        "fit_rows": telemetry.get("fit_rows"), "stopping_rows": telemetry.get("stopping_rows"),
        "score_rows": telemetry.get("score_rows"), "iterations": telemetry.get("iterations"),
        "selected_iteration": telemetry.get("selected_iteration"),
        "epochs_completed": telemetry.get("epochs_completed"),
        "environment_steps_requested": telemetry.get("environment_steps_requested"),
        "environment_steps_completed": telemetry.get("environment_steps_completed"),
        "gradient_updates": telemetry.get("gradient_updates"),
        "parameter_count": telemetry.get("parameter_count"),
        "convergence_warning": telemetry.get("convergence_warning"),
        "parameters_json": json.dumps(telemetry.get("parameters", {}), sort_keys=True),
        "backend_metrics_json": json.dumps(telemetry.get("backend_metrics", {}), sort_keys=True),
        "warnings_json": json.dumps(telemetry.get("warnings", []), sort_keys=True),
        "telemetry_status": "available" if telemetry else "unavailable",
        "unavailable_reason": None if telemetry else "No telemetry object was recorded",
    }


def _fit_telemetry(run: Path) -> pl.DataFrame:
    sources = (("fit_ledger.jsonl", "inner_hpo"),
               ("outer_fit_ledger.jsonl", "outer_evaluation"),
               ("holdout_fit_ledger.jsonl", "descriptive_holdout"),
               ("sensitivity_fit_ledger.jsonl", "controlled_sensitivity"))
    rows = [_telemetry_row(item, stage) for name, stage in sources
            for item in _jsonl(run / name)]
    if not rows:
        raise ValueError("No fit telemetry was recorded")
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "evaluation_stage", "arm_id", "fold_id", "trial_id")


def _controlled_sensitivity(run: Path) -> pl.DataFrame:
    design = _json(run / "sensitivity_design.json")
    ledger = _jsonl(run / "sensitivity_trial_ledger.jsonl")
    proposed, terminal = {}, {}
    for item in ledger:
        target = proposed if item.get("status") == "proposed" else terminal
        trial_id = item.get("trial_id")
        if not trial_id or trial_id in target:
            raise ValueError("Invalid or duplicate controlled-sensitivity ledger row")
        target[trial_id] = item
    if not proposed or set(proposed) != set(terminal):
        raise ValueError("Controlled-sensitivity proposal/terminal matrix is incomplete")
    indexed = {(p["arm_id"], p["outer_fold_id"], p.get("risk_scenario"), p["proposal_index"]):
               (p, terminal[trial_id]) for trial_id, p in proposed.items()}
    rows = []
    for arm_id, arm_design in sorted(design.get("arms", {}).items()):
        for contrast in arm_design.get("contrasts", []):
            contrast_index = contrast["contrast_proposal_index"]
            reference_index = contrast["reference_proposal_index"]
            contexts = [key for key in indexed if key[0] == arm_id and key[3] == contrast_index]
            if not contexts:
                raise ValueError(f"Sensitivity contrast has no executed cell: {arm_id}/{contrast_index}")
            for key in contexts:
                reference_key = (key[0], key[1], key[2], reference_index)
                if reference_key not in indexed:
                    raise ValueError(f"Sensitivity reference cell is missing: {reference_key}")
                contrast_proposal, contrast_result = indexed[key]
                reference_proposal, reference_result = indexed[reference_key]
                for result in (contrast_result, reference_result):
                    if (result.get("status") != "complete"
                            or result.get("selection_eligible") is not False
                            or result.get("completed_cells") != result.get("expected_cells")):
                        raise ValueError("Sensitivity evidence is incomplete or selection-eligible")
                metric = ("mean_certainty_equivalent" if "mean_certainty_equivalent" in contrast_result
                          else "mean_ic")
                reference_value, contrast_value = reference_result.get(metric), contrast_result.get(metric)
                if reference_value is None or contrast_value is None:
                    raise ValueError("Sensitivity contrast lacks its declared primary metric")
                rows.append({
                    "arm_id": arm_id, "component_id": arm_design["component_id"],
                    "outer_fold_id": key[1], "risk_scenario": key[2],
                    "factor_name": contrast["factor_name"],
                    "directly_changed_fields_json": json.dumps(contrast["directly_changed_fields"]),
                    "reference_proposal_index": reference_index,
                    "contrast_proposal_index": contrast_index,
                    "reference_level_json": json.dumps(contrast["reference_level"], sort_keys=True),
                    "contrast_level_json": json.dumps(contrast["contrast_level"], sort_keys=True),
                    "reference_parameters_json": json.dumps(reference_proposal["parameters"], sort_keys=True),
                    "contrast_parameters_json": json.dumps(contrast_proposal["parameters"], sort_keys=True),
                    "primary_metric": metric,
                    "reference_metric_value": reference_value,
                    "contrast_metric_value": contrast_value,
                    "contrast_minus_reference": contrast_value - reference_value,
                    "reference_rmse": reference_result.get("mean_rmse"),
                    "contrast_rmse": contrast_result.get("mean_rmse"),
                    "reference_resource_seconds": reference_result.get("resource_seconds"),
                    "contrast_resource_seconds": contrast_result.get("resource_seconds"),
                    "selection_eligible": False,
                    "interpretation_scope": "local_reduced_fidelity_descriptive_contrast",
                })
    if not rows:
        raise ValueError("Controlled-sensitivity design has no contrasts")
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "arm_id", "outer_fold_id", "risk_scenario", "factor_name")


def _availability(protocol: dict, telemetry: pl.DataFrame) -> pl.DataFrame:
    seeds = protocol["reproducibility"]["seeds"]
    missing_telemetry = telemetry.filter(pl.col("telemetry_status") != "available").height
    return pl.DataFrame([
        {"item_id": "seed_dispersion", "status": "available" if len(seeds) > 1 else "unavailable",
         "reason": None if len(seeds) > 1 else
                   "The approved protocol predeclared one seed only; seed dispersion cannot be estimated."},
        {"item_id": "fit_telemetry", "status": "available" if not missing_telemetry else "partial",
         "reason": None if not missing_telemetry else
                   f"{missing_telemetry} fit rows have no recorded telemetry object."},
        {"item_id": "rl_predictive_ic_rmse", "status": "not_applicable",
         "reason": "RL selectors emit policy actions and certainty-equivalent evidence, not return predictions."},
        {"item_id": "controlled_sensitivity_selection", "status": "selection_ineligible",
         "reason": "The approved protocol forbids controlled-sensitivity results from HPO, family locks or holdout selection."},
        {"item_id": "final_holdout_role", "status": "descriptive_only",
         "reason": "The sealed final holdout cannot select or tune models, scenarios or parameters."},
    ])


def _write_tables(destination: Path, tables: dict[str, pl.DataFrame]) -> list[dict]:
    outputs = []
    for name, frame in sorted(tables.items()):
        if frame.is_empty():
            raise ValueError(f"Assignment table is empty: {name}")
        parquet, csv = destination / f"{name}.parquet", destination / f"{name}.csv"
        frame.write_parquet(parquet)
        frame.write_csv(csv)
        for path in (parquet, csv):
            outputs.append({"relative_path": path.name, "sha256": _hash(path),
                            "bytes": path.stat().st_size, "rows": frame.height})
    return outputs


def generate_assignment_export(*, repository_root: str | Path, run_id: str,
                               version: str = "v1") -> Path:
    """Generate assignment tables from verified sources without altering them."""
    root = Path(repository_root).resolve()
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    if not version.startswith("v") or not version[1:].isdigit():
        raise ValueError("version must have the form v<number>")
    run = (root / "runs" / "expanded_closeout" / run_id).resolve()
    metadata, protocol, run_hashes = _verified_run(root, run)
    if (metadata.get("claim_status") != "exploratory_closeout_not_confirmatory"
            or metadata.get("holdout_role") != "descriptive_only"):
        raise ValueError("Assignment export requires a descriptive expanded closeout")
    wp7 = (root / "reports" / "expanded_closeout" / run_id).resolve()
    wp7_manifest, wp7_outputs = _verify_wp7(root, wp7, run_id, metadata)
    features, columns, feature_sources = _feature_inputs(root, protocol)
    partitions, window_sources = _window_partitions(root, protocol)

    feature_dictionary = _feature_dictionary(protocol, columns)
    feature_statistics = _feature_statistics(features, columns, partitions)
    predictive = _predictive_diagnostics(run, protocol)
    telemetry = _fit_telemetry(run)
    sensitivity = _controlled_sensitivity(run)
    availability = _availability(protocol, telemetry)

    destination = (root / "reports" / "assignment" / run_id / version).resolve()
    if (not destination.is_relative_to(root / "reports" / "assignment")
            or destination.exists()):
        raise FileExistsError(f"Assignment export destination must be new: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    generated_at = datetime.now(timezone.utc).isoformat()
    code_version = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=root, text=True).strip()
    provenance = pl.DataFrame([{
        "source_run_id": run_id,
        "source_run_path": run.relative_to(root).as_posix(),
        "source_run_audit_sha256": _hash(run / "audit.json"),
        "source_run_completion_sha256": _hash(run / "completion.json"),
        "source_wp7_report_path": wp7.relative_to(root).as_posix(),
        "source_wp7_manifest_sha256": _hash(wp7 / "research_evidence_manifest.json"),
        "protocol_sha256": metadata["protocol_sha256"],
        "source_vintage_id": metadata["vintage_id"],
        "source_git_commit": metadata.get("git_commit"),
        "exporter_code_version": code_version,
        "exporter_module_sha256": _hash(Path(__file__)),
        "generated_at_utc": generated_at,
        "claim_status": metadata["claim_status"],
        "holdout_role": metadata["holdout_role"],
        "seeds_json": json.dumps(protocol["reproducibility"]["seeds"]),
        "seed_count": len(protocol["reproducibility"]["seeds"]),
        "seed_dispersion_status": "available" if len(protocol["reproducibility"]["seeds"]) > 1
                                  else "unavailable_single_predeclared_seed",
    }])
    tables = {
        "controlled_sensitivity_summary": sensitivity,
        "evidence_availability": availability,
        "feature_descriptive_statistics": feature_statistics,
        "feature_dictionary": feature_dictionary,
        "fit_telemetry": telemetry,
        "predictive_diagnostics": predictive,
        "provenance_table": provenance,
    }
    try:
        outputs = _write_tables(destination, tables)
        input_hashes = {
            (run / "audit.json").relative_to(root).as_posix(): _hash(run / "audit.json"),
            (run / "completion.json").relative_to(root).as_posix(): _hash(run / "completion.json"),
            wp7.relative_to(root).joinpath("research_evidence_manifest.json").as_posix():
                _hash(wp7 / "research_evidence_manifest.json"),
            **feature_sources, **window_sources,
            **{wp7.relative_to(root).joinpath(name).as_posix(): sha
               for name, sha in wp7_outputs.items()},
        }
        manifest = {
            "schema_version": 1, "source_run_id": run_id,
            "source_run_audit_sha256": _hash(run / "audit.json"),
            "source_run_completion_sha256": _hash(run / "completion.json"),
            "source_wp7_manifest_sha256": _hash(wp7 / "research_evidence_manifest.json"),
            "source_wp7_generated_at_utc": wp7_manifest["generated_at_utc"],
            "protocol_sha256": metadata["protocol_sha256"],
            "claim_status": metadata["claim_status"], "holdout_role": metadata["holdout_role"],
            "generated_at_utc": generated_at, "code_version": code_version,
            "generator_module_sha256": _hash(Path(__file__)),
            "input_artifacts": [{"relative_path": name, "sha256": sha}
                                for name, sha in sorted(input_hashes.items())],
            "outputs": outputs,
            "source_run_hashed_artifact_count": len(run_hashes),
            "limitations": [row for row in availability.to_dicts()
                            if row["status"] in {"unavailable", "partial", "not_applicable"}],
        }
        (destination / "assignment_evidence_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
    except Exception:
        # Preserve fail-closed diagnostic output outside runs/; a new version is required.
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    print(generate_assignment_export(repository_root=args.repository_root,
                                     run_id=args.run_id, version=args.version))


if __name__ == "__main__":
    main()
