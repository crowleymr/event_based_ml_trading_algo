"""Authoritative, manifest-bound supervised lane for an expanded study.

This module is called only by ``trading_pipeline.run`` after authority verification.
It never reads a legacy run. Unsupported arms fail before any score-bearing data I/O.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from copy import deepcopy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
import uuid

import numpy as np
import polars as pl
import psutil
from scipy.stats import spearmanr

from trading_pipeline.experiments import FitContext
from trading_pipeline.experiments.authority import VerifiedStudyAuthority
from trading_pipeline.experiments.default_registry import default_registry
from trading_pipeline.experiments.deep_study_adapter import fit_score_deep_study
from trading_pipeline.experiments.rl_episode_producer import (
    MODEL_SLEEVES, produce_selector_episode, selector_feature_frame,
)
from trading_pipeline.experiments.schema import ResolvedStudy
from trading_pipeline.experiments.study_resume import (
    CheckpointStore, code_state_sha256, sha256, cell_key, process_still_running,
)
from trading_pipeline.features import F0, F1
from trading_pipeline.optimisation.search import controlled_one_factor_design, random_proposals
from trading_pipeline.rl.study_integration import execute_study_rl_trial


FEATURE_VIEWS = {"F0", "F1", "F1_CAUSAL_STACK_V1"}
LOGGER = logging.getLogger(__name__)


def _study_bridge(spec) -> str:
    """Resolve an audited bridge solely from trusted registration metadata."""
    capabilities = spec.capabilities
    if not capabilities.get("study_adapter"):
        raise ValueError(f"No audited study bridge for {spec.component_id}")
    if spec.interface == "RLPolicy" and capabilities.get("discrete_actions"):
        return "rl"
    if spec.interface == "SupervisedModel":
        views = [name for name in ("tabular_view", "sequence_view")
                 if capabilities.get(name)]
        if len(views) == 1:
            return "deep" if views[0] == "sequence_view" else "tabular"
    raise ValueError(f"No audited study bridge for {spec.component_id}")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_file(root: Path, relative: str, expected: str) -> Path:
    """Verify a file named and hashed by the approved, self-hashed protocol."""
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Protocol input requires a path and SHA-256")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or _hash(path) != expected:
        raise ValueError(f"Protocol input is absent or hash-mismatched: {relative}")
    return path


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, default=str, allow_nan=False)


def _append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str, allow_nan=False) + "\n")


def _append_many(path: Path, values: list[dict]) -> None:
    """Append one cell's ordered JSONL evidence with a single file open."""
    if not values:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(
            json.dumps(value, sort_keys=True, default=str, allow_nan=False) + "\n"
            for value in values
        )


def _manifest(root: Path, relative: str, authority: VerifiedStudyAuthority) -> dict:
    if relative not in authority.input_sha256:
        raise PermissionError(f"Manifest is not authority-pinned: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or _hash(path) != authority.input_sha256[relative]:
        raise ValueError(f"Authority manifest changed: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def _dates(fold: dict, name: str) -> tuple[date, ...]:
    values = fold.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Study fold requires explicit {name}")
    dates = tuple(date.fromisoformat(value) for value in values)
    if dates != tuple(sorted(set(dates))):
        raise ValueError(f"Study fold {name} must be sorted and unique")
    return dates


def _feature_path(root: Path, snapshot: dict) -> Path:
    relative, expected = snapshot.get("feature_path"), snapshot.get("feature_sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Pinned snapshot requires feature_path and feature_sha256")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or _hash(path) != expected:
        raise ValueError("Canonical feature snapshot is absent or hash-mismatched")
    return path


def _validate_declared_arms(config: dict, registry, root: Path) -> list[dict]:
    """Bind the YAML roster to trusted components and its approved capability gate."""
    arms = config["experiment_arms"]
    if not isinstance(arms, list) or not arms:
        raise ValueError("Study requires declared experiment arms")
    ids = [arm["id"] for arm in arms]
    components = [arm["component_id"] for arm in arms]
    if len(set(ids)) != len(ids) or len(set(components)) != len(components):
        raise ValueError("Study arm and component IDs must each be unique")
    for arm in arms:
        spec = registry.spec(arm["component_id"])
        if arm["interface"] != spec.interface or spec.interface not in {"SupervisedModel", "RLPolicy"}:
            raise ValueError(f"No score-bearing interface for {arm['id']}")
        if arm.get("feature_set_id") not in FEATURE_VIEWS:
            raise ValueError(f"Unsupported feature view for {arm['id']}")
        expected_objective = ("supervised_ic_v1" if spec.interface == "SupervisedModel"
                              else "rl_certainty_equivalent_v1")
        if arm.get("objective_id") != expected_objective:
            raise ValueError(f"Unsupported objective for {arm['id']}")
        _study_bridge(spec)
    authority = config["authority"]
    gate_path = _protocol_file(root, authority.get("capability_gate"),
                               authority.get("capability_gate_sha256"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    capabilities = gate.get("arm_capabilities")
    if (gate.get("study_id") != config["study_id"]
            or gate.get("score_blind_calibration") is not True
            or gate.get("outcome_rankings_opened") is not False
            or not isinstance(capabilities, dict) or set(capabilities) != set(ids)
            or any(not isinstance(item, dict)
                   or item.get("synthetic_verification_passed") is not True
                   or item.get("real_data_bridge_passed") is not True
                   or item.get("component_id") != arm["component_id"]
                   or item.get("interface") != arm["interface"]
                   for arm in arms for item in [capabilities[arm["id"]]])):
        raise PermissionError("Approved capability gate must cover the exact declared arms")
    return arms


def _preflight(study: ResolvedStudy, authority: VerifiedStudyAuthority, root: Path):
    config = study.config
    if study.engineering_only or authority.study_id != config["study_id"]:
        raise PermissionError("Verified approved authority is required")
    registry = default_registry()
    arms = _validate_declared_arms(config, registry, root)
    if any(arm["interface"] == "RLPolicy" for arm in arms):
        rl_contract = config["data"].get("rl_inputs")
        if not isinstance(rl_contract, dict) or not {
            "bars_path", "bars_sha256", "model_outputs_path", "model_outputs_sha256",
            "model_sleeve_components",
        } <= rl_contract.keys():
            raise PermissionError("Study arm has no audited score-bearing bridge: RL input contract is missing")
        if rl_contract.get("eligibility_policy") == "complete_causal_upstream_dates_plus_observed_price_history_v2":
            if rl_contract.get("price_eligibility_policy") != "observed_history_and_endpoint_valuation_v1":
                raise PermissionError("Revised RL price eligibility declaration is missing")
        elif rl_contract.get("price_eligibility_policy") == "observed_history_and_endpoint_valuation_v1":
            raise PermissionError("Revised RL eligibility requires paired policy declaration")
        _protocol_file(root, rl_contract["bars_path"], rl_contract["bars_sha256"])
        _protocol_file(root, rl_contract["model_outputs_path"], rl_contract["model_outputs_sha256"])
        training = config["search"].get("rl_training", {})
        if type(training.get("total_timesteps")) is not int or training["total_timesteps"] < 1:
            raise ValueError("RL total timesteps must be frozen in the approved protocol")
    if config["search"].get("method") not in {"deterministic_grid", "deterministic_random_with_multifidelity_for_deep_models"}:
        raise ValueError("Supervised lane requires declared deterministic search")
    if any(arm["feature_set_id"] == "F1_CAUSAL_STACK_V1" for arm in arms):
        contract = config["data"].get("augmented_feature_contract")
        if not isinstance(contract, dict) or not {
            "feature_path", "feature_sha256", "feature_columns", "source_manifest",
            "source_manifest_sha256",
        } <= contract.keys():
            raise ValueError("F1_CAUSAL_STACK_V1 requires an explicit supplied augmented feature contract")
        if (not isinstance(contract["feature_columns"], list)
                or not contract["feature_columns"]
                or len(set(contract["feature_columns"])) != len(contract["feature_columns"])
                or not set(F1) < set(contract["feature_columns"])
                or any(name in {"forward_return_5d", "label_end_date", "split"}
                       for name in contract["feature_columns"])):
            raise ValueError("Augmented feature columns must explicitly extend F1 without targets")
        _protocol_file(root, contract["source_manifest"], contract["source_manifest_sha256"])
    snapshot = _manifest(root, config["data"]["snapshot_manifest"], authority)
    outer = _manifest(root, config["validation"]["outer_windows_manifest"], authority)["folds"]
    inner = _manifest(root, config["validation"]["inner_windows_manifest"], authority)["folds"]
    holdout = _manifest(root, config["validation"]["final_holdout"]["manifest"], authority)
    feature_path = _feature_path(root, snapshot)
    if not outer or not inner or not holdout.get("sealed"):
        raise ValueError("Study windows and sealed holdout are required")
    grouped: dict[str, list[dict]] = {item["fold_id"]: [] for item in outer}
    for fold in inner:
        grouped[fold["outer_fold_id"]].append(fold)
        fit, stop, score = (_dates(fold, key) for key in ("fit_dates", "stopping_dates", "score_dates"))
        if not max(fit) < min(stop) or not max(stop) < min(score):
            raise ValueError("Inner fit/stopping/scoring windows must be distinct and chronological")
    if any(not group for group in grouped.values()):
        raise ValueError("Every outer fold needs an inner fold")
    for fold in outer:
        _dates(fold, "fit_dates")
        _dates(fold, "score_dates")
    return registry, arms, outer, grouped, holdout, feature_path, snapshot


def _feature_rows(root: Path, config: dict, authority: VerifiedStudyAuthority,
                  feature_path: Path, feature_sha256: str) -> list[dict]:
    canonical = pl.read_parquet(feature_path, columns=[name for name in
        pl.read_parquet_schema(feature_path) if name != "forward_return_5d"])
    if not any(arm["feature_set_id"] == "F1_CAUSAL_STACK_V1"
               for arm in config["experiment_arms"]):
        return canonical.to_dicts()
    contract = config["data"]["augmented_feature_contract"]
    relative = contract["feature_path"]
    path = _protocol_file(root, relative, contract["feature_sha256"])
    source = json.loads((root / contract["source_manifest"]).read_text(encoding="utf-8"))
    if (source.get("study_id") != config["study_id"]
            or source.get("feature_sha256") != feature_sha256
            or source.get("table_sha256") != contract["feature_sha256"]):
        raise ValueError("Augmented feature provenance does not bind to canonical snapshot")
    augmented = pl.read_parquet(path)
    keys = ["security_id", "session_date"]
    columns = contract["feature_columns"]
    if (not set([*keys, *columns]) <= set(augmented.columns)
            or "forward_return_5d" in augmented.columns
            or augmented.height != canonical.height
            or augmented.select(keys).is_duplicated().any()):
        raise ValueError("Augmented feature table has incomplete keys or forbidden labels")
    common = [*keys, *F1]
    if not augmented.select(common).sort(keys).equals(canonical.select(common).sort(keys)):
        raise ValueError("Augmented feature table changes canonical F1 rows")
    extra = [name for name in columns if name not in F1]
    return canonical.join(augmented.select([*keys, *extra]), on=keys,
                          how="inner", validate="1:1").to_dicts()


def _score(rows: list[dict], predictions: np.ndarray) -> tuple[float, float]:
    if len(rows) != len(predictions) or not np.isfinite(predictions).all():
        raise ValueError("Missing or nonfinite study prediction")
    by_day: dict[date, list[tuple[float, float]]] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        actual = row.get("forward_return_5d")
        if actual is None or not math.isfinite(float(actual)):
            raise ValueError("Scoring requires finite label")
        by_day.setdefault(row["session_date"], []).append((float(actual), float(prediction)))
    daily = []
    for values in by_day.values():
        actual, predicted = zip(*values)
        if len(values) < 2:
            raise ValueError("Daily cross-sectional IC requires at least two securities")
        actual_array = np.asarray(actual, dtype=np.float64)
        predicted_array = np.asarray(predicted, dtype=np.float64)
        # A constant cross-section contains no rank information. Treat its IC as
        # neutral explicitly instead of asking scipy to emit ConstantInputWarning
        # and return NaN. RMSE below still evaluates the predictions normally.
        if (np.all(actual_array == actual_array[0])
                or np.all(predicted_array == predicted_array[0])):
            daily.append(0.0)
            continue
        correlation = spearmanr(actual_array, predicted_array).statistic
        daily.append(0.0 if not math.isfinite(correlation) else float(correlation))
    rmse = math.sqrt(sum((float(row["forward_return_5d"]) - float(pred)) ** 2
                         for row, pred in zip(rows, predictions, strict=True)) / len(rows))
    return sum(daily) / len(daily), rmse


def _matrix(rows: list[dict], columns: tuple[str, ...], *, labels: bool) -> tuple[np.ndarray, np.ndarray | None]:
    x = np.asarray([[np.nan if row.get(column) is None else float(row[column])
                     for column in columns] for row in rows], dtype=np.float64)
    if not len(rows) or np.isinf(x).any():
        raise ValueError("Empty or infinite feature matrix")
    if not labels:
        return x, None
    y = np.asarray([row["forward_return_5d"] for row in rows], dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("Fit and scoring labels must be finite")
    return x, y


def _partition(rows_by_day: dict[date, list[dict]], dates: tuple[date, ...]) -> list[dict]:
    if any(day not in rows_by_day for day in dates):
        raise ValueError("Feature snapshot is missing a declared fold session")
    return [row for day in dates for row in rows_by_day[day]]


def _labelled(feature_path: Path, rows_by_day: dict[date, list[dict]],
              dates: tuple[date, ...]) -> list[dict]:
    """Project target values only for the partition currently authorised to score."""
    base = _partition(rows_by_day, dates)
    labels = (pl.scan_parquet(feature_path)
              .filter(pl.col("session_date").is_in(list(dates)))
              .select("security_id", "session_date", "forward_return_5d")
              .collect())
    lookup = {(row["security_id"], row["session_date"]): row["forward_return_5d"]
              for row in labels.to_dicts()}
    if len(lookup) != len(base):
        raise ValueError("Target rows do not match the declared feature partition")
    return [{**row, "forward_return_5d": lookup[(row["security_id"], row["session_date"])]}
            for row in base]


class _DevelopmentLabelCache:
    """Reuse only declared development labels, loading each partition on demand."""

    def __init__(self, feature_path: Path, rows_by_day: dict[date, list[dict]],
                 outer: list[dict], inner: dict[str, list[dict]],
                 holdout_dates: tuple[date, ...]):
        self.feature_path = feature_path
        self.rows_by_day = rows_by_day
        partitions = [*_partition_dates(outer),
                      *_partition_dates([fold for group in inner.values() for fold in group])]
        self.allowed_dates = frozenset(day for dates in partitions for day in dates)
        if not self.allowed_dates or self.allowed_dates.intersection(holdout_dates):
            raise ValueError("Development label cache overlaps sealed holdout or is empty")
        if not self.allowed_dates <= rows_by_day.keys():
            raise ValueError("Development label cache has missing feature sessions")
        self._labelled_by_day: dict[date, list[dict]] = {}
        self._loaded_dates: set[date] = set()

    def partitions(self, *partitions: tuple[date, ...]) -> tuple[list[dict], ...]:
        requested = frozenset(day for dates in partitions for day in dates)
        if not requested or not requested <= self.allowed_dates:
            raise PermissionError("Label partition is outside declared development windows")
        missing = requested - self._loaded_dates
        if missing:
            dates = tuple(sorted(missing))
            base = _partition(self.rows_by_day, dates)
            labels = (pl.scan_parquet(self.feature_path)
                      .filter(pl.col("session_date").is_in(list(dates)))
                      .select("security_id", "session_date", "forward_return_5d")
                      .collect())
            lookup = {(row["security_id"], row["session_date"]): row["forward_return_5d"]
                      for row in labels.to_dicts()}
            expected = {(row["security_id"], row["session_date"]) for row in base}
            if len(lookup) != labels.height or lookup.keys() != expected:
                raise ValueError("Target rows do not match declared development features")
            self._labelled_by_day.update({day: [
                {**row, "forward_return_5d": lookup[(row["security_id"], day)]}
                for row in self.rows_by_day[day]] for day in dates})
            self._loaded_dates.update(missing)
        return tuple([row for day in dates for row in self._labelled_by_day[day]]
                     for dates in partitions)


def _partition_dates(folds: list[dict]):
    for fold in folds:
        for name in ("fit_dates", "stopping_dates", "score_dates"):
            if name in fold:
                yield _dates(fold, name)


def _holdout_labelled(feature_path: Path, rows_by_day: dict[date, list[dict]],
                      dates: tuple[date, ...], family_lock_path: Path) -> list[dict]:
    """Open descriptive labels only after the completed family lock is sealed."""
    if not family_lock_path.is_file():
        raise PermissionError("Family lock required before holdout labels")
    lock = json.loads(family_lock_path.read_text(encoding="utf-8"))
    if lock.get("holdout_selection") != "forbidden" or not lock.get("final_choices"):
        raise PermissionError("Complete family lock required before holdout labels")
    return _labelled(feature_path, rows_by_day, dates)


def _columns(arm: dict, config: dict) -> tuple[str, ...]:
    view = arm["feature_set_id"]
    if view == "F0":
        return tuple(F0)
    if view == "F1":
        return tuple(F1)
    if view == "F1_CAUSAL_STACK_V1":
        return tuple(config["data"]["augmented_feature_contract"]["feature_columns"])
    raise ValueError(f"Unsupported supervised feature view: {view}")


def _fit_score(registry, arm: dict, params: dict, seed: int, trial_id: str,
               fold_id: str, fit: list[dict], stop: list[dict], score: list[dict],
               *, config: dict, feature_pool: list[dict], predict: bool = False):
    columns = _columns(arm, config)
    bridge = _study_bridge(registry.spec(arm["component_id"]))
    if bridge == "deep":
        training = config["search"].get("deep_training", {})
        if not {"epochs", "patience"} <= training.keys():
            raise ValueError("Deep execution requires frozen epochs and patience")
        result = fit_score_deep_study(
            component_id=arm["component_id"], feature_set_id=arm["feature_set_id"],
            parameters=params, seed=seed, study_id=config["study_id"],
            trial_id=trial_id, fold_id=fold_id, feature_pool=feature_pool,
            fit_rows=fit, stopping_rows=stop,
            score_rows=[{**row, "split": fold_id} for row in score],
            feature_columns=columns,
            requested_device=arm.get(
                "execution_device", config["reproducibility"].get("requested_device", "cpu")),
            epochs=training["epochs"], patience=training["patience"],
        )
        ic, rmse = _score(score, result.predictions)
        return None, ic, rmse, dict(result.telemetry), result.predictions if predict else None
    if bridge != "tabular":
        raise ValueError(f"No supervised fit bridge for {arm['component_id']}")
    fit_x, fit_y = _matrix(fit, columns, labels=True)
    score_x, _ = _matrix(score, columns, labels=False)
    spec = registry.spec(arm["component_id"])
    kwargs = {"params": params}
    if spec.capabilities.get("cuda"):
        kwargs["device"] = arm.get(
            "execution_device", config["reproducibility"].get("requested_device", "cpu"))
    model = registry.create(arm["component_id"], **kwargs)
    context = FitContext("expanded", trial_id, fold_id, seed, {})
    stopping = None
    if model.spec.capabilities.get("stopping_data"):
        stop_x, stop_y = _matrix(stop, columns, labels=True)
        stopping = (stop_x, stop_y)
    started = time.perf_counter()
    model.fit(fit_x, fit_y, context=context, stopping_data=stopping)
    predictions = model.predict(score_x)
    duration = time.perf_counter() - started
    ic, rmse = _score(score, predictions)
    telemetry = {**model.telemetry(), "wall_seconds": duration,
                 "fit_rows": len(fit), "stopping_rows": len(stop), "score_rows": len(score),
                 "device": model.telemetry().get("actual_device", "cpu")}
    return model, ic, rmse, telemetry, predictions if predict else None


def _proposals(registry, arm: dict, config: dict) -> list[dict]:
    if registry.spec(arm["component_id"]).interface != "SupervisedModel":
        raise ValueError("Supervised proposal generation cannot handle RL arms")
    component = registry.create(arm["component_id"])
    space = component.search_space()["parameters"]
    deep = _study_bridge(registry.spec(arm["component_id"])) == "deep"
    budget_key = "deep_supervised_per_family" if deep else "classical_per_family"
    ceiling = config["search"]["budgets"][budget_key]
    if type(ceiling) is not int or ceiling < 1:
        raise ValueError(f"Invalid frozen proposal budget for {arm['id']}")
    declared = config["search"].get("proposals", {}).get(arm["id"])
    if declared is not None:
        if not isinstance(declared, list) or not declared or any(not isinstance(p, dict) for p in declared):
            raise ValueError("Declared proposals must be a nonempty list of parameter mappings")
        proposals = declared
    elif deep:
        if config["search"].get("method") != "deterministic_random_with_multifidelity_for_deep_models":
            raise ValueError("Deep architecture search requires deterministic random proposals")
        proposals = random_proposals(space, ceiling, config["reproducibility"]["seeds"][0])
    else:
        proposals = random_proposals(space, ceiling, config["reproducibility"]["seeds"][0])
    if len({json.dumps(p, sort_keys=True) for p in proposals}) != len(proposals):
        raise ValueError("Duplicate proposals")
    if len(proposals) != ceiling:
        raise ValueError(f"{arm['id']} has {len(proposals)} proposals; frozen tier requires {ceiling}")
    for proposal in proposals:
        if set(proposal) != set(space):
            raise ValueError(f"{arm['id']} proposal must declare exact model parameters")
        registry.create(arm["component_id"], params=proposal)
    return proposals


def _rl_proposals(registry, arm: dict, config: dict) -> list[dict]:
    """Freeze valid architecture proposals before any outcome is opened."""
    component_id = arm["component_id"]
    if _study_bridge(registry.spec(component_id)) != "rl":
        raise ValueError("RL proposals require a registered RL arm")
    ceiling = config["search"]["budgets"]["rl_per_family_per_risk_scenario"]
    steps = config["search"].get("rl_training", {}).get("total_timesteps")
    if type(ceiling) is not int or ceiling < 1 or type(steps) is not int or steps < 1:
        raise ValueError("RL requires a positive frozen proposal budget and total timesteps")
    space = registry.create(component_id, total_timesteps=steps).search_space()["parameters"]
    declared = config["search"].get("proposals", {}).get(arm["id"])
    if declared is not None:
        if not isinstance(declared, list) or len(declared) != ceiling:
            raise ValueError("Declared RL proposal count differs from frozen budget")
        candidates = declared
    else:
        candidates = random_proposals(space, ceiling * 20,
                                      config["reproducibility"]["seeds"][0])
    proposals, seen = [], set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate or not set(candidate) <= set(space):
            if declared is not None:
                raise ValueError("RL proposal has missing or unknown architecture fields")
            continue
        try:
            registry.create(component_id, parameters=candidate, total_timesteps=steps)
        except (TypeError, ValueError):
            if declared is not None:
                raise
            continue
        key = json.dumps(candidate, sort_keys=True)
        if key in seen:
            raise ValueError("Duplicate RL proposal")
        proposals.append(candidate)
        seen.add(key)
        if len(proposals) == ceiling:
            break
    if len(proposals) != ceiling:
        raise ValueError("RL search could not produce its frozen valid proposal budget")
    return proposals


def _controlled_sensitivity_designs(registry, arms: list[dict], config: dict) -> dict[str, dict]:
    declaration = config.get("search", {}).get("controlled_sensitivity")
    if declaration is None:
        return {}
    required = {"enabled", "design", "inner_folds_per_outer", "selection_eligible",
                "deep_training", "rl_training"}
    if (not isinstance(declaration, dict) or not required <= declaration.keys()
            or declaration["enabled"] is not True
            or declaration["design"] != "deterministic_controlled_one_factor_v1"
            or declaration["selection_eligible"] is not False):
        raise ValueError("Controlled sensitivity must be enabled, score-blind and selection-ineligible")
    folds = declaration["inner_folds_per_outer"]
    deep = declaration["deep_training"]
    rl = declaration["rl_training"]
    if (type(folds) is not int or folds < 1
            or type(deep.get("epochs")) is not int or deep["epochs"] < 1
            or type(deep.get("patience")) is not int or deep["patience"] < 1
            or type(rl.get("total_timesteps")) is not int or rl["total_timesteps"] < 1):
        raise ValueError("Controlled sensitivity requires positive fold and fidelity declarations")
    if (deep["epochs"] > config["search"]["deep_training"]["epochs"]
            or rl["total_timesteps"] > config["search"]["rl_training"]["total_timesteps"]):
        raise ValueError("Sensitivity fidelity cannot exceed selection fidelity")
    designs = {}
    for arm in arms:
        component_id = arm["component_id"]
        if arm["interface"] == "RLPolicy":
            component = registry.create(component_id, total_timesteps=rl["total_timesteps"])

            def validate(parameters, component_id=component_id):
                registry.create(component_id, parameters=parameters,
                                total_timesteps=rl["total_timesteps"])
        else:
            component = registry.create(component_id)

            def validate(parameters, component_id=component_id):
                registry.create(component_id, params=parameters)
        design = controlled_one_factor_design(component.search_space()["parameters"], validate)
        designs[arm["id"]] = {**design, "arm_id": arm["id"],
                              "component_id": component_id,
                              "selection_eligible": False}
    return designs


def _rl_inputs(root: Path, config: dict, feature_path: Path, feature_sha256: str):
    """Open only protocol-hashed bars and long-form causal model outputs."""
    contract = config["data"].get("rl_inputs")
    if not isinstance(contract, dict) or not {
        "bars_path", "bars_sha256", "model_outputs_path", "model_outputs_sha256",
        "model_sleeve_components",
    } <= contract.keys():
        raise PermissionError("Study arm has no audited score-bearing bridge: RL input contract is missing")
    bars_path = _protocol_file(root, contract["bars_path"], contract["bars_sha256"])
    if bars_path != feature_path or contract["bars_sha256"] != feature_sha256:
        raise ValueError("RL bars must bind to the canonical feature snapshot")
    outputs_path = _protocol_file(root, contract["model_outputs_path"],
                                  contract["model_outputs_sha256"])
    bars = pl.read_parquet(bars_path, columns=["security_id", "session_date", "adjusted_close"])
    source = pl.read_parquet(outputs_path)
    required = {"upstream_component_id", "security_id", "session_date", "prediction",
                "fit_cutoff_date", "fit_max_label_end_date", "prediction_role"}
    mapping = contract["model_sleeve_components"]
    if (not isinstance(mapping, dict) or set(mapping) != set(MODEL_SLEEVES)
            or len(set(mapping.values())) != len(mapping)
            or not required <= set(source.columns)
            or set(source["upstream_component_id"].unique()) != set(mapping.values())):
        raise ValueError("RL model outputs require an explicit complete E1–E4 component map")
    outputs = source.with_columns(
        pl.col("upstream_component_id").replace({value: key for key, value in mapping.items()})
          .alias("model_id"),
        pl.col("prediction").alias("predicted_return_5d"),
    )
    return bars, outputs, contract["model_outputs_sha256"]


def _rl_episode(*, split: str, dates: tuple[date, ...], rows: list[dict],
                bars: pl.DataFrame, outputs: pl.DataFrame, config: dict,
                feature_sha256: str, outputs_sha256: str, fold_sha256: str,
                protocol_sha256: str, scenario: str):
    portfolio = config["portfolio"]
    constraints = portfolio["risk_scenarios"][scenario]
    e5_source = portfolio.get("e5_source")
    lookback = portfolio.get("volatility_lookback_sessions")
    top_k = portfolio.get("top_k")
    if e5_source not in MODEL_SLEEVES or any(type(value) is not int or value < 1
                                              for value in (lookback, top_k)):
        raise ValueError("RL needs a validation-locked E5 source, top-K and volatility lookback")
    calendar = tuple(sorted({row["session_date"] for row in rows}))
    return produce_selector_episode(
        split=split, signal_sessions=dates, calendar=calendar,
        features=selector_feature_frame(rows),
        model_outputs=outputs, bars=bars,
        source_hashes={"canonical_features": feature_sha256,
                       "model_outputs": outputs_sha256,
                       "adjusted_close": feature_sha256,
                       "fold_manifest": fold_sha256,
                       "protocol": protocol_sha256},
        e5_source=e5_source, top_k=top_k,
        gross_exposure_cap=constraints["gross_exposure_cap"],
        max_position=constraints["max_position"],
        annualised_volatility_target=constraints["annualised_volatility_target"],
        volatility_lookback_sessions=lookback,
        allowed_prediction_roles=("inner_oof", "outer_oof", "holdout_score_only"),
        price_eligibility_policy=config["data"]["rl_inputs"].get(
            "price_eligibility_policy", "complete_daily_valuation_v1"),
    )


def _rl_trial_cell(*, study: ResolvedStudy, authority: VerifiedStudyAuthority,
                   arm: dict, params: dict, seed: int, trial_id: str, fold_id: str,
                   scenario: str, fit_dates: tuple[date, ...], score_dates: tuple[date, ...],
                   rows: list[dict], bars: pl.DataFrame, outputs: pl.DataFrame,
                   feature_sha256: str, outputs_sha256: str, fold_sha256: str,
                   output: Path, total_timesteps: int | None = None,
                   episode_cache: dict | None = None):
    steps = (study.config["search"]["rl_training"]["total_timesteps"]
             if total_timesteps is None else total_timesteps)
    if type(steps) is not int or steps < 1:
        raise ValueError("RL trial requires positive declared environment steps")
    available_dates = set(outputs["session_date"].unique().to_list())
    fit_dates = tuple(day for day in fit_dates if day in available_dates)
    score_dates = tuple(day for day in score_dates if day in available_dates)
    if len(fit_dates) < 2 or len(score_dates) < 2:
        raise ValueError("RL fold lacks two causal-output weekly signal sessions")
    common = dict(rows=rows, bars=bars, outputs=outputs, config=study.config,
                  feature_sha256=feature_sha256, outputs_sha256=outputs_sha256,
                  fold_sha256=fold_sha256, protocol_sha256=authority.protocol_sha256,
                  scenario=scenario)
    # Episode construction scans the full verified feature/output/price views and is
    # invariant to policy family, architecture, hyperparameters, seed and trial ID.
    # Scope this cache in callers to one outer fold so at most the three declared
    # risk-scenario pairs remain resident. Candidate training and evidence remain
    # independent; only their immutable input objects are reused.
    cache_key = (fit_dates, score_dates, scenario)
    cached = episode_cache.get(cache_key) if episode_cache is not None else None
    if cached is None:
        train = _rl_episode(split="fit", dates=fit_dates, **common)
        score = _rl_episode(split="score", dates=score_dates, **common)
        if episode_cache is not None:
            # Risk scenarios change frozen sleeve weights, not the fold calendar
            # or canonical price lookup. Intern those dominant immutable objects
            # across scenario entries so the bounded cache does not replicate the
            # multi-year price dictionaries three times.
            prior = next((pair for (prior_fit, prior_score, _), pair in
                          episode_cache.items()
                          if prior_fit == fit_dates and prior_score == score_dates), None)
            if prior is not None:
                train = replace(train, dataset=replace(
                    train.dataset, prices=prior[0].dataset.prices,
                    calendar=prior[0].dataset.calendar))
                score = replace(score, dataset=replace(
                    score.dataset, prices=prior[1].dataset.prices,
                    calendar=prior[1].dataset.calendar))
            episode_cache[cache_key] = (train, score)
    else:
        train, score = cached
    price_rows = {"rl_price_eligibility.jsonl": [], "rl_price_exclusions.jsonl": []}
    for episode in (train, score):
        for summary in episode.eligibility:
            row = {
                "trial_id": trial_id, "fold_id": fold_id, "seed": seed,
                "risk_scenario": scenario,
                "split": episode.dataset.split, "source_hashes": episode.source_hashes,
                **summary,
            }
            price_rows["rl_price_eligibility.jsonl"].append(row)
        for exclusion in episode.exclusions:
            row = {
                "trial_id": trial_id, "fold_id": fold_id, "seed": seed,
                "risk_scenario": scenario,
                "split": episode.dataset.split, "source_hashes": episode.source_hashes,
                **exclusion,
            }
            price_rows["rl_price_exclusions.jsonl"].append(row)
    for name, evidence in price_rows.items():
        _append_many(output / name, evidence)
    result = execute_study_rl_trial(
        study=study, authority=authority, component_id=arm["component_id"],
        context=FitContext(authority.study_id, trial_id, fold_id, seed,
                           {"environment_steps": steps}),
        train_dataset=train.dataset, score_dataset=score.dataset,
        risk_scenario=scenario, parameters=params,
        total_timesteps=steps,
        output_dir=output,
    )
    primary = result.metrics["certainty_equivalent_mean"]
    if not isinstance(primary, (int, float)) or not math.isfinite(primary):
        raise ValueError("RL trial returned an invalid certainty-equivalent objective")
    return {"certainty_equivalent_mean": float(primary),
            "telemetry": dict(result.telemetry), "metrics": dict(result.metrics),
            "actions": result.actions, "equity_curve": result.equity_curve,
            "price_rows": price_rows}


def _rl_scenarios(config: dict) -> tuple[str, ...]:
    objective = config["objectives"].get("rl_certainty_equivalent_v1", {})
    scenarios = objective.get("risk_scenarios", {})
    if (objective.get("primary") != "weekly_net_certainty_equivalent_return"
            or set(scenarios) != {"conservative", "balanced", "aggressive"}
            or set(config["portfolio"].get("risk_scenarios", {})) != set(scenarios)):
        raise ValueError("RL requires the three approved certainty-equivalent scenarios")
    return tuple(sorted(scenarios))


def _rl_select(candidates: list[tuple], tolerance: float) -> tuple:
    if not candidates:
        raise RuntimeError("No complete RL candidate for the required fold/seed matrix")
    best = max(item[2] for item in candidates)
    equivalent = [item for item in candidates if best - item[2] <= tolerance]
    return min(equivalent, key=lambda item: (len(json.dumps(item[1], sort_keys=True)),
                                              item[3], item[0]))


def _rl_checkpoint_rows(output: Path, trial_id: str, fold_id: str, seed: int,
                        scenario: str) -> tuple[dict, dict, tuple[Path, ...]]:
    def matching(path: Path, *, completed: bool) -> dict:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        selected = [row for row in rows if row.get("trial_id") == trial_id
                    and row.get("fold_id") == fold_id and row.get("seed") == seed
                    and row.get("risk_scenario") == scenario
                    and (not completed or row.get("status") == "complete")]
        if len(selected) != 1:
            raise ValueError(f"RL cell evidence is incomplete or duplicated: {trial_id}/{fold_id}")
        return selected[0]

    complete = matching(output / "rl_trial_ledger.jsonl", completed=True)
    resource = matching(output / "rl_resource_ledger.jsonl", completed=False)
    directory = output / "rl_cells" / complete["cell_id"]
    paths = tuple(directory / name for name in sorted(complete["artefact_sha256"]))
    if any(not path.is_file() or _hash(path) != complete["artefact_sha256"][path.name]
           for path in paths):
        raise ValueError("RL cell artefacts are incomplete or hash-mismatched")
    return complete, resource, paths


def _restore_rl_price_rows(output: Path, price_rows: dict[str, list[dict]]) -> None:
    for name, rows in price_rows.items():
        if name not in {"rl_price_eligibility.jsonl", "rl_price_exclusions.jsonl"}:
            raise ValueError("Unknown RL price evidence file")
        _append_many(output / name, rows)


def _sensitivity_terminal_failure(*, trial_id: str, proposal_index: int,
                                  started: float, completed_cells: int,
                                  expected_cells: int, active_cell: dict | None,
                                  exc: Exception) -> dict:
    return {
        "trial_id": trial_id,
        "proposal_index": proposal_index,
        "status": "failed",
        "selection_eligible": False,
        "exception_type": type(exc).__name__,
        "reason": f"{type(exc).__name__}: {exc}",
        "diagnostic_traceback": traceback.format_exc(),
        "failed_cell": active_cell,
        "completed_cells": completed_cells,
        "expected_cells": expected_cells,
        "resource_seconds": time.perf_counter() - started,
    }


def _run_controlled_sensitivity(
    *, study: ResolvedStudy, authority: VerifiedStudyAuthority, registry,
    arms: list[dict], outer: list[dict], inner: dict[str, list[dict]],
    designs: dict[str, dict], seeds: list[int], development_labels,
    rows: list[dict], bars: pl.DataFrame | None, outputs: pl.DataFrame | None,
    feature_sha256: str, outputs_sha256: str | None, fold_sha256: str,
    output: Path, checkpoints: CheckpointStore, source: Path | None,
) -> None:
    """Collect selection-ineligible matched contrasts before holdout access."""
    if not designs:
        return
    declaration = study.config["search"]["controlled_sensitivity"]
    fold_count = declaration["inner_folds_per_outer"]
    if any(len(inner[fold["fold_id"]]) < fold_count for fold in outer):
        raise ValueError("Controlled sensitivity requests more inner folds than exist")
    sensitivity_config = deepcopy(study.config)
    sensitivity_config["search"]["deep_training"] = dict(declaration["deep_training"])
    rl_steps = declaration["rl_training"]["total_timesteps"]
    _write_new(output / "sensitivity_design.json", {
        "schema_version": 1,
        "study_id": authority.study_id,
        "protocol_sha256": authority.protocol_sha256,
        "selection_eligible": False,
        "fold_policy": "first_declared_inner_folds_per_outer",
        "inner_folds_per_outer": fold_count,
        "seeds": seeds,
        "deep_training": declaration["deep_training"],
        "rl_training": declaration["rl_training"],
        "arms": designs,
    })
    LOGGER.info("Controlled sensitivity started arms=%d folds_per_outer=%d",
                len(arms), fold_count)
    total_candidates = sum(
        len(designs[arm["id"]]["proposals"])
        * (len(_rl_scenarios(study.config)) if arm["interface"] == "RLPolicy" else 1)
        for _outer_fold in outer for arm in arms
    )
    total_cells = total_candidates * fold_count * len(seeds)
    candidate_number = 0
    completed_total = 0
    for outer_fold in outer:
        outer_id = outer_fold["fold_id"]
        sensitivity_folds = inner[outer_id][:fold_count]
        # Shared by DQN and PPO for this outer fold only. The three scenario
        # entries avoid the former unbounded cross-fold episode retention.
        rl_episode_cache = {}
        for arm in arms:
            arm_id = arm["id"]
            design = designs[arm_id]
            for proposal_index, params in enumerate(design["proposals"]):
                scenarios = (_rl_scenarios(study.config)
                             if arm["interface"] == "RLPolicy" else (None,))
                for scenario in scenarios:
                    candidate_number += 1
                    scenario_token = f"-{scenario}" if scenario else ""
                    trial_id = f"SENS-{arm_id}{scenario_token}-{outer_id}-{proposal_index + 1:03d}"
                    proposal = {
                        "trial_id": trial_id, "arm_id": arm_id,
                        "component_id": arm["component_id"], "outer_fold_id": outer_id,
                        "risk_scenario": scenario, "proposal_index": proposal_index,
                        "status": "proposed", "selection_eligible": False,
                        "parameters": params,
                    }
                    _append(output / "sensitivity_trial_ledger.jsonl", proposal)
                    expected = len(sensitivity_folds) * len(seeds)
                    completed = 0
                    active_cell = None
                    started = time.perf_counter()
                    LOGGER.info(
                        "Controlled sensitivity candidate started candidate=%d/%d "
                        "cells=%d/%d arm=%s fold=%s scenario=%s trial=%s",
                        candidate_number, total_candidates, completed_total, total_cells,
                        arm_id, outer_id, scenario or "not_applicable", trial_id,
                    )
                    scores = []
                    try:
                        for part in sensitivity_folds:
                            fit_dates, stop_dates, score_dates = (_dates(part, name) for name in
                                ("fit_dates", "stopping_dates", "score_dates"))
                            if arm["interface"] == "SupervisedModel":
                                fit, stop, score = development_labels.partitions(
                                    fit_dates, stop_dates, score_dates)
                                if any(row["label_end_date"] >= stop_dates[0] for row in fit):
                                    raise ValueError("Sensitivity fit label reaches stopping window")
                                if any(row["label_end_date"] >= score_dates[0] for row in stop):
                                    raise ValueError("Sensitivity stopping label reaches score window")
                            for seed in seeds:
                                active_cell = {"inner_fold_id": part["fold_id"], "seed": seed,
                                               "risk_scenario": scenario}
                                identity = {"trial_id": trial_id, **active_cell,
                                            "parameters": params,
                                            "selection_eligible": False}
                                kind = ("supervised_sensitivity" if arm["interface"] == "SupervisedModel"
                                        else "rl_sensitivity")
                                prior_cell = checkpoints.get(kind, identity)
                                if arm["interface"] == "SupervisedModel":
                                    if prior_cell is None:
                                        _, ic, rmse, telemetry, _ = _fit_score(
                                            registry, arm, params, seed, trial_id, part["fold_id"],
                                            fit, stop, score, config=sensitivity_config,
                                            feature_pool=rows)
                                        record = {**active_cell, "trial_id": trial_id,
                                                  "outer_fold_id": outer_id,
                                                  "proposal_index": proposal_index,
                                                  "status": "complete", "ic": ic, "rmse": rmse,
                                                  "telemetry": telemetry,
                                                  "selection_eligible": False}
                                    else:
                                        record = prior_cell["record"]
                                    checkpoints.save(kind, identity, record,
                                                     reused_from=source.name if prior_cell else None)
                                else:
                                    if prior_cell is None:
                                        result = _rl_trial_cell(
                                            study=study, authority=authority, arm=arm, params=params,
                                            seed=seed, trial_id=trial_id, fold_id=part["fold_id"],
                                            scenario=scenario, fit_dates=fit_dates,
                                            score_dates=score_dates, rows=rows, bars=bars,
                                            outputs=outputs, feature_sha256=feature_sha256,
                                            outputs_sha256=outputs_sha256, fold_sha256=fold_sha256,
                                            output=output, total_timesteps=rl_steps,
                                            episode_cache=rl_episode_cache)
                                        record = {**active_cell, "trial_id": trial_id,
                                                  "outer_fold_id": outer_id,
                                                  "proposal_index": proposal_index,
                                                  "status": "complete",
                                                  "certainty_equivalent_mean":
                                                      result["certainty_equivalent_mean"],
                                                  "metrics": result["metrics"],
                                                  "telemetry": result["telemetry"],
                                                  "selection_eligible": False}
                                        complete_row, resource, paths = _rl_checkpoint_rows(
                                            output, trial_id, part["fold_id"], seed, scenario)
                                        price_rows = result["price_rows"]
                                    else:
                                        record = prior_cell["record"]["fit"]
                                        complete_row = prior_cell["record"]["rl_complete"]
                                        resource = prior_cell["record"]["rl_resource"]
                                        price_rows = prior_cell["record"].get("rl_price", {})
                                        paths = tuple(checkpoints.copy_artefact(prior_cell, relative)
                                                      for relative in prior_cell["artefact_sha256"])
                                        _append(output / "rl_trial_ledger.jsonl", complete_row)
                                        _append(output / "rl_resource_ledger.jsonl", resource)
                                        _restore_rl_price_rows(output, price_rows)
                                    checkpoints.save(kind, identity,
                                        {"fit": record, "rl_complete": complete_row,
                                         "rl_resource": resource, "rl_price": price_rows}, paths,
                                        reused_from=source.name if prior_cell else None)
                                _append(output / "sensitivity_fit_ledger.jsonl", record)
                                scores.append(record)
                                completed += 1
                                completed_total += 1
                        terminal = {
                            "trial_id": trial_id, "proposal_index": proposal_index,
                            "status": "complete", "selection_eligible": False,
                            "completed_cells": completed, "expected_cells": expected,
                            "resource_seconds": time.perf_counter() - started,
                        }
                        if arm["interface"] == "SupervisedModel":
                            terminal.update(
                                mean_ic=sum(item["ic"] for item in scores) / expected,
                                mean_rmse=sum(item["rmse"] for item in scores) / expected)
                        else:
                            terminal["mean_certainty_equivalent"] = sum(
                                item["certainty_equivalent_mean"] for item in scores) / expected
                        _append(output / "sensitivity_trial_ledger.jsonl", terminal)
                        LOGGER.info(
                            "Controlled sensitivity candidate completed candidate=%d/%d "
                            "cells=%d/%d arm=%s fold=%s scenario=%s trial=%s "
                            "elapsed_seconds=%.3f",
                            candidate_number, total_candidates, completed_total, total_cells,
                            arm_id, outer_id, scenario or "not_applicable", trial_id,
                            terminal["resource_seconds"],
                        )
                    except Exception as exc:
                        if active_cell is not None:
                            _append(output / "sensitivity_fit_ledger.jsonl", {
                                **active_cell, "trial_id": trial_id,
                                "outer_fold_id": outer_id, "proposal_index": proposal_index,
                                "status": "failed", "selection_eligible": False,
                                "exception_type": type(exc).__name__,
                                "reason": f"{type(exc).__name__}: {exc}"})
                        terminal = _sensitivity_terminal_failure(
                            trial_id=trial_id, proposal_index=proposal_index,
                            started=started, completed_cells=completed,
                            expected_cells=expected, active_cell=active_cell, exc=exc)
                        _append(output / "sensitivity_trial_ledger.jsonl", terminal)
                        LOGGER.warning(
                            "Controlled sensitivity candidate failed candidate=%d/%d "
                            "cells=%d/%d arm=%s fold=%s scenario=%s trial=%s error=%s: %s",
                            candidate_number, total_candidates, completed_total, total_cells,
                            arm_id, outer_id, scenario or "not_applicable", trial_id,
                            type(exc).__name__, exc,
                        )
    LOGGER.info("Controlled sensitivity completed candidates=%d cells=%d/%d",
                total_candidates, completed_total, total_cells)


def _run_rl_outer(*, study: ResolvedStudy, authority: VerifiedStudyAuthority,
                  arm: dict, outer_fold: dict, inner: list[dict], proposals: list[dict],
                  seeds: list[int], rows: list[dict], bars: pl.DataFrame,
                  outputs: pl.DataFrame, feature_sha256: str, outputs_sha256: str,
                  fold_sha256: str, output: Path, tolerance: float,
                  checkpoints: CheckpointStore, source: Path | None) -> None:
    outer_id = outer_fold["fold_id"]
    for scenario in _rl_scenarios(study.config):
        # The scenario loop is outermost, so three fold-specific pairs cover both
        # inner folds and the selected outer evaluation without retaining other
        # risk scenarios in memory.
        episode_cache = {}
        candidates = []
        for index, params in enumerate(proposals):
            trial_id = f"{arm['id']}-{scenario}-{outer_id}-{index + 1:03d}"
            _append(output / "trial_ledger.jsonl", {"trial_id": trial_id,
                "arm_id": arm["id"], "risk_scenario": scenario,
                "outer_fold_id": outer_id, "proposal_index": index,
                "status": "proposed", "parameters": params})
            evidence = []
            expected = len(inner) * len(seeds)
            active_cell = None
            trial_started = time.perf_counter()
            try:
                for fold in inner:
                    fit_dates, score_dates = (_dates(fold, name) for name in
                                               ("fit_dates", "score_dates"))
                    for seed in seeds:
                        active_cell = {"inner_fold_id": fold["fold_id"], "seed": seed,
                                       "risk_scenario": scenario}
                        identity = {"trial_id": trial_id, "fold_id": fold["fold_id"],
                                    "seed": seed, "scenario": scenario, "parameters": params}
                        prior_cell = checkpoints.get("rl_inner", identity)
                        if prior_cell is None:
                            result = _rl_trial_cell(
                                study=study, authority=authority, arm=arm, params=params,
                                seed=seed, trial_id=trial_id, fold_id=fold["fold_id"],
                                scenario=scenario, fit_dates=fit_dates, score_dates=score_dates,
                                rows=rows, bars=bars, outputs=outputs,
                                feature_sha256=feature_sha256, outputs_sha256=outputs_sha256,
                                fold_sha256=fold_sha256, output=output,
                                episode_cache=episode_cache)
                            record = {"trial_id": trial_id, "arm_id": arm["id"],
                                      "risk_scenario": scenario, "outer_fold_id": outer_id,
                                      "inner_fold_id": fold["fold_id"], "seed": seed,
                                      "certainty_equivalent_mean": result["certainty_equivalent_mean"],
                                      "telemetry": result["telemetry"]}
                            complete, resource, paths = _rl_checkpoint_rows(
                                output, trial_id, fold["fold_id"], seed, scenario)
                            price_rows = result["price_rows"]
                        else:
                            record = prior_cell["record"]["fit"]
                            complete = prior_cell["record"]["rl_complete"]
                            resource = prior_cell["record"]["rl_resource"]
                            price_rows = prior_cell["record"].get("rl_price", {})
                            paths = tuple(checkpoints.copy_artefact(prior_cell, relative)
                                          for relative in prior_cell["artefact_sha256"])
                            _append(output / "rl_trial_ledger.jsonl", complete)
                            _append(output / "rl_resource_ledger.jsonl", resource)
                            _restore_rl_price_rows(output, price_rows)
                        checkpoints.save("rl_inner", identity,
                                         {"fit": record, "rl_complete": complete,
                                          "rl_resource": resource, "rl_price": price_rows}, paths,
                                         reused_from=source.name if prior_cell else None)
                        _append(output / "fit_ledger.jsonl", record)
                        evidence.append(record)
                        active_cell = None
                if len(evidence) != expected:
                    raise ValueError("Incomplete RL inner fold/seed matrix")
                mean = sum(item["certainty_equivalent_mean"] for item in evidence) / expected
                duration = sum(item["telemetry"].get("duration_seconds", 0.0)
                               for item in evidence)
                candidates.append((trial_id, params, mean, duration))
                _append(output / "trial_ledger.jsonl", {"trial_id": trial_id,
                        "status": "complete", "mean_certainty_equivalent": mean,
                        "resource_seconds": duration})
            except Exception as exc:
                if active_cell is not None:
                    _append(output / "fit_ledger.jsonl", {
                        "trial_id": trial_id, "arm_id": arm["id"],
                        "outer_fold_id": outer_id, **active_cell, "status": "failed",
                        "exception_type": type(exc).__name__,
                        "reason": f"{type(exc).__name__}: {exc}"})
                _append(output / "trial_ledger.jsonl", {"trial_id": trial_id,
                        "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                        "exception_type": type(exc).__name__,
                        "diagnostic_traceback": traceback.format_exc(),
                        "failed_cell": active_cell,
                        "completed_cells": len(evidence), "expected_cells": expected,
                        "resource_seconds": time.perf_counter() - trial_started})
        selected = _rl_select(candidates, tolerance)
        lock = output / "locks" / f"{arm['id']}-{scenario}-{outer_id}.json"
        _write_new(lock, {"study_id": authority.study_id, "arm_id": arm["id"],
            "risk_scenario": scenario, "outer_fold_id": outer_id,
            "trial_id": selected[0], "parameters": selected[1],
            "inner_mean_certainty_equivalent": selected[2],
            "trial_ledger_sha256": _hash(output / "trial_ledger.jsonl"),
            "fit_ledger_sha256": _hash(output / "fit_ledger.jsonl"),
            "sealed_at": datetime.now(timezone.utc).isoformat()})
        fit_dates, score_dates = (_dates(outer_fold, name) for name in
                                   ("fit_dates", "score_dates"))
        for seed in seeds:
            identity = {"trial_id": selected[0], "fold_id": outer_id,
                        "seed": seed, "scenario": scenario, "parameters": selected[1]}
            prior_cell = checkpoints.get("rl_outer", identity)
            if prior_cell is None:
                result = _rl_trial_cell(
                    study=study, authority=authority, arm=arm, params=selected[1],
                    seed=seed, trial_id=selected[0], fold_id=outer_id,
                    scenario=scenario, fit_dates=fit_dates, score_dates=score_dates,
                    rows=rows, bars=bars, outputs=outputs,
                    feature_sha256=feature_sha256, outputs_sha256=outputs_sha256,
                    fold_sha256=fold_sha256, output=output,
                    episode_cache=episode_cache)
                record = {"arm_id": arm["id"], "risk_scenario": scenario,
                          "outer_fold_id": outer_id, "seed": seed,
                          "certainty_equivalent_mean": result["certainty_equivalent_mean"],
                          "telemetry": result["telemetry"]}
                complete, resource, paths = _rl_checkpoint_rows(
                    output, selected[0], outer_id, seed, scenario)
                price_rows = result["price_rows"]
            else:
                record = prior_cell["record"]["fit"]
                complete = prior_cell["record"]["rl_complete"]
                resource = prior_cell["record"]["rl_resource"]
                price_rows = prior_cell["record"].get("rl_price", {})
                paths = tuple(checkpoints.copy_artefact(prior_cell, relative)
                              for relative in prior_cell["artefact_sha256"])
                _append(output / "rl_trial_ledger.jsonl", complete)
                _append(output / "rl_resource_ledger.jsonl", resource)
                _restore_rl_price_rows(output, price_rows)
            record = record | {"selection_lock_sha256": _hash(lock)}
            checkpoints.save("rl_outer", identity,
                             {"fit": record, "rl_complete": complete,
                              "rl_resource": resource, "rl_price": price_rows}, paths,
                             reused_from=source.name if prior_cell else None)
            _append(output / "outer_fit_ledger.jsonl", record)


def run_study(study: ResolvedStudy, authority: VerifiedStudyAuthority, *, repository_root: Path,
              resume_from: Path | None = None, allow_code_drift: bool = False) -> Path:
    """Execute classical nested selection with no legacy-run or report dependency."""
    root = repository_root.resolve()
    registry, arms, outer, inner, holdout, feature_path, snapshot = _preflight(study, authority, root)
    config = study.config
    supervised_arms = [arm for arm in arms if arm["interface"] == "SupervisedModel"]
    rl_arms = [arm for arm in arms if arm["interface"] == "RLPolicy"]
    proposals = {
        arm["id"]: (_proposals(registry, arm, config) if arm in supervised_arms
                    else _rl_proposals(registry, arm, config))
        for arm in arms
    }
    sensitivity_designs = _controlled_sensitivity_designs(registry, arms, config)
    required = {"session_date", "security_id", "ticker", "label_end_date", "forward_return_5d", *F1}
    schema = pl.read_parquet_schema(feature_path)
    if not required <= set(schema):
        raise ValueError(f"Canonical feature snapshot lacks {sorted(required - set(schema))}")
    # The outcome column is deliberately absent from the initial feature view.
    rows = _feature_rows(root, config, authority, feature_path, snapshot["feature_sha256"])
    if rl_arms:
        bars, rl_outputs, rl_outputs_sha256 = _rl_inputs(
            root, config, feature_path, snapshot["feature_sha256"])
    else:
        bars, rl_outputs, rl_outputs_sha256 = None, None, None
    if len({(row["security_id"], row["session_date"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate security/session feature key")
    by_day: dict[date, list[dict]] = {}
    for row in rows:
        by_day.setdefault(row["session_date"], []).append(row)
    for fold in [*outer, *[part for group in inner.values() for part in group]]:
        fit, score = (_dates(fold, key) for key in ("fit_dates", "score_dates"))
        for row in _partition(by_day, fit):
            if row["label_end_date"] is None or row["label_end_date"] >= score[0]:
                raise ValueError("Training label reaches scoring boundary")
    holdout_dates = tuple(date.fromisoformat(day) for day in holdout["session_dates"])
    if holdout_dates != tuple(sorted(set(holdout_dates))):
        raise ValueError("Invalid holdout dates")
    development_labels = _DevelopmentLabelCache(feature_path, by_day, outer, inner,
                                                holdout_dates)
    runs_root = (root / config["outputs"]["runs_root"]).resolve()
    if not runs_root.is_relative_to(root):
        raise ValueError("Study runs root escapes repository")
    code_sha = code_state_sha256(root)
    source = None
    source_code_sha = None
    if allow_code_drift and resume_from is None:
        raise ValueError("Code-drift override requires a resume source")
    if resume_from is not None:
        source = Path(resume_from).resolve()
        if source.parent != runs_root or not source.is_dir():
            raise ValueError("Resume source must be one run in the approved runs root")
        started_path = source / "started.json"
        if not started_path.is_file() or (source / "completion.json").exists():
            raise ValueError("Resume source must be an incomplete expanded study")
        prior = json.loads(started_path.read_text(encoding="utf-8"))
        source_code_sha = prior.get("code_state_sha256")
        if (prior.get("protocol_sha256") != authority.protocol_sha256
                or prior.get("input_sha256") != dict(authority.input_sha256)
                or (source_code_sha != code_sha and not allow_code_drift)
                or prior.get("study_id") != authority.study_id
                or prior.get("run_id") != source.name
                or not (source / "protocol.json").is_file()
                or json.loads((source / "protocol.json").read_text(encoding="utf-8")) != config):
            raise ValueError("Resume source protocol, input or code state mismatch")
        unmarked = not (source / "failure.json").is_file()
        if unmarked and process_still_running(prior):
            raise ValueError("Resume source process is still running")
        # Validate every indexed checkpoint and the terminal marker before creating a run.
        CheckpointStore(source, protocol_sha256=authority.protocol_sha256,
                        code_sha256=code_sha, source=source,
                        source_code_sha256=source_code_sha,
                        allow_unmarked=unmarked)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = runs_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"run_id": run_id, "study_id": authority.study_id,
                "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": authority.protocol_sha256,
                "input_sha256": dict(authority.input_sha256),
                "feature_path": str(feature_path.relative_to(root)),
                "feature_sha256": snapshot["feature_sha256"],
                "vintage_id": authority.vintage_id, "claim_status": authority.claim_status,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "code_state_sha256": code_sha,
                "pid": os.getpid(),
                "process_create_time": psutil.Process(os.getpid()).create_time()}
    if source is not None:
        source_index = source / "checkpoint_index.jsonl"
        metadata["resume_lineage"] = {
            "source_run_id": source.name,
            "source_started_sha256": sha256(source / "started.json"),
            "source_failure_sha256": sha256(source / "failure.json")
            if (source / "failure.json").is_file() else None,
            "source_checkpoint_index_sha256": sha256(source_index)
            if source_index.is_file() else None,
            "relation": "verified_cell_continuation_across_authorised_code_drift"
            if source_code_sha != code_sha else "verified_cell_continuation",
            "code_drift_authorised": bool(source_code_sha != code_sha),
            "source_code_state_sha256": source_code_sha,
            "resumed_code_state_sha256": code_sha,
            "source_terminal_status": "failed" if (source / "failure.json").is_file()
            else "interrupted_without_marker"}
    _write_new(output / "protocol.json", config)
    _write_new(output / "started.json", metadata)
    checkpoints = CheckpointStore(output, protocol_sha256=authority.protocol_sha256,
                                  code_sha256=code_sha, source=source,
                                  source_code_sha256=source_code_sha,
                                  allow_unmarked=source is not None and not
                                  (source / "failure.json").is_file(),
                                  code_guard=lambda: code_state_sha256(root))
    LOGGER.info("Expanded study started run_id=%s outer_folds=%d arms=%d seeds=%d",
                run_id, len(outer), len(arms), len(config["reproducibility"]["seeds"]))
    seeds = config["reproducibility"]["seeds"]
    tolerance = float(config["selection"]["practical_equivalence_tolerance"])
    predictions_out = []
    try:
        _run_controlled_sensitivity(
            study=study, authority=authority, registry=registry, arms=arms,
            outer=outer, inner=inner, designs=sensitivity_designs, seeds=seeds,
            development_labels=development_labels, rows=rows, bars=bars,
            outputs=rl_outputs, feature_sha256=snapshot["feature_sha256"],
            outputs_sha256=rl_outputs_sha256,
            fold_sha256=authority.input_sha256[
                config["validation"]["inner_windows_manifest"]],
            output=output, checkpoints=checkpoints, source=source,
        )
        for outer_fold in outer:
            outer_id = outer_fold["fold_id"]
            LOGGER.info("Outer fold started fold=%s", outer_id)
            for arm in supervised_arms:
                arm_id = arm["id"]
                LOGGER.info("Supervised arm started fold=%s arm=%s candidates=%d",
                            outer_id, arm_id, len(proposals[arm_id]))
                candidates = []
                for index, params in enumerate(proposals[arm_id]):
                    trial_id = f"{arm_id}-{outer_id}-{index + 1:03d}"
                    LOGGER.info("Inner candidate started fold=%s arm=%s trial=%s",
                                outer_id, arm_id, trial_id)
                    _append(output / "trial_ledger.jsonl", {"trial_id": trial_id, "arm_id": arm_id,
                            "outer_fold_id": outer_id, "proposal_index": index, "status": "proposed",
                            "parameters": params})
                    trial_evidence = []
                    expected = len(inner[outer_id]) * len(seeds)
                    active_cell = None
                    trial_started = time.perf_counter()
                    try:
                        for part in inner[outer_id]:
                            fit_dates, stop_dates, score_dates = (_dates(part, name) for name in
                                ("fit_dates", "stopping_dates", "score_dates"))
                            fit, stop, score = development_labels.partitions(
                                fit_dates, stop_dates, score_dates)
                            if any(row["label_end_date"] >= stop_dates[0] for row in fit):
                                raise ValueError("Fit label reaches stopping window")
                            if any(row["label_end_date"] >= score_dates[0] for row in stop):
                                raise ValueError("Stopping label reaches score window")
                            for seed in seeds:
                                active_cell = {"inner_fold_id": part["fold_id"], "seed": seed}
                                identity = {"trial_id": trial_id, "inner_fold_id": part["fold_id"],
                                            "seed": seed, "parameters": params}
                                prior_cell = checkpoints.get("supervised_inner", identity)
                                if prior_cell is None:
                                    _, ic, rmse, telemetry, _ = _fit_score(registry, arm, params, seed,
                                        trial_id, part["fold_id"], fit, stop, score,
                                        config=config, feature_pool=rows)
                                    evidence = {"trial_id": trial_id, "outer_fold_id": outer_id,
                                                "inner_fold_id": part["fold_id"], "seed": seed,
                                                "ic": ic, "rmse": rmse, "telemetry": telemetry}
                                else:
                                    evidence = prior_cell["record"]
                                checkpoints.save("supervised_inner", identity, evidence,
                                                 reused_from=source.name if prior_cell else None)
                                _append(output / "fit_ledger.jsonl", evidence)
                                trial_evidence.append(evidence)
                                active_cell = None
                        if len(trial_evidence) != expected:
                            raise ValueError("Incomplete inner fold/seed matrix")
                        mean_ic = sum(item["ic"] for item in trial_evidence) / expected
                        mean_rmse = sum(item["rmse"] for item in trial_evidence) / expected
                        resources = sum(item["telemetry"]["wall_seconds"] for item in trial_evidence)
                        candidates.append((trial_id, params, mean_ic, mean_rmse, resources))
                        _append(output / "trial_ledger.jsonl", {"trial_id": trial_id,
                                "status": "complete", "mean_ic": mean_ic, "mean_rmse": mean_rmse,
                                "resource_seconds": resources})
                        LOGGER.info("Inner candidate completed fold=%s arm=%s trial=%s",
                                    outer_id, arm_id, trial_id)
                    except Exception as exc:
                        if active_cell is not None:
                            _append(output / "fit_ledger.jsonl", {
                                "trial_id": trial_id, "arm_id": arm_id,
                                "outer_fold_id": outer_id, **active_cell, "status": "failed",
                                "exception_type": type(exc).__name__,
                                "reason": f"{type(exc).__name__}: {exc}"})
                        _append(output / "trial_ledger.jsonl", {"trial_id": trial_id,
                                "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                                "exception_type": type(exc).__name__,
                                "diagnostic_traceback": traceback.format_exc(),
                                "failed_cell": active_cell,
                                "completed_cells": len(trial_evidence),
                                "expected_cells": expected,
                                "resource_seconds": time.perf_counter() - trial_started})
                        LOGGER.warning("Inner candidate failed fold=%s arm=%s trial=%s error=%s: %s",
                                       outer_id, arm_id, trial_id, type(exc).__name__, exc)
                if not candidates:
                    raise RuntimeError(f"No complete candidates for {arm_id}/{outer_id}")
                best_ic = max(item[2] for item in candidates)
                equivalent = [item for item in candidates if best_ic - item[2] <= tolerance]
                selected = min(equivalent, key=lambda item: (len(json.dumps(item[1], sort_keys=True)),
                                                                item[3], item[4], item[0]))
                lock_path = output / "locks" / f"{arm_id}-{outer_id}.json"
                _write_new(lock_path, {"study_id": authority.study_id, "arm_id": arm_id,
                         "outer_fold_id": outer_id, "trial_id": selected[0],
                         "parameters": selected[1], "inner_mean_ic": selected[2],
                         "trial_ledger_sha256": _hash(output / "trial_ledger.jsonl"),
                         "fit_ledger_sha256": _hash(output / "fit_ledger.jsonl"),
                         "sealed_at": datetime.now(timezone.utc).isoformat()})
                # Outer labels enter only after the corresponding selection lock exists.
                fit_dates, score_dates = (_dates(outer_fold, name) for name in ("fit_dates", "score_dates"))
                stop_dates = fit_dates[-max(5, min(20, len(fit_dates) // 10)):]
                fit_dates = tuple(day for day in fit_dates if day < stop_dates[0] and
                                  all(row["label_end_date"] < stop_dates[0] for row in by_day[day]))
                fit, stop, score = development_labels.partitions(
                    fit_dates, stop_dates, score_dates)
                if any(row["label_end_date"] >= score_dates[0] for row in stop):
                    raise ValueError("Outer stopping label reaches score window")
                for seed in seeds:
                    LOGGER.info("Outer evaluation started fold=%s arm=%s seed=%s",
                                outer_id, arm_id, seed)
                    identity = {"arm_id": arm_id, "outer_fold_id": outer_id,
                                "seed": seed, "trial_id": selected[0],
                                "parameters": selected[1]}
                    prior_cell = checkpoints.get("supervised_outer", identity)
                    relative = f"cell_predictions/{cell_key('supervised_outer', identity)}.parquet"
                    if prior_cell is None:
                        _, ic, rmse, telemetry, predictions = _fit_score(registry, arm, selected[1], seed,
                            selected[0], outer_id, fit, stop, score, config=config,
                            feature_pool=rows, predict=True)
                        cell_rows = [{"arm_id": arm_id, "partition": "outer", "fold_id": outer_id,
                                      "seed": seed, "session_date": row["session_date"],
                                      "security_id": row["security_id"], "ticker": row["ticker"],
                                      "actual_forward_return_5d": row["forward_return_5d"],
                                      "predicted_return_5d": float(value)}
                                     for row, value in zip(score, predictions, strict=True)]
                        cell_path = output / relative
                        cell_path.parent.mkdir(parents=True, exist_ok=True)
                        if cell_path.exists():
                            raise FileExistsError(cell_path)
                        pl.DataFrame(cell_rows).write_parquet(cell_path)
                        evidence = {"arm_id": arm_id, "outer_fold_id": outer_id,
                                    "seed": seed, "ic": ic, "rmse": rmse,
                                    "telemetry": telemetry,
                                    "selection_lock_sha256": _hash(lock_path)}
                    else:
                        cell_path = checkpoints.copy_artefact(prior_cell, relative)
                        cell_rows = pl.read_parquet(cell_path).to_dicts()
                        evidence = prior_cell["record"] | {
                            "selection_lock_sha256": _hash(lock_path)}
                        if (len(cell_rows) != len(score)
                                or [(r["session_date"], r["security_id"]) for r in cell_rows]
                                != [(r["session_date"], r["security_id"]) for r in score]):
                            raise ValueError("Resumed outer prediction rows do not match frozen score window")
                    checkpoints.save("supervised_outer", identity, evidence, (cell_path,),
                                     reused_from=source.name if prior_cell else None)
                    _append(output / "outer_fit_ledger.jsonl", evidence)
                    predictions_out.extend(cell_rows)
            for arm in rl_arms:
                LOGGER.info("RL arm started fold=%s arm=%s scenarios=%d candidates=%d",
                            outer_id, arm["id"], len(_rl_scenarios(config)),
                            len(proposals[arm["id"]]))
                _run_rl_outer(
                    study=study, authority=authority, arm=arm,
                    outer_fold=outer_fold, inner=inner[outer_id],
                    proposals=proposals[arm["id"]], seeds=seeds, rows=rows,
                    bars=bars, outputs=rl_outputs,
                    feature_sha256=snapshot["feature_sha256"],
                    outputs_sha256=rl_outputs_sha256,
                    fold_sha256=authority.input_sha256[config["validation"]["inner_windows_manifest"]],
                    output=output, tolerance=tolerance,
                    checkpoints=checkpoints, source=source,
                )
        # Pool only complete inner evidence across outer histories. Outer scores
        # remain diagnostic and never determine final holdout parameters.
        fit_evidence = [json.loads(line) for line in
                        (output / "fit_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
        complete_trials = {json.loads(line)["trial_id"] for line in
                           (output / "trial_ledger.jsonl").read_text(encoding="utf-8").splitlines()
                           if json.loads(line)["status"] == "complete"}
        final_choices = {}
        for arm in supervised_arms:
            ranked = []
            for index, params in enumerate(proposals[arm["id"]]):
                records = [record for record in fit_evidence
                           if record["trial_id"] in {
                               f"{arm['id']}-{fold['fold_id']}-{index + 1:03d}" for fold in outer}]
                expected = sum(len(inner[fold["fold_id"]]) for fold in outer) * len(seeds)
                if len(records) != expected or any(
                    f"{arm['id']}-{fold['fold_id']}-{index + 1:03d}" not in complete_trials
                    for fold in outer):
                    continue
                # Each inner fold and seed receives equal weight.
                ranked.append((index, params,
                               sum(record["ic"] for record in records) / expected,
                               sum(record["rmse"] for record in records) / expected,
                               sum(record["telemetry"]["wall_seconds"] for record in records)))
            if not ranked:
                raise RuntimeError(f"No globally complete inner candidate for {arm['id']}")
            best_ic = max(item[2] for item in ranked)
            equivalent = [item for item in ranked if best_ic - item[2] <= tolerance]
            chosen = min(equivalent, key=lambda item: (len(json.dumps(item[1], sort_keys=True)),
                                                         item[3], item[4], item[0]))
            final_choices[arm["id"]] = {"proposal_index": chosen[0], "parameters": chosen[1],
                                        "pooled_inner_mean_ic": chosen[2],
                                        "pooled_inner_mean_rmse": chosen[3]}
        # The same sealed holdout episode inputs serve both policy families. Keep
        # one pair per risk scenario; policies still train and evaluate separately.
        holdout_rl_episode_cache = {}
        for arm in rl_arms:
            for scenario in _rl_scenarios(config):
                ranked = []
                for index, params in enumerate(proposals[arm["id"]]):
                    trial_ids = {
                        f"{arm['id']}-{scenario}-{fold['fold_id']}-{index + 1:03d}"
                        for fold in outer
                    }
                    records = [record for record in fit_evidence
                               if record.get("trial_id") in trial_ids
                               and record.get("risk_scenario") == scenario]
                    expected = sum(len(inner[fold["fold_id"]]) for fold in outer) * len(seeds)
                    if len(records) != expected or not trial_ids <= complete_trials:
                        continue
                    ranked.append((index, params,
                        sum(record["certainty_equivalent_mean"] for record in records) / expected,
                        sum(record.get("telemetry", {}).get("duration_seconds", 0.0)
                            for record in records)))
                selected = _rl_select(
                    [(f"{arm['id']}-{scenario}-global-{item[0] + 1:03d}",
                      item[1], item[2], item[3]) for item in ranked], tolerance)
                proposal_index = next(index for index, params in enumerate(proposals[arm["id"]])
                                      if params == selected[1])
                final_choices[f"{arm['id']}::{scenario}"] = {
                    "proposal_index": proposal_index,
                    "parameters": selected[1],
                    "pooled_inner_mean_certainty_equivalent": selected[2],
                }
        # A final lock is required before any descriptive holdout label or prediction is opened.
        _write_new(output / "family_locks.json", {"study_id": authority.study_id,
            "outer_lock_hashes": {p.name: _hash(p) for p in sorted((output / "locks").glob("*.json"))},
            "inner_fit_ledger_sha256": _hash(output / "fit_ledger.jsonl"),
            "final_choices": final_choices,
            "sealed_at": datetime.now(timezone.utc).isoformat(),
            "holdout_selection": "forbidden"})
        LOGGER.info("Family locks sealed run_id=%s; descriptive holdout may now open", run_id)
        if any(day not in by_day for day in holdout_dates):
            raise ValueError("Feature snapshot lacks holdout session")
        # The globally complete inner evidence determines final parameters.
        final_fit = tuple(day for day in sorted(by_day) if day < holdout_dates[0]
                          and all(row["label_end_date"] is not None and row["label_end_date"] < holdout_dates[0]
                                  for row in by_day[day]))
        final_stop = final_fit[-max(5, min(20, len(final_fit) // 10)):]
        final_train = tuple(day for day in final_fit if day < final_stop[0] and
                            all(row["label_end_date"] < final_stop[0] for row in by_day[day]))
        # Final training may include pre-holdout dates outside the nested folds.
        # Keep those labels separate from the exact nested-development cache.
        final_train_rows = _labelled(feature_path, by_day, final_train)
        final_stop_rows = _labelled(feature_path, by_day, final_stop)
        holdout_rows = _holdout_labelled(feature_path, by_day, holdout_dates,
                                         output / "family_locks.json")
        for arm in supervised_arms:
            choice = final_choices[arm["id"]]
            fit, stop, score = final_train_rows, final_stop_rows, holdout_rows
            for seed in seeds:
                LOGGER.info("Descriptive holdout fit started arm=%s seed=%s", arm["id"], seed)
                trial_id = f"{arm['id']}-global-{choice['proposal_index'] + 1:03d}"
                identity = {"arm_id": arm["id"], "seed": seed, "trial_id": trial_id,
                            "parameters": choice["parameters"]}
                prior_cell = checkpoints.get("supervised_holdout", identity)
                relative = f"cell_predictions/{cell_key('supervised_holdout', identity)}.parquet"
                if prior_cell is None:
                    _, ic, rmse, telemetry, predictions = _fit_score(
                        registry, arm, choice["parameters"], seed, trial_id,
                        "holdout", fit, stop, score, config=config, feature_pool=rows,
                        predict=True)
                    cell_rows = [{"arm_id": arm["id"], "partition": "descriptive_holdout",
                                  "fold_id": "holdout", "seed": seed,
                                  "session_date": row["session_date"],
                                  "security_id": row["security_id"], "ticker": row["ticker"],
                                  "actual_forward_return_5d": row["forward_return_5d"],
                                  "predicted_return_5d": float(value)}
                                 for row, value in zip(score, predictions, strict=True)]
                    cell_path = output / relative
                    cell_path.parent.mkdir(parents=True, exist_ok=True)
                    if cell_path.exists():
                        raise FileExistsError(cell_path)
                    pl.DataFrame(cell_rows).write_parquet(cell_path)
                    record = {"arm_id": arm["id"], "seed": seed,
                              "ic": ic, "rmse": rmse, "telemetry": telemetry}
                else:
                    cell_path = checkpoints.copy_artefact(prior_cell, relative)
                    cell_rows = pl.read_parquet(cell_path).to_dicts()
                    record = prior_cell["record"]
                    if (len(cell_rows) != len(score)
                            or [(r["session_date"], r["security_id"]) for r in cell_rows]
                            != [(r["session_date"], r["security_id"]) for r in score]):
                        raise ValueError("Resumed holdout prediction rows do not match sealed calendar")
                record = record | {"family_locks_sha256": _hash(output / "family_locks.json")}
                checkpoints.save("supervised_holdout", identity, record, (cell_path,),
                                 reused_from=source.name if prior_cell else None)
                _append(output / "holdout_fit_ledger.jsonl", record)
                predictions_out.extend(cell_rows)
        for arm in rl_arms:
            for scenario in _rl_scenarios(config):
                choice = final_choices[f"{arm['id']}::{scenario}"]
                for seed in seeds:
                    LOGGER.info("Descriptive RL holdout fit started arm=%s scenario=%s seed=%s",
                                arm["id"], scenario, seed)
                    trial_id = f"{arm['id']}-{scenario}-global-{choice['proposal_index'] + 1:03d}"
                    identity = {"trial_id": trial_id, "fold_id": "holdout",
                                "seed": seed, "scenario": scenario,
                                "parameters": choice["parameters"]}
                    prior_cell = checkpoints.get("rl_holdout", identity)
                    if prior_cell is None:
                        result = _rl_trial_cell(
                            study=study, authority=authority, arm=arm,
                            params=choice["parameters"], seed=seed, trial_id=trial_id,
                            fold_id="holdout", scenario=scenario,
                            fit_dates=final_train, score_dates=holdout_dates,
                            rows=rows, bars=bars, outputs=rl_outputs,
                            feature_sha256=snapshot["feature_sha256"],
                            outputs_sha256=rl_outputs_sha256,
                            fold_sha256=authority.input_sha256[
                                config["validation"]["final_holdout"]["manifest"]],
                            output=output,
                            episode_cache=holdout_rl_episode_cache)
                        record = {"arm_id": arm["id"], "risk_scenario": scenario,
                                  "seed": seed,
                                  "certainty_equivalent_mean": result["certainty_equivalent_mean"],
                                  "telemetry": result["telemetry"]}
                        complete, resource, paths = _rl_checkpoint_rows(
                            output, trial_id, "holdout", seed, scenario)
                        price_rows = result["price_rows"]
                    else:
                        record = prior_cell["record"]["fit"]
                        complete = prior_cell["record"]["rl_complete"]
                        resource = prior_cell["record"]["rl_resource"]
                        price_rows = prior_cell["record"].get("rl_price", {})
                        paths = tuple(checkpoints.copy_artefact(prior_cell, relative)
                                      for relative in prior_cell["artefact_sha256"])
                        _append(output / "rl_trial_ledger.jsonl", complete)
                        _append(output / "rl_resource_ledger.jsonl", resource)
                        _restore_rl_price_rows(output, price_rows)
                    record = record | {"family_locks_sha256": _hash(output / "family_locks.json")}
                    checkpoints.save("rl_holdout", identity,
                                     {"fit": record, "rl_complete": complete,
                                      "rl_resource": resource, "rl_price": price_rows}, paths,
                                     reused_from=source.name if prior_cell else None)
                    _append(output / "holdout_fit_ledger.jsonl", record)
        pl.DataFrame(predictions_out).write_parquet(output / "predictions.parquet")
        if code_state_sha256(root) != code_sha:
            raise ValueError("Study code changed before completion audit")
        _write_new(output / "metadata.json", metadata | {"status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "arm_ids": [arm["id"] for arm in arms],
            "outer_fold_ids": [fold["fold_id"] for fold in outer],
            "holdout_role": "descriptive_only"})
        artefacts = {str(path.relative_to(output)).replace("\\", "/"): _hash(path)
                     for path in sorted(output.rglob("*")) if path.is_file()}
        _write_new(output / "audit.json", {"status": "passed", "artefact_sha256": artefacts,
                   "audited_at": datetime.now(timezone.utc).isoformat()})
        _write_new(output / "completion.json", {"run_id": run_id, "status": "complete",
                   "audit_sha256": _hash(output / "audit.json"),
                   "completed_at": datetime.now(timezone.utc).isoformat()})
        LOGGER.info("Expanded study completed run_id=%s", run_id)
        return output
    except BaseException as exc:
        checkpoint_index = output / "checkpoint_index.jsonl"
        _write_new(output / "failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "failed_at": datetime.now(timezone.utc).isoformat(),
                   "checkpoint_index_sha256": sha256(checkpoint_index)
                   if checkpoint_index.is_file() else None})
        LOGGER.exception("Expanded study failed run_id=%s", run_id)
        raise
