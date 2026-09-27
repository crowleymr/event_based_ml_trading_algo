"""Fold-owned, fixed-specification F1 predictions for a causal stacked view.

The authoritative study runner must supply approved, pinned partitions and call
this module. This module does not select a model, read a holdout, or open a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import polars as pl

from trading_pipeline.experiments.contracts import FitContext
from trading_pipeline.experiments.registry import ComponentRegistry
from trading_pipeline.features import F1
from trading_pipeline.features.model_outputs import INPUT_COLUMNS, build_oof_store, load_oof_store


@dataclass(frozen=True)
class BaseLayer:
    component_id: str
    parameters: Mapping[str, object]
    seed: int


@dataclass(frozen=True)
class EnsembleLayer:
    component_id: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class CrossFitWindow:
    fold_id: str
    fit_dates: tuple[date, ...]
    stopping_dates: tuple[date, ...]
    score_dates: tuple[date, ...]


def closeout_f1_base_layers() -> tuple[tuple[BaseLayer, ...], EnsembleLayer]:
    """Human-declared fixed upstream layer; approval still belongs to study authority."""
    layers = (
        BaseLayer("supervised.elastic_net.v1", {"alpha": 0.001, "l1_ratio": 0.5}, 41),
        BaseLayer("supervised.hist_gbt.v1", {"max_leaf_nodes": 15,
                                            "l2_regularization": 10.0}, 41),
        BaseLayer("supervised.xgboost.v1", {"max_depth": 3, "reg_lambda": 10.0}, 41),
    )
    return layers, EnsembleLayer("derived.equal_weight_mean.v1",
                                 tuple(layer.component_id for layer in layers))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dates(values: Sequence[date], name: str) -> None:
    if not values or any(type(value) is not date for value in values) or tuple(values) != tuple(sorted(set(values))):
        raise ValueError(f"{name} must contain sorted unique session dates")


def expanding_crossfit_windows(
    *, calendar: Sequence[date], max_label_end_by_session: Mapping[date, date],
    protected_start_date: date, minimum_fit_sessions: int,
    stopping_sessions: int, score_block_sessions: int,
) -> tuple[CrossFitWindow, ...]:
    """Build deterministic blocks, freezing fit/stop data before the holdout.

    Protected-period rows are prediction-only.  Their labels never enter a later
    fit or stopping partition, while all protected sessions still receive the
    same causal upstream information available to every downstream family.
    """
    days = tuple(calendar)
    _dates(days, "calendar")
    if protected_start_date not in set(days):
        raise ValueError("Protected start must be a canonical session")
    if (type(minimum_fit_sessions) is not int or minimum_fit_sessions < 1
            or type(stopping_sessions) is not int or stopping_sessions < 1
            or type(score_block_sessions) is not int or score_block_sessions < 1):
        raise ValueError("Cross-fit session counts must be positive integers")
    if set(days) != set(max_label_end_by_session):
        raise ValueError("Every canonical session requires a maximum label end")
    first_score = minimum_fit_sessions + stopping_sessions
    if first_score >= len(days):
        raise ValueError("Cross-fit history exhausts the calendar")
    protected_index = days.index(protected_start_date)
    starts = list(range(first_score, protected_index, score_block_sessions))
    if protected_index >= first_score:
        starts.append(protected_index)
        starts.extend(range(protected_index + score_block_sessions, len(days), score_block_sessions))
    windows = []
    for index, start in enumerate(starts):
        boundary = protected_index if start >= protected_index else start
        score_end = min(start + score_block_sessions,
                        protected_index if start < protected_index else len(days))
        score = days[start:score_end]
        eligible_stop = tuple(day for day in days[:boundary]
                              if max_label_end_by_session[day] < score[0])
        stop = eligible_stop[-stopping_sessions:]
        if len(stop) != stopping_sessions:
            raise ValueError("Stopping history is incomplete")
        fit = tuple(day for day in days[:days.index(stop[0])]
                    if max_label_end_by_session[day] < stop[0])
        if len(fit) < minimum_fit_sessions:
            continue
        if score:
            windows.append(CrossFitWindow(f"upstream_{index:04d}", fit, stop, score))
    covered = {day for window in windows for day in window.score_dates}
    if not windows:
        raise ValueError("No cross-fit window survives label-boundary purging")
    expected = set(days[days.index(windows[0].score_dates[0]):])
    if covered != expected:
        raise ValueError("Cross-fit windows do not cover every eligible session")
    return tuple(windows)


def _validate(frame: pl.DataFrame, layers: Sequence[BaseLayer], windows: Sequence[CrossFitWindow],
              registry: ComponentRegistry, protected_start_date: date,
              minimum_fit_sessions: int, score_block_sessions: int,
              ensemble: EnsembleLayer | None) -> pl.DataFrame:
    if type(protected_start_date) is not date:
        raise ValueError("An explicit protected holdout start date is required")
    needed = {"security_id", "session_date", "label_end_date", "forward_return_5d", *F1}
    if needed - set(frame.columns):
        raise ValueError(f"Canonical F1 frame lacks {sorted(needed - set(frame.columns))}")
    frame = frame.with_columns(pl.col("session_date").cast(pl.Date),
                               pl.col("label_end_date").cast(pl.Date))
    if frame.is_empty() or frame.select("security_id", "session_date").is_duplicated().any():
        raise ValueError("Canonical F1 keys must be nonempty and unique")
    if frame.select(pl.any_horizontal(pl.col("security_id").is_null(),
                                      pl.col("session_date").is_null()).any()).item():
        raise ValueError("Canonical F1 keys may not be null")
    if not layers or not windows:
        raise ValueError("Frozen upstream base_layers and cross_fit_windows are required")
    if type(minimum_fit_sessions) is not int or minimum_fit_sessions < 1:
        raise ValueError("minimum_fit_sessions must be positive")
    if type(score_block_sessions) is not int or score_block_sessions < 1:
        raise ValueError("score_block_sessions must be positive")
    ids = [layer.component_id for layer in layers]
    if len(set(ids)) != len(ids):
        raise ValueError("Upstream component IDs must be unique")
    if ensemble is not None and (not ensemble.component_id or ensemble.component_id in ids
                                 or len(ensemble.members) < 2
                                 or len(set(ensemble.members)) != len(ensemble.members)
                                 or set(ensemble.members) != set(ids)
                                 or len({layer.seed for layer in layers}) != 1):
        raise ValueError("Derived ensemble must name every unique declared base component")
    for layer in layers:
        if not isinstance(layer.parameters, Mapping) or not layer.parameters or type(layer.seed) is not int:
            raise ValueError("Every base layer requires fixed parameters and integer seed")
        if registry.spec(layer.component_id).interface != "SupervisedModel":
            raise ValueError("Upstream base layer must be a registered supervised model")
    seen_scores: set[date] = set()
    previous_score: date | None = None
    previous_fit: set[date] = set()
    by_day = set(frame["session_date"].to_list())
    for window in windows:
        if not window.fold_id:
            raise ValueError("Every cross-fit window requires a fold ID")
        for name in ("fit_dates", "stopping_dates", "score_dates"):
            _dates(getattr(window, name), name)
            if set(getattr(window, name)) - by_day:
                raise ValueError(f"Cross-fit {name} absent from canonical snapshot")
        if (len(window.fit_dates) < minimum_fit_sessions
                or not 1 <= len(window.score_dates) <= score_block_sessions):
            raise ValueError("Cross-fit window violates declared fit minimum or score block size")
        if not (window.fit_dates[-1] < window.stopping_dates[0]
                and window.stopping_dates[-1] < window.score_dates[0]):
            raise ValueError("Cross-fit windows must be distinct and chronological")
        if (window.score_dates[0] >= protected_start_date
                and (window.fit_dates[-1] >= protected_start_date
                     or window.stopping_dates[-1] >= protected_start_date)):
            raise ValueError("Protected-period labels cannot enter upstream fitting")
        if previous_score is not None and (window.score_dates[0] <= previous_score
                                           or not previous_fit <= set(window.fit_dates)):
            raise ValueError("Cross-fit score dates must advance with expanding fit history")
        if seen_scores.intersection(window.score_dates):
            raise ValueError("Cross-fit score dates overlap")
        seen_scores.update(window.score_dates)
        previous_score = window.score_dates[-1]
        previous_fit = set(window.fit_dates)
        fit = frame.filter(pl.col("session_date").is_in(window.fit_dates))
        stop = frame.filter(pl.col("session_date").is_in(window.stopping_dates))
        for partition, boundary in ((fit, window.stopping_dates[0]),
                                    (stop, window.score_dates[0])):
            if partition.filter(pl.col("label_end_date").is_null()
                                | (pl.col("label_end_date") >= boundary)
                                | pl.col("forward_return_5d").is_null()).height:
                raise ValueError("Training or stopping label reaches protected boundary")
    if len({window.fold_id for window in windows}) != len(windows):
        raise ValueError("Cross-fit fold IDs must be unique")
    return frame


def _matrix(frame: pl.DataFrame, *, label: bool) -> tuple[np.ndarray, np.ndarray | None]:
    x = np.asarray([[np.nan if row[column] is None else float(row[column])
                     for column in F1] for row in frame.to_dicts()], dtype=np.float64)
    if not len(x) or np.isinf(x).any():
        raise ValueError("Empty or infinite F1 matrix")
    if not label:
        return x, None
    y = np.asarray(frame["forward_return_5d"].to_list(), dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("Fit labels must be finite")
    return x, y


def produce_causal_stack(*, frame: pl.DataFrame, registry: ComponentRegistry,
                         layers: Sequence[BaseLayer], windows: Sequence[CrossFitWindow],
                         study_id: str, protected_start_date: date,
                         minimum_fit_sessions: int, score_block_sessions: int,
                         ensemble: EnsembleLayer | None = None,
                         output_dir: str | Path) -> dict:
    """Persist source rows compatible with ``build_oof_store``.

    The caller must bind the result to its pinned canonical feature manifest.
    No proposal search or score-based upstream selection occurs here.
    """
    if not study_id:
        raise ValueError("study_id is required")
    frame = _validate(frame, layers, windows, registry, protected_start_date,
                      minimum_fit_sessions, score_block_sessions, ensemble)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    source_rows: list[dict] = []
    fits: list[dict] = []
    try:
        for window in windows:
            fit = frame.filter(pl.col("session_date").is_in(window.fit_dates)).sort(["session_date", "security_id"])
            stop = frame.filter(pl.col("session_date").is_in(window.stopping_dates)).sort(["session_date", "security_id"])
            score = frame.filter(pl.col("session_date").is_in(window.score_dates)).sort(["session_date", "security_id"])
            fit_x, fit_y = _matrix(fit, label=True)
            stop_x, stop_y = _matrix(stop, label=True)
            score_x, _ = _matrix(score, label=False)
            member_values: dict[str, np.ndarray] = {}
            member_fits: dict[str, dict] = {}
            for layer in layers:
                model = registry.create(layer.component_id, params=dict(layer.parameters))
                context = FitContext(study_id, f"base-{layer.component_id}", window.fold_id,
                                     layer.seed, {})
                stopping = (stop_x, stop_y) if model.spec.capabilities.get("stopping_data") else None
                model.fit(fit_x, fit_y, context=context, stopping_data=stopping)
                values = np.asarray(model.predict(score_x), dtype=np.float64)
                if values.shape != (score.height,) or not np.isfinite(values).all():
                    raise ValueError("Upstream model emitted missing or nonfinite predictions")
                model_path = destination / "fits" / f"{window.fold_id}-{layer.component_id}-{layer.seed}.joblib"
                model.save(model_path)
                fit_hash = _hash(model_path)
                max_label_end = max(stop["label_end_date"]) if stopping is not None else max(fit["label_end_date"])
                cutoff = max(window.stopping_dates[-1] if stopping is not None else window.fit_dates[-1],
                             max_label_end)
                if cutoff >= window.score_dates[0]:
                    raise ValueError("Upstream fit information reaches scoring session")
                fits.append({"fold_id": window.fold_id, "upstream_component_id": layer.component_id,
                             "seed": layer.seed, "parameters": dict(layer.parameters),
                             "fit_sha256": fit_hash, "fit_path": str(model_path),
                             "fit_cutoff_date": cutoff.isoformat(),
                             "fit_max_label_end_date": max_label_end.isoformat(),
                             "fit_rows": fit.height, "stopping_rows": stop.height,
                             "score_rows": score.height})
                member_values[layer.component_id] = values
                member_fits[layer.component_id] = fits[-1]
                for row, value in zip(score.select("security_id", "session_date").to_dicts(), values, strict=True):
                    source_rows.append({**row, "prediction": float(value),
                                        "upstream_component_id": layer.component_id,
                                        "fold_id": window.fold_id, "seed": layer.seed,
                                        "fit_sha256": fit_hash, "fit_cutoff_date": cutoff,
                                        "fit_max_label_end_date": max_label_end,
                                        "prediction_role": "inner_oof"})
            if ensemble is not None:
                members = [member_fits[name] for name in ensemble.members]
                derived_path = destination / "fits" / f"{window.fold_id}-{ensemble.component_id}-derived.json"
                with derived_path.open("x", encoding="utf-8") as stream:
                    json.dump({"formula": "equal_weight_arithmetic_mean_v1",
                               "members": [{"component_id": item["upstream_component_id"],
                                            "fit_sha256": item["fit_sha256"],
                                            "seed": item["seed"]} for item in members]},
                              stream, sort_keys=True)
                derived_hash = _hash(derived_path)
                derived_cutoff = max(date.fromisoformat(item["fit_cutoff_date"]) for item in members)
                derived_label_end = max(date.fromisoformat(item["fit_max_label_end_date"])
                                        for item in members)
                fits.append({"fold_id": window.fold_id, "upstream_component_id": ensemble.component_id,
                             "seed": layers[0].seed, "parameters": {"formula": "equal_weight_arithmetic_mean_v1",
                                                                  "members": list(ensemble.members)},
                             "fit_sha256": derived_hash, "fit_path": str(derived_path),
                             "fit_cutoff_date": derived_cutoff.isoformat(),
                             "fit_max_label_end_date": derived_label_end.isoformat(),
                             "fit_rows": fit.height, "stopping_rows": stop.height,
                             "score_rows": score.height})
                values = np.mean(np.stack([member_values[name] for name in ensemble.members]), axis=0)
                for row, value in zip(score.select("security_id", "session_date").to_dicts(), values, strict=True):
                    source_rows.append({**row, "prediction": float(value),
                                        "upstream_component_id": ensemble.component_id,
                                        "fold_id": window.fold_id, "seed": layers[0].seed,
                                        "fit_sha256": derived_hash, "fit_cutoff_date": derived_cutoff,
                                        "fit_max_label_end_date": derived_label_end,
                                        "prediction_role": "inner_oof"})
        source_path = destination / "source_predictions.parquet"
        pl.DataFrame(source_rows).select(INPUT_COLUMNS).write_parquet(source_path)
        manifest = {"schema_version": 1, "study_id": study_id,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "generator_sha256": _hash(Path(__file__)),
                    "source_prediction_path": str(source_path),
                    "source_prediction_sha256": _hash(source_path),
                    "source_rows": len(source_rows),
                    "components": [layer.component_id for layer in layers]
                                  + ([ensemble.component_id] if ensemble else []),
                    "feature_columns": list(F1), "feature_view": "F1_CAUSAL_STACK_V1",
                    "protected_start_date": protected_start_date.isoformat(),
                    "minimum_fit_sessions": minimum_fit_sessions,
                    "score_block_sessions": score_block_sessions,
                    "prediction_role": "inner_oof", "fits": fits}
        with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
        return manifest
    except Exception:
        # A failed output stays visibly incomplete and cannot be mistaken for a sealed source.
        raise


def build_stack_store(*, producer_manifest: Mapping[str, object], feature_path: str | Path,
                      feature_sha256: str, feature_manifest_path: str | Path,
                      feature_manifest_sha256: str, output_dir: str | Path) -> dict:
    """Bind produced source rows to the canonical full-key OOF store."""
    return build_oof_store(feature_path=feature_path, feature_sha256=feature_sha256,
                           feature_manifest_path=feature_manifest_path,
                           feature_manifest_sha256=feature_manifest_sha256,
                           source_prediction_path=str(producer_manifest["source_prediction_path"]),
                           source_prediction_sha256=str(producer_manifest["source_prediction_sha256"]),
                           components=tuple(producer_manifest["components"]),
                           study_id=str(producer_manifest["study_id"]), output_dir=output_dir)


def augmented_tabular_view(*, canonical: pl.DataFrame, store_manifest_path: str | Path,
                           feature_sha256: str, study_id: str,
                           training_keys: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Return one shared wide view and train-partition-only fill metadata.

    Prediction values stay null where unavailable. Consumers may apply the
    returned medians to their own matrices; the availability masks remain.
    """
    store = load_oof_store(store_manifest_path, expected_feature_sha256=feature_sha256,
                           expected_study_id=study_id)
    keys = ["security_id", "session_date"]
    if training_keys.is_empty() or training_keys.select(keys).is_duplicated().any():
        raise ValueError("Training partition keys must be nonempty and unique")
    if canonical.select(keys).is_duplicated().any():
        raise ValueError("Canonical view keys must be unique")
    view = canonical.drop("forward_return_5d", strict=False)
    metadata = {"study_id": study_id, "feature_sha256": feature_sha256,
                "imputation_scope": "training_partition_only", "columns": {}}
    for index, component in enumerate(sorted(store["upstream_component_id"].unique().to_list())):
        name = f"upstream_prediction_{index}"
        mask = f"{name}_is_available"
        part = store.filter(pl.col("upstream_component_id") == component).select(
            *keys, pl.col("prediction").alias(name), pl.col("is_available").alias(mask))
        view = view.join(part, on=keys, how="left")
        if view[mask].is_null().any():
            raise ValueError("OOF store does not cover all canonical keys")
        train = view.join(training_keys.select(keys), on=keys, how="inner")
        if train.height != training_keys.height:
            raise ValueError("Training partition keys absent from canonical view")
        observed = train.filter(pl.col(mask))[name]
        if observed.is_empty():
            raise ValueError(f"No observed training predictions for {component}")
        metadata["columns"][name] = {"component_id": component, "mask_column": mask,
                                      "train_median": float(observed.median()),
                                      "train_observed_rows": len(observed),
                                      "train_rows": train.height}
    return view, metadata
