"""Point-in-time weekly dataset derived from immutable Slice 1 artefacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import hashlib
import json

import numpy as np
import polars as pl

from .protocol import ACTION_IDS


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


@dataclass(frozen=True)
class PilotDataset:
    split: str
    signal_dates: tuple[date, ...]
    execution_dates: tuple[date, ...]
    end_dates: tuple[date, ...]
    base_observations: np.ndarray
    observation_fields: tuple[str, ...]
    targets: dict[tuple[int, int], dict[str, float]]
    prices: dict[tuple[date, str], float]
    calendar: tuple[date, ...]
    reference_equity: dict[str, dict[date, float]]
    source_hashes: dict[str, str]
    valuation_policy: str = "complete_daily_valuation_v1"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _weekly_dates(calendar: list[date]) -> list[date]:
    result, seen = [], set()
    for day in calendar:
        week = day.isocalendar()[:2]
        if week not in seen:
            result.append(day)
            seen.add(week)
    return result


def _finite_mean(values: np.ndarray, label: str) -> float:
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Non-finite or missing observation input: {label}")
    return float(values.mean())


def prepare_dataset(reference_run: str | Path, feature_path: str | Path,
                    expected_feature_sha256: str, split: str) -> PilotDataset:
    """Build one chronological episode without modifying source artefacts."""
    root, feature_path = Path(reference_run).resolve(), Path(feature_path).resolve()
    metadata = _load_json(root / "metadata.json")
    manifest = _load_json(root / "dataset_manifest.json")
    if metadata.get("status") != "complete":
        raise ValueError("RL preparation requires a completed reference run")
    actual_hash = sha256(feature_path)
    if actual_hash != expected_feature_sha256 or actual_hash != manifest["feature_sha256"]:
        raise ValueError("Reference feature SHA-256 does not match dataset_manifest.json")
    if split not in {"validation", "test"}:
        raise ValueError("Pilot episodes are restricted to validation or descriptive test")

    features = pl.read_parquet(feature_path).filter(pl.col("split") == split)
    bars_path = root / "datasets/market_bars.parquet"
    positions_path = root / "positions.parquet"
    curves_path = root / "equity_curve.parquet"
    predictions_path = root / "predictions.parquet"
    bars = pl.read_parquet(bars_path)
    positions = pl.read_parquet(positions_path).filter(pl.col("split") == split)
    curves = pl.read_parquet(curves_path).filter(pl.col("split") == split)
    predictions = pl.read_parquet(predictions_path).filter(pl.col("split") == split)

    split_calendar = features["session_date"].unique().sort().to_list()
    signals = _weekly_dates(split_calendar)
    market_calendar = bars["session_date"].unique().sort().to_list()
    market_index = {day: index for index, day in enumerate(market_calendar)}
    valid = []
    for index, signal in enumerate(signals[:-1]):
        market_position = market_index.get(signal)
        if market_position is None or market_position + 1 >= len(market_calendar):
            raise ValueError(f"Signal date lacks a next-session execution date: {signal}")
        execution = market_calendar[market_position + 1]
        end = signals[index + 1]
        if execution > end:
            raise ValueError("Execution date cannot follow transition end")
        valid.append((signal, execution, end))
    if not valid:
        raise ValueError(f"No complete weekly transitions in split {split}")

    by_feature = {day: frame for day, frame in (
        (group["session_date"][0], group) for group in features.partition_by("session_date")
    )}
    curve_rows = curves.sort(["experiment", "session_date"])
    curve_history = {
        experiment: group for experiment, group in (
            (group["experiment"][0], group) for group in curve_rows.partition_by("experiment")
        )
    }
    prediction_groups = {
        (group["session_date"][0], group["model_id"][0]): group
        for group in predictions.partition_by(["session_date", "model_id"])
    }
    base_rows = []
    fields = ["market_return_5d_mean", "market_vol_20d_mean", "market_dispersion_5d"]
    fields += [f"sleeve_return_5d_{name}" for name in ACTION_IDS.values() if name != "CASH"]
    fields += [f"sleeve_vol_20d_{name}" for name in ACTION_IDS.values() if name != "CASH"]
    fields += [f"prediction_mean_{name}" for name in ("E1", "E2", "E3", "E4")]
    fields += [f"prediction_dispersion_{name}" for name in ("E1", "E2", "E3", "E4")]
    for signal, _, _ in valid:
        cross = by_feature.get(signal)
        if cross is None:
            raise ValueError(f"Missing feature cross-section at {signal}")
        returns = cross["return_5d"].to_numpy()
        row = [
            _finite_mean(returns, "market_return_5d_mean"),
            _finite_mean(cross["vol_20d"].to_numpy(), "market_vol_20d_mean"),
            float(np.std(returns)),
        ]
        for experiment in ("E0", "E1", "E2", "E3", "E4", "E5"):
            history = curve_history[experiment].filter(pl.col("session_date") <= signal).tail(5)
            row.append(float(np.prod(1 + history["daily_return"].to_numpy()) - 1))
        for experiment in ("E0", "E1", "E2", "E3", "E4", "E5"):
            history = curve_history[experiment].filter(pl.col("session_date") <= signal).tail(20)
            row.append(float(np.std(history["daily_return"].to_numpy()) * np.sqrt(252)))
        for experiment in ("E1", "E2", "E3", "E4"):
            values = prediction_groups[(signal, experiment)]["predicted_return_5d"].to_numpy()
            row.append(_finite_mean(values, f"prediction_mean_{experiment}"))
        for experiment in ("E1", "E2", "E3", "E4"):
            values = prediction_groups[(signal, experiment)]["predicted_return_5d"].to_numpy()
            row.append(float(np.std(values)))
        base_rows.append(row)
    observations = np.asarray(base_rows, dtype=np.float32)
    if observations.shape != (len(valid), len(fields)) or not np.isfinite(observations).all():
        raise ValueError("Prepared observations are missing or non-finite")

    target_rows = positions.filter(pl.col("target_weight").is_not_null())
    targets: dict[tuple[int, int], dict[str, float]] = {}
    for step, (_, execution, _) in enumerate(valid):
        targets[(step, 0)] = {}
        for action, experiment in ACTION_IDS.items():
            if action == 0:
                continue
            rows = target_rows.filter(
                (pl.col("experiment") == experiment)
                & (pl.col("session_date") == execution)
            )
            if rows.is_empty():
                raise ValueError(f"Missing frozen {experiment} sleeve at {execution}")
            weights = dict(zip(rows["security_id"].to_list(), rows["target_weight"].to_list()))
            if not weights or any(value is None or not np.isfinite(value) for value in weights.values()):
                raise ValueError(f"Invalid frozen {experiment} target at {execution}")
            if sum(weights.values()) > 1 + 1e-12 or min(weights.values()) < 0:
                raise ValueError("Frozen sleeve violates long-only/no-leverage constraints")
            targets[(step, action)] = weights

    first, last = valid[0][0], valid[-1][2]
    episode_calendar = tuple(day for day in market_calendar if first <= day <= last)
    prices = {
        (row["session_date"], row["security_id"]): row["adjusted_close"]
        for row in bars.filter(pl.col("session_date").is_between(first, last)).select(
            "session_date", "security_id", "adjusted_close"
        ).to_dicts()
    }
    if any(not np.isfinite(value) or value <= 0 for value in prices.values()):
        raise ValueError("Market bars contain invalid prices")
    reference_equity = {
        experiment: dict(zip(group["session_date"].to_list(), group["equity"].to_list()))
        for experiment, group in curve_history.items()
    }
    sources = [feature_path, bars_path, positions_path, curves_path, predictions_path,
               root / "dataset_manifest.json", root / "split_manifest.json"]
    return PilotDataset(
        split=split,
        signal_dates=tuple(item[0] for item in valid),
        execution_dates=tuple(item[1] for item in valid),
        end_dates=tuple(item[2] for item in valid),
        base_observations=observations,
        observation_fields=tuple(fields),
        targets=targets,
        prices=prices,
        calendar=episode_calendar,
        reference_equity=reference_equity,
        source_hashes={str(path): sha256(path) for path in sources},
    )
