"""Causal, central-runner-owned inputs for the frozen RL sleeve selector.

This module accepts verified in-memory study views. It never opens a completed
Slice 1 run or chooses a model from observed evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import re
from typing import Mapping

import numpy as np
import polars as pl

from trading_pipeline.portfolio.signals import select_weights
from trading_pipeline.rl.study_integration import build_study_selector_dataset
from trading_pipeline.rl.dataset import PilotDataset


MODEL_SLEEVES = ("E1", "E2", "E3", "E4")
ALL_SLEEVES = ("E0", *MODEL_SLEEVES, "E5")
REQUIRED_HASHES = frozenset({"canonical_features", "model_outputs", "adjusted_close",
                             "fold_manifest", "protocol"})


@dataclass(frozen=True)
class SelectorEpisode:
    dataset: PilotDataset
    transitions: tuple[dict, ...]
    observations: tuple[dict, ...]
    sleeves: tuple[dict, ...]
    scaling: tuple[dict, ...]
    exclusions: tuple[dict, ...]
    eligibility: tuple[dict, ...]
    source_hashes: dict[str, str]


def _rows(frame: pl.DataFrame, fields: set[str], label: str) -> list[dict]:
    if not isinstance(frame, pl.DataFrame) or not fields <= set(frame.columns):
        raise ValueError(f"{label} lacks required columns: {sorted(fields - set(getattr(frame, 'columns', [])))}")
    return frame.to_dicts()


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be finite{' and positive' if positive else ''}")
    return number


def produce_selector_episode(
    *, split: str, signal_sessions: tuple[date, ...], calendar: tuple[date, ...],
    features: pl.DataFrame, model_outputs: pl.DataFrame, bars: pl.DataFrame,
    source_hashes: Mapping[str, str], e5_source: str, top_k: int,
    gross_exposure_cap: float, max_position: float,
    annualised_volatility_target: float, volatility_lookback_sessions: int,
    allowed_prediction_roles: tuple[str, ...],
    price_eligibility_policy: str = "complete_daily_valuation_v1",
) -> SelectorEpisode:
    """Build one fold episode with T+1 fills and frozen E0–E5 targets.

    Inputs must come from the runner's verified, common feature/model-output view.
    Model output rows require ``model_id, security_id, session_date,
    predicted_return_5d, fit_cutoff_date, fit_max_label_end_date,
    prediction_role``. The declared lookback uses adjusted closes ending at T;
    no price later than T enters target volatility scaling or observations.
    The producer scales weights to the minimum of the volatility target, gross
    cap and position cap; the RL bridge may apply its identical gross/position
    caps again, which is idempotent for these already constrained targets.
    """
    if not split or e5_source not in MODEL_SLEEVES or type(top_k) is not int or top_k <= 0:
        raise ValueError("A split, declared E5 source and positive top-K are required")
    if price_eligibility_policy not in {"complete_daily_valuation_v1",
                                        "observed_history_and_endpoint_valuation_v1"}:
        raise ValueError("RL price eligibility policy is undeclared or unsupported")
    if type(volatility_lookback_sessions) is not int or volatility_lookback_sessions < 2:
        raise ValueError("Volatility lookback must be explicitly declared with at least two returns")
    if not allowed_prediction_roles or any(role == "legacy_observed_final" for role in allowed_prediction_roles):
        raise ValueError("Prediction roles must be declared and exclude legacy final evidence")
    cap = _finite(gross_exposure_cap, "gross cap", positive=True)
    position_cap = _finite(max_position, "position cap", positive=True)
    target_vol = _finite(annualised_volatility_target, "annualised volatility target", positive=True)
    if cap > 1 or position_cap > 1:
        raise ValueError("Leverage and positions above one are forbidden")
    hashes = dict(source_hashes)
    if not REQUIRED_HASHES <= hashes.keys() or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in hashes.values()
    ):
        raise ValueError("All verified canonical source hashes are required")
    if not calendar or calendar != tuple(sorted(set(calendar))):
        raise ValueError("Calendar must be sorted and unique")
    if not signal_sessions or signal_sessions != tuple(sorted(set(signal_sessions))):
        raise ValueError("Fold sessions must be sorted and unique")
    index = {day: i for i, day in enumerate(calendar)}
    if any(day not in index for day in signal_sessions):
        raise ValueError("Fold signal session is absent from the canonical calendar")
    signal_dates, seen = [], set()
    for day in signal_sessions:
        week = day.isocalendar()[:2]
        if week not in seen:
            signal_dates.append(day)
            seen.add(week)
    if len(signal_dates) < 2:
        raise ValueError("At least two weekly signals are required for a complete transition")
    transitions = []
    for signal, end in zip(signal_dates, signal_dates[1:]):
        execution_index = index[signal] + 1
        if execution_index >= len(calendar) or calendar[execution_index] > end:
            raise ValueError("Weekly transition lacks a T+1 execution session")
        transitions.append({"signal_date": signal, "execution_date": calendar[execution_index],
                            "end_date": end})
    active_signals = {row["signal_date"] for row in transitions}

    feature_rows = _rows(features, {"session_date", "security_id", "return_5d",
                                    "return_20d", "vol_20d"}, "Canonical feature view")
    feature_by_day: dict[date, dict[str, dict]] = {}
    for row in feature_rows:
        day, security = row["session_date"], row["security_id"]
        if day not in active_signals:
            continue
        if not security or security in feature_by_day.setdefault(day, {}):
            raise ValueError("Duplicate or null canonical feature key")
        feature_by_day[day][security] = row
    output_rows = _rows(model_outputs, {"session_date", "security_id", "model_id",
                                        "predicted_return_5d", "fit_cutoff_date",
                                        "fit_max_label_end_date", "prediction_role"},
                        "Causal model-output view")
    output_by_key: dict[tuple[date, str], dict[str, float]] = {}
    for row in output_rows:
        day, model, security = row["session_date"], row["model_id"], row["security_id"]
        if day not in active_signals:
            continue
        if model not in MODEL_SLEEVES or row["prediction_role"] not in allowed_prediction_roles:
            raise ValueError("Undeclared model or prediction role in RL output view")
        if not (row["fit_max_label_end_date"] <= row["fit_cutoff_date"] < day):
            raise ValueError("Model output violates the point-in-time fit cutoff")
        if security not in feature_by_day.get(day, {}):
            raise ValueError("Model output has no canonical feature key")
        by_model = output_by_key.setdefault((day, model), {})
        if security in by_model:
            raise ValueError("Duplicate model output key")
        by_model[security] = _finite(row["predicted_return_5d"], "model output")
    price_rows = _rows(bars, {"session_date", "security_id", "adjusted_close"},
                       "Canonical adjusted-close bars")
    prices = {}
    for row in price_rows:
        key = (row["session_date"], row["security_id"])
        if key in prices:
            raise ValueError("Duplicate canonical adjusted-close key")
        prices[key] = _finite(row["adjusted_close"], "adjusted close", positive=True)

    observations, sleeves, scaling, exclusions, eligibility = [], [], [], [], []
    fields = ("market_return_5d_mean", "market_vol_20d_mean", "market_dispersion_5d",
              *(f"prediction_mean_{name}" for name in MODEL_SLEEVES),
              *(f"prediction_dispersion_{name}" for name in MODEL_SLEEVES))
    for transition in transitions:
        signal = transition["signal_date"]
        cross = feature_by_day.get(signal, {})
        if not cross:
            raise ValueError(f"Missing canonical signal cross-section: {signal}")
        left = index[signal] - volatility_lookback_sessions
        if left < 0:
            raise ValueError("Insufficient pre-signal adjusted closes for volatility estimation")
        history = calendar[left:index[signal] + 1]
        ready = {}
        for security, row in cross.items():
            if not all(row.get(name) is not None and math.isfinite(float(row[name]))
                       for name in ("return_5d", "return_20d", "vol_20d")) or float(row["vol_20d"]) <= 0:
                exclusions.append({"signal_date": signal, "security_id": security,
                                   "reason": "invalid_signal_features"})
                continue
            missing = [day for day in history if (day, security) not in prices]
            if missing and price_eligibility_policy == "observed_history_and_endpoint_valuation_v1":
                exclusions.append({"signal_date": signal, "security_id": security,
                                   "reason": "missing_observed_volatility_history",
                                   "missing_sessions": [day.isoformat() for day in missing]})
                continue
            ready[security] = row
        if not ready:
            raise ValueError(f"No eligible signal securities: {signal}")
        reasons = {}
        for excluded in (item for item in exclusions if item["signal_date"] == signal):
            reasons[excluded["reason"]] = reasons.get(excluded["reason"], 0) + 1
        eligibility.append({"signal_date": signal, "candidate_count": len(cross),
                            "eligible_count": len(ready), "excluded_count": len(cross) - len(ready),
                            "reason_counts": reasons})
        returns = np.asarray([float(row["return_5d"]) for row in ready.values()])
        obs = {"signal_date": signal, "market_return_5d_mean": float(returns.mean()),
               "market_vol_20d_mean": float(np.mean([row["vol_20d"] for row in ready.values()])),
               "market_dispersion_5d": float(np.std(returns))}
        scores = {"E0": {security: float(row["return_20d"]) for security, row in ready.items()}}
        for model in MODEL_SLEEVES:
            available = output_by_key.get((signal, model), {})
            if set(ready) - set(available):
                raise ValueError(f"Missing causal model outputs for {model} at {signal}")
            values = {security: available[security] for security in ready}
            array = np.asarray(list(values.values()))
            obs[f"prediction_mean_{model}"] = float(array.mean())
            obs[f"prediction_dispersion_{model}"] = float(np.std(array))
            scores[model] = values
        observations.append(obs)
        scores["E5"] = scores[e5_source]
        for action in ALL_SLEEVES:
            predictions = pl.DataFrame({
                "security_id": list(ready),
                "predicted_return_5d": [scores[action][security] for security in ready],
                "vol_20d": [float(ready[security]["vol_20d"]) for security in ready],
            })
            raw = select_weights(predictions, top_k=top_k, inverse=(action == "E5"))
            if not raw:
                raise ValueError(f"Empty frozen {action} sleeve at {signal}")
            daily = []
            for previous, current in zip(history, history[1:]):
                weighted = 0.0
                for security, weight in raw.items():
                    first, second = prices.get((previous, security)), prices.get((current, security))
                    if first is None or second is None:
                        raise ValueError(f"Missing pre-signal adjusted close: {security} {current}")
                    weighted += weight * (second / first - 1)
                daily.append(weighted)
            estimated_vol = float(np.std(daily, ddof=1) * np.sqrt(252))
            if not math.isfinite(estimated_vol) or estimated_vol <= 0:
                raise ValueError("Annualised volatility estimate is undefined")
            raw_gross = sum(raw.values())
            factor = min(1.0, target_vol / estimated_vol, cap / raw_gross,
                         min(position_cap / weight for weight in raw.values() if weight > 0))
            constrained = {security: weight * factor for security, weight in raw.items()}
            scaling.append({"signal_date": signal, "action_id": action,
                            "volatility_lookback_sessions": volatility_lookback_sessions,
                            "estimated_annualised_volatility": estimated_vol,
                            "annualised_volatility_target": target_vol,
                            "gross_exposure_cap": cap, "max_position": position_cap,
                            "applied_scale": factor, "resulting_gross_exposure": sum(constrained.values())})
            sleeves.extend({"signal_date": signal, "action_id": action,
                            "security_id": security, "target_weight": weight}
                           for security, weight in constrained.items())
            valuation_dates = ((transition["execution_date"], transition["end_date"])
                               if price_eligibility_policy == "observed_history_and_endpoint_valuation_v1"
                               else calendar[index[transition["execution_date"]]:index[transition["end_date"]] + 1])
            for day in valuation_dates:
                for security in constrained:
                    if (day, security) not in prices:
                        if price_eligibility_policy == "complete_daily_valuation_v1":
                            raise ValueError(f"Missing held-security adjusted close: {security} {day}")
                        raise ValueError(f"Missing required endpoint adjusted close: {security} {day}")
    first, last = transitions[0]["signal_date"], transitions[-1]["end_date"]
    episode_calendar = [day for day in calendar if first <= day <= last]
    episode_prices = [{"session_date": day, "security_id": security, "adjusted_close": value}
                      for (day, security), value in prices.items() if first <= day <= last]
    dataset = build_study_selector_dataset(
        split=split, transitions=transitions, observations=observations,
        sleeves=sleeves, prices=episode_prices, calendar=episode_calendar,
        observation_fields=fields, source_hashes=hashes,
        valuation_policy=price_eligibility_policy)
    return SelectorEpisode(dataset, tuple(transitions), tuple(observations),
                           tuple(sleeves), tuple(scaling), tuple(exclusions),
                           tuple(eligibility), hashes)
SELECTOR_FEATURE_COLUMNS = ("security_id", "session_date", "return_5d", "return_20d", "vol_20d")


def selector_feature_frame(rows) -> pl.DataFrame:
    """Materialise sparse expanded rows without sampling-based Null inference."""
    return pl.DataFrame(rows, infer_schema_length=None).select(*SELECTOR_FEATURE_COLUMNS)
