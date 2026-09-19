from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from trading_pipeline.rl.dataset import PilotDataset, prepare_dataset
from trading_pipeline.rl.environment import StrategySelectorEnv


def tiny_dataset(missing=False):
    days = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(15))
    signals = (days[0], days[5], days[10])
    executions = (days[1], days[6], days[11])
    ends = (days[5], days[10], days[14])
    targets = {}
    for step in range(3):
        targets[(step, 0)] = {}
        for action in range(1, 7):
            targets[(step, action)] = {"A": 1.0}
    prices = {(day, "A"): float(index + 1) for index, day in enumerate(days)}
    if missing:
        del prices[(days[3], "A")]
    return PilotDataset(
        split="validation", signal_dates=signals, execution_dates=executions,
        end_dates=ends, base_observations=np.zeros((3, 23), dtype=np.float32),
        observation_fields=tuple(f"field_{i}" for i in range(23)), targets=targets,
        prices=prices, calendar=days, reference_equity={}, source_hashes={},
    )


def test_environment_checks_and_seeded_transitions_are_deterministic():
    check_env(StrategySelectorEnv(tiny_dataset()), skip_render_check=True)
    sb3_check_env(StrategySelectorEnv(tiny_dataset()), warn=False)
    first, second = StrategySelectorEnv(tiny_dataset()), StrategySelectorEnv(tiny_dataset())
    assert np.array_equal(first.reset(seed=42)[0], second.reset(seed=42)[0])
    for action in (1, 0, 2):
        left, right = first.step(action), second.step(action)
        assert np.array_equal(left[0], right[0])
        assert left[1:] == right[1:]


def test_t_plus_one_cost_reward_and_constraints():
    free, paid = StrategySelectorEnv(tiny_dataset(), 0), StrategySelectorEnv(tiny_dataset(), 10)
    free.reset(); paid.reset()
    _, free_reward, _, _, free_info = free.step(1)
    _, paid_reward, _, _, paid_info = paid.step(1)
    assert paid_info["execution_date"] == date(2024, 1, 2)
    assert paid_info["execution_date"] > paid_info["signal_date"]
    assert paid_info["cost"] > 0
    assert paid_reward < free_reward
    assert paid_info["gross_exposure"] <= 1 + 1e-12
    assert paid_info["cash_weight"] >= -1e-12
    with pytest.raises(ValueError, match="Invalid action"):
        paid.step(99)


def test_missing_held_bar_and_nonfinite_observation_fail():
    env = StrategySelectorEnv(tiny_dataset(missing=True))
    env.reset()
    with pytest.raises(ValueError, match="Missing held"):
        env.step(1)
    data = tiny_dataset()
    data.base_observations[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite observation"):
        StrategySelectorEnv(data).reset()


REFERENCE = Path("D:/repos/event_based_ml_trading_algo/runs/20260912T071137Z-9899fd9a")
FEATURES = Path("D:/repos/event_based_ml_trading_algo/data/features/20260912T071137Z-9899fd9a/features.parquet")
FEATURE_HASH = "83b4d0e3ff45f27d54ecc858a0a7ce695b18cfed218ab6c33af4cef2ce9cd8a3"


@pytest.mark.skipif(not REFERENCE.is_dir() or not FEATURES.is_file(), reason="preserved reference data unavailable")
def test_real_dataset_hash_boundaries_and_fixed_sleeve_parity():
    data = prepare_dataset(REFERENCE, FEATURES, FEATURE_HASH, "validation")
    assert data.signal_dates[0] >= date(2022, 1, 24)
    assert data.end_dates[-1] <= date(2024, 5, 6)
    for action, experiment in enumerate(("E0", "E1", "E2", "E3", "E4", "E5"), start=1):
        env = StrategySelectorEnv(data)
        env.reset(seed=42)
        terminated = False
        info = None
        while not terminated:
            _, _, terminated, truncated, info = env.step(action)
            assert not truncated
        expected = data.reference_equity[experiment][info["end_date"]] / data.reference_equity[experiment][data.signal_dates[0]]
        assert info["equity"] == pytest.approx(expected, rel=1e-9, abs=1e-11)


@pytest.mark.skipif(not REFERENCE.is_dir() or not FEATURES.is_file(), reason="preserved reference data unavailable")
def test_feature_hash_mismatch_fails():
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_dataset(REFERENCE, FEATURES, "0" * 64, "validation")
