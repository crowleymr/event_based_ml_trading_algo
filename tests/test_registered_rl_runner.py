from __future__ import annotations

from types import SimpleNamespace
import importlib.util

import pytest

from trading_pipeline.experiments import FitContext
from trading_pipeline.rl.runner import run_registered_policy_trial


class _Environment:
    action_space = SimpleNamespace(n=7)
    observation_fields = ("market_return", "cash")
    cost_bps = 10.0


class _Backend:
    def __init__(self, policy, environment, **kwargs):
        self.device = kwargs["device"]
        self.num_timesteps = 0
        self._n_updates = 0
        self.logger = SimpleNamespace(name_to_value={"train/loss": 0.5})

    def learn(self, *, total_timesteps, progress_bar):
        self.num_timesteps = total_timesteps
        self._n_updates = 4

    def predict(self, observation, *, deterministic):
        assert deterministic
        return 2, None


@pytest.mark.parametrize("component_id", ["rl_dqn_sb3_v1", "rl_ppo_categorical_sb3_v1"])
def test_registered_policies_share_no_learning_evaluation(component_id):
    result = run_registered_policy_trial(
        component_id=component_id,
        context=FitContext("engineering", "trial", "inner_1", 7, {"environment_steps": 16}),
        train_environment=_Environment, evaluation_environment=_Environment,
        parameters={}, total_timesteps=16, data_role="synthetic",
        backend_factory=_Backend,
        evaluator=lambda env, act, policy_id, seed: ([act([0.0])], [1.0], {"reward_sum": 0.1}),
    )
    assert result.actions == [2]
    assert result.telemetry["environment_steps_completed"] == 16
    assert result.telemetry["gradient_updates"] == 4
    assert result.telemetry["evaluation_state_unchanged"] is True


def test_registered_runner_rejects_real_data_and_mismatched_environment():
    kwargs = dict(
        component_id="rl_dqn_sb3_v1",
        context=FitContext("engineering", "trial", "inner_1", 7),
        train_environment=_Environment, evaluation_environment=_Environment,
        parameters={}, total_timesteps=16, backend_factory=_Backend,
        evaluator=lambda *args: ([], [], {}),
    )
    with pytest.raises(PermissionError, match="central study orchestration"):
        run_registered_policy_trial(**kwargs, data_role="real")
    class ChangedEnvironment(_Environment):
        cost_bps = 20.0
    with pytest.raises(ValueError, match="cost_bps"):
        run_registered_policy_trial(**{**kwargs, "evaluation_environment": ChangedEnvironment},
                                    data_role="synthetic")


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None,
                    reason="optional stable-baselines3")
def test_categorical_ppo_trains_with_real_backend_and_freezes_evaluation():
    import gymnasium as gym
    import numpy as np

    class TinySelector(gym.Env):
        observation_fields = ("known_signal",)
        cost_bps = 10.0

        def __init__(self):
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
            self.index = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.index = 0
            return np.array([0.25], dtype=np.float32), {}

        def step(self, action):
            self.index += 1
            return np.array([0.25], dtype=np.float32), float(action == 1), self.index == 4, False, {}

    def evaluate_tiny(env, choose, policy_id, seed):
        observation, _ = env.reset(seed=seed)
        actions = []
        done = False
        while not done:
            action = choose(observation)
            actions.append(action)
            observation, _, done, _, _ = env.step(action)
        return actions, [], {"reward_sum": float(sum(actions))}

    result = run_registered_policy_trial(
        component_id="rl_ppo_categorical_sb3_v1",
        context=FitContext("engineering", "ppo_real_backend", "inner_1", 7,
                           {"environment_steps": 16}),
        train_environment=TinySelector, evaluation_environment=TinySelector,
        parameters={"n_steps": 8, "batch_size": 4, "n_epochs": 1,
                    "policy_widths": [8], "value_widths": [8]},
        total_timesteps=16, data_role="synthetic", device="cpu",
        evaluator=evaluate_tiny,
    )
    assert len(result.actions) == 4
    assert result.telemetry["algorithm"] == "PPO"
    assert result.telemetry["environment_steps_completed"] == 16
    assert result.telemetry["gradient_updates"] is not None
    assert result.telemetry["parameter_count"] > 0
    assert result.telemetry["evaluation_state_unchanged"] is True
