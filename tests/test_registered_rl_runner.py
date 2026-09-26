from __future__ import annotations

from types import SimpleNamespace

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
    with pytest.raises(PermissionError, match="synthetic"):
        run_registered_policy_trial(**kwargs, data_role="real")
    class ChangedEnvironment(_Environment):
        cost_bps = 20.0
    with pytest.raises(ValueError, match="cost_bps"):
        run_registered_policy_trial(**{**kwargs, "evaluation_environment": ChangedEnvironment},
                                    data_role="synthetic")
