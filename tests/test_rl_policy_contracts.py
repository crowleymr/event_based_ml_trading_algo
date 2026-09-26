from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trading_pipeline.experiments import FitContext, RLPolicy
from trading_pipeline.rl.policy_models import CategoricalPPOPolicy, DQNPolicy


class _Discrete:
    n = 7


class _Environment:
    action_space = _Discrete()


class _Action:
    def __init__(self, value: int):
        self.value = value

    def item(self) -> int:
        return self.value


class _FakeBackend:
    instances = []

    def __init__(self, policy, environment, **kwargs):
        self.policy = policy
        self.environment = environment
        self.kwargs = kwargs
        self.device = kwargs["device"]
        self.num_timesteps = 0
        self._n_updates = 0
        self.logger = SimpleNamespace(name_to_value={"train/loss": 0.25, "bad": object()})
        self.saved = None
        self.__class__.instances.append(self)

    def learn(self, *, total_timesteps, progress_bar):
        assert progress_bar is False
        self.num_timesteps = total_timesteps
        self._n_updates = 9
        return self

    def predict(self, observation, *, deterministic):
        return _Action(3 if deterministic else 4), None

    def save(self, path):
        self.saved = path


@pytest.fixture(autouse=True)
def _clear_fake_instances():
    _FakeBackend.instances.clear()


def _context(**fidelity):
    return FitContext(
        study_id="engineering-study",
        trial_id="trial-001",
        fold_id="fold-01",
        seed=41,
        fidelity=fidelity,
    )


@pytest.mark.parametrize("policy_type", [DQNPolicy, CategoricalPPOPolicy])
def test_policy_contract_search_space_and_research_gate(policy_type):
    policy = policy_type(total_timesteps=128, backend_factory=_FakeBackend)
    assert isinstance(policy, RLPolicy)
    assert policy.spec.interface == "RLPolicy"
    assert policy.spec.research_enabled is False
    assert policy.spec.capabilities["discrete_actions"] is True
    assert policy.search_space()["resource_axis"] == "environment_steps"
    json.dumps(policy.search_space())
    json.dumps(policy.telemetry())


@pytest.mark.parametrize("policy_type", [DQNPolicy, CategoricalPPOPolicy])
def test_policy_learn_act_save_and_telemetry(policy_type, tmp_path: Path):
    policy = policy_type(
        total_timesteps=128,
        device="cpu",
        backend_factory=_FakeBackend,
    )
    assert policy.learn(_Environment(), context=_context(environment_steps=64)) is policy
    backend = _FakeBackend.instances[-1]
    assert backend.kwargs["seed"] == 41
    assert policy.act([0.0], deterministic=True) == 3
    assert policy.act([0.0], deterministic=False) == 4
    policy.save(tmp_path / "models" / "policy")
    assert backend.saved == str(tmp_path / "models" / "policy")
    telemetry = policy.telemetry()
    assert telemetry["status"] == "fitted"
    assert telemetry["environment_steps_requested"] == 64
    assert telemetry["environment_steps_completed"] == 64
    assert telemetry["gradient_updates"] == 9
    assert telemetry["seed"] == 41
    assert telemetry["backend_metrics"] == {"train/loss": 0.25}
    json.dumps(telemetry)


def test_dqn_validates_conditional_parameters_before_training():
    with pytest.raises(ValueError, match="batch_size"):
        DQNPolicy(
            total_timesteps=10,
            parameters={"buffer_size": 16, "batch_size": 32},
        )
    with pytest.raises(ValueError, match="learning_starts"):
        DQNPolicy(
            total_timesteps=10,
            parameters={"buffer_size": 100, "learning_starts": 100},
        )
    with pytest.raises(ValueError, match="exploration_final_eps"):
        DQNPolicy(
            total_timesteps=10,
            parameters={"exploration_initial_eps": 0.1, "exploration_final_eps": 0.2},
        )


def test_ppo_validates_architecture_and_rollout_divisibility():
    with pytest.raises(ValueError, match="matched architecture"):
        CategoricalPPOPolicy(
            total_timesteps=10,
            parameters={
                "architecture_mode": "matched",
                "policy_widths": [64],
                "value_widths": [32],
            },
        )
    with pytest.raises(ValueError, match="divisible"):
        CategoricalPPOPolicy(
            total_timesteps=10,
            parameters={"n_steps": 100, "batch_size": 64},
        )


@pytest.mark.parametrize("policy_type", [DQNPolicy, CategoricalPPOPolicy])
def test_policy_rejects_non_discrete_environment_and_unknown_fidelity(policy_type):
    policy = policy_type(total_timesteps=10, backend_factory=_FakeBackend)
    with pytest.raises(ValueError, match="discrete action space"):
        policy.learn(SimpleNamespace(action_space=SimpleNamespace(shape=(2,))), context=_context())
    with pytest.raises(ValueError, match="Unsupported RL fidelity"):
        policy.learn(_Environment(), context=_context(epochs=2))


@pytest.mark.parametrize("policy_type", [DQNPolicy, CategoricalPPOPolicy])
def test_policy_requires_learning_before_action_or_save(policy_type, tmp_path: Path):
    policy = policy_type(total_timesteps=10, backend_factory=_FakeBackend)
    with pytest.raises(RuntimeError, match="learned before act"):
        policy.act([0.0])
    with pytest.raises(RuntimeError, match="learned before save"):
        policy.save(tmp_path / "unused")
