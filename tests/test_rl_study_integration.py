from __future__ import annotations

from datetime import date, timedelta
import json
from types import SimpleNamespace

import pytest

from trading_pipeline.experiments import FitContext
from trading_pipeline.experiments.authority import VerifiedStudyAuthority
from trading_pipeline.experiments.schema import ResolvedStudy, protocol_content_sha256
from trading_pipeline.rl.study_integration import (
    build_study_selector_dataset, execute_study_rl_trial,
)


class _Backend:
    mutate_at_evaluation = False

    def __init__(self, policy, environment, **kwargs):
        self.environment = environment
        self.device = kwargs["device"]
        self.num_timesteps = 0
        self._n_updates = 0
        self.logger = SimpleNamespace(name_to_value={"train/loss": 0.5})

    def learn(self, *, total_timesteps, progress_bar):
        self.num_timesteps = total_timesteps
        self._n_updates = 2

    def predict(self, observation, *, deterministic):
        assert deterministic
        if self.mutate_at_evaluation:
            self._n_updates += 1
        return 1, None


def _episode(start: date, split: str):
    calendar = [start + timedelta(days=day) for day in range(3)]
    return build_study_selector_dataset(
        split=split,
        transitions=[{"signal_date": calendar[0], "execution_date": calendar[1],
                      "end_date": calendar[2]}],
        observations=[{"signal_date": calendar[0], "known_signal": 0.25}],
        sleeves=[{"signal_date": calendar[0], "action_id": f"E{action}",
                  "security_id": "A", "target_weight": 0.1}
                 for action in range(6)],
        prices=[{"session_date": day, "security_id": "A",
                 "adjusted_close": 100.0 + index}
                for index, day in enumerate(calendar)],
        calendar=calendar, observation_fields=("known_signal",),
        source_hashes={"canonical_feature": "a" * 64, "sleeves": "b" * 64},
    )


def _study():
    config = {
        "study_id": "expanded-test", "status": "approved",
        "authority": {"protocol_sha256": None},
        "search": {"real_data_execution": "enabled"},
        "experiment_arms": [
            {"interface": "RLPolicy", "component_id": component}
            for component in ("rl_dqn_sb3_v1", "rl_ppo_categorical_sb3_v1")],
        "reproducibility": {"seeds": [41, 42, 43], "requested_device": "cpu"},
        "objectives": {"rl_certainty_equivalent_v1": {
            "primary": "weekly_net_certainty_equivalent_return",
            "risk_scenarios": {"conservative": {"risk_aversion": 8.0},
                               "balanced": {"risk_aversion": 4.0},
                               "aggressive": {"risk_aversion": 2.0}}}},
        "portfolio": {"execution": "action_at_T_filled_at_T_plus_1_close",
                      "cost_bps_one_way": 10,
                      "risk_scenarios": {
                          name: {"gross_exposure_cap": cap, "max_position": cap}
                          for name, cap in (("conservative", 0.6),
                                            ("balanced", 0.8),
                                            ("aggressive", 1.0))}},
    }
    config["authority"]["protocol_sha256"] = protocol_content_sha256(config)
    study = ResolvedStudy(config, "test", False)
    authority = VerifiedStudyAuthority("expanded-test", config["authority"]["protocol_sha256"],
                                       {}, "fresh", "exploratory_closeout_not_confirmatory")
    return study, authority


@pytest.mark.parametrize("component", ["rl_dqn_sb3_v1", "rl_ppo_categorical_sb3_v1"])
@pytest.mark.parametrize("scenario", ["conservative", "balanced", "aggressive"])
def test_authorised_study_trial_writes_complete_evidence(tmp_path, component, scenario):
    study, authority = _study()
    result = execute_study_rl_trial(
        study=study, authority=authority, component_id=component,
        context=FitContext("expanded-test", "trial-1", "inner-1", 41,
                           {"environment_steps": 16}),
        train_dataset=_episode(date(2026, 1, 5), "fit"),
        score_dataset=_episode(date(2026, 1, 12), "score"),
        risk_scenario=scenario,
        parameters=({"n_steps": 8, "batch_size": 4} if "ppo" in component else
                    {"buffer_size": 32, "batch_size": 8, "learning_starts": 0}),
        total_timesteps=16, output_dir=tmp_path, backend_factory=_Backend,
    )
    assert result.telemetry["evaluation_state_unchanged"] is True
    assert result.metrics["reward_convention"] == "net_log_return_minus_half_lambda_squared_log_return"
    assert len(result.actions) == 1
    ledger = [json.loads(line) for line in (tmp_path / "rl_trial_ledger.jsonl").read_text().splitlines()]
    assert [row["status"] for row in ledger] == ["started", "complete"]
    assert len(list((tmp_path / "rl_cells").rglob("actions.parquet"))) == 1
    assert len((tmp_path / "rl_resource_ledger.jsonl").read_text().splitlines()) == 1


def test_study_trial_rejects_learning_during_evaluation(tmp_path):
    study, authority = _study()
    class MutatingBackend(_Backend):
        mutate_at_evaluation = True
    with pytest.raises(RuntimeError, match="mutated policy"):
        execute_study_rl_trial(
            study=study, authority=authority, component_id="rl_dqn_sb3_v1",
            context=FitContext("expanded-test", "trial-mutate", "inner-1", 41),
            train_dataset=_episode(date(2026, 1, 5), "fit"),
            score_dataset=_episode(date(2026, 1, 12), "score"),
            risk_scenario="balanced",
            parameters={"buffer_size": 32, "batch_size": 8, "learning_starts": 0},
            total_timesteps=16, output_dir=tmp_path, backend_factory=MutatingBackend,
        )
    ledger = [json.loads(line) for line in (tmp_path / "rl_trial_ledger.jsonl").read_text().splitlines()]
    assert [row["status"] for row in ledger] == ["started", "failed"]


def test_study_trial_rejects_invalid_params_before_writing(tmp_path):
    study, authority = _study()
    with pytest.raises(ValueError, match="divisible"):
        execute_study_rl_trial(
            study=study, authority=authority, component_id="rl_ppo_categorical_sb3_v1",
            context=FitContext("expanded-test", "trial-bad", "inner-1", 41),
            train_dataset=_episode(date(2026, 1, 5), "fit"),
            score_dataset=_episode(date(2026, 1, 12), "score"),
            risk_scenario="balanced", parameters={"n_steps": 9, "batch_size": 4},
            total_timesteps=16, output_dir=tmp_path, backend_factory=_Backend,
        )
    assert not (tmp_path / "rl_trial_ledger.jsonl").exists()


def test_builder_rejects_missing_sleeve_and_future_overlap(tmp_path):
    base = _episode(date(2026, 1, 5), "fit")
    with pytest.raises(ValueError, match="missing or invalid frozen sleeve"):
        build_study_selector_dataset(
            split="fit", transitions=[{"signal_date": base.signal_dates[0],
                                        "execution_date": base.execution_dates[0],
                                        "end_date": base.end_dates[0]}],
            observations=[{"signal_date": base.signal_dates[0], "known_signal": 0.25}],
            sleeves=[], prices=[{"session_date": day, "security_id": "A",
                                 "adjusted_close": 100.0} for day in base.calendar],
            calendar=list(base.calendar), observation_fields=("known_signal",),
            source_hashes={"features": "a" * 64},
        )
    study, authority = _study()
    with pytest.raises(ValueError, match="overlaps score"):
        execute_study_rl_trial(
            study=study, authority=authority, component_id="rl_dqn_sb3_v1",
            context=FitContext("expanded-test", "trial-overlap", "inner-1", 41),
            train_dataset=base, score_dataset=base,
            risk_scenario="balanced", parameters={}, total_timesteps=16,
            output_dir=tmp_path, backend_factory=_Backend,
        )
