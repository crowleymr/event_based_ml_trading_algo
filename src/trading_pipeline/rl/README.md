# Phase 2 exploratory RL pilot

This package consumes a completed Slice 1 run without changing it. The frozen
weekly Gymnasium environment selects cash or one of E0-E5 at T and executes the
underlying frozen sleeve weights at the next close. It reuses the Slice 1
self-financing cost solver.

Install and run from the repository root:

```powershell
.venv/Scripts/python -m pip install -e ".[test,phase2]"
.venv/Scripts/python -m trading_pipeline.rl.runner --config configs/rl_pilot.yaml
```

Protocol v2 accepts `cpu`, `cuda`, or `auto`, records requested/actual device and any
fallback reason, and writes common `training_trace.parquet`, `training_summary.parquet`
and a short `device_benchmark.parquet`. Check the installed build and actual tensor
placement with `python -m trading_pipeline.rl.device --device auto`.

Use `--skip-dqn` only when the optional ML stack cannot be installed or the fixed
training budget exceeds the stop condition. Outputs are new immutable directories
under `runs/rl/`; the reference run and source feature file remain untouched. Reruns on
the observed test must be labelled `diagnostic_reproduction_not_model_selection`.

## Phase 3 policy adapters

`policy_models/` contains research-disabled DQN and categorical PPO implementations of
the shared `RLPolicy` contract. Optional Stable-Baselines3 and PyTorch imports are lazy.
The adapters declare architecture/search spaces, validate discrete-action and resource
constraints, and emit seed/device/resource telemetry. They do not replace or modify the
frozen Phase 2 runner. Real-data DQN/PPO optimisation remains blocked until the
walk-forward protocol, causal upstream lineage, economic objective, risk constraints,
seeds and compute budget receive explicit approval.

## Expanded study bridge

`study_integration.py` is the central runner's fold-level bridge. The runner
must supply explicit transition, observation, frozen sleeve, price and source-hash
rows to `build_study_selector_dataset`; this builder never reads the legacy run.
It requires all six noncash sleeves on every signal date, a sorted market calendar,
next-session close execution, complete held-price valuation and equal source
lineage across fit and scoring episodes. Causal observation construction and
upstream model-output hashes are the caller's responsibility.

`execute_study_rl_trial` requires approved verified study authority and a declared
fold, seed, risk scenario and search proposal. DQN and categorical PPO use the
same seven-action environment and scenario-scaled frozen sleeves. The reward is
the after-cost weekly log return minus `0.5 * risk_aversion * log_return**2`;
the unpenalised net return remains in the equity curve. Each fit appends started,
complete or failed records, resource telemetry and hashed cell artefacts. The
deterministic evaluation checks that policy parameters, update counters, replay
position and observation normaliser state have not changed. A direct invocation
of the pilot registered runner on real data remains forbidden.

The bridge is not itself an expanded research result. The authoritative runner
must still create causal upstream observations and fold-specific sleeves, invoke
this bridge for every proposal/fold/seed/scenario, seal selection before holdout,
and audit the complete matrix.
