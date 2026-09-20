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
