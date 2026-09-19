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

Use `--skip-dqn` only when the optional ML stack cannot be installed or the fixed
training budget exceeds the stop condition. Outputs are new immutable directories
under `runs/rl/`; the reference run and source feature file remain untouched.
