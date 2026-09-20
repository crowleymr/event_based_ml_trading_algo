# Read-only Streamlit dashboard

The dashboard reads an already generated Slice 1 report and, optionally, one
completed audited RL run. It cannot train, tune, regenerate reports or modify runs.

```powershell
.venv/Scripts/python -m trading_pipeline.dashboard `
  --report reports/20260912T071137Z-9899fd9a/v2 `
  --rl-run runs/rl/<run_id>
```

The fixed/random RL harness can be omitted by leaving out `--rl-run`. Report schema v2
adds Overview/Evidence, Data & Splits, Model Diagnostics, Training Diagnostics, RL Gym,
Backtest & Benchmark, and Securities views. Older runs show explicit not-recorded states;
the UI never reconstructs unavailable training histories.
