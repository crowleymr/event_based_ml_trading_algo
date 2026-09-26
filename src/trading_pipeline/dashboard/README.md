# Read-only Streamlit dashboard

The dashboard reads an already generated Slice 1 report and, optionally, one
completed audited RL run. It cannot train, tune, regenerate reports or modify runs.

```powershell
.venv/Scripts/python -m trading_pipeline.dashboard `
  --report reports/20260920T010939Z-1225374b/v1 `
  --rl-run runs/rl/20260920T002714Z-9c9692fc `
  --catalog reports/catalog/v1
```

The fixed/random RL harness can be omitted by leaving out `--rl-run`. Report schema v2
adds Overview/Evidence, Data & Splits, Model Diagnostics, Training Diagnostics, RL Gym,
Backtest & Benchmark, and Securities views. Older runs show explicit not-recorded states;
the UI never reconstructs unavailable training histories.

Training Diagnostics provides one family/metric-filtered surface over the common trace
contract. Histogram GBT and XGBoost use boosting round on the x-axis and decimal-return
RMSE on the y-axis; DQN retains its recorded step and native reward/loss semantics.
Run ID, device, split role and protocol status are shown with summaries and CPU/GPU
benchmarks. Elastic Net explicitly reports that no conventional epoch/boosting curve
applies.

The optional catalogue adds run/config/time selection and persisted multi-run metric
inspection. The dashboard warns when selected runs have different compatibility keys;
it does not pool them. Catalogue and generated-report checksums are verified before
display when their versioned provenance supplies output manifests.
