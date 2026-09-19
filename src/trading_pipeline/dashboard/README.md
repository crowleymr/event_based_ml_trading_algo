# Read-only Streamlit dashboard

The dashboard reads an already generated Slice 1 report and, optionally, one
completed audited RL run. It cannot train, tune, regenerate reports or modify runs.

```powershell
.venv/Scripts/python -m trading_pipeline.dashboard `
  --report reports/20260912T071137Z-9899fd9a/v1 `
  --rl-run runs/rl/<run_id>
```

The fixed/random RL harness can be omitted by leaving out `--rl-run`.
