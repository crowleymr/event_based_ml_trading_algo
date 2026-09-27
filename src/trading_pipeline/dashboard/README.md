# Read-only Streamlit dashboard

The dashboard reads completed, generated evidence. It cannot train, tune,
regenerate reports or modify runs. From the repository checkout, the normal
operator command is simply:

```powershell
.venv/Scripts/python -m trading_pipeline.dashboard
```

It selects the newest completed, audited expanded report. The main research
tabs show that run only. The **Model monitoring / run comparisons** tab shows
compatible earlier runs and labels them with completion time, study/budget,
implemented arm-to-component/device mappings, protocol hash and code commit.
Optional explicit paths remain available for forensic inspection of a specific
report or the legacy Slice 1/RL evidence surfaces.

The fixed/random RL harness can be omitted by leaving out `--rl-run`. Report schema v2
adds Overview/Evidence, Data & Splits, Model Diagnostics, Training Diagnostics, RL Gym,
Backtest & Benchmark, and Securities views. Older runs show explicit not-recorded states;
the UI never reconstructs unavailable training histories.

Training Diagnostics provides one family/metric-filtered surface over the common trace
contract. Histogram GBT and XGBoost use boosting round on the x-axis and decimal-return
RMSE on the y-axis; DQN retains its recorded step and native reward/loss semantics.
The overview is a graphical executive summary: it introduces the experiment families,
shows validation and descriptive test outcomes side by side, identifies the highest and
lowest descriptive Sharpe under the selected evidence lens, and places portfolio growth
against the market front and centre. Tables and hash detail remain available in expanders.
When a catalogue is supplied, its audited runs are explored through a single run dropdown.

Run ID, device, split role and protocol status are shown with summaries and CPU/GPU
benchmarks. Elastic Net explicitly reports that no conventional epoch/boosting curve
applies.

The optional catalogue adds run/config/time selection and persisted multi-run metric
inspection. The dashboard warns when selected runs have different compatibility keys;
it does not pool them. Catalogue and generated-report checksums are verified before
display when their versioned provenance supplies output manifests.

## Completed expanded close-out report

After the authoritative study run is complete and audited, the ordinary operator
flow is two commands from the repository checkout:

```powershell
.venv/Scripts/python -m trading_pipeline.reporting.expanded_closeout
.venv/Scripts/python -m trading_pipeline.dashboard
```

The first command discovers the newest completed, audited expanded run and creates
its report, or verifies and reuses that report if it already exists. The second
discovers verified expanded reports, opens the newest by source completion time,
and compares reports with the same generated evaluation calendar, execution, cost,
annualisation and benchmark contract in the Model monitoring / run comparisons tab.
Different protocol hashes are visibly flagged, with budget, folds, seeds, requested
devices and arm mappings retained for interpretation. Partial, smoke and tampered
reports are skipped. If none exists,
the dashboard prints the simple generation command. Explicit `--run-id` and
`--expanded-report` remain available for a particular run. The expanded report is
self-contained. `--expanded-report` and the legacy `--report` are mutually exclusive
entry points. The dashboard checks the report manifest, all
eleven table hashes, schemas and row counts before display; it reads no raw data or
source-run outputs. Its tabs cover pipeline/security evidence, architecture/HPO,
declared risk and frontier views, the complete descriptive final test bench, and the
generated question/assumption register. Model and risk controls filter presentation
only; the SPY benchmark remains visible in the test bench. A missing or incomplete
report fails closed. No test outcome is used to choose a model or scenario.
