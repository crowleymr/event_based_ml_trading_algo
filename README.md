# ML-Driven Short-Horizon Systematic Equity Trading Research

This repository is a local, reproducible research application for testing whether
machine-learning models improve weekly cross-sectional US-equity ranking and
after-cost portfolio outcomes. It is an academic proof of concept, not a live trading
system or investment recommendation.

There are two distinct research tracks:

- **Locked Slice 1** is the original approximately 100-stock study. Its authoritative
  route is `trading_pipeline.run --config ...`; its completed test is descriptive and
  must not be reused for model selection.
- **Expanded close-out study** admits as many as 500 candidates and compares Elastic
  Net, Histogram GBT, XGBoost, LSTM, Transformer, DQN and PPO under one point-in-time
  feature contract. Its approved protocol is separate from, and does not rewrite,
  Slice 1.

All training and backtesting goes through `trading_pipeline.run`. Reports, the
dashboard and the notebook only read completed immutable artefacts.

## Repository layout

~~~text
event_based_ml_trading_algo/
├── configs/                         # Slice 1 and approved expanded protocols
├── data/                            # Git-ignored source caches and derived data
├── docs/                            # Specification, decisions, evidence and guides
├── notebooks/final_evidence.ipynb   # Read-only end-to-end evidence narrative
├── reports/                         # Generated reports and operational monitoring
├── runs/                            # Git-ignored immutable experiment artefacts
├── .tmp/                            # Git-ignored disposable test/tool scratch
├── src/trading_pipeline/            # Authoritative application code
├── tests/                           # Unit, leakage, accounting and integration tests
├── pyproject.toml                   # Package and optional dependency groups
├── requirements.txt                 # Editable core/test/XGBoost installation
└── requirements-lock.txt            # Verified core Windows/Python environment
~~~

## Fresh-repository shakedown

The commands below use Windows PowerShell. Run every command from the repository
root—the directory containing this README and `pyproject.toml`. Python 3.12 is the
verified version; Python 3.11 or newer is supported. Git is needed for provenance.

If you downloaded an archive, extract it and use `Set-Location` to enter the extracted
directory. If you have a repository URL, clone it first:

~~~powershell
git clone <repository-url> event_based_ml_trading_algo
Set-Location event_based_ml_trading_algo
~~~

### 1. Create the environment and install dependencies

For Slice 1, tests and the offline smoke run:

~~~powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
~~~

For the expanded study, dashboard and notebook, also install the Phase 2 and notebook
extras:

~~~powershell
.venv\Scripts\python.exe -m pip install -e ".[test,phase2,notebook]"
.venv\Scripts\python.exe -m pip check
~~~

Linux and macOS users can create the environment with `python3.12 -m venv .venv`
and replace `.venv\Scripts\python.exe` with `.venv/bin/python`. CUDA is optional and
is not a CI dependency. The approved expanded protocol requests CUDA for XGBoost,
LSTM and Transformer, while Elastic Net, Histogram GBT, DQN and PPO run on CPU.

To inspect actual PyTorch/CUDA placement before a long run:

~~~powershell
.venv\Scripts\python.exe -m trading_pipeline.rl.device --device auto
~~~

`auto` records the requested and actual device and reports an explicit CPU fallback.
If a CUDA-enabled PyTorch build is required on Windows, the currently verified local
wheel can be installed separately and checked again:

~~~powershell
.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m trading_pipeline.rl.device --device auto
~~~

### 2. Run tests and the offline smoke experiment

~~~powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m trading_pipeline.run --config configs/smoke.yaml
~~~

The smoke experiment uses deterministic synthetic data and requires no network
access. It verifies the installed software, temporal controls, model interfaces,
portfolio accounting and artefact audit. Its results are not research evidence.
The application assigns its internal output directory automatically; you do not need
to construct or remember a run name.

Put disposable test and tool output under `.tmp/<tool>/<task>/`. To review and clean
completed scratch, run `python -m trading_pipeline.operations.clean_temp` to list its
immediate contents, then add `--purge` to remove them. The purge retains `.tmp/` and
does not target research data, runs or reports. See
[Temporary workspace policy](docs/operations/TEMPORARY_WORKSPACE.md).

## Download and validate Slice 1 data

Live SEC requests require an honest application/operator contact. Set it for the
current PowerShell session:

~~~powershell
$env:SEC_USER_AGENT = 'TradingResearchPOC Your Real Name your-real-contact@email.org'
~~~

A Git-ignored `.sec-user-agent` file containing the same value is also supported.
Do not use a fabricated contact.

Download and validate Slice 1 source data without starting a training run:

~~~powershell
.venv\Scripts\python.exe -m trading_pipeline.data.download --config configs/poc.yaml
~~~

The config defaults to `configs/poc.yaml`, so the common invocation can be shortened
to `.venv\Scripts\python.exe -m trading_pipeline.data.download`.

This step downloads or reuses Yahoo daily market data and SEC Company Facts, validates
their schemas and identifiers, writes canonical Parquet tables under `data/curated/`,
maintains source caches under `data/raw/yahoo/` and `data/raw/sec/`, and creates
`data/catalog.duckdb`. It does **not** train a model or run a backtest. Failed downloads
leave caches and `data/ingestion_errors.json` in place so the command can be retried.

Raw caches are never silently replaced. A genuinely new source-data vintage should use
a new data directory and a newly approved protocol rather than overwriting evidence.

## Execute the locked Slice 1 experiment

After the ingestion-only check succeeds:

~~~powershell
.venv\Scripts\python.exe -m trading_pipeline.run --config configs/poc.yaml
~~~

The authoritative command reuses the caches, then curates data, builds point-in-time
features, creates purged chronological splits, tunes on training/validation only,
freezes selection before opening the descriptive test, runs T+1 after-cost backtests,
and audits the persisted outputs. A new immutable output directory is assigned
automatically on every invocation.

Slice 1 deliberately excludes the expanded neural-network and RL comparison. Review
[Slice 1 completion evidence](docs/SLICE1_COMPLETION_REPORT.md) before interpreting
its results.

## Execute the approved expanded close-out study

The approved expanded protocol is hash-bound to a specific admitted-universe,
canonical-feature, causal-stack, benchmark and validation-manifest bundle. Its exact
relative paths and hashes are declared in the approved YAML. Those large immutable
inputs are Git-ignored. A source-only clone therefore cannot reproduce the approved
expanded run merely by running the Slice 1 downloader: copy or restore the matching
evidence bundle first.
There is currently no public one-command downloader for that pinned bundle. Do not
substitute newly downloaded files into the approved protocol; their hashes and data
vintage would differ and require a new documented approval.

When the pinned bundle is present, launch through the read-only resource supervisor:

~~~powershell
.venv\Scripts\python.exe -m trading_pipeline.operations.study_supervisor --study configs/studies/expanded_closeout_approved_v5.yaml --log-dir reports/operations/expanded-closeout-v5-attempt-1
~~~

The supervisor starts the authoritative `trading_pipeline.run --study ...` process,
prints concise phase/warning/error/completion messages, and writes detailed console,
event and CPU/RAM/GPU/VRAM telemetry under the friendly log directory supplied above.
It does not train, select, repair or inspect scores itself. Choose another unused,
human-readable log directory for each attempt.

### Resume an interrupted or failed expanded attempt

Resume creates a new immutable child run and never modifies the failed parent. It
reuses only complete, schema-validated, hash-verified checkpoints and refuses changed
protocols, inputs, source code or dependencies. Do not edit code or update packages
between failure and resume.

The following PowerShell selects the newest failed checkpointed attempt, so you do not
have to type its generated directory name:

~~~powershell
$failedRun = Get-ChildItem runs\expanded_closeout -Directory |
    Where-Object { Test-Path (Join-Path $_.FullName 'failure.json') } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

.venv\Scripts\python.exe -m trading_pipeline.operations.study_supervisor `
    --study configs/studies/expanded_closeout_approved_v5.yaml `
    --resume-from $failedRun.FullName `
    --log-dir reports/operations/expanded-closeout-resume-1
~~~

The runner fails closed if the selected attempt predates resumable checkpoints or is
otherwise incompatible. See [Expanded study monitoring](docs/operations/STUDY_MONITORING.md)
for log contents and failure semantics.

## Generate and inspect final results

After an expanded run completes and passes its audit, these commands deliberately use
the newest verified completed run. No run identifier or repository path is required:

~~~powershell
.venv\Scripts\python.exe -m trading_pipeline.reporting.expanded_closeout
.venv\Scripts\python.exe -m trading_pipeline.dashboard
~~~

The report generator writes a separate hash-backed report without changing the source
run. The dashboard opens the newest verified report and offers compatible historical
run comparisons in its model-monitoring view. It never trains or tunes models.

For the educational narrative, open `notebooks/final_evidence.ipynb` in Jupyter and
run all cells:

~~~powershell
.venv\Scripts\python.exe -m jupyter lab notebooks/final_evidence.ipynb
~~~

With no `TRADING_REPORT_DIR` override, the notebook discovers the newest generated
expanded report, verifies its manifest and table hashes, and explains the data pipeline,
architecture/HPO evidence, risk scenarios, model-conditioned efficient frontiers,
realised risk-return curves and final benchmark. Student reflection prompts require
the student's own evidence and wording.

## Research and evidence rules

- The completed final test is descriptive. Never use it to select, tune or reject a
  feature, model, architecture, risk scenario or parameter.
- All reported numbers, tables and charts must be generated from canonical data or
  immutable run artefacts. Correct derivation code and regenerate downstream outputs;
  never hand-patch a published number.
- Preserve point-in-time joins, purge/embargo boundaries, T+1 execution and transaction
  costs. Missing required prices fail closed; they are not silently substituted.
- Keep raw run artefacts immutable. Reports and operational logs live outside `runs/`.
- The dashboard and notebook are read-only consumers, not alternative orchestration
  paths.

## Key documentation

| Document | Purpose |
|---|---|
| [Functional Specification v2](docs/FSD_v2.md) | Authoritative scope, research controls and artefact contract |
| [Expanded close-out plan](docs/planning/FINAL_5_DAY_EXPANDED_RESEARCH_CLOSEOUT_PLAN.md) | Current expanded-study critical path and acceptance gates |
| [Expanded study monitoring](docs/operations/STUDY_MONITORING.md) | Supervisor, telemetry, failure and resume operations |
| [Temporary workspace policy](docs/operations/TEMPORARY_WORKSPACE.md) | Central `.tmp/` layout, safe purge boundary and legacy cleanup rules |
| [Architecture](docs/ARCHITECTURE.md) | Module ownership and component status |
| [Decisions and limitations](docs/DECISIONS.md) | Research and engineering decisions that constrain interpretation |
| [Slice 1 completion evidence](docs/SLICE1_COMPLETION_REPORT.md) | Locked original-study completion and limitations |
| [Experiment registry](docs/EXPERIMENT_REGISTRY.md) | Stable experiment identifiers and presentation labels |
| [Glossary](docs/GLOSSARY.md) | Metrics, concepts and research controls |
| [Public notebook preparation](docs/PUBLIC_NOTEBOOK.md) | Read-only notebook and optional publication workflow |
| [AI use and verification](docs/AI_USE_AND_VERIFICATION.md) | AI-assisted actions and student-owned reflection boundary |
| [Implementation log](docs/IMPLEMENTATION_LOG.md) | Chronological implementation and verification record |

## Known limitations

The studies retain survivorship bias, current-identifier and retrospective Yahoo
adjustment limitations, incomplete exact-tag SEC coverage, mixed fiscal durations,
and a five-session label that is not identical to the realised T+1 holding interval.
The expanded close-out protocol is deadline-constrained exploratory evidence, not a
confirmatory claim. CUDA improves selected production-scale models but does not make
cross-hardware floating-point results bit-identical.
