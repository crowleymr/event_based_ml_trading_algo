# ML-Driven Short-Horizon Systematic Equity Trading POC

This repository is a reproducible research application for testing whether machine-learning
models can improve the weekly cross-sectional ranking of approximately 100 large-cap US
equities. It combines daily Yahoo market data with point-in-time public SEC fundamentals,
trains two required model families, converts predictions into long-only portfolios, and
evaluates them out of sample after transaction costs.

The mission is to demonstrate a complete, leakage-safe machine-learning workflow for a
real financial decision problem. It is an academic proof of concept, not a live trading
system or investment recommendation.

## Objectives

- Preserve and validate raw Yahoo OHLCV and SEC Company Facts data.
- Enforce the point-in-time rule: market session date must be after the SEC filed date.
- Build market-only F0 and market-plus-SEC F1 feature sets.
- Predict five-trading-day forward adjusted-close returns.
- Select models using chronological training and validation data only.
- Run E0-E5 with weekly, next-session execution and 10 bps one-way costs.
- Compare equal-weight and inverse-volatility top-10 portfolios.
- Persist datasets, models, predictions, holdings, trades, metrics, plots, and audit evidence.

Slice 1 deliberately excludes NLP, reinforcement learning, mean-variance optimisation,
intraday data, Australian equities, live execution, and a production user interface.

## Repository layout

~~~text
event_based_ml_trading_algo/
├── README.md                       # Operator and reviewer entry point
├── pyproject.toml                  # Package metadata and portable dependencies
├── requirements.txt               # Editable development installation
├── requirements-lock.txt          # Verified Windows/Python 3.12 environment
├── configs/
│   ├── poc.yaml                    # Full live research run
│   ├── smoke.yaml                  # Deterministic offline smoke run
│   └── universe.txt                # Fixed 100-equity US universe
├── data/                           # Git-ignored, preserved source and derived data
│   ├── raw/{yahoo,sec}/
│   ├── curated/
│   └── features/
├── src/trading_pipeline/
│   ├── data/                       # Universe, clients, schemas, identifiers, ingestion
│   ├── features/                   # Market features, PIT fundamentals, feature assembly
│   ├── modelling/                  # Target, split, models, tuning, prediction, ML metrics
│   ├── portfolio/                  # Signals, weighting engines, T+1 backtest
│   ├── tracking/                   # Run artefacts and plots
│   ├── validation/                 # Persisted-run leakage and integrity audit
│   ├── reporting/                  # Read-only versioned report generation
│   ├── config.py                   # Locked configuration validation
│   ├── environment.py              # CPU/GPU and reproducibility inventory
│   ├── audit.py                    # Audit command-line entry point
│   └── run.py                      # End-to-end application entry point
├── tests/                          # Unit, leakage, accounting, reproducibility, layout tests
├── notebooks/
│   └── poc_results.ipynb           # Read-only exploration of persisted results
├── runs/                           # Git-ignored immutable experiment artefacts
├── docs/                           # Specification, decisions, logs, architecture, evidence
└── .github/workflows/smoke.yml     # Manual CPU-only test and smoke workflow
~~~

The detailed module-to-requirement map is in
[Architecture](docs/ARCHITECTURE.md). Each substantive source package also has a
concise local README defining its interfaces and ownership boundary.

## Quickstart

The supported local runtime is Python 3.11 or newer; the verified environment uses
Python 3.12. Run commands from the repository root.

### Install on Windows PowerShell

~~~powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
~~~

For the exact package versions used by the completed Windows run, install
requirements-lock.txt first, then install requirements.txt to register the local package.
On Linux or macOS, create the environment with python3.12 -m venv .venv and replace
.venv/Scripts/python below with .venv/bin/python.

### Run the smoke pipeline

~~~powershell
.venv/Scripts/python -m trading_pipeline.run --config configs/smoke.yaml
~~~

Smoke mode uses deterministic synthetic data, requires no network access, runs E0-E5,
and writes a new directory under runs/smoke/. It verifies software behavior only;
its performance is not research evidence.

### Run the main research pipeline

SEC automated requests require a real application/operator contact. Set it for the
current PowerShell session:

~~~powershell
$env:SEC_USER_AGENT = 'TradingResearchPOC Your Real Name your-real-contact@email.org'
.venv/Scripts/python -m trading_pipeline.run --config configs/poc.yaml
~~~

A Git-ignored .sec-user-agent file containing the same value is also supported.
The main run downloads or reuses Yahoo and SEC data, validates and curates it, builds
features and splits, freezes validation-selected models, evaluates the untouched test
split, executes E0-E5 plus the SPY benchmark, and audits persisted outputs. Successful
output appears under runs/<run_id>/.

To download and validate data without training:

~~~powershell
.venv/Scripts/python -m trading_pipeline.run --config configs/poc.yaml --ingest-only
~~~

To re-audit the completed reference run without retraining:

~~~powershell
.venv/Scripts/python -m trading_pipeline.audit --run runs/20260912T071137Z-9899fd9a
~~~

To generate a versioned read-only report from that exact run:

~~~powershell
.venv/Scripts/python -m trading_pipeline.reporting --run runs/20260912T071137Z-9899fd9a --output reports/20260912T071137Z-9899fd9a/v1
~~~

The output directory must be new. It contains CSV and Parquet tables, a Markdown
report and provenance JSON. Reporting never retrains, selects or changes the source run.

### Run the Phase 2 exploratory selector and dashboard

Install the optional packages separately from the Slice 1 core runtime:

~~~powershell
.venv/Scripts/python -m pip install -e ".[test,phase2]"
~~~

Run the frozen CPU-first selector protocol:

~~~powershell
.venv/Scripts/python -m trading_pipeline.rl.runner --config configs/rl_pilot.yaml
~~~

The v2 protocol accepts `cpu`, `cuda`, or `auto`. CUDA remains optional and is not a
CI dependency. For a local CUDA 12.8 environment, install the official wheel separately
and verify actual tensor placement before a run:

~~~powershell
.venv/Scripts/python -m pip install --force-reinstall torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m trading_pipeline.rl.device --device auto
~~~

`auto` records the requested and actual device and falls back safely to CPU with an
explicit reason. Each new RL run emits schema-versioned training traces/summaries and a
short CPU/GPU timing diagnostic. These diagnostics never select a device, model or seed.

Start the read-only dashboard from an existing generated report and optional completed
RL run:

~~~powershell
.venv/Scripts/python -m trading_pipeline.dashboard `
  --report reports/20260912T071137Z-9899fd9a/v2 `
  --rl-run runs/rl/<run_id>
~~~

The RL run is exploratory and the completed Slice 1 final test is descriptive only.
The dashboard never trains, tunes, regenerates reports or modifies source artefacts.

Raw caches are not overwritten. Use a new data_dir for a new source-data vintage.
Every full invocation creates a new run ID and preserves previous experiment artefacts.

## Key documentation

| Document | Purpose |
|---|---|
| [Functional Specification v2](docs/FSD_v2.md) | Authoritative research scope, policies, schemas, experiments, and Definition of Done |
| [Slice 1 Implementation Plan](docs/planning/CODEX_SLICE1_IMPLEMENTATION_PLAN.md) | Authoritative phased build plan and target repository layout |
| [Architecture and Component Status](docs/ARCHITECTURE.md) | Actual module ownership, planned-layout conformance, and component status |
| [Slice 1 Completion Report](docs/SLICE1_COMPLETION_REPORT.md) | Definition-of-Done evidence, reproduction commands, limitations, and stage-gate risks |
| [Documentation Index](docs/README.md) | Current, generated and future documentation map |
| [Backlog](docs/BACKLOG.md) | Gated Slice 1 close-out and Slice 2+ roadmap |
| [Experiment Registry](docs/EXPERIMENT_REGISTRY.md) | Immutable machine IDs and semantic labels |
| [Glossary](docs/GLOSSARY.md) | Project terminology, metrics and research controls |
| [Assignment Support](docs/assignment/README.md) | Living paper/presentation material and student-owned reflection prompts |
| [Implementation Log](docs/IMPLEMENTATION_LOG.md) | Chronological implementation and verification record |
| [AI Use and Verification](docs/AI_USE_AND_VERIFICATION.md) | Challenges, AI-assisted actions, independent checks and student prompts |
| [Public Notebook Preparation](docs/PUBLIC_NOTEBOOK.md) | Self-contained generated-report workflow; no publication claim |
| [Decisions and Limitations](docs/DECISIONS.md) | Technical choices and explicitly retained limitations |
| [Phase 2 RL Gymnasium Pilot](docs/RL_GYM.md) | Frozen MDP, leakage controls, artefact contract and confirmatory boundary |
| [Planning Index](docs/planning/README.md) | Current and archived planning material |

## Outputs and operating notes

- data/raw/yahoo/ and data/raw/sec/ preserve original source payloads.
- data/curated/ is canonical Parquet; data/catalog.duckdb exposes query views.
- data/features/<run_id>/ preserves the feature and split snapshot.
- runs/<run_id>/selection.json records validation-only model selection before test scoring.
- runs/<run_id>/ contains models, predictions, positions, trades, costs, equity curves,
  metrics, plots, manifests, environment metadata, summary, logs, and audit.json.
- Generated reports are versioned derivatives outside the immutable source run and
  record the run ID, input hashes, code revision and generation timestamp.

The completed reference run covers 100 equities and passed all persisted-run integrity
checks. Known research limitations include survivorship bias, retrospective Yahoo
adjustments, incomplete exact-tag EPS coverage for Visa, mixed SEC fiscal durations,
and a five-day close label that is not identical to the realised T+1 execution window.
Review [the completion report](docs/SLICE1_COMPLETION_REPORT.md) before interpreting results.
