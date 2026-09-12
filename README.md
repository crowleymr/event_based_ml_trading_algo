# Slice 1 daily equity research

Authoritative scope: [FSD v2](docs/FSD_v2.md) and
[implementation plan](docs/planning/CODEX_SLICE1_IMPLEMENTATION_PLAN.md).
Fixed 100 US equity universe, Yahoo daily prices, public SEC EPS/Net Income,
strict filing availability, five-session labels, chronological purged validation,
Elastic Net and histogram GBT, weekly equal/inverse-volatility top-K portfolios.

## Reproduce (PowerShell, repository root)

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m trading_pipeline.run --config configs/smoke.yaml
$env:SEC_USER_AGENT = 'TradingResearchPOC Your Real Name your-real-contact@email.org'
.venv/Scripts/python -m trading_pipeline.run --config configs/poc.yaml
```

Use your actual contact; the example must be replaced. A Git-ignored `.sec-user-agent`
file is also supported. The supplied local contact is already configured on this
machine. It is excluded from run configuration snapshots and Git.
For the exact verified dependency set install `requirements-lock.txt` before
`requirements.txt`. This lock was generated on Python 3.12/Windows; CI uses the
portable package constraints and records resolved versions in run metadata.
On Linux use `python3.12` to create the environment and `.venv/bin/python` thereafter.
`--ingest-only` downloads and validates without training. Repeating a command reuses
raw caches and creates a new run directory; it never overwrites previous runs.

Smoke uses deterministic **synthetic** fixtures, clearly isolated in `data/smoke`
and `runs/smoke`. It proves software flow, not market performance. The live config
uses 2015-01-01 through 2026-09-12 (end exclusive) and never substitutes providers.
Failed securities stop a complete research run; see `data/ingestion_errors.json`.

## Method

Market features use bars through T. SEC facts join only when session date is
strictly later than filed date. Latest period within the latest public filing
wins; shortest duration breaks multiple-period ties. Annual/quarterly levels
remain mixed and growth features are deferred. Restatements retain their own
filed dates. USD and USD/share are the only model-input units.

The shared calendar after 20-session warmup is split 60/20/20. Labels reaching the
next boundary are purged and the next split's first five sessions are embargoed.
Imputation/scaling and models fit on train only. Each family has four fixed
parameter settings for each feature set. Validation mean daily Spearman IC selects
settings and the E5 source, with validation RMSE and fixed order as tie-breaks.
`selection.json` and fitted models are saved **before** test predictions. No test
performance controls model, portfolio or parameter selection. Test-tail rows with
unavailable labels remain eligible for portfolio valuation, not ML metrics.

Signals form after the first session of each ISO week and execute at T+1 **close**.
Existing holdings earn the return into that close; new holdings earn subsequent
returns. Positions drift between trades. Costs are 10 bps of absolute traded
dollars (both buys and sells), solved against post-cost NAV. Cash starts at 1;
initial entry costs are included. Sharpe uses 252 sessions and zero risk-free rate.
Live runs also attempt B0 (SPY buy-and-hold) from Yahoo, with a single T+1-close
entry and the same costs. It is a benchmark, not a candidate in the equity universe.
Turnover is gross absolute buys plus sells / pre-trade NAV, averaged over all days.
Accumulated costs are in initial-capital units. Final holdings are marked, not
liquidated. Missing held or execution bars cause an explicit failure.

## Artefacts

- `data/raw/yahoo/`: complete yfinance daily responses and ingestion metadata.
- `data/raw/sec/`: original official SEC mapping and Company Facts JSON.
- `data/curated/<entity>/part.parquet`: idempotent canonical current tables.
- `data/catalog.duckdb`: views querying canonical Parquet.
- `data/features/<run_id>/features.parquet`: immutable feature/split snapshots.
- `runs/<run_id>/`: config, metadata, raw and snapshot hashes, split manifest,
  frozen selection, saved models, daily IC, metrics, predictions, daily positions,
  trades/costs, equity curves, comparison CSV/Parquet, coefficients, plots, summary.

Runs and raw data are deliberately Git-ignored, not deleted. Back them up separately.
Manual GitHub Actions runs tests, offline smoke and uploads both data and artefacts.
Each complete run also passes the persisted-data integrity audit (`audit.json`).
To re-audit without retraining:

```powershell
.venv/Scripts/python -m trading_pipeline.audit --run runs/20260912T071137Z-9899fd9a
```

Raw caches are intentionally immutable. For a new source vintage, choose a new
`data_dir` in a copied config; preserve the old directory and run manifests.
Hardware inventory, CPU model/core count, NVIDIA model/VRAM, driver compatibility,
installed CUDA toolkit/runtime and actual CPU model devices are recorded. Missing
CUDA/tools do not prevent CPU runs. No GPU dependencies are required.

## Limitations and review

Accepted survivorship bias; current identifiers and retrospectively adjusted Yahoo
history; no delisting model or investability reconstruction. SEC facts retain
fiscal metadata but are not a general comparable-quarter XBRL engine. Event table
covers filings carrying the selected facts. Missing SEC features are train-imputed.
Dollar volume uses unadjusted close; price returns use adjusted close. Vendor
back-adjustments can revise historic prices/volumes: raw download hashes preserve
the research vintage, but do not supply historical vendor vintages.
Five-day close labels differ from executable T+1 returns. No spread/impact model
beyond fixed cost. No inferential alpha claims or automatic stage-gate success.
No NLP, RL, MPT, intraday, Australian equities or production UI.
