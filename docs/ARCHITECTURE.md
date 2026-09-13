# Slice 1 architecture and component status

This document maps the implementation to the repository layout in the authoritative
[Slice 1 implementation plan](planning/CODEX_SLICE1_IMPLEMENTATION_PLAN.md).
The package follows the planned responsibility boundaries. Two small orchestration
modules, data/pipeline.py and modelling/training.py, are deliberate additions: they
coordinate sibling components without moving policy into the command-line runner.

## Pipeline

~~~mermaid
flowchart LR
    C[Config and fixed universe] --> I[Yahoo and SEC ingestion]
    I --> V[Schema and coverage validation]
    V --> P[(Raw and curated Parquet)]
    P --> F[F0 and PIT-safe F1 features]
    F --> S[Purged 60/20/20 split]
    S --> M[Elastic Net and GBT]
    M --> L[Validation-only selection lock]
    L --> R[Weekly ranking and top-K weights]
    R --> B[T+1 backtest and 10 bps costs]
    B --> A[Artefacts, metrics, plots, audit]
~~~

## Planned layout conformance

| Planned component | Implemented module | Responsibility | Status |
|---|---|---|---|
| data/universe.py | src/trading_pipeline/data/universe.py | Checked-in fixed universe | Built and tested |
| data/yahoo_client.py | src/trading_pipeline/data/yahoo_client.py | Yahoo download, retry, raw cache, canonical bars | Built and live verified |
| data/sec_client.py | src/trading_pipeline/data/sec_client.py | SEC contact, retry, raw JSON, fact/event normalization | Built and live verified |
| data/security_master.py | src/trading_pipeline/data/security_master.py | Ticker/CIK bridge and stable IDs | Built and tested |
| data/schemas.py | src/trading_pipeline/data/schemas.py | Schemas, atomic Parquet, hashes, validation | Built and tested |
| ingestion coordinator | src/trading_pipeline/data/pipeline.py | Live/synthetic orchestration and DuckDB views | Built; supporting module |
| features/market.py | src/trading_pipeline/features/market.py | Nine backward-looking F0 features | Built and tested |
| features/fundamentals.py | src/trading_pipeline/features/fundamentals.py | Strict session-after-filing joins | Built and PIT tested |
| features/build.py | src/trading_pipeline/features/build.py | F0/F1 assembly and label attachment | Built |
| modelling/targets.py | src/trading_pipeline/modelling/targets.py | Exact five-session target | Built and tested |
| modelling/splits.py | src/trading_pipeline/modelling/splits.py | 60/20/20, purge, embargo | Built and tested |
| modelling/elastic_net.py | src/trading_pipeline/modelling/elastic_net.py | Train-only preprocessing and fixed grid | Built and tested |
| modelling/gbt.py | src/trading_pipeline/modelling/gbt.py | CPU histogram GBT and fixed grid | Built and tested |
| modelling/evaluate.py | src/trading_pipeline/modelling/evaluate.py | MAE, RMSE, daily and mean IC | Built and persisted |
| training coordinator | src/trading_pipeline/modelling/training.py | Matrix, validation selection, test prediction | Built; isolation tested |
| portfolio/signals.py | src/trading_pipeline/portfolio/signals.py | Shared deterministic top-K selection | Built and tested |
| portfolio/equal_weight.py | src/trading_pipeline/portfolio/equal_weight.py | Long-only equal weights | Built and tested |
| portfolio/inverse_vol.py | src/trading_pipeline/portfolio/inverse_vol.py | Normalized inverse-volatility weights | Built and tested |
| portfolio/backtest.py | src/trading_pipeline/portfolio/backtest.py | Weekly T+1 execution, drift, costs, metrics | Built and tested |
| tracking/artefacts.py | src/trading_pipeline/tracking/artefacts.py | JSON and plot persistence | Built and tested |
| validation/leakage.py | src/trading_pipeline/validation/leakage.py | PIT, purge, T+1, cost and hash audit | Built; 17 live checks passed |
| run.py | src/trading_pipeline/run.py | Single-command orchestration | Built; smoke/live passed |
| notebooks/poc_results.ipynb | notebooks/poc_results.ipynb | Persisted-result exploration only | Built |

config.py, environment.py, and the audit.py CLI facade support the planned components.
Package initializers expose stable imports while implementation ownership remains in
the modules above.

## Functional coverage

All mandatory Slice 1 components are designed and built: fixed universe and security
master; Yahoo and SEC caches; canonical Parquet and DuckDB views; filing events; F0/F1;
five-day target; purged split; Elastic Net and GBT; validation-only selection; E0-E5;
equal and inverse-volatility portfolios; T+1 backtesting; costs; metrics; artefacts;
leakage tests; a single command; and CPU-only CI smoke.

The limited optional areas are explicit. SEC comparable-period growth was cut to avoid
unsafe XBRL assumptions; tree feature importance was cut; the notebook reads persisted
results instead of creating another execution path; remote Actions has not been
dispatched; and no production UI was built. These cuts follow the plan.

## Dependency direction

~~~text
data -> features -> modelling -> portfolio -> tracking/validation -> run
~~~

run.py owns experiment sequencing. Data-source code does not import features or models.
Portfolio code consumes prediction tables and prices, not fitted estimators. Validation
reads completed artefacts and never tunes or fits models. This keeps leakage policy
reviewable and prevents notebooks or plotting from becoming alternate pipelines.

## Independent architecture review

A separate high-capability review task inspected the refactor and ran the full suite.
It found no missing mandatory component or regression in the locked research rules.
Its concrete findings were addressed here: recursive source hashing; rejection of null
market keys and conflicting SEC fact identities; rejection of null predictions/weights
and missing filing provenance; run-local feature snapshots; and an input-vintage
manifest limited to raw files actually consumed by the run.
