# Implementation log

## 2026-09-12 — Bootstrap
Read FSD v2 and the complete implementation plan (located in docs/planning).
Repository initially contained only licence, ignore rules and untracked planning documents.
Created Python package, fixed 100-security universe, separate research/synthetic smoke configs.
Python 3.12 selected from installed runtimes. No research policy changed.

## 2026-09-12 — Data phase
Implemented official SEC mapping/Company Facts, Yahoo daily raw caching, retries,
canonical Parquet, DuckDB views, synthetic fixtures and schema/coverage checks.
Verification: 4 tests passed. Windows sandbox temp-file and Git-index permissions
required approved elevated commands; no data provider was changed.

## 2026-09-12 — Features, models and portfolio
Added Polars rolling features, strict backward filing-date as-of joins, labels,
purge/embargo splits, train-only Elastic Net/GBT pipelines and small fixed grids.
Added weekly T+1-close holdings accounting with drift and self-financing costs.
Verification: 13 tests passed. Full synthetic E0–E5 run completed at
runs/smoke/20260912T070326Z-3265a9d6. This is not empirical trading evidence.
Real SEC mapping fetched; corrected verified MMC -> MRSH rename for the same issuer.
