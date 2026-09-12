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

## 2026-09-12 — Runner and verification
Implemented immutable run snapshots/hashes, fitted model persistence, selection lock,
E0–E5 metrics/plots and manual GitHub Actions smoke upload. Added SPY B0 benchmark
using the existing approved Yahoo provider. Verification: 15 tests passed before
benchmark addition, including two identical seeded complete smoke runs and a
final-test-label perturbation test confirming unchanged validation selection.

## 2026-09-12 — Real run and audit
Full live run completed: runs/20260912T071137Z-9899fd9a (core commit 9cadfe9).
100 equities, 294000 daily bars, 2015-01-02 through 2026-09-11; 51934 SEC facts.
SEC issuer coverage: basic EPS 99%, Net Income 100%. Visa lacks the exact basic EPS
tag; retained with train-only imputation. No company or fundamental tag substituted.
E0–E5 and SPY B0 completed. E5 selected E3 from validation IC, before final test.
Independent 17-check audit passed, including SHA256, PIT, purge, identical E5
predictions/selections, T+1, cost and equity reconciliation. Equity plot inspected.
No test performance was used to change settings. 16-test suite passed after audit addition.

## 2026-09-12 — Concurrent authoritative document update
Observed user changes to FSD/plan adding hardware/reproducibility metadata. Reviewed
the complete diff and implemented best-effort CPU/GPU and CUDA inventory, actual
CPU devices, deterministic settings and fallback explanation. The existing live
metadata was supplemented explicitly after execution on the same host, without
rerunning or selecting on test outcomes. Future runs record this at startup.
Detected Ryzen 5 3600 (6 physical/12 logical cores), RTX 4060 Ti (16380 MiB), installed
CUDA toolkit/runtime 11.8; driver CUDA compatibility was not discoverable and is
explicitly unknown. No GPU execution or scope expansion.

## 2026-09-12 — Final verification
17 tests passed in 17.77 seconds. Final synthetic single-command smoke completed at
runs/smoke/20260912T072009Z-573f1f57 with startup hardware metadata and automatic
17-check integrity audit. Live-run audit passed 17/17 separately. Repaired Markdown
table spacing in the persisted live summary only; no predictions/metrics changed.
Created docs/SLICE1_COMPLETION_REPORT.md covering every DoD item, commands, artefacts,
known limitations and stage-gate risks. Remote GitHub workflow not dispatched.
