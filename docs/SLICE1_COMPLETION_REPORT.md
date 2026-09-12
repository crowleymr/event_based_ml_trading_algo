# Slice 1 completion evidence

Date: 12 September 2026. Research run: `20260912T071137Z-9899fd9a`.
This records engineering completion evidence; declaring research/stage-gate success
remains a human decision under FSD section 28.

## Definition of Done

| Requirement | Status / evidence |
|---|---|
| Fixed approximately 100-stock universe | PASS: 100 checked-in US equities; verified MMC/MRSH rename retains issuer |
| SEC CIK mapping | PASS: 100/100 official SEC mapping coverage |
| Yahoo OHLCV cached | PASS: 294,000 bars; 2,940 sessions per equity; 2015-01-02 to 2026-09-11 |
| EPS + Net Income raw SEC payloads | PASS with coverage limitation: 100 Company Facts payloads; EPS tag available for 99 issuers, Net Income for 100 |
| Curated Parquet | PASS: security master, identifiers, market bars, facts, events |
| DuckDB queries Parquet | PASS: all five canonical views count-checked during ingestion |
| F0 features | PASS: all nine candidates, rolling windows and temporal tests |
| F1 features | PASS: strictly PIT latest EPS/Net Income; missing values train-imputed |
| Five-day target | PASS: exact adjusted-close ratio and explicit label-end dates |
| Leakage-safe split | PASS: 60/20/20, boundary purges and five-session embargo |
| Elastic Net | PASS: train-only imputation/scaling; four fixed configurations |
| GBT | PASS: CPU sklearn HistGradientBoostingRegressor; four fixed configurations |
| E0–E4 runnable | PASS: all completed on real data |
| Equal-weight backtest | PASS: weekly top-10, next-session close, drifted holdings |
| Inverse-volatility engine | PASS: E5 completed using identical E3 predictions and selected stocks |
| 10 bps costs | PASS: self-financing traded-dollar accounting, initial entry included |
| ML metrics | PASS: MAE, RMSE, daily Spearman IC and mean IC |
| Financial metrics | PASS: return, annualisation, volatility, Sharpe, drawdown, turnover and cost |
| Plots/summary | PASS: three PNGs, Markdown, comparison CSV/Parquet; equity plot visually checked |
| PIT/leakage tests | PASS: regression tests plus 17 persisted-run integrity checks |
| Full local one-command run | PASS: live pipeline completed; failed download attempts preserved caches |
| Smoke GitHub Action | IMPLEMENTED: manual CPU-only tests/smoke/upload; remote dispatch not performed |

Additional implemented evidence: SPY B0 buy-and-hold, saved fitted models, frozen
selection manifest, immutable dataset/feature snapshots, raw SHA256 manifest,
package lock, environment/device metadata, CPU-only detection regression test.

Final verification: **17 tests passed in 17.77 seconds**. Updated single-command
smoke completed at `runs/smoke/20260912T072009Z-573f1f57`, including startup hardware
metadata and the automatic 17-check audit. Live-run integrity audit also passed
17/17 checks. GitHub-hosted execution was not performed.

## Temporal and selection record

- Train: 2015-02-02 to 2022-01-06; 174,700 rows.
- Validation: 2022-01-24 to 2024-05-06; 57,400 rows.
- Test: 2024-05-21 to 2026-09-11; 57,900 rows (unlabelled final five sessions retained for valuation).
- 2,000 boundary rows excluded across purge and embargo regions.
- E5 source **E3**: selected on validation mean daily IC, with RMSE tie-break.
- Selection persisted before final-test prediction; models never refit using test.
- Final-test results are descriptive in the run summary. They did not drive any configuration changes.

## Reproduce

From `D:\repos\event_based_ml_trading_algo` in PowerShell:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m trading_pipeline.run --config configs/smoke.yaml
.venv/Scripts/python -m trading_pipeline.run --config configs/poc.yaml
.venv/Scripts/python -m trading_pipeline.audit --run runs/20260912T071137Z-9899fd9a
```

The supplied SEC contact is stored locally in Git-ignored `.sec-user-agent`.
On a different machine set `SEC_USER_AGENT` to your actual application/contact.
Existing raw caches avoid new downloads. Every pipeline invocation creates a new
run ID. Exact reproduction requires the preserved data vintage and dependency set.
To refresh sources, use a new data directory rather than replacing old caches.

## Artefact locations

- `data/raw/yahoo/`: 100 equity responses plus SPY, original yfinance columns/actions and timestamp sidecars.
- `data/raw/sec/`: SEC ticker mapping and 100 original Company Facts JSON payloads.
- `data/curated/` and `data/catalog.duckdb`: canonical tables and query views.
- `data/features/20260912T071137Z-9899fd9a/features.parquet`: research features, labels, PIT dates and splits.
- `runs/20260912T071137Z-9899fd9a/`: complete live run contract (~25 MB), including `audit.json`, `selection.json`, `models/`, `datasets/`, and `plots/`.
- `runs/smoke/`: preserved synthetic software-verification runs.

Source, tests and implementation documentation were committed in logical phases.
The live core was run at commit `9cadfe9`; metadata records its source hashes.
Later additions concern integrity auditing, hardware inventory and documentation.
The live hardware metadata explicitly states that inventory was captured after
execution on the same host. Authoritative document changes made concurrently by
the user were reviewed and preserved.

## Unresolved limitations and stage-gate risks

1. **Historical universe bias:** fixed present-day survivors, current identifiers and
   retrospective Yahoo adjustment factors. No historical constituent, delisting,
   or vendor-vintage reconstruction. Corporate restructurings can reduce historical
   issuer comparability. These results cannot establish unbiased historical alpha.
2. **SEC coverage/comparability:** Visa lacks the exact basic EPS tag. Across feature
   rows, EPS is 4.18% null and Net Income 4.45% null before train-only imputation.
   Latest facts mix fiscal durations; annual/quarterly comparability and growth
   features are deferred. No alternate concept was silently substituted.
3. **Execution/label mismatch:** labels use T to T+5 closes; tradable positions enter
   at T+1 close. No same-close execution, but the label is not the realised holding
   period return. Fixed 10 bps cost excludes an explicit spread/impact model.
4. **Statistical scope:** one chronological split, small grids and no multiple-testing
   inference, DSR/PBO or formal significance claim. Test is now evaluated and must
   not be reused for model/configuration selection in follow-up work.
5. **Operational scope:** missing held-price bars fail rather than being filled;
   final positions are marked without terminal liquidation. No live execution or
   production UI. Local files need separate backup; Git intentionally excludes data.
6. **Runtime reproducibility:** same-host seeded smoke reproduces predictions/metrics;
   cross-version/hardware floating-point identity is not guaranteed. GPU is detected
   but unused. Installed toolkit/runtime are 11.8; driver-reported CUDA compatibility
   is explicitly unknown, not inferred from the toolkit or GPU model.
7. **Remote CI:** workflow exists and its commands passed locally; no GitHub dispatch
   or remote artefact-upload result is claimed.

No known failing correctness tests remain. The engineering evidence supports review
against the GREEN criteria, with the above coverage and research caveats. No deferred
NLP, RL, MPT, intraday, Australian equity, neural-network or production UI work was added.
