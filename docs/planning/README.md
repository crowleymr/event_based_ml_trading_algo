# Trading POC Synthesis v2

This package supersedes earlier weekend planning where there is a conflict.

## Files
- `FSD_v2.md` — authoritative functional specification.
- `CODEX_SLICE1_IMPLEMENTATION_PLAN.md` — autonomous Codex implementation contract.
- `SYNTHESIS_DECISIONS.md` — rationale for accepted/reduced/deferred research recommendations.

## Locked Slice 1
- ~100 US equities
- daily market data via yfinance
- SEC EDGAR PIT fundamentals/events
- EPS + Net Income
- Parquet + DuckDB + Polars
- 5-day return regression/ranking
- Elastic Net vs GBT
- market-only vs market+SEC
- equal-weight vs inverse-vol
- weekly rebalance
- T+1 execution
- 10 bps one-way cost
- chronological split + purge/embargo
- local-first automated pipeline

## Deferred
Australia, intraday, news/FinBERT, RNN/Transformers, MPT, RL, live trading and production cloud architecture.

Start Codex using the prompt at the bottom of `CODEX_SLICE1_IMPLEMENTATION_PLAN.md`.
