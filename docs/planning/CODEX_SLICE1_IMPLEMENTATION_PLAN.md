# Codex Autonomous Implementation Plan — Slice 1
## ML-Driven Short-Horizon Systematic Equity Trading POC

**Authority:** `FSD_v2.md`  
**Goal:** Deliver the first complete leakage-safe vertical slice autonomously.

## 1. Mission
Implement:
```text
~100 US equities
→ Yahoo daily OHLCV + SEC ticker/CIK + Company Facts
→ Parquet / DuckDB
→ Polars feature pipeline
→ 5-day forward return
→ leakage-safe temporal split
→ Elastic Net + Gradient Boosted Trees
→ ranking
→ equal-weight top-K + inverse-vol top-K
→ weekly T+1 backtest
→ ML + financial metrics
→ persisted artefacts
```

Do not implement deferred components until the Definition of Done passes.

## 2. Priority
1. correctness;
2. PIT integrity;
3. reproducibility;
4. end-to-end automation;
5. readable code;
6. sophistication.

## 3. Preferred Stack
- Python 3.11+
- Polars
- DuckDB
- PyArrow
- scikit-learn
- yfinance
- requests/httpx
- PyYAML
- pytest
- matplotlib
- MLflow only if low-friction

GBT: LightGBM/XGBoost if frictionless; otherwise sklearn HistGradientBoostingRegressor.

## 4. Repository Layout
```text
trading-ml-poc/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── poc.yaml
│   └── smoke.yaml
├── data/
│   ├── raw/yahoo/
│   ├── raw/sec/
│   ├── curated/
│   └── features/
├── src/trading_pipeline/
│   ├── data/
│   │   ├── universe.py
│   │   ├── yahoo_client.py
│   │   ├── sec_client.py
│   │   ├── security_master.py
│   │   └── schemas.py
│   ├── features/
│   │   ├── market.py
│   │   ├── fundamentals.py
│   │   └── build.py
│   ├── modelling/
│   │   ├── targets.py
│   │   ├── splits.py
│   │   ├── elastic_net.py
│   │   ├── gbt.py
│   │   └── evaluate.py
│   ├── portfolio/
│   │   ├── signals.py
│   │   ├── equal_weight.py
│   │   ├── inverse_vol.py
│   │   └── backtest.py
│   ├── tracking/artefacts.py
│   ├── validation/leakage.py
│   └── run.py
├── tests/
├── notebooks/poc_results.ipynb
├── runs/
└── docs/
    ├── FSD_v2.md
    ├── IMPLEMENTATION_LOG.md
    └── DECISIONS.md
```

## 5. Phase A — Bootstrap
- initialise environment;
- dependencies;
- config loader;
- logging;
- pytest;
- executable entry point.

Acceptance:
```bash
pytest
python -m trading_pipeline.run --config configs/smoke.yaml
```

## 6. Phase B — Universe + Security Master
- fixed checked-in ~100-stock large-cap US universe;
- fetch official SEC ticker→CIK map;
- create stable internal `security_id`;
- persist Parquet;
- expose DuckDB view/table;
- record SEC mapping coverage.

Do not scrape an index list on every run.

## 7. Phase C — Market Ingestion
Implement yfinance ingestion:
- default 2015-present;
- daily OHLCV/adjusted close;
- throttle, retry/backoff;
- raw cache;
- idempotent curated Parquet;
- sensible partitioning.

Validate uniqueness, monotonic dates, nonnegative volume and coverage.

## 8. Phase D — SEC Ingestion
Use official SEC APIs.

Mandatory:
- ticker/CIK map;
- Company Facts JSON;
- only `EarningsPerShareBasic` and `NetIncomeLoss`.

Persist raw JSON before normalising. Preserve value, units, fiscal end, filed date, form, accession where available.

SEC User-Agent must be configurable via environment/config. Do not hard-code fake personal contact details.

## 9. Phase E — Event Table
Build minimal SEC filing event table:
- security/CIK;
- filed date;
- form;
- accession ID.

No filing text NLP.

## 10. Phase F — PIT Join
Use Polars `join_asof` or equivalent DuckDB as-of logic.

Hard rule with daily filing availability:
```text
session_date > filed_date
```

Tests must show zero violations. Never backfill future facts.

## 11. Phase G — Market Features
Implement F0:
- return_5d;
- return_10d;
- return_20d;
- vol_5d;
- vol_20d;
- avg_dollar_volume_20d;
- volume_ratio_20d;
- ma_distance_20d;
- price_to_52w_high.

Signal uses data through T and executes at T+1.

## 12. Phase H — Fundamental Features
Implement F1 by extending F0:
- latest filed EPS;
- latest filed Net Income;
- prior comparable values/change only if straightforward and PIT-safe.

If SEC accounting duplication/comparability becomes ambiguous, keep latest PIT values and document the limitation rather than building a general XBRL engine.

## 13. Phase I — Target
```text
forward_return_5d = adj_close(t+5)/adj_close(t)-1
```
Unit test on synthetic data.

## 14. Phase J — Temporal Split
- chronological 60/20/20;
- 5-day horizon-aware purge at boundaries;
- simple ~5-day embargo;
- persist split dates.

No CPCV.

## 15. Phase K — Models
### Elastic Net
sklearn Pipeline with train-only imputation/scaling.

### GBT
Robust tree implementation; no scaling needed.

Hyperparameter budget: <= 8–12 fixed configurations per family on validation only.

## 16. Phase L — ML Metrics
Calculate:
- MAE;
- RMSE;
- cross-sectional Spearman IC per date;
- mean IC.

Persist prediction schema:
```text
session_date
security_id
ticker
actual_forward_return_5d
predicted_return_5d
predicted_rank
split
model_id
feature_set
```

## 17. Phase M — Momentum Baseline
Implement E0 independently:
- 20-day momentum rank;
- weekly top K=10;
- equal weight;
- T+1 execution;
- same universe, costs and backtester.

## 18. Phase N — Portfolio Engines
### Equal weight
Top K=10, long-only.

### Inverse volatility
Same top-K list, trailing 20-day volatility, normalised weights, optional cap.

No optimiser.

## 19. Phase O — Backtester
Rules:
```text
signal uses data through T
execution at T+1
weekly rebalance
long-only
10 bps one-way transaction costs
```

Persist target weights, trades, costs, daily returns and equity curve.

Tests must prove no same-close execution and costs are correctly applied.

## 20. Phase P — Financial Metrics
- total return;
- annualised return;
- annualised volatility;
- Sharpe;
- max drawdown;
- average turnover;
- cumulative transaction cost.

## 21. Phase Q — Experiment Runner
Exact experiment IDs:
```text
E0 momentum + equal weight
E1 market + elastic net + equal weight
E2 market + GBT + equal weight
E3 market+SEC + elastic net + equal weight
E4 market+SEC + GBT + equal weight
E5 best validation ML prediction + inverse volatility
```

E5 selection uses validation only.

## 22. Phase R — Artefacts
Per run:
```text
config.yaml
metadata.json
dataset_manifest.json
split_manifest.json
metrics.json
predictions.parquet
positions.parquet
trades.parquet
equity_curve.parquet
feature_importance.parquet (if applicable)
summary.md
plots/
```

Also produce a top-level experiment comparison CSV/Parquet.

## 23. Phase S — Tracking
Implement filesystem artefacts first. Add MLflow only if trivial after core pipeline works.

## 24. Phase T — Tests
Minimum:
- security mapping;
- market schema;
- SEC schema;
- ingestion idempotency;
- target math;
- market temporal correctness;
- SEC PIT join;
- split chronology;
- purge boundary;
- train-only preprocessing;
- portfolio weights;
- inverse-vol weights;
- T+1 execution;
- transaction-cost accounting;
- reproducibility smoke run.

## 25. Phase U — GitHub Actions
Only after local green.
Manual dispatch:
1. checkout;
2. install;
3. pytest;
4. smoke pipeline;
5. upload smoke artefacts.

Do not make full 10-year research training mandatory on every push.

## 26. Implementation Log
Append factual records to `docs/IMPLEMENTATION_LOG.md`:
```text
timestamp
task
agent action
result
verification
technical issue
```
Do not invent human reflection. Human decisions go in `docs/DECISIONS.md`.

## 27. Autonomy
Proceed autonomously on routine implementation, tests, debugging, retries, refactors, and performance improvements.

Stop/log approval needed for:
- alternative data provider;
- material universe change;
- changing 5-day target;
- PIT-policy change;
- final split change;
- execution-timing change;
- test-driven model selection;
- NLP/RL/MPT;
- removing/replacing mandatory experiment.

## 28. Scope Cut Order
If behind schedule, cut:
1. MLflow;
2. dashboard;
3. feature-importance extras;
4. fundamental growth/change features;
5. E5 inverse-vol comparison.

Never cut PIT joins, leakage tests, Elastic Net, GBT, market baseline, SEC raw ingestion, backtester, costs or persisted results.

## 29. Definition of Done
```text
[ ] fixed ~100-stock universe
[ ] SEC CIK mapping
[ ] Yahoo OHLCV cached
[ ] EPS + Net Income SEC raw payloads cached
[ ] curated Parquet
[ ] DuckDB queries curated Parquet
[ ] F0 market features
[ ] F1 SEC-enhanced features
[ ] 5-day target
[ ] leakage-safe split
[ ] Elastic Net
[ ] GBT
[ ] E0–E4 runnable
[ ] equal-weight backtest
[ ] inverse-vol engine implemented or queued for final block
[ ] 10 bps costs
[ ] ML metrics persisted
[ ] financial metrics persisted
[ ] plots/summary persisted
[ ] PIT/leakage tests pass
[ ] full local run from one command
[ ] smoke GitHub Action if time permits
```

Finish with:
1. build summary;
2. test results;
3. experiment status;
4. unresolved defects;
5. exact reproduce commands;
6. stage-gate risks.

## 30. Initial Codex Prompt
> Read `docs/FSD_v2.md` and `CODEX_SLICE1_IMPLEMENTATION_PLAN.md` in full. Treat them as authoritative. Implement Slice 1 autonomously and incrementally. Do not implement NLP, RL, MPT, intraday data, Australian equities, or a production UI. Prioritise a leakage-safe end-to-end pipeline over sophistication. Run tests after each major phase, preserve raw data and experiment artefacts, and commit logical changes. Stop only for decisions explicitly listed under Autonomy. Otherwise diagnose and fix routine failures without human intervention. Do not use final-test performance to choose models or configurations. At completion report Definition-of-Done status, reproduce commands, artefact locations, unresolved limitations and stage-gate risks.
