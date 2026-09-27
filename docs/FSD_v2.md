# Functional Specification Document v2
## ML-Driven Short-Horizon Systematic Equity Trading POC

**Status:** Authoritative weekend POC specification  
**Assignment deadline:** 29 September 2026  
**POC stage gate:** End of current 1.5-day weekend  
**Primary implementation agent:** Codex  
**Human role:** requirements, research judgement, review, stage-gate decisions

## 1. Purpose
Build a compact, reproducible ML trading research system demonstrating practical and theoretical applications of machine learning to a real-world financial decision problem.

The POC is not intended to prove a publishable market anomaly, nor to build a production trading platform. It must show understanding of the learning task, data/target alignment, model choice, temporal validation, leakage, portfolio construction, risk-adjusted evaluation, practical ML system design, and AI-assisted implementation/verification.

## 2. Locked POC Scope
- **Universe:** approximately 100 large-cap US equities.
- **Known limitation:** survivorship bias is accepted for the weekend POC and must be documented.
- **Geography:** US only. Australia deferred.
- **Frequency:** daily bars.
- **Prediction horizon:** 5-trading-day forward return.
- **Rebalance:** weekly.
- **No intraday, news/NLP, RL, MPT, live trading, or Australian fundamentals in Slice 1.**

## 3. Research Questions
### Primary
> Can machine-learning models using market features and point-in-time public SEC fundamentals improve short-horizon cross-sectional equity ranking relative to simple baselines, and does this improvement translate into better out-of-sample risk-adjusted portfolio performance after transaction costs?

### Secondary
> Does risk-aware portfolio construction improve drawdown and risk-adjusted return when applied to identical ML predictions?

## 4. Hypotheses
- **H1:** Gradient-boosted trees outperform a regularised linear model on ranking and/or out-of-sample portfolio metrics.
- **H2:** Point-in-time EPS and Net Income features provide incremental information beyond market-only features.
- **H3:** Inverse-volatility weighting reduces volatility/drawdown relative to equal-weight top-K, possibly at the cost of raw return.

A valid null result is acceptable.

## 5. Data Sources
### Market
**Primary:** `yfinance` / Yahoo Finance, used only for historical daily OHLCV/adjusted pricing and corporate-action-adjusted data where available.

Requirements:
- academic/personal POC only;
- throttling, retry, exponential backoff;
- cache raw downloads;
- never use Yahoo fundamentals as historical point-in-time features.

### Fundamentals
**Primary:** SEC EDGAR `data.sec.gov` Company Facts JSON API.

Extract only:
- `EarningsPerShareBasic`
- `NetIncomeLoss`

Preserve value, unit, fiscal period end, filed date, form, and accession metadata where available.

### Corporate Events
For Slice 1, SEC filing metadata is the event stream. Persist at minimum security/CIK, filed date/time where available, form, accession identifier. General news is deferred.

## 6. Identifier Strategy
Create an internal security master. Use SEC ticker-to-CIK mapping as the bridge between ticker, CIK and internal `security_id`.

Logical entities:
- `security_master`
- `identifier_map`
- `market_bars`
- `fundamental_facts`
- `corporate_events`

Do not join fundamentals to prices by company name.

## 7. Point-in-Time Policy
A fundamental/event record may only be used after it became public.

For daily data, conservative rule:
```text
market.session_date > fundamental.filed_date
```

Never:
- use fiscal period end as availability date;
- backfill future filings into prior dates;
- use present-day restated Yahoo fundamentals historically.

## 8. Data Technology Stack
### Canonical storage
**Parquet**.

Suggested layout:
```text
data/
  raw/
    yahoo/
    sec/
  curated/
    security_master/
    market_bars/
    fundamental_facts/
    corporate_events/
  features/
```

### Query layer
**DuckDB** for local SQL/querying and native Parquet scans. A persistent `.duckdb` file may store views/metadata while Parquet remains canonical.

### DataFrame engine
**Polars** for feature transforms, rolling/group operations, as-of joins and Parquet IO. Pandas may be used only for dependency compatibility or clear simplicity gains.

Do not introduce Spark or remote DB infrastructure.

## 9. Canonical Schemas
### security_master
```text
security_id
company_id
ticker
exchange
currency
cik
is_active
```

### market_bars
```text
security_id
ticker
session_date
open
high
low
close
adjusted_close
volume
source
ingested_at
```
Primary key: `security_id, session_date`

### fundamental_facts
```text
company_id
security_id
cik
fact_name
fact_value
unit
fiscal_period_end
filed_date
form
accession_number
source
ingested_at
```

### corporate_events
```text
company_id
security_id
cik
event_date
event_datetime
event_type
form
accession_number
source
ingested_at
```

## 10. Feature Sets
### F0 — Market-only
Mandatory candidates:
- 5/10/20-day returns;
- 5/20-day rolling volatility;
- 20-day average dollar volume;
- volume ratio vs 20-day average;
- distance from 20-day moving average;
- distance from 52-week high where sufficient history exists.

### F1 — Market + SEC
F0 plus:
- latest filed EPS;
- latest filed Net Income;
- simple change versus prior comparable filing where readily reliable.

If prior-period comparability becomes an XBRL edge-case sink, retain PIT latest values and document the limitation.

## 11. Target
Primary target:
```text
forward_return_5d = adjusted_close[t+5] / adjusted_close[t] - 1
```

Use regression and cross-sectional ranking. The future return is label-only and never part of features.

## 12. ML Models
Exactly two mandatory families:

### M1 — Elastic Net
`sklearn.linear_model.ElasticNet` in a preprocessing pipeline.

### M2 — Gradient-Boosted Trees
Prefer LightGBM/XGBoost if installation is reliable; otherwise sklearn `HistGradientBoostingRegressor`.

No neural networks in Slice 1.

### Local hardware and device policy
The development machine has an NVIDIA RTX 4060 Ti with 16 GB VRAM. GPU use is optional local acceleration, not a POC dependency. Mandatory weekend models (Elastic Net and GBT) remain CPU-first unless GPU acceleration is trivial and stable. If XGBoost/LightGBM GPU support is used, provide a CPU fallback and record any fallback reason.

Use fixed seeds and deterministic/reproducible settings where practical; record settings and any remaining nondeterminism.

## 13. Mandatory Model/Feature Matrix
| ID | Features | Model |
|---|---|---|
| MF1 | F0 Market | Elastic Net |
| MF2 | F0 Market | GBT |
| MF3 | F1 Market + SEC | Elastic Net |
| MF4 | F1 Market + SEC | GBT |

Use a small fixed grid or <= 8–12 tuning configurations per family. No 50-trial Bayesian search.

## 14. Temporal Validation
- chronological 60/20/20 train/validation/test;
- purge observations around split boundaries for overlapping 5-day labels;
- simple embargo aligned approximately to the 5-day label horizon;
- preprocessing and feature selection fit on training only;
- final test is untouched during design/model selection.

Full CPCV/PBO infrastructure is deferred.

## 15. ML Evaluation
Mandatory:
- MAE;
- RMSE;
- cross-sectional Spearman rank correlation / IC.

Directional accuracy is supplementary.

## 16. Signal Generation
At each weekly rebalance:
1. score eligible stocks with predicted 5-day return;
2. rank cross-sectionally;
3. select top K, default `K=10`.

The same predictions feed both portfolio engines.

## 17. Portfolio / Risk Engines
### P1 — Equal-Weight Top-K
Top K, long-only, `weight = 1/K`.

### P2 — Inverse-Volatility Top-K
Same top-K securities, weight proportional to `1 / trailing_volatility`, normalised to sum to one. Optional max-position cap.

Deferred: MPT, shorting, RL.

## 18. Backtest
Execution convention:
```text
features through session T
signal created after T
position entered at next trading session T+1
```

- weekly rebalance;
- long-only;
- configurable default transaction cost = 10 bps one-way;
- persist positions, weights, turnover, costs, daily return and equity curve.

## 19. Financial Metrics
Mandatory:
- total return;
- annualised return;
- annualised volatility;
- Sharpe ratio;
- maximum drawdown;
- turnover;
- cumulative transaction cost.

Desirable: benchmark-relative return and hit rate.

Deferred: DSR, PBO, HAC inference.

## 20. Benchmarks
- **B0:** broad-market buy-and-hold proxy where easily available.
- **B1:** simple momentum top-K strategy without ML.

## 21. Weekend Experiment Matrix
| Experiment | Features | Model | Portfolio |
|---|---|---|---|
| E0 | Momentum rule | None | Equal Weight |
| E1 | Market | Elastic Net | Equal Weight |
| E2 | Market | GBT | Equal Weight |
| E3 | Market + SEC | Elastic Net | Equal Weight |
| E4 | Market + SEC | GBT | Equal Weight |
| E5 | Best validation ML configuration | Same predictions | Inverse Volatility |

E5 must be chosen from validation results, never final-test performance.

## 22. MLOps Architecture
Single-command local-first pipeline:
```text
config → ingest → validate → curate → features → split → train → predict → portfolio → backtest → metrics → artefacts
```

Example:
```bash
python -m trading_pipeline.run --config configs/poc.yaml
```

### Tracking
Filesystem artefact contract is mandatory. MLflow is preferred only if it adds little friction.

Long-running local studies must also support a **read-only supervisor/heartbeat**.
The supervisor may launch and observe the authoritative `trading_pipeline.run`
process, but must not implement training, selection, tuning, backtesting, artefact
repair or outcome-dependent control. It must preserve timestamped stdout/stderr and
structured lifecycle events, and periodically record wall-clock elapsed time,
heartbeat time, process-tree RSS, system RAM availability/utilisation, CPU load and
utilisation, and—when available—GPU utilisation and VRAM used/total. Missing GPU or
platform counters must be recorded explicitly and must not fail CPU-only execution.
Logs must survive normal completion, model failure, interruption and supervisor
failure, and must remain separate from immutable research-result artefacts.

Expanded studies must support fail-closed continuation through immutable derived
attempts. A resumed attempt must bind its parent run, exact approved protocol and
input hashes, code-state hash, seeds and completed-cell checkpoint hashes. It may
reuse only complete, schema-validated, untampered cells and their persisted
predictions; incomplete cells are rerun. Parent failures and logs remain immutable.
Protocol or code-state drift, missing prediction checkpoints, ledger disagreement or
holdout-boundary ambiguity must reject resume. Holdout access remains forbidden until
the derived attempt verifies and seals the complete family locks.

For the expanded weekly RL selector, portfolio reward must use exact observed T+1
execution and weekly endpoint prices. Intervening daily marks are not required by the
weekly return definition and must not be filled or substituted. A missing required
endpoint fails closed. Security eligibility at a signal must use only the declared
contiguous trailing observed-price history then available; exclusions and missing-date
reasons are auditable per signal, and full-future-panel completeness screening is
forbidden as lookahead. This expanded-study rule does not alter locked Slice 1.

### GitHub Actions
Weekend target: tests + smoke run + artefact upload. Full 10-year training may remain local. CI/GitHub Actions must run on CPU without requiring a GPU, CUDA, or GPU-specific dependencies.

## 23. Run Artefact Contract
```text
runs/<run_id>/
  config.yaml
  metadata.json
  dataset_manifest.json
  split_manifest.json
  metrics.json
  predictions.parquet
  positions.parquet
  trades.parquet
  equity_curve.parquet
  feature_importance.parquet
  plots/
    equity_curve.png
    drawdown.png
    model_comparison.png
  summary.md
```

Metadata includes Git commit, timestamp, model, feature set, portfolio, label horizon, split dates, cost assumption and seed. Detect and record CPU/GPU availability (including CPU model/core count and GPU model/VRAM where available), CUDA toolkit/runtime version where installed, relevant package versions, and the actual chosen device per model. Distinguish driver-reported CUDA compatibility from an installed toolkit/runtime; record unavailable or unknown values explicitly. Detection must not fail a CPU-only run. Persist reproducibility settings and any GPU fallback reason.

For supervised long-running execution, write operational logs to a separate
versioned directory containing the exact child command, supervisor metadata,
timestamped combined console output, structured events, periodic resource samples
and a terminal status/exit code. The monitoring directory is operational evidence,
not a source of model metrics and not an input to model selection.

## 24. Mandatory Tests
### Data
- market schema;
- SEC schema;
- unique keys;
- ticker↔CIK mapping coverage;
- idempotent ingestion.

### PIT / leakage
- no fundamental fact before filing availability;
- no feature reads future bars;
- target horizon correct;
- preprocessing fit on training only;
- temporal split ordering;
- purge boundary valid.

### Portfolio/backtest
- weights sum <= 1;
- no negative POC weights;
- P1/P2 use same selected predictions;
- transaction costs reduce portfolio value;
- T+1 execution enforced.

## 25. Dashboard
Not a blocker. If core pipeline is green, add a thin Streamlit dashboard reading persisted artefacts only. Suggested panels: experiment comparison, equity curves, Sharpe/return/drawdown, turnover/cost, feature importance, holdings.

No training inside Streamlit.

## 26. Stage Gate
### GREEN
- ~100 equities ingested;
- EPS + Net Income PIT pipeline;
- F0/F1 built;
- Elastic Net + GBT trained;
- E0–E5 executed or nearly complete;
- P1/P2 working;
- costs and results persisted;
- leakage tests pass.

Then select at most ONE major enhancement.

### AMBER
Core runs but results/features are incomplete. Fix PIT, validation and simple features. Do not add NLP/RL.

### RED
End-to-end flow is unreliable. Freeze enhancements and repair core.

## 27. Post-Stage-Gate Backlog
Choose selectively:
1. SEC event text / FinBERT or other NLP;
2. shallow NN / LSTM sequence representation;
3. constrained MPT;
4. RL risk engine;
5. Australian market expansion.

PyTorch/CUDA may be used for optional neural-network, NLP/FinBERT, sequence-model, or RL experiments after the stage gate. The 16 GB VRAM is sufficient for moderate local inference, fine-tuning and NN experiments, subject to model size, batch size and precision; it does not change the current weekend scope.

No enhancement is required just to make the project look complex.

## 28. AI / Human Boundary
### Codex may decide autonomously
- module decomposition;
- tests/logging;
- routine package choice;
- refactors;
- retries/error handling;
- performance optimisation.

### Human approval required
- research question changes;
- universe expansion;
- target/horizon changes;
- data-source substitution;
- PIT rule changes;
- split-policy changes;
- headline transaction-cost changes;
- adding/removing mandatory models;
- declaring success.

## 29. Non-Goals
This is not production investment advice, HFT, live execution, proof of persistent alpha, a fully unbiased historical simulation, an A* finance paper, an RL showcase or an LLM showcase.
