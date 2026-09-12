# Functional Specification Document (FSD)
## ML-Driven Event-Based Systematic Trading Strategy — POC

**Version:** 0.1  
**POC target:** End of first 1.5-day build weekend  
**Assignment deadline:** 29 September 2026  
**Primary objective:** Produce a reproducible, research-ready proof of concept that can be trained, backtested and evaluated autonomously during the following week.

---

## 1. Executive Summary

The system will investigate whether machine learning can improve systematic equity-selection/trading decisions when evaluated using **time-aware validation and risk-adjusted portfolio performance**, rather than relying only on conventional ML prediction metrics.

The preferred architecture is an **ML-Driven Event-Based Systematic Trading Strategy**, but the first POC deliberately uses a constrained version that can be implemented and verified within 1.5 days.

### POC principle

> Build one complete vertical slice that works end-to-end before adding sophistication.

The system must be able to:

1. acquire/reproduce historical market data;
2. construct leakage-safe features;
3. generate a prediction/signal;
4. construct a portfolio;
5. backtest the portfolio;
6. calculate ML and financial metrics;
7. persist results and artefacts;
8. run the complete process automatically.

The POC should be modular enough that news sentiment, corporate events, RNN-derived features, fundamentals, MPT and reinforcement learning can be added later without redesigning the core pipeline.

---

# 2. Research Question

### Primary research question

**Can machine-learning models improve systematic equity-selection decisions when evaluated using rigorous temporal validation and risk-adjusted portfolio performance?**

### Event-oriented extension

If the event-data pipeline becomes feasible without jeopardising the POC:

**Can machine learning identify situations in which public corporate information contains exploitable information about subsequent abnormal equity returns?**

The first question is the mandatory POC question. The event-oriented question is an enhancement.

---

# 3. Scope

## 3.1 In scope for POC

- US equities initially.
- Historical daily market data.
- A reproducible stock universe.
- Rules-based market/price features.
- At least two candidate predictive models.
- At least two portfolio/risk approaches.
- Time-based train/validation/test methodology.
- Transaction-cost assumption.
- Backtesting.
- Financial and ML metrics.
- Experiment/result persistence.
- Automated execution locally.
- GitHub-compatible execution.
- Clear artefact structure for report generation.

## 3.2 Explicitly deferred

These are valuable enhancements but **not required for the first stage gate**:

- Australian equities.
- Intraday trading.
- Live trading.
- Autonomous web/news agents.
- Annual-report scraping.
- Large-scale fundamental extraction.
- LLM-based financial reasoning.
- Reinforcement-learning portfolio allocation.
- Cloud database infrastructure.
- Production trading execution.
- Real-money deployment.

The architecture should permit these later.

---

# 4. Recommended POC Design

## 4.1 Investment horizon

**Primary:** short-term, daily frequency, holding periods of approximately 5–20 trading days.

Reason:

- substantially easier than intraday data;
- sufficient observations for ML;
- avoids excessive infrastructure;
- allows meaningful transaction-cost modelling;
- supports event/news enhancements later.

## 4.2 Geography

**POC: US equities.**

Rationale:

- plentiful historical data;
- broad universe;
- mature public-data ecosystem;
- easier experimentation;
- no need to solve two markets simultaneously.

Australia can be an extension after the stage gate.

## 4.3 Data storage

**POC recommendation: local Parquet files.**

Suggested structure:

```text
data/
  raw/
    prices/
  processed/
    prices/
    features/
  metadata/
```

Use SQLite only for experiment metadata if useful.

Do **not** introduce Supabase/AWS/Azure as a dependency for the POC unless there is a clear problem that local files cannot solve.

---

# 5. System Architecture

```text
                HISTORICAL DATA
                      |
          +-----------+-----------+
          |                       |
      Market Data             Optional Events
          |                       |
          +-----------+-----------+
                      |
              Data Validation
                      |
              Feature Engineering
                      |
          +-----------+-----------+
          |                       |
    Rules-Based Features     ML-Derived Features
          |                       |
          +-----------+-----------+
                      |
               Prediction Model
                      |
               Signal Generator
                      |
          +-----------+-----------+
          |                       |
      Equal/Rule Based       MPT Portfolio
      Baseline Portfolio      Optimisation
          |                       |
          +-----------+-----------+
                      |
              Portfolio / Risk
                      |
                  Backtest
                      |
       +--------------+--------------+
       |                             |
   ML Metrics                  Financial Metrics
       |                             |
       +--------------+--------------+
                      |
             Experiment Store
                      |
              Report Artefacts
```

RL portfolio optimisation is an **optional post-gate component**, not a POC dependency.

---

# 6. Data Specification

## 6.1 Required market data

Minimum fields:

- ticker/symbol
- trading date
- open
- high
- low
- close
- adjusted close if available
- volume

Derived data:

- daily return
- rolling volatility
- momentum
- moving averages
- relative strength
- volume change
- drawdown
- cross-sectional ranks

## 6.2 Data requirements

The pipeline must:

- use historical observations only;
- make the information timestamp available to the feature pipeline;
- avoid using future prices in features;
- document missing-data handling;
- document survivorship-bias limitations;
- ensure train/test chronology is preserved.

## 6.3 Optional event data

Candidate event sources can include:

- corporate announcements;
- earnings announcements;
- SEC filings;
- news headlines.

Event data should only be introduced if timestamps can be aligned reliably with market observations.

---

# 7. Feature Engineering Candidates

The POC should compare at least two feature approaches and ideally four.

## Candidate F1 — Rules-Based Market Features

Features:

- 5/10/20-day momentum;
- 5/20-day moving-average relationship;
- rolling volatility;
- volume momentum;
- recent drawdown;
- market-relative return;
- cross-sectional ranks.

**Advantages:** fast, transparent, robust baseline.

**Role:** mandatory baseline.

---

## Candidate F2 — Statistical Feature Selection

Start from a larger rules-based feature set and apply:

- correlation filtering;
- mutual information;
- univariate predictive screening;
- recursive/embedded selection where appropriate.

**Research value:** tests whether reducing feature redundancy improves out-of-sample portfolio performance.

---

## Candidate F3 — Sequence Model Features

Use an RNN/LSTM/GRU to encode a rolling sequence of market observations into a latent representation.

**Advantages:**

- captures temporal patterns;
- demonstrates deeper ML;
- visually impressive.

**Risk:** higher complexity and overfitting.

**POC status:** optional if F1/F2 pipeline is stable.

---

## Candidate F4 — News Sentiment

Use a pretrained BERT-style sentiment model to transform news into numerical features.

Possible aggregation:

```text
headline sentiment
        ↓
ticker + date
        ↓
daily sentiment score
        ↓
rolling sentiment statistics
```

An agentic LLM evaluation of news is explicitly **not required for the POC**.

Reason: it creates additional latency, cost, reproducibility and leakage risks.

---

# 8. Prediction Model Candidates

At least two predictive models should be compared.

## M1 — Logistic Regression / Linear Baseline

Predict whether future return exceeds a defined threshold.

Purpose:

- transparent baseline;
- low variance;
- establishes whether complex ML adds value.

## M2 — Gradient Boosted Trees

Candidate implementation:

- XGBoost;
- LightGBM;
- HistGradientBoostingClassifier/Regressor.

Purpose:

- strong tabular baseline;
- handles nonlinear interactions;
- computationally inexpensive.

## M3 — MLP

Optional neural-network model using the engineered feature vector.

Purpose:

- demonstrates PyTorch experience;
- tests whether nonlinear representation adds value.

## M4 — Sequence Model

Optional LSTM/GRU.

Only implement if the core pipeline is already robust.

---

# 9. Prediction Target

Preferred POC target:

### Forward N-day return

For each stock at time `t`:

```text
target(t) = Close(t+N) / Close(t) - 1
```

or a classification target:

```text
target(t) = 1 if forward return > threshold
            0 otherwise
```

Recommended starting point:

**5-trading-day forward return regression.**

The signal can then be derived from predicted returns.

This avoids conflating the ML task with the financial portfolio objective.

---

# 10. Portfolio / Risk Engine Candidates

At least two should be implemented.

## P1 — Equal-Weight Top-K

1. Rank stocks by predicted return.
2. Select top K.
3. Equal-weight selected securities.
4. Rebalance at the defined frequency.

This is the mandatory baseline.

## P2 — Volatility-Adjusted Top-K

Modify weights according to estimated volatility:

```text
raw_weight_i ∝ 1 / volatility_i
```

Then normalise weights.

Purpose:

- introduces explicit risk control;
- simple enough to verify.

## P3 — Mean-Variance / MPT

Use expected returns and covariance estimates to optimise the portfolio subject to constraints.

Possible constraints:

- maximum position weight;
- maximum turnover;
- long-only;
- minimum/maximum diversification.

Compare against P1 and P2.

## P4 — Reinforcement Learning

RL agent learns portfolio allocation/rebalancing decisions.

**Status: deferred enhancement.**

It is attractive academically but should not be allowed to jeopardise the end-to-end POC.

---

# 11. Experimental Matrix

The minimum experiment matrix:

| Experiment | Features | Prediction | Portfolio |
|---|---|---|---|
| E0 | Market baseline | Simple return/momentum rule | Equal weight |
| E1 | Rules-based | Logistic/linear | Equal weight |
| E2 | Rules-based | Gradient boosting | Equal weight |
| E3 | Rules-based | Gradient boosting | Volatility adjusted |
| E4 | Rules-based | Gradient boosting | MPT |

Optional:

| Experiment | Features | Prediction | Portfolio |
|---|---|---|---|
| E5 | Sequence | MLP/GBM | Vol-adjusted |
| E6 | News sentiment | GBM | MPT |
| E7 | News + market | GBM | MPT |
| E8 | Market + news | GBM | RL |

**Do not implement E5–E8 before E0–E4 run reproducibly.**

---

# 12. Validation Methodology

## 12.1 Temporal split

Never randomly shuffle observations across time.

Example:

```text
TRAIN        VALIDATION       TEST
---------    ------------     ----------
2018-2022    2023             2024-2025
```

Exact dates depend on data availability.

## 12.2 Walk-forward evaluation

Preferred final methodology:

```text
Train → Validate → Test
       ↓
roll forward
       ↓
Train → Validate → Test
       ↓
roll forward
```

The test period must remain untouched during model selection.

## 12.3 Leakage controls

The system must explicitly guard against:

- future prices appearing in features;
- future news being associated with earlier observations;
- normalising using future data;
- feature selection using the complete dataset;
- using test-set information to tune hyperparameters;
- look-ahead in portfolio construction;
- unrealistic execution prices.

---

# 13. Backtesting Specification

Backtest assumptions must be explicit.

Minimum:

- starting capital;
- rebalance frequency;
- position sizing;
- transaction cost;
- slippage assumption;
- long-only/short constraints;
- cash treatment;
- execution timing.

Recommended POC:

**Long-only, daily data, rebalance weekly, top-K selection, fixed transaction cost/slippage assumption.**

---

# 14. Evaluation Metrics

## 14.1 ML metrics

Depending on target:

- RMSE;
- MAE;
- directional accuracy;
- rank correlation;
- precision/recall if classification is used.

## 14.2 Financial metrics

Mandatory:

- cumulative return;
- annualised return;
- annualised volatility;
- Sharpe ratio;
- maximum drawdown;
- turnover;
- number of trades;
- transaction costs.

Optional:

- Sortino ratio;
- Calmar ratio;
- hit rate;
- average winning/losing trade.

## 14.3 Research comparison

The key comparison is not:

> Which model has the highest accuracy?

It is:

> Does the more sophisticated ML approach produce a statistically and economically meaningful improvement in out-of-sample portfolio performance after costs?

---

# 15. MLOps Specification

## 15.1 Required automated workflow

```text
1. Environment setup
2. Data acquisition/load
3. Data validation
4. Feature generation
5. Temporal split
6. Model training
7. Validation
8. Model selection
9. Backtest
10. Metrics calculation
11. Artefact generation
12. Experiment registration
13. Summary report generation
```

The workflow should execute from a single command.

Example:

```bash
python -m trading_pipeline.run
```

## 15.2 Experiment artefacts

Each run should produce:

```text
runs/
  YYYYMMDD_HHMMSS/
    config.yaml
    data_manifest.json
    model/
    metrics.json
    predictions.parquet
    portfolio.parquet
    trades.parquet
    equity_curve.csv
    feature_importance.csv
    plots/
      equity_curve.png
      drawdown.png
      returns.png
    summary.md
```

## 15.3 Experiment tracking

POC recommendation:

**Use MLflow locally first.**

Reasons:

- simple local setup;
- experiment tracking;
- parameters and metrics;
- artefact logging;
- reproducibility;
- no external service dependency.

W&B or ClearML can be added later if desired.

## 15.4 CI/CD

GitHub Actions should eventually execute:

```text
push / manual trigger
        ↓
install environment
        ↓
run tests
        ↓
run pipeline
        ↓
store experiment artefacts
```

The first weekend should prove the workflow locally before making GitHub Actions responsible for expensive training.

---

# 16. Proposed Repository Structure

```text
trading-ml-poc/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── baseline.yaml
│   └── experiment.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   └── trading_pipeline/
│       ├── __init__.py
│       ├── data.py
│       ├── validation.py
│       ├── features.py
│       ├── targets.py
│       ├── models.py
│       ├── portfolio.py
│       ├── backtest.py
│       ├── metrics.py
│       ├── tracking.py
│       └── run.py
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_backtest.py
│   └── test_no_leakage.py
│
├── notebooks/
│   └── poc_analysis.ipynb
│
├── runs/
│
├── docs/
│   ├── FSD.md
│   ├── EXPERIMENT_PLAN.md
│   └── IMPLEMENTATION_LOG.md
│
└── .github/
    └── workflows/
        └── train_backtest.yml
```

---

# 17. Dashboard Specification

Dashboard is secondary to research validity.

Minimum display:

- portfolio value;
- cumulative return;
- Sharpe ratio;
- maximum drawdown;
- benchmark comparison;
- current model;
- selected portfolio holdings;
- model comparison table;
- feature importance;
- equity curve;
- drawdown curve.

Recommended technology:

**Streamlit.**

The dashboard should read persisted run artefacts rather than execute expensive training interactively.

---

# 18. Acceptance Criteria — POC Stage Gate

The POC is successful if all mandatory criteria are met.

### A. Data

- [ ] Historical data can be reproduced/downloaded.
- [ ] Data validation executes.
- [ ] Dataset is stored reproducibly.
- [ ] Data limitations are documented.

### B. ML

- [ ] At least two predictive approaches run.
- [ ] Features are generated automatically.
- [ ] Training is deterministic/reproducible where possible.
- [ ] Validation is time-aware.

### C. Portfolio

- [ ] At least two portfolio/risk approaches run.
- [ ] A simple baseline exists.
- [ ] Transaction costs are modelled.
- [ ] Portfolio construction does not use future information.

### D. Backtest

- [ ] Backtest executes automatically.
- [ ] Equity curve is generated.
- [ ] Drawdown is generated.
- [ ] Financial metrics are calculated.

### E. MLOps

- [ ] Entire workflow runs from one command.
- [ ] Run configuration is saved.
- [ ] Metrics are saved.
- [ ] Predictions and portfolio outputs are saved.
- [ ] Plots are saved.
- [ ] Errors are surfaced clearly.
- [ ] A fresh environment can reproduce the run.

### F. Research

- [ ] Baseline is defined.
- [ ] At least one meaningful ML comparison exists.
- [ ] Test set remains separated from model selection.
- [ ] Results can support a paper discussion.

---

# 19. Stage-Gate Decision Rules

At the next weekend, make the decision based on evidence.

## GREEN

Core pipeline is reliable and results are interesting.

Proceed with:

- event/news features;
- improved portfolio optimisation;
- RNN features;
- dashboard;
- report.

## AMBER

Pipeline works but results are weak or unstable.

Prioritise:

- validation improvements;
- leakage audit;
- feature engineering;
- stronger baselines;
- robustness analysis.

Do not add complexity merely to make results look better.

## RED

Pipeline cannot reliably reproduce results.

Freeze new features.

Fix:

- data;
- leakage;
- temporal splitting;
- backtesting;
- experiment reproducibility.

---

# 20. AI Agent Autonomy Boundary

## Agent may autonomously

- create/refactor implementation;
- write unit tests;
- implement feature transformations;
- run experiments;
- tune hyperparameters within defined ranges;
- generate plots;
- calculate metrics;
- diagnose routine errors;
- update documentation;
- maintain experiment artefacts;
- improve code quality.

## Agent must request/record human approval before

- changing the research question;
- changing the prediction target;
- changing train/test methodology;
- introducing a new external dataset;
- changing leakage controls;
- changing the financial objective;
- changing transaction-cost assumptions materially;
- interpreting anomalous results as evidence;
- declaring the strategy successful;
- removing a failed experiment from the record.

This boundary is also useful for the university implementation log because it distinguishes mechanical AI assistance from human research judgement.

---

# 21. POC Priority Order

### Priority 1 — Must work

1. Data
2. Features
3. Baseline model
4. Second model
5. Baseline portfolio
6. Second portfolio
7. Backtest
8. Metrics
9. Experiment artefacts

### Priority 2 — Strongly desirable

10. MLflow
11. automated tests
12. GitHub Actions
13. dashboard

### Priority 3 — Enhancement

14. news sentiment
15. event detection
16. RNN features
17. MPT improvements
18. RL portfolio engine
19. fundamentals extraction
20. Australian market

---

# 22. Non-Goals

The POC is not:

- a production trading platform;
- investment advice;
- a claim that ML can reliably beat markets;
- a live brokerage integration;
- a high-frequency trading system;
- an autonomous financial agent.

The research objective is empirical investigation under controlled and reproducible assumptions.
