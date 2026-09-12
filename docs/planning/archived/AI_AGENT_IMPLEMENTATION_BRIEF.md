# AI Agent Implementation Brief
## ML-Driven Event-Based Systematic Trading Strategy — POC

This document is intended to be handed to Claude Code, Codex, or Cursor as the implementation contract.

---

# 1. Mission

Build a working research POC for an ML-driven systematic equity trading system.

The POC must prioritise **reproducibility, leakage prevention and end-to-end execution** over sophistication.

The implementation must produce usable evidence for a university ML assignment.

---

# 2. Non-Negotiable Constraints

1. Daily-frequency US equities first.
2. Local execution must work before cloud automation.
3. Local Parquet is the default data store.
4. No live trading.
5. No broker integration.
6. No random train/test shuffling.
7. No future-information leakage.
8. Every experiment must save its configuration and results.
9. The complete pipeline must be executable from one command.
10. Do not add complex components until the core pipeline is green.
11. Preserve failed experiments and record why they failed.
12. Never silently alter the research question or evaluation methodology.

---

# 3. Implementation Strategy

Use incremental vertical slices.

## Slice 1

```text
data → simple features → baseline prediction → equal-weight portfolio → backtest → metrics
```

## Slice 2

```text
data → richer features → second model → volatility-adjusted portfolio → backtest → compare
```

## Slice 3

Add MPT.

## Slice 4

Add MLflow and automated experiment execution.

## Slice 5

Only if time remains, add news/RNN/event features.

---

# 4. Technology Recommendation

Preferred:

- Python 3.11 or 3.12
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- pyarrow
- PyYAML
- MLflow
- pytest
- yfinance or another reproducible historical-data provider
- Streamlit later

Do not add heavyweight packages unless justified.

If using XGBoost/LightGBM, isolate them as optional model dependencies where practical.

PyTorch is optional for the POC. Do not introduce it merely to demonstrate neural networks.

---

# 5. Required Interfaces

Create modular functions/classes approximately matching:

```python
load_market_data(...)
validate_market_data(...)
build_features(...)
build_target(...)
temporal_split(...)
train_model(...)
predict(...)
generate_signals(...)
build_equal_weight_portfolio(...)
build_volatility_adjusted_portfolio(...)
build_mpt_portfolio(...)
run_backtest(...)
calculate_ml_metrics(...)
calculate_financial_metrics(...)
save_run_artifacts(...)
run_experiment(...)
```

Exact design is agent discretion provided behaviour remains testable.

---

# 6. Configuration

Do not hard-code experimental parameters.

Example:

```yaml
data:
  universe: "US"
  start_date: "2018-01-01"
  end_date: "2025-12-31"
  frequency: "1d"

target:
  horizon_days: 5
  type: "forward_return"

features:
  momentum_windows: [5, 10, 20]
  volatility_windows: [5, 20]
  volume_windows: [5, 20]

model:
  name: "gradient_boosting"
  random_state: 42

portfolio:
  method: "top_k_equal_weight"
  top_k: 10
  rebalance_frequency: "weekly"
  max_position_weight: 0.10

backtest:
  transaction_cost_bps: 10
  slippage_bps: 5

validation:
  train_end: "2022-12-31"
  validation_end: "2023-12-31"
  test_end: "2025-12-31"
```

The exact dates may be adjusted after data availability is confirmed.

---

# 7. Leakage Audit

Create explicit tests for:

### Feature leakage

For any feature at date `t`, verify it only uses data dated `<= t`.

### Target leakage

Forward return must only be used as the target.

### Scaling leakage

Fit scalers only on training data.

### Feature-selection leakage

Feature selection must occur inside the training process.

### Temporal leakage

Test observations must never influence model fitting or hyperparameter selection.

### Portfolio leakage

Portfolio decisions at `t` must only use information available by the decision time.

The agent must add automated tests for at least the first five.

---

# 8. Baselines

Implement a deliberately simple financial baseline.

Example:

- momentum ranking;
- top 10;
- equal weighted;
- weekly rebalance.

This baseline is important because an ML model should beat something meaningful, not merely random noise.

Also include buy-and-hold market benchmark if data permits.

---

# 9. Models

Mandatory:

### Model A

Linear/logistic baseline.

### Model B

Gradient-boosted tree.

Optional:

### Model C

MLP.

### Model D

LSTM/GRU.

The agent must not implement C/D until A/B and the backtest are working.

---

# 10. Portfolio Engines

Mandatory:

### Portfolio A

Top-K equal weight.

### Portfolio B

Inverse-volatility weighting.

Preferred if time:

### Portfolio C

Mean-variance/MPT optimisation.

Deferred:

### Portfolio D

Reinforcement-learning allocator.

If MPT implementation introduces instability, retain A/B and document the limitation rather than spending the entire POC fixing an optimiser.

---

# 11. Backtesting Rules

The backtester must make the execution convention explicit.

Recommended:

```text
Signal generated after market close at t
Position entered using next available trading price
Holding period/rebalance determined by configuration
Costs applied to turnover
```

Do not assume execution at a price that was not available when the signal was generated.

Record:

- positions;
- weights;
- trades;
- turnover;
- costs;
- daily portfolio value;
- daily returns.

---

# 12. Metrics

Return a structured metrics dictionary.

Minimum:

```python
{
    "total_return": ...,
    "annualised_return": ...,
    "annualised_volatility": ...,
    "sharpe_ratio": ...,
    "max_drawdown": ...,
    "turnover": ...,
    "transaction_costs": ...,
    "rmse": ...,
    "mae": ...,
    "directional_accuracy": ...
}
```

Do not report a metric that is not meaningful for the selected target.

---

# 13. Experiment Tracking

Use MLflow locally.

Log:

- run ID;
- git commit;
- dataset period;
- feature configuration;
- model;
- hyperparameters;
- portfolio method;
- transaction costs;
- validation dates;
- test dates;
- ML metrics;
- financial metrics;
- plots;
- model artefacts.

If MLflow creates excessive setup friction, fall back temporarily to a filesystem-based JSON/CSV experiment registry.

Do not allow tracking infrastructure to block the research pipeline.

---

# 14. Automated Run

The following should work:

```bash
python -m trading_pipeline.run --config configs/baseline.yaml
```

It should:

1. load data;
2. validate data;
3. build features;
4. create target;
5. split data;
6. train;
7. validate;
8. generate predictions;
9. construct portfolio;
10. backtest;
11. calculate metrics;
12. save artefacts;
13. register experiment;
14. write a human-readable summary.

Return a non-zero exit code on failure.

---

# 15. Artefact Contract

Every run must create:

```text
runs/<run_id>/
├── config.yaml
├── metadata.json
├── metrics.json
├── predictions.parquet
├── portfolio.parquet
├── trades.parquet
├── equity_curve.csv
├── feature_importance.csv
├── model/
├── plots/
│   ├── equity_curve.png
│   ├── drawdown.png
│   └── returns_distribution.png
└── summary.md
```

If an artefact is not applicable, record that fact rather than silently omitting it.

---

# 16. Testing Requirements

At minimum:

- data schema test;
- missing-value test;
- feature calculation test;
- target calculation test;
- temporal split test;
- leakage test;
- portfolio weights sum test;
- no-position-over-limit test;
- transaction-cost test;
- backtest accounting test;
- reproducibility smoke test.

Use small synthetic datasets for unit tests.

Do not run the full historical dataset for every unit test.

---

# 17. Failure Handling

The agent should classify failures:

### Data failure

Bad source, missing columns, malformed dates.

### Validation failure

Leakage or chronology problem.

### Model failure

Training error or invalid prediction.

### Portfolio failure

Invalid weights, optimiser failure.

### Backtest failure

Accounting inconsistency.

### Infrastructure failure

Dependency, environment or tracking problem.

Errors should identify the stage and provide enough context for diagnosis.

---

# 18. Research Logging

Create:

```text
docs/IMPLEMENTATION_LOG.md
```

Record:

- date;
- experiment;
- objective;
- AI-generated changes;
- human decision;
- result;
- verification;
- unresolved issue.

The agent may append factual implementation details but must not invent human decisions.

---

# 19. Git Workflow

Use small logical commits.

Suggested:

```text
feat: add market data ingestion
feat: add leakage-safe feature pipeline
feat: add baseline model
feat: add gradient boosting model
feat: add portfolio engine
feat: add backtester
test: add leakage and accounting tests
feat: add experiment tracking
ci: add training workflow
```

Never rewrite history unless explicitly instructed.

---

# 20. GitHub Actions

After local execution succeeds, create:

```text
.github/workflows/train_backtest.yml
```

The workflow should:

1. checkout;
2. install Python;
3. install dependencies;
4. run tests;
5. execute selected experiment;
6. upload run artefacts.

Do not schedule frequent cloud runs during the POC.

Prefer manual dispatch initially.

---

# 21. Dashboard

Build only after reproducible runs exist.

Streamlit pages/components:

```text
Systematic Trading Lab

[Portfolio Value] [Sharpe] [Max Drawdown] [Return]

Equity Curve

Model Comparison

Portfolio Comparison

Current/Test Holdings

Feature Importance

Drawdown

Experiment Configuration
```

Dashboard must consume saved artefacts.

---

# 22. Agent Stop Conditions

Stop and ask for human judgement if:

- data source terms/availability require a substantive change;
- the target must change;
- leakage cannot be resolved unambiguously;
- the test methodology needs changing;
- an experiment produces suspiciously strong results;
- adding a component materially changes the research question;
- a model appears successful only after repeated tuning on the test set;
- a proposed optimisation removes an important baseline;
- financial assumptions materially change.

Routine coding errors do not require stopping.

---

# 23. Definition of Done

The POC is DONE when:

```text
[ ] Fresh environment installs successfully
[ ] Data can be loaded/reproduced
[ ] Data validation passes
[ ] Features generated
[ ] Target generated
[ ] Temporal train/validation/test split works
[ ] Model A trains
[ ] Model B trains
[ ] Portfolio A works
[ ] Portfolio B works
[ ] Backtest works
[ ] Transaction costs included
[ ] Financial metrics calculated
[ ] ML metrics calculated
[ ] Artefacts persisted
[ ] Tests pass
[ ] Full pipeline runs from one command
[ ] Results are reproducible
[ ] Summary.md is generated
```

Anything beyond this is enhancement.

---

# 24. First 1.5-Day Execution Plan

## Block 1 — 2 hours

- create repository;
- create environment;
- implement data loader;
- confirm data source;
- save a small historical sample.

## Block 2 — 2 hours

- validation;
- feature engineering;
- forward-return target;
- temporal split;
- leakage tests.

## Block 3 — 2 hours

- baseline model;
- gradient boosting;
- predictions;
- metrics.

## Block 4 — 2 hours

- equal-weight portfolio;
- inverse-volatility portfolio;
- backtester;
- transaction costs.

## Block 5 — 1.5 hours

- end-to-end orchestration;
- artefact persistence;
- tests.

## Block 6 — 1.5 hours

- MLflow;
- experiment comparison;
- plots.

## Block 7 — 1 hour

- README;
- implementation log;
- preliminary research observations;
- Git commit.

## Remaining time

Buffer for failures.

**Do not spend the buffer adding RL/news/RNN features unless the entire core pipeline is already green.**

---

# 25. Expected First Experiments

Run in this order:

```text
E0: Momentum baseline
E1: Linear model + rules features + equal weight
E2: Gradient boosting + rules features + equal weight
E3: Gradient boosting + rules features + inverse volatility
E4: Gradient boosting + rules features + MPT
```

Then compare:

- out-of-sample return;
- Sharpe;
- drawdown;
- turnover;
- transaction costs;
- stability across time windows.

---

# 26. Research Interpretation Rule

Never conclude:

> "Model X is better because its accuracy is higher."

Instead ask:

1. Did predictive performance improve?
2. Did portfolio performance improve?
3. Did risk-adjusted performance improve?
4. Did improvement survive transaction costs?
5. Did it survive multiple time periods?
6. Did complexity produce enough economic benefit to justify itself?
7. Could the result plausibly be leakage or overfitting?

The research conclusion should follow the evidence.

---

# 27. Priority Instruction to the Coding Agent

**A boring pipeline that runs correctly is more valuable than an impressive model that cannot be trusted.**

Optimise first for:

1. correctness;
2. reproducibility;
3. leakage prevention;
4. automated evaluation;
5. meaningful baselines;
6. then sophistication.
