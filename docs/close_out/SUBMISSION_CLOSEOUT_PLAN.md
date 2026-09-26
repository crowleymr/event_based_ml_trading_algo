# Submission Close-Out Plan
## ML-Driven Event-Based Systematic Trading Strategy

**Deadline:** 29 September 2026  
**Planning date:** 26 September 2026  
**Purpose:** Freeze feature expansion and close the analytical/evidentiary gaps required for a strong A2 submission.

> IMPORTANT: This document is based on the assignment specification, FSD v2, synthesis decisions, and literature review available to ChatGPT. The latest local repository and five Codex threads were not directly inspectable in this session. Items marked **VERIFY IN REPO** must be checked against the actual implementation before being claimed in the report.

---

## 1. Submission Requirements

The assignment requires one PDF containing:

1. A plain-text URL to a publicly accessible, self-contained Python notebook.
2. A comprehensive project report:
   - problem definition;
   - training/deployment input-output specification;
   - ML approach and model justification;
   - training procedure/hyperparameters;
   - evaluation metrics;
   - results;
   - discussion, limitations and future work.
3. A detailed implementation log:
   - challenges and solutions;
   - AI-tool use and critical review;
   - knowledge gaps;
   - why uncertain components were necessary;
   - how correctness was verified;
   - attempts made to understand them.
4. Presentation materials are optional for A2 but useful for A3.

The implementation must be self-contained, use real non-trivial data, and use a proper validation scheme.

---

# 2. Freeze Principle

From now until submission:

> **No new feature is justified unless it closes a rubric/evidence gap.**

Do NOT implement before submission unless already nearly complete:
- RL;
- NLP/FinBERT;
- LSTM/Transformers;
- live trading;
- ASX;
- intraday trading;
- major cloud infrastructure;
- complex portfolio optimisation;
- extensive hyperparameter search.

A coherent, validated system with critical analysis is more valuable than another unfinished model family.

---

# 3. Current Analytical Foundation — Expected from FSD

The authoritative planned baseline is:

- US large-cap equity universe;
- daily market data;
- SEC EDGAR point-in-time fundamentals;
- EPS and Net Income;
- 5-trading-day forward-return target;
- market-only and market+SEC feature sets;
- Elastic Net linear baseline;
- gradient-boosted-tree challenger;
- chronological train/validation/test;
- purge/embargo around overlapping labels;
- weekly cross-sectional ranking;
- equal-weight and inverse-volatility portfolio engines;
- T+1 execution;
- configurable transaction costs, planned default 10 bps;
- ML metrics: MAE, RMSE, Spearman IC;
- portfolio metrics: return, volatility, Sharpe, drawdown, turnover, costs;
- reproducible run artefacts.

**VERIFY IN REPO:** which of these are actually implemented and which were superseded by later Codex work.

---

# 4. P0 — Submission Blockers

These must be closed before polishing the prose.

## P0.1 Freeze a canonical final experiment

Create one configuration, e.g.:

```text
configs/submission.yaml
```

It must specify:
- universe;
- data history;
- features;
- target horizon;
- split dates;
- purge/embargo;
- model configurations;
- portfolio rules;
- transaction costs;
- random seed.

The run must produce a stable `run_id`.

### Evidence to capture

```text
Git commit:
Run ID:
Command:
Start/end time:
Universe count:
Market date range:
SEC fact count:
Train dates:
Validation dates:
Test dates:
Models:
Transaction cost:
Tests passed:
```

---

## P0.2 Verify point-in-time correctness

Produce explicit evidence that:

1. SEC values are unavailable before their filing date.
2. Future 5-day returns appear only in the target.
3. rolling features do not read future rows;
4. scalers/imputers/model fitting use training data only;
5. test data do not select hyperparameters/models;
6. signal formed using information through T executes at T+1;
7. overlapping labels do not cross train/validation/test boundaries.

This is one of the strongest technical aspects of the project. It should be demonstrated, not merely asserted.

Recommended report artefact:

**Table: Leakage Risk → Control → Test/Evidence**

---

## P0.3 Capture model-learning evidence

For each final model/feature-set combination capture:

- training loss/error;
- validation error;
- test error;
- MAE;
- RMSE;
- Spearman IC;
- hyperparameters;
- feature count;
- observations;
- training time.

If learning curves exist, preserve them.

If they do not, generate a simple sample-size learning curve only if inexpensive.

The purpose is to discuss:
- under/overfitting;
- nonlinear vs linear hypothesis space;
- whether extra SEC information improves generalisation.

---

## P0.4 Capture economic evaluation

For every headline strategy/baseline, use the same test interval and report:

- total return;
- annualised return;
- annualised volatility;
- Sharpe;
- maximum drawdown;
- turnover;
- transaction costs.

Where available also include:
- beta;
- alpha;
- correlation with broad market;
- hit rate.

Do not select the headline model using final-test Sharpe.

---

## P0.5 Establish credible baselines

Minimum:
1. simple momentum strategy;
2. broad US equity benchmark (S&P 500/SPY or implemented equivalent);
3. equal-weight universe if implemented;
4. random/dartboard baseline if implemented.

Useful if already present:
- NASDAQ;
- DJIA;
- risk-free/Treasury context;
- internal market-cap-weighted benchmark.

Do not delay submission to build all optional benchmarks.

---

## P0.6 Public self-contained notebook

The assignment explicitly requires a public cloud notebook URL.

The notebook should:
1. install dependencies;
2. obtain/download required data OR download an archived reproducible dataset/run;
3. execute a representative pipeline;
4. train at least the core models;
5. evaluate;
6. display headline results;
7. explain how the notebook corresponds to the full repository.

A full 500-security/10-year rebuild does not need to run interactively if it is impractical. A self-contained reproducible subset plus archived full-run results is preferable, provided this is transparent.

**VERIFY with teaching staff wording if necessary:** the assignment says the notebook must contain the complete implementation and all data preparation.

---

# 5. P1 — Highest-Value Analytical Improvements

## P1.1 Loss-function vs task-objective analysis

Make this central to the report.

The supervised learner minimises prediction error, but the practical objective is not low MSE/MAE by itself.

The real objective is approximately:

```text
useful cross-sectional ranking
→ investable portfolio
→ return after costs
→ acceptable volatility/drawdown
```

Analyse cases such as:
- Model A has lower RMSE but worse IC.
- Model B has better IC but worse after-cost return because of turnover.
- Equal-weight produces more return but inverse-vol improves drawdown/Sharpe.
- SEC features improve training fit but not test performance.

This directly addresses the strongest part of Criterion C.

---

## P1.2 Feature-set ablation

At minimum compare:

```text
Market only
vs
Market + SEC
```

This answers H2 directly.

If possible report:
- delta MAE/RMSE;
- delta IC;
- delta Sharpe;
- delta drawdown.

A null or negative result is valid.

---

## P1.3 Linear vs nonlinear model analysis

Compare Elastic Net against HistGBT/XGBoost/implemented GBT.

Discuss:
- linear additive hypothesis;
- regularisation;
- nonlinear splits/interactions;
- sensitivity to scaling;
- overfitting risk;
- interpretability;
- observed train-validation-test gap.

Do not simply state that GBT is “more powerful.”

---

## P1.4 Portfolio-engine isolation

Use identical predictions to compare:
- equal weight;
- inverse volatility;
- any additional implemented risk engine.

This isolates allocation from forecasting.

---

## P1.5 Failure/error analysis

Produce at least two of:

- worst prediction periods;
- largest drawdown;
- industries/securities with poor performance;
- high-error observations;
- regime dependence;
- feature missingness/coverage;
- turnover/cost spikes.

Tie future work to observed failure modes.

---

# 6. P2 — Defer Unless Already Complete

- expansion from 100 to 500 if it destabilises ingestion;
- Power BI;
- FastAPI;
- new deep-learning model;
- RL gym;
- NLP;
- Australian equities;
- live broker integration;
- sophisticated statistical tests such as PBO/CPCV/DSR.

If already implemented and stable, report them. Otherwise put them in Future Work.

---

# 7. Rubric Maximisation Map

## Criterion A — Task Definition

To target Excellent:
- define training input tensor/table precisely;
- define target mathematically;
- define deployment input/output;
- explain weekly ranking and portfolio output;
- explain why ML is plausible;
- explain why traditional rules alone may be insufficient;
- give practical context;
- state research question and hypotheses.

Avoid vague “predict the stock market” language.

## Criterion B — Model/System Understanding

To target Excellent:
- include end-to-end architecture;
- map each theoretical component to code;
- explain Elastic Net objective;
- explain GBT sequential residual/error correction at a conceptual/technical level;
- explain regularisation and key hyperparameters;
- explain PIT join;
- explain temporal split/purge;
- explain why regression/ranking;
- explain portfolio construction;
- discuss implementation challenges and alternatives.

Prepare to identify the exact functions/classes implementing each item during Q&A.

## Criterion C — Evaluation

To target Excellent:
- separate training loss from real objective;
- use train/validation/test;
- show multiple metrics;
- include IC and economic metrics;
- include costs;
- compare baselines;
- analyse failure cases;
- discuss where predictive loss and portfolio performance disagree;
- make future work emerge from observed weaknesses.

Do not claim profitability proves predictive validity.

---

# 8. Required Final Tables/Figures

Aim for a compact evidence set.

## Tables

1. Dataset and feature specification.
2. Train/validation/test periods and counts.
3. Model configuration/hyperparameters.
4. Predictive metrics by feature set/model.
5. Portfolio metrics by model/risk engine/baseline.
6. Leakage controls.
7. Key limitations and consequences.

## Figures

1. System workflow/architecture.
2. Equity curves.
3. Drawdown curves.
4. Learning curves or train-vs-validation diagnostic.
5. Model/experiment comparison.
6. Feature importance or coefficient magnitude.
7. Optional: IC over time or rolling strategy performance.

---

# 9. Claims That Require Repository Evidence

Do not make these statements until verified:

- “500 equities were used.”
- “XGBoost used GPU acceleration.”
- “SEC features improved performance.”
- “GBT beat Elastic Net.”
- “inverse volatility reduced drawdown.”
- “the strategy beat the S&P 500.”
- “transaction costs were 10 bps in all experiments.”
- “all leakage tests passed.”
- “GitHub Actions reproduces the full run.”
- “the Streamlit dashboard supports X.”
- any numerical performance claim.

---

# 10. Five-Day Close-Out Schedule

## Day 1 — Evidence freeze

- run tests;
- run canonical submission experiment;
- freeze run ID and Git commit;
- export all metrics/plots/tables;
- verify PIT/leakage controls;
- record exact commands;
- stop changing model design.

## Day 2 — Analytical audit

- compare models/features;
- compare portfolio engines;
- compare baselines;
- investigate largest failure modes;
- write conclusions supported by results;
- finalise figures/tables.

## Day 3 — Submission notebook + report rewrite

- make public notebook self-contained;
- verify from clean runtime;
- deeply rewrite Project Report draft in your own words;
- check every numerical claim against run artefacts;
- replace placeholders.

## Day 4 — Implementation log + Q&A mastery

- rewrite implementation log in your own words;
- review Git/Codex history;
- document genuine challenges and decisions;
- study theory-to-code mapping;
- prepare answers to likely questions.

## Day 5 — Final assembly

- verify public URL;
- assemble single PDF;
- proofread;
- check citations;
- check figures;
- check plain-text notebook URL;
- run notebook one final time;
- archive exact repo/run used for submission.

---

# 11. Stop Rule

At T-48 hours:

> No new model, data source, universe, or architecture change unless the existing submission is technically invalid.

Use the remaining time to improve understanding, evidence and communication.
