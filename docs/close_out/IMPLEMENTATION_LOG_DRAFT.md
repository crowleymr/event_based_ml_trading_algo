# Implementation Log — First Draft / Reconstruction Template
## ML-Driven Event-Based Systematic Trading Strategy

> **IMPORTANT:** The assignment asks for a detailed development log and critical disclosure of AI use. This draft must be reconciled against Git history, the five Codex threads, actual run logs, and your own recollection. Do not present reconstructed entries as contemporaneous notes if they were reconstructed after the fact.

---

## 1. Development Approach

I used AI coding/research assistants extensively to accelerate implementation, documentation and literature/data-source research. I retained responsibility for:
- choosing the problem;
- defining scope;
- approving research questions;
- deciding the data and validation principles;
- reviewing architecture;
- selecting which proposed features were in/out of scope;
- checking experimental claims;
- understanding the final implementation sufficiently to explain it.

AI assistance included ChatGPT for project planning/research synthesis and Codex for repository implementation/debugging. Perplexity was used for data-source and academic literature research.

[VERIFY exact tools and subscriptions actually used.]

---

## 2. Project Evolution

### Stage 1 — Problem selection

The initial project exploration considered several domains. Systematic trading was selected because it offered:
- accessible public data;
- a self-contained experiment;
- clear ML and financial evaluation layers;
- an opportunity to compare predictive and decision objectives;
- a feasible path to a demonstrable system within the assignment timeframe.

The scope was deliberately narrowed to daily US equities rather than intraday/Australian data.

### Stage 2 — Data-source research

A research task compared market, fundamental and corporate-event sources.

The resulting baseline architecture selected:
- [VERIFY] for daily market data;
- SEC EDGAR for point-in-time fundamentals;
- SEC filing metadata as the initial corporate-event representation.

A key lesson was that historical fundamentals must be joined using the date they became public, not the fiscal period they describe.

### Stage 3 — Baseline design

The baseline research question was reduced to:
- market-only vs market+SEC features;
- regularised linear vs nonlinear tree models;
- equal-weight vs inverse-volatility portfolio construction.

This reduction followed YAGNI: deep learning, NLP, RL and Australian equities were deferred until the baseline was reliable.

### Stage 4 — Codex implementation

[RECONSTRUCT FROM FIVE CODEX THREADS.]

For each thread record:
- thread identifier/date;
- objective;
- files/modules changed;
- tests executed;
- failures;
- design decisions;
- human review/approval.

---

## 3. Chronological Log

### Entry template

**Date/time:**  
**Goal:**  
**AI/tool used:**  
**What I asked it to do:**  
**What it produced/changed:**  
**Problem/challenge:**  
**How correctness was checked:**  
**What I understood/learned:**  
**What remained uncertain:**  
**Decision:**  

---

### Entry A — Functional specification and scope

**Goal:** Define a feasible POC rather than implement every proposed trading component.

**AI/tool used:** ChatGPT + Perplexity research.

**Key decision:** Limit the first slice to US daily equities, a five-day target, two SEC fundamentals, two supervised model families and two portfolio approaches.

**Critical review:** The initial architecture contained possible NLP, RNN, MPT and RL components. These were judged unnecessary for demonstrating the core research question and were deferred.

**Learning:** Model sophistication is not equivalent to experimental quality. Point-in-time correctness and validation are more important in a financial backtest.

---

### Entry B — Data architecture

**Challenge:** Market data are naturally keyed by ticker/date, while SEC fundamentals use CIK and refer to accounting periods whose values are only known later.

**Solution:** Build an identifier mapping/security master and preserve SEC filing availability dates.

**Verification:** [VERIFY exact tests/functions.]

**Knowledge gap:** SEC Company Facts can contain repeated facts, multiple units/forms and restatements. The POC deliberately avoids building a complete XBRL accounting engine.

**Why acceptable:** Only two facts are required to demonstrate PIT feature integration.

**Attempt to understand:** [Add documentation/articles/code review you personally completed.]

---

### Entry C — Point-in-time feature joining

**Challenge:** A naïve join on fiscal period end leaks information.

**Solution:** Use a backward/as-of availability join based on filing date and permit the fact only on later sessions.

**Verification:** [INSERT test names and example.]

**Critical point:** This is a stronger correctness requirement than simply obtaining high predictive accuracy.

---

### Entry D — Target and validation

**Challenge:** Five-day forward-return labels overlap, so adjacent observations are not independent.

**Solution:** Chronological train/validation/test and [VERIFY implemented purge/embargo].

**Verification:** [test names / split manifest.]

**Knowledge gap:** Full combinatorial purged cross-validation/backtest-overfitting analysis was outside project scope.

**Reason:** The assignment requires proper validation, but a journal-grade finance-validation framework would have displaced effort from the core ML system.

---

### Entry E — Elastic Net

**Purpose:** Provide an interpretable, regularised linear baseline.

**Theory reviewed:** squared-error regression plus L1/L2 regularisation.

**Implementation:** [VERIFY module/class.]

**Hyperparameters:** [VERIFY.]

**Verification:** train/validation/test metrics, coefficients, deterministic seed where applicable.

**What I need to be able to explain:** effect of alpha, L1/L2 mixture, scaling, why regularisation is useful with correlated/noisy features.

---

### Entry F — Gradient-boosted model

**Purpose:** Test whether nonlinear feature interactions improve generalisation.

**Implementation:** [VERIFY HistGBT/XGBoost and module.]

**Hyperparameters:** [VERIFY.]

**GPU:** The development machine has an RTX 4060 Ti 16 GB, but GPU acceleration was not made a baseline dependency. [VERIFY whether final XGBoost uses GPU.]

**What I need to explain:** sequential boosting, weak learners, learning rate, tree depth/leaves, regularisation/early stopping where used, and overfitting trade-offs.

---

### Entry G — Portfolio/backtest

**Challenge:** Good predictions are not identical to good investment decisions.

**Solution:** Rank predicted returns, select top K, then apply separate portfolio weighting.

**Execution:** information through T → trade T+1.

**Costs:** [VERIFY.]

**Verification:** [test names.]

**Learning:** RMSE can improve while portfolio performance worsens because ranking, turnover and concentration matter.

---

### Entry H — MLOps/reproducibility

**Goal:** Make training/backtesting repeatable and preserve report evidence.

**Implementation:** [VERIFY CLI, config, run directories, manifests, MLflow/ClearML if used, GitHub Actions.]

**Verification:** [exact one-command run and tests.]

**Limitation:** Cloud/GPU CI is unnecessary if local execution and public notebook are reproducible.

---

### Entry I — Reporting/UI

[VERIFY actual implementation.]

Record:
- run logs;
- Parquet/CSV/XLSX outputs;
- Streamlit;
- learning curves;
- model comparisons;
- security/industry/universe views.

Explain which reporting features materially supported analysis versus cosmetic presentation.

---

## 4. AI Use — Critical Review

### Appropriate uses

AI was particularly effective for:
- boilerplate/module implementation;
- test generation;
- repetitive data transformations;
- documentation scaffolding;
- identifying candidate APIs/literature;
- debugging library/API issues;
- producing first-draft prose for subsequent human review.

### Risks

AI assistance introduced risks:
- plausible but unsupported claims;
- scope expansion;
- incorrect assumptions about financial timing;
- code that passes syntax checks but contains leakage;
- over-complex architecture;
- citations requiring independent verification.

### Controls

The project used or should document:
- FSD as authoritative scope;
- explicit human-approval boundaries;
- automated tests;
- run artefacts;
- PIT/leakage tests;
- literature/source checking;
- final manual review of claims;
- final report rewritten by the student.

---

## 5. Knowledge Gaps and Verification

Use this table candidly.

| Topic | Why needed | Initial uncertainty | Verification | Current understanding |
|---|---|---|---|---|
| SEC Company Facts | PIT fundamentals | XBRL duplicates/restatements | [VERIFY] | [YOUR WORDS] |
| Purge/embargo | overlapping labels | exact boundary logic | [VERIFY] | [YOUR WORDS] |
| Elastic Net | linear baseline | regularisation effects | [VERIFY] | [YOUR WORDS] |
| Gradient boosting | nonlinear model | boosting/hyperparameters | [VERIFY] | [YOUR WORDS] |
| IC/Spearman | ranking evaluation | relation to RMSE | [VERIFY] | [YOUR WORDS] |
| Backtest costs | economic realism | simplified friction model | [VERIFY] | [YOUR WORDS] |
| Inverse volatility | risk allocation | assumptions/limitations | [VERIFY] | [YOUR WORDS] |

---

## 6. Main Challenges and Solutions

Complete from evidence.

Recommended candidates:
1. ticker↔CIK mapping;
2. SEC fact normalisation;
3. PIT joins;
4. overlapping labels;
5. dependency/library compatibility;
6. market-data throttling;
7. experiment/run reproducibility;
8. learning-curve/model diagnostic integration;
9. reporting/dashboard consistency.

For each, include the actual error/failure and the test/evidence that demonstrated resolution.

---

## 7. Reflection

Write this section yourself.

Suggested prompts:

- Which AI recommendation did I reject, and why?
- Which bug or methodological risk changed my understanding most?
- Where did I initially confuse predictive accuracy with trading usefulness?
- What part of the final code could I explain from memory?
- What part still requires reference to documentation?
- If I had one additional week, what would I change based on observed results rather than novelty?
- Which result contradicted my initial hypothesis?
- What would make me distrust the backtest?

---

## 8. Final Verification Checklist

Before submission:

```text
[ ] Every AI-generated factual claim checked
[ ] Every result copied from canonical run
[ ] Every citation independently verified
[ ] Notebook runs in clean cloud environment
[ ] Git commit recorded
[ ] Tests recorded
[ ] Knowledge gaps stated honestly
[ ] Can explain each core model without reading code
[ ] Can locate theory-to-code implementation quickly
[ ] Can explain why test set was not used for selection
[ ] Can explain loss vs task objective
[ ] Can explain PIT join and T+1 execution
```
