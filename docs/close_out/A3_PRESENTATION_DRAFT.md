# A3 Presentation — First Draft Outline
## Five-minute presentation + Q&A preparation

> Presentation materials are optional in A2, but the assignment's A3 component expects a 5-minute presentation and approximately 10 minutes of questions.

---

## Slide 1 — Problem and research question (~40 sec)

**Title:** Can better return prediction produce a better trading decision?

Explain:
- ~[VERIFY] US equities;
- daily public market + SEC data;
- predict 5-day forward returns;
- rank stocks weekly;
- evaluate after costs.

Research question:
> Can ML using market and point-in-time SEC information improve short-horizon equity ranking, and does predictive improvement translate into better risk-adjusted portfolio performance?

---

## Slide 2 — System / architecture (~55 sec)

Show:

```text
Market + SEC
→ PIT data
→ features
→ Elastic Net / GBT
→ predicted 5-day return
→ cross-sectional rank
→ equal weight / inverse vol
→ T+1 backtest
→ ML + economic metrics
```

Emphasise:
- SEC filing date, not fiscal period end;
- T+1 execution;
- prediction and allocation are separate.

---

## Slide 3 — ML theory-to-code (~70 sec)

Explain two models.

### Elastic Net
- linear hypothesis;
- squared-error objective;
- L1/L2 regularisation;
- low-complexity baseline.

### GBT
- sequential tree ensemble;
- nonlinear thresholds/interactions;
- higher capacity;
- greater overfit risk.

Point to exact repository modules/functions during Q&A.

Mention chronological split and overlapping-label purge.

---

## Slide 4 — Results and central insight (~75 sec)

Insert one compact table:

| Model/features | RMSE | IC | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| [final experiments] | | | | |

Show equity curve or model comparison.

Central analytical message:

> Prediction loss and investment value are different objectives.

Use one real example from results where predictive and portfolio metrics differ.

---

## Slide 5 — Critical evaluation (~60 sec)

Three columns:

**What worked**
- [VERIFY]

**What failed/surprised**
- [VERIFY]

**Limitations**
- survivorship bias;
- simplified transaction costs;
- limited SEC facts;
- non-stationarity;
- daily timestamp granularity.

End with one evidence-driven future experiment.

---

# Likely Q&A

## Why not classification?
Regression preserves magnitude and supports ranking; up/down classification discards magnitude and requires an arbitrary threshold.

## Why Elastic Net?
It is a regularised linear baseline. It establishes whether nonlinear model complexity is actually useful.

## Why GBT?
Tabular features may interact nonlinearly; boosting can model those interactions while remaining practical on the dataset size.

## What is the training loss?
[VERIFY exact objective.] Explain formula and implementation.

## Why isn't RMSE enough?
The deployment task selects/ranks securities. Small errors on irrelevant securities can dominate RMSE while not changing the portfolio. IC and portfolio metrics therefore complement prediction error.

## Why Spearman IC?
It evaluates cross-sectional ordering rather than exact return magnitude, aligning more closely with top-K selection.

## How did you avoid look-ahead bias?
Explain:
- SEC filing availability;
- historical rolling features;
- chronological splits;
- purge/embargo;
- train-only preprocessing;
- T+1 execution.

## Why purge?
A five-day label at time T includes returns through T+5. Observations around a split can otherwise share future return intervals.

## Why not random k-fold?
Time order and overlapping labels make random row splits inappropriate.

## Why can a model with lower RMSE have worse Sharpe?
Portfolio outcomes depend on rank, concentration, turnover, costs and covariance, not only prediction magnitude error.

## Why inverse volatility?
It reduces allocation to historically volatile selected securities and tests whether simple risk-aware weighting improves the same predictions.

## What would invalidate the result?
Leakage, test-set tuning, inaccurate timestamps, survivorship bias, unrealistic execution assumptions, or a result that disappears under modest costs/alternative periods.

## Did AI build this?
AI materially accelerated implementation. Explain the decisions you personally made, which AI suggestions you rejected, the tests used to verify code, and one component you studied in depth.

---

# Two-minute “question bait” topic

Recommended:

> **Why a lower ML loss can produce a worse trading strategy.**

This lets you demonstrate:
- learning objective;
- ranking;
- portfolio construction;
- transaction costs;
- model evaluation;
- critical thinking.

It connects directly to the assignment rubric.
