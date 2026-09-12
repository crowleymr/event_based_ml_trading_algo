# Experiment Plan
## ML-Driven Systematic Trading Strategy — Stage-Gate POC

## 1. Objective

Determine whether an ML-driven stock-selection process produces better out-of-sample risk-adjusted portfolio performance than simple systematic baselines.

---

## 2. Core Hypotheses

### H0

ML-based stock selection does not materially improve risk-adjusted out-of-sample performance relative to a simple systematic baseline after transaction costs.

### H1

At least one ML-based approach produces a meaningful improvement in out-of-sample risk-adjusted performance after transaction costs.

### Secondary hypothesis

Risk-aware portfolio construction can improve the risk-adjusted performance of identical ML predictions.

---

## 3. Baseline Ladder

Build complexity gradually.

### B0 — Buy and Hold

Market benchmark.

### B1 — Momentum

Rules-based ranking + top-K equal weight.

### B2 — Linear ML

Rules features + linear/logistic model + top-K equal weight.

### B3 — Gradient Boosting

Rules features + gradient boosting + top-K equal weight.

### B4 — Gradient Boosting + Risk

Rules features + gradient boosting + inverse-volatility portfolio.

### B5 — Gradient Boosting + MPT

Rules features + gradient boosting + MPT.

---

## 4. Evaluation

For every experiment record:

- RMSE/MAE where applicable;
- directional accuracy;
- total return;
- annualised return;
- annualised volatility;
- Sharpe;
- maximum drawdown;
- turnover;
- transaction costs.

---

## 5. Robustness

If time permits, evaluate:

- multiple holding horizons;
- multiple top-K values;
- different cost assumptions;
- multiple test sub-periods;
- feature ablations;
- model stability across seeds.

Do not tune all of these against the final test set.

---

## 6. Feature Ablation

Useful comparisons:

```text
Price only
Price + momentum
Price + volatility
Price + volume
All market features
```

This creates a useful research discussion even without news/RNNs.

---

## 7. Future Enhancements

After the stage gate:

### News

Compare:

- pretrained sentiment;
- event classification;
- LLM/agentic evaluation.

### Market sequences

Compare:

- handcrafted features;
- RNN representation.

### Fundamentals

Compare:

- structured fundamentals;
- annual-report extraction.

### Portfolio

Compare:

- MPT;
- RL.

The research value of these extensions should be judged by whether they improve out-of-sample economic performance, not by model complexity alone.
