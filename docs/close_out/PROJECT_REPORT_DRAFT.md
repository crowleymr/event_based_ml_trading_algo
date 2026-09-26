# Project Report — First Draft
## ML-Driven Event-Based Systematic Trading Strategy

> **DRAFT FOR CRITICAL HUMAN REWRITE.** Text in `[VERIFY: ...]` must be replaced using final repository/run evidence. Do not submit numerical or implementation claims that have not been checked against the canonical final run.

---

## 1. Problem Definition

### 1.1 Practical problem

Equity investors must choose among many securities using information that arrives at different frequencies. Market prices and volumes change daily, while company fundamentals become public through periodic filings. A practical systematic trading system therefore needs to transform heterogeneous, time-dependent information into investment decisions without using information that was unavailable at the decision time.

This project investigates a deliberately constrained version of that problem: selecting US equities over a short holding horizon using daily market information and point-in-time public SEC fundamental information.

The project does **not** attempt to prove that machine learning can consistently beat financial markets. Instead, it asks whether a small set of supervised ML models can extract useful cross-sectional ranking information and whether predictive improvements survive the transition from prediction to an investable portfolio after basic trading frictions.

### 1.2 Research question

> Can machine-learning models using market features and point-in-time SEC fundamentals improve short-horizon cross-sectional equity ranking relative to simple baselines, and does any predictive improvement translate into better out-of-sample risk-adjusted portfolio performance after transaction costs?

A secondary question is:

> When the predictions are held fixed, does risk-aware portfolio weighting improve risk-adjusted performance relative to equal weighting?

### 1.3 Hypotheses

H1: A nonlinear gradient-boosted-tree model can capture interactions missed by a regularised linear model and therefore may improve out-of-sample ranking.

H2: Point-in-time EPS and Net Income features may add information beyond market-derived features.

H3: Inverse-volatility weighting may reduce portfolio volatility or drawdown relative to equal weighting, although it may sacrifice raw return.

These are empirical hypotheses rather than assumptions. A null result is considered informative.

---

## 2. Technical Task Definition

### 2.1 Training input

Each training observation represents security `i` at trading session `t`.

The planned feature vector contains market-derived variables such as recent returns, rolling volatility, volume/liquidity measures, moving-average distance and distance from the 52-week high. A second feature set adds the latest SEC-reported EPS and Net Income that were publicly available before the trading session.

[VERIFY: insert exact final feature list from the feature manifest.]

Fundamental values are joined using a point-in-time rule. A filing with SEC filing date `T` is not available to a model decision on or before `T`; the conservative daily-data rule is to allow it only on a later trading session.

### 2.2 Training output / target

The supervised target is the five-trading-day forward adjusted return:

\[
y_{i,t} =
\frac{P_{i,t+5}}{P_{i,t}} - 1
\]

where \(P\) is the adjusted closing price.

This is used only as the learning label and is never included in the feature set.

### 2.3 Deployment input

At a weekly decision point, the deployed model receives only features available through the current session for eligible securities.

### 2.4 Deployment output

The model outputs an estimated five-day return for each eligible security. Securities are ranked cross-sectionally by this score. The top `K` securities are passed to a portfolio-construction rule.

[VERIFY: final K and rebalance rule.]

Execution is delayed until the next trading session to avoid assuming that a model using closing information can trade at the same closing price.

---

## 3. System Architecture

The system is designed as a modular research pipeline:

```text
PUBLIC DATA
  ├── Market OHLCV
  └── SEC filings / fundamentals
          ↓
Raw immutable/cache layer
          ↓
Identifier normalisation
          ↓
Point-in-time curated data
          ↓
Feature engineering
  ├── Market-only
  └── Market + SEC
          ↓
Supervised prediction
  ├── Elastic Net
  └── Gradient-boosted trees
          ↓
Cross-sectional ranking
          ↓
Portfolio / risk engine
  ├── Equal weight
  └── Inverse volatility
          ↓
T+1 backtest + transaction costs
          ↓
Predictive + financial evaluation
          ↓
Run artefacts / reports / dashboard
```

[VERIFY: adapt diagram to the final repository architecture.]

The design deliberately separates prediction from portfolio allocation. This allows the same predictions to be evaluated under different risk engines and avoids attributing an allocation improvement to the forecasting model.

---

## 4. Data

### 4.1 Market data

Daily OHLCV/adjusted market data are obtained from [VERIFY: final source]. Raw data are cached so experiments can be reproduced without repeatedly downloading the same history.

[VERIFY: number of securities, start/end dates, observation count, missingness.]

### 4.2 Fundamentals and events

SEC EDGAR is used as the point-in-time source for structured company facts. The baseline design limits fundamentals to:

- `EarningsPerShareBasic`;
- `NetIncomeLoss`.

The important timestamp is the filing/public availability date rather than the fiscal period end. Using fiscal period end as though the value were known at that time would introduce look-ahead bias.

[VERIFY: final SEC fact counts and coverage.]

### 4.3 Universe and identifiers

A security master maps market tickers to SEC CIK identifiers.

[VERIFY: universe construction, size, mapping success rate, industry/exchange metadata.]

### 4.4 Known data limitations

The planned universe is based on currently available large US equities rather than a historically reconstructed constituent universe. This creates survivorship/selection bias. The POC therefore should not be interpreted as an unbiased historical estimate of a deployable strategy.

Other limitations include [VERIFY against repo/data]:
- missing SEC facts;
- changes in ticker/company identifiers;
- daily rather than intraday filing availability;
- unofficial Yahoo interface if used;
- limited corporate-event representation.

---

## 5. Feature Engineering

### 5.1 Market feature set

[VERIFY exact implemented features.]

The planned features include returns over several lookback windows, rolling volatility, dollar volume, volume ratios, moving-average distance and distance from recent highs.

These features are deterministic and interpretable. They represent momentum/reversal, risk and liquidity information without requiring a learned sequence model.

### 5.2 Market + SEC feature set

The second feature family augments market features with point-in-time SEC information.

This comparison is an ablation experiment: if the model using SEC data improves out-of-sample behaviour, the improvement can be associated with the additional information set rather than simply a different model.

### 5.3 Leakage controls

The project treats information availability as part of feature definition.

[VERIFY tests and implementation.]

Key controls should include:
- backward/as-of joins for SEC data;
- no backward filling of future filings;
- rolling features based only on historical observations;
- target excluded from feature construction;
- training-only fitting of preprocessing;
- T+1 execution.

---

## 6. Machine Learning Approach

### 6.1 Why regression and ranking?

A binary up/down classification discards return magnitude and requires an arbitrary threshold. This project instead predicts a continuous forward return and evaluates whether the relative ordering of predictions is useful for stock selection.

The prediction itself is therefore an intermediate representation. The deployment decision is based primarily on ranking.

### 6.2 Elastic Net

Elastic Net provides a regularised linear baseline. Conceptually it estimates coefficients by balancing squared prediction error with L1 and L2 penalties:

\[
\min_{\beta}
\frac{1}{2n}\|y-X\beta\|_2^2
+
\lambda\left[
\rho\|\beta\|_1
+
\frac{1-\rho}{2}\|\beta\|_2^2
\right]
\]

The L1 component can shrink some coefficients to zero, while L2 stabilises correlated predictors. This is useful as a low-complexity baseline for noisy financial features.

[VERIFY exact sklearn implementation, alpha/l1_ratio/grid and preprocessing.]

### 6.3 Gradient-boosted trees

The nonlinear challenger is [VERIFY: HistGradientBoosting/XGBoost/final model].

Gradient boosting builds an additive sequence of trees, where each new learner is fitted to improve the current ensemble with respect to the training objective. Unlike the linear model, tree splits can represent nonlinear thresholds and feature interactions.

The model is appropriate for tabular financial data but has greater capacity to overfit. The project therefore evaluates train/validation/test behaviour rather than assuming additional complexity is beneficial.

[VERIFY final hyperparameters and implementation.]

### 6.4 Training procedure

The data are split chronologically rather than randomly.

Planned design:
- 60% training;
- 20% validation;
- 20% test;
- purge/embargo around boundaries because five-day forward-return labels overlap.

[VERIFY exact final dates/counts.]

Hyperparameters are selected using validation data only. The test set is reserved for final evaluation.

---

## 7. Loss Function vs Practical Objective

This distinction is central to the project.

The supervised models optimise prediction error. However, the practical objective is not simply to minimise the numerical difference between predicted and realised returns.

The investment objective depends on:

- correct relative ranking;
- concentration;
- volatility;
- drawdown;
- turnover;
- transaction costs;
- market exposure.

A model can therefore achieve lower RMSE but still create a worse portfolio. For example, small prediction errors on low-ranked securities may reduce RMSE while having no effect on top-K selection. Conversely, a model with somewhat larger absolute errors can be useful if it consistently ranks the strongest securities above weaker ones.

For this reason the project evaluates three layers:

1. **Prediction:** MAE/RMSE.
2. **Ranking:** cross-sectional Spearman information coefficient.
3. **Economic outcome:** after-cost return, Sharpe, volatility, drawdown and turnover.

[VERIFY: insert one concrete example from final results where metrics disagree. This is a high-value discussion point.]

---

## 8. Portfolio Construction

### 8.1 Equal-weight top-K

At each rebalance, the highest-ranked `K` securities receive equal weights.

This is transparent and prevents the portfolio optimiser from obscuring whether the forecasting model contains useful information.

### 8.2 Inverse-volatility top-K

The second risk engine applies weights approximately proportional to inverse trailing volatility:

\[
w_i =
\frac{1/\sigma_i}
{\sum_{j \in K}1/\sigma_j}
\]

This uses the same selected securities/predictions as the equal-weight portfolio and therefore isolates the effect of risk allocation.

[VERIFY final volatility lookback and any caps.]

---

## 9. Backtesting

The backtester follows a conservative information/execution sequence:

```text
information available through T
→ feature vector
→ prediction/rank
→ target portfolio
→ execution from T+1
```

[VERIFY exact execution price convention.]

A transaction-cost assumption of [VERIFY] basis points per traded notional is applied.

The backtest records positions, weights, turnover, transaction costs, daily returns and equity value.

### 9.1 Baselines

[VERIFY implemented baselines.]

Potential final comparison set:
- simple momentum;
- S&P 500 / SPY;
- equal-weight universe;
- random/dartboard portfolios;
- other implemented indices.

The purpose of baselines is not to prove market outperformance but to give context to the ML result.

---

## 10. Evaluation Metrics

### Predictive metrics

**MAE:** average absolute prediction error.

**RMSE:** penalises larger prediction errors more strongly.

**Spearman IC:** measures whether predicted and realised returns have similar cross-sectional rank ordering.

### Financial metrics

- total return;
- annualised return;
- annualised volatility;
- Sharpe ratio;
- maximum drawdown;
- turnover;
- transaction costs.

[VERIFY definitions used by code, including annualisation factor and risk-free rate.]

---

## 11. Results

> Replace this section from the canonical final run. Do not select models using final-test performance.

### 11.1 Data summary

| Item | Final value |
|---|---:|
| Securities | [VERIFY] |
| Market observations | [VERIFY] |
| Date range | [VERIFY] |
| SEC EPS observations | [VERIFY] |
| SEC Net Income observations | [VERIFY] |
| Train observations | [VERIFY] |
| Validation observations | [VERIFY] |
| Test observations | [VERIFY] |

### 11.2 Predictive performance

| Feature set | Model | Train RMSE | Validation RMSE | Test RMSE | Test MAE | Test IC |
|---|---|---:|---:|---:|---:|---:|
| Market | Elastic Net | [ ] | [ ] | [ ] | [ ] | [ ] |
| Market | GBT | [ ] | [ ] | [ ] | [ ] | [ ] |
| Market + SEC | Elastic Net | [ ] | [ ] | [ ] | [ ] | [ ] |
| Market + SEC | GBT | [ ] | [ ] | [ ] | [ ] | [ ] |

### 11.3 Portfolio performance

| Strategy | Total return | Ann. return | Ann. vol | Sharpe | Max DD | Turnover | Costs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Momentum | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Broad market | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Selected ML + Equal Weight | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Same ML + Inverse Vol | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

### 11.4 Figures

Insert:
1. equity curves;
2. drawdowns;
3. train/validation diagnostic or learning curve;
4. experiment comparison;
5. feature importance/coefficient chart.

---

## 12. Discussion

### 12.1 Research question

[WRITE AFTER FINAL RUN.]

A useful structure:

- Did nonlinear modelling improve predictive generalisation?
- Did SEC features add incremental information?
- Did better predictive metrics correspond to better portfolio outcomes?
- Did risk weighting improve the risk-return profile?
- How did ML compare with simple baselines?
- Which result was surprising?

Avoid interpreting one historical backtest as evidence of persistent alpha.

### 12.2 Expected vs actual model behaviour

[INSERT specific examples.]

Possible interpretations:
- nonlinear model fits training data substantially better but gains little out of sample → capacity/noise/overfitting;
- SEC features do not improve ranking → sparse/low-frequency fundamentals may add little at a five-day horizon;
- IC improves but Sharpe does not → ranking signal is consumed by turnover/cost/risk concentration;
- inverse-vol improves drawdown but reduces return → explicit risk-return trade-off.

Only use interpretations supported by final evidence.

---

## 13. Limitations

1. **Survivorship/selection bias.** A present-day universe does not reconstruct the historical investable universe.
2. **Data-source limitations.** [VERIFY Yahoo/SEC implementation.]
3. **Daily timestamp granularity.** Filing information may be treated conservatively because exact intraday availability is not always represented.
4. **Simplified execution.** Fixed transaction costs do not model bid-ask spread, slippage, market impact or partial fills.
5. **Limited fundamentals.** EPS and Net Income represent only a small subset of company information.
6. **Short research horizon.** The project compares a deliberately small model family and limited hyperparameter search.
7. **Non-stationarity.** Financial relationships can change across market regimes.
8. **Statistical uncertainty.** A single historical test period is not proof of future performance.

[VERIFY which limitations apply.]

---

## 14. Future Work

Future work should be driven by observed failure modes rather than model novelty.

Potential extensions:
- richer SEC/event features and filing text;
- FinBERT/event embeddings;
- learned sequence features;
- stronger cross-validation/backtest-overfitting analysis;
- additional cost/slippage/liquidity modelling;
- MPT/risk-parity portfolio engines;
- an RL environment for dynamic allocation;
- longer/shorter horizons;
- other asset classes;
- prospective simulated-live and paper-trading evaluation.

---

## 15. Conclusion

This project frames short-horizon equity selection as a layered ML decision problem rather than only a return-prediction exercise. The architecture deliberately separates point-in-time information, supervised prediction, ranking, portfolio construction and economic evaluation.

[REWRITE after results.]

The most important conclusion should not be “the model made money.” It should state what the experiments show about:
- model complexity;
- information content;
- generalisation;
- prediction-vs-economic objective mismatch;
- risk allocation;
- limitations of historical evaluation.

---

## References

Build the final reference list from the sources you personally verify.

Priority literature from the research review includes:
- Gu, Kelly & Xiu on ML in empirical asset pricing;
- Jegadeesh & Titman on momentum;
- López de Prado on purged/embargoed validation;
- Bailey et al. on backtest overfitting;
- relevant SEC/Yahoo/data documentation;
- references specifically supporting the implemented models/metrics.

Do not copy unverified AI-generated citations into the final submission.
