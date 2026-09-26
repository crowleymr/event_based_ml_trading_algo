# Glossary

This document defines both repository terminology and the financial/ML concepts needed
to interpret the experiments. Metrics must be considered together: no single value is
sufficient evidence that a trading strategy is useful or robust.

## Core evaluation concepts

### Return

Return measures the proportional change in value. A daily return of `0.01` means 1%.
Total return measures the complete backtest; annualised return expresses the compounded
result on a 252-trading-session basis. Return is the practical objective, but a higher
return is not automatically preferable if it requires substantially greater risk,
turnover, concentration or cost.

### Volatility

Annualised volatility is the sample standard deviation of daily returns multiplied by
the square root of 252. It measures variability, not simply losses. It is relevant
because two strategies with similar returns may expose the operator to very different
uncertainty. It does not fully describe tail risk or the size and duration of drawdowns.

### Sharpe ratio

The Sharpe ratio compares excess return with volatility:

```text
Sharpe = annualised mean excess daily return / annualised volatility
```

This project currently uses a zero risk-free-rate convention unless an artefact states
otherwise. Sharpe is relevant because it places return and variability on a common
risk-adjusted scale. It should not be treated as proof of skill: it is sensitive to the
sample period, non-normal returns, serial dependence, costs and multiple testing.

### Maximum drawdown

Maximum drawdown is the largest peak-to-trough percentage decline in the equity curve.
It describes the worst experienced capital loss from a previous high and is often more
intuitive than volatility. It does not say how long recovery took or predict the worst
possible future loss.

### Information coefficient (IC)

Daily IC is the Spearman rank correlation between predicted and realised five-session
returns across securities on one date. Mean IC asks whether a model consistently orders
stocks correctly, which is closer to the top-K portfolio task than exact return
prediction. A useful IC can coexist with mediocre RMSE, and a positive IC need not
produce an attractive after-cost portfolio.

### MAE and RMSE

Mean absolute error averages absolute prediction errors. Root mean squared error gives
more weight to large errors. Both use decimal-return units and measure forecast accuracy,
not portfolio usefulness. They must be interpreted with IC, turnover, costs and realised
portfolio metrics.

### Turnover and transaction cost

Turnover is absolute traded value divided by pre-trade equity. Higher turnover typically
causes higher implementation costs and can erase a small forecasting advantage. The
backtest deducts the configured one-way transaction cost through a self-financing
rebalance calculation; costs are not subtracted later as an informal adjustment.

### Beta, tracking error and information ratio

Beta measures the strategy's co-movement with SPY relative to SPY variance. Tracking
error is the annualised volatility of strategy return minus benchmark return. The
information ratio divides annualised active return by tracking error. These measures
help distinguish market exposure from benchmark-relative behaviour, but remain
sample-dependent.

### Efficient frontier and realised risk–return curve

A model-conditioned efficient frontier is the set of portfolios implied by a supervised
model's expected-return vector, a declared covariance estimate and common constraints.
It is conditional on those estimates and is not universal to the stock universe.
DQN/PPO sleeve selectors do not naturally define a stock-level classical frontier.

A realised model risk–return curve connects conservative, balanced and aggressive
backtest outcomes for one model when their declared risk-control parameter is monotonic.
It is empirical sensitivity evidence, not a mathematical efficient frontier. Dominated
and non-monotonic outcomes are retained.

### Train, validation, outer evaluation and final holdout

Training data fit model parameters. Inner validation/stopping data choose architectures
and hyperparameters. Outer walk-forward folds estimate how the complete selection
procedure generalises. A final holdout is accessed only after selection is locked. The
legacy final test has already been observed and is descriptive; it cannot select new
models or settings.

### Purge and embargo

Purging removes observations whose forward label interval crosses a split boundary.
Embargo withholds additional sessions around that boundary. Both reduce leakage caused
by overlapping five-session targets; ordinary shuffled cross-validation is unsuitable.

### Point-in-time information

Point-in-time (PIT) processing ensures a value is used only after it was publicly
available. SEC facts use the conservative rule `market.session_date > filed_date`.
Fiscal-period end dates are not availability dates. This prevents future filings from
being backfilled into historical decisions.

### Hyperparameter and architecture optimisation

Hyperparameters are choices not learned directly from the fit data, such as
regularisation, learning rate or tree depth. For DQN, PPO, LSTM and Transformer,
architecture—depth, width, recurrent size, attention heads and related choices—is also
a hyperparameter. Proposals are evaluated only within the predeclared temporal protocol
and compute budget. HPO improves the selected configuration within that search; it does
not prove that it is globally optimal.

### Risk appetite scenarios

Conservative, balanced and aggressive scenarios are a sensitivity grid over frozen
exposure, volatility, position and RL risk-aversion controls. They show how each model
responds to different assumptions. The project reports every scenario rather than using
the final holdout to choose an investor's preferred appetite.

### Reinforcement-learning reward

An RL reward is the feedback used to train a policy. It is not automatically the same as
investment success. DQN and PPO use a frozen sleeve-selector environment and an
after-cost risk-aware reward; evaluation also reports return, volatility, drawdown,
turnover and action stability so reward optimisation cannot hide undesirable behaviour.

| Term | Meaning in this repository |
|---|---|
| After-cost objective | Portfolio result after the configured transaction-cost deductions; distinct from a model loss |
| B0 | Immutable SPY buy-and-hold benchmark ID |
| Cross-sectional rank | Ordering securities against one another on the same session |
| Daily IC | Spearman correlation between predicted and realised five-session returns across securities for one date |
| Embargo | Sessions withheld after a split boundary to reduce temporal overlap |
| Final test | Once-only held-out period for descriptive evaluation; no longer eligible for selection |
| F0 / F1 | Market-only / market-plus-PIT-SEC feature contracts |
| Immutable run | A completed `runs/<run_id>/` directory treated as read-only evidence |
| Label horizon | Five trading sessions from T to T+5 |
| MAE / RMSE | Mean absolute error / root mean squared error, expressed in return units |
| PIT | Point in time: information is usable only after it was public; SEC facts require session date strictly after filed date |
| P1 / P2 | Equal-weight / inverse-volatility top-K portfolio engines |
| Purge | Removal of samples whose label window reaches a later split |
| Selection lock | `selection.json`, persisted before final-test prediction |
| Sharpe ratio | Annualised mean return divided by annualised volatility under the stated risk-free-rate convention |
| Semantic label | Human-readable experiment name joined onto, but never replacing, an immutable ID |
| Shadow/paper trading | Non-capital operational rehearsal; not implemented and not equivalent to validated live trading |
| Split | Train, validation, test or excluded chronological partition |
| T+1 close | Signal uses data through T; trade occurs at the next market session's close |
| Turnover | Absolute traded value divided by pre-trade equity; reported as a fraction |
| Universe | Fixed checked-in list of approximately 100 current large-cap US equities; survivorship bias is accepted in Slice 1 |
