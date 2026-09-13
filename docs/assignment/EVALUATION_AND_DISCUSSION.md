# Evaluation and discussion guide

## Methodology

Evaluation separates fitting, selection and final description. Models fit only the training period. A small fixed grid is scored on validation data. Mean daily cross-sectional Spearman IC is primary, validation RMSE is the tie-break, and stable grid order resolves any remaining tie. The chosen models and E5 source are persisted before final-test prediction.

The same predictions feed weekly top-10 portfolio construction. The backtest uses first observed ISO-week session signals, next-session close fills, drifted holdings and 10 bps one-way costs. Persisted audits reconcile selection, T+1 dates, costs, equity compounding, weights, hashes and strict SEC availability.

## What each result answers

| Evidence | Question | Unit/interpretation |
|---|---|---|
| MAE | Typical absolute prediction error | Decimal return |
| RMSE | Prediction error with extra penalty for large misses | Decimal return |
| Daily/mean IC | Whether predicted order agrees with realised cross-sectional order | Spearman correlation; dimensionless |
| Total/annualised return | Compounded portfolio growth | Decimal fraction |
| Annualised volatility | Dispersion of daily portfolio returns | Decimal fraction per year |
| Sharpe | Mean return relative to volatility, with zero risk-free rate in code | Dimensionless |
| Maximum drawdown | Worst peak-to-trough equity decline | Negative decimal fraction |
| Turnover/cost | Trading intensity and deductions | Fraction of pre-trade equity / currency-equivalent NAV units |

Use generated report metric definitions with every final table or chart.

## Expected behavior

Elastic Net should provide a stable linear baseline and may suppress weak features. Histogram GBT can capture thresholds and interactions but has greater capacity to fit unstable patterns. F1 can help only if the two latest SEC facts add timely cross-sectional information beyond market features; similar F0/F1 results are a valid null. Inverse-volatility weighting is expected to reduce concentration in volatile selected names, but need not improve return or drawdown in every period.

These are expectations, not reported findings. Findings must be derived from the specified immutable run.

## Failure modes

- SEC facts attached before their filed date, or restated/current values backfilled historically.
- Labels or preprocessing crossing split boundaries.
- Selecting hyperparameters, features or narratives after viewing final-test outcomes.
- Conflating low prediction error with useful ranking or after-cost performance.
- Unstable rank ties, missing execution bars or same-close execution.
- Turnover/cost errors, unrecorded cash or terminal-liquidation assumptions.
- Survivorship, delisting, corporate-action and retrospective-adjustment biases.
- Multiple comparisons and weak inference from a single chronological split.
- Treating smoke fixtures as empirical market evidence.

## Limitations and implications

The fixed present-day universe accepts survivorship bias and does not reconstruct historical membership or delistings. Yahoo adjusted histories can reflect retrospective adjustments. The SEC feature design uses two exact concepts and latest values that may mix fiscal durations; it is not a general accounting normalization system. The five-session T-to-T+5 label differs from the executable T+1 holding window. Fixed costs omit a calibrated spread/impact model. One split and small grids do not establish statistical significance or persistent alpha.

Consequently, any positive final-test outcome is conditional evidence, not proof of generalization. Any negative outcome is still useful evidence about this specification and does not justify mining the test for repairs. Stronger claims require a new untouched period/data vintage and a predeclared follow-up protocol.

Industry, historical market cap, P/E and similar analyses are deferred because no approved PIT-safe source/policy exists. Do not approximate them from present-day metadata.

## Future work

Prioritize reproducible reporting and student interpretation first. Then define a new holdout protocol and improve historical-universe and execution realism. Only after those gates should multiple predictive models, risk engines, alternative assets or a real-data Gymnasium RL environment be considered. Shadow/paper trading requires a separate operational-readiness gate.

## Student interpretation prompts

- **STUDENT TO COMPLETE:** Explain, in your own words, which metric best answers the research question and why.
- **STUDENT TO COMPLETE:** Inspect generated validation and test tables. Describe one result that surprised you without using it to tune the system.
- **STUDENT TO COMPLETE:** Identify the limitation you believe most weakens the conclusion and justify your choice.
- **STUDENT TO COMPLETE:** Explain what evidence would change your mind about the value of the SEC features.
