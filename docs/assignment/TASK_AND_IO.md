# Task definition, motivation and I/O

## Precise task

For each eligible US equity and session T, estimate the adjusted-close return over the next five trading sessions. Rank the cross-section by that estimate. On the first observed session of each ISO week, choose the top 10 and enter at T+1 close. Compare a momentum rule, two regression families, two feature sets and two weighting rules, after 10 bps one-way transaction costs.

The practical decision is which equities to hold and at what weights. The learning target is continuous, but success is not defined by prediction error alone: forecasts must order opportunities well enough to support an after-cost portfolio under a fixed execution convention.

## Motivation

Short-horizon equity returns are noisy, nonlinear and non-stationary. The task exposes the central gap between a statistically fitted model and an investable decision: availability timing, temporal validation, ranking, turnover, transaction costs and portfolio construction. Comparing market-only and market-plus-SEC inputs tests whether a small set of public accounting facts contributes incremental information without unsafe historical backfilling.

## Training contract

### Inputs

- Entity key: stable `security_id`, with ticker and CIK retained for traceability.
- Observation time: daily session T after close.
- F0: returns over 5/10/20 sessions, realised volatility over 5/20, 20-session average dollar volume and volume ratio, 20-session moving-average distance and distance from the 252-session high.
- F1: F0 plus latest basic EPS and Net Income whose SEC filed date is strictly earlier than T.
- Label: `forward_return_5d = adjusted_close[T+5] / adjusted_close[T] - 1`.
- Split: chronological 60/20/20; labels crossing a boundary are purged and the first five sessions after each boundary are embargoed.

### Outputs

- One fitted pipeline for each of E1-E4.
- Candidate parameters and validation MAE, RMSE and mean daily IC.
- Frozen selected parameter index per model and E5 source.
- Validation predictions, followed only after the selection lock by once-only final-test predictions.
- Provenance: source hashes, data manifests, split policy, seed, environment and code revision.

## Deployment-style inference contract

Slice 1 does not deploy or trade live. Its backtest interface represents the eventual inference boundary:

1. receive PIT-safe features through T;
2. emit a predicted five-session return per eligible security;
3. rank all securities for the same T with deterministic tie-breaking;
4. create top-K target weights;
5. execute at the next session's close;
6. emit auditable trades, positions, costs and daily equity.

The prediction table includes session, security, ticker, actual label when available, prediction, rank, split, model ID and feature-set ID. Reporting preserves machine IDs and adds semantic display labels.

## Scope and claims

The implemented result is evidence about one fixed universe, data vintage, split and cost convention. It is not proof of persistent alpha, an unbiased historical index simulation, a live system or investment advice. The observed final test may be described but never used to revise the chosen system.
