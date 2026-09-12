# Synthesis Decision Log
## Research Outputs → POC Architecture

This records the deliberate reductions/decisions after reviewing the data-feed research, ML trading literature, existing FSD and weekend constraint.

1. **US only:** ASX deferred because PIT fundamentals/events are materially harder with free sources.
2. **~100 securities, not 500:** system completeness matters more than sample breadth for the POC.
3. **Daily, 5-day horizon:** short-term but much easier to implement defensibly than intraday.
4. **Two SEC facts only:** `EarningsPerShareBasic`, `NetIncomeLoss`.
5. **SEC filing metadata is the initial event stream:** general news/NLP deferred.
6. **Polars + DuckDB + Parquet:** local, fast, native Parquet, strong as-of/SQL workflow.
7. **Regression/ranking:** predict 5-day return then rank stocks rather than binary direction.
8. **Elastic Net + GBT:** low-cost linear/nonlinear comparison; deep models deferred.
9. **Four model-feature comparisons:** market/Elastic, market/GBT, market+SEC/Elastic, market+SEC/GBT.
10. **Two portfolio engines:** equal-weight top-K vs inverse-vol top-K; MPT/RL deferred.
11. **Weekly rebalance:** reduces turnover and POC noise.
12. **10 bps one-way cost:** simple but non-frictionless.
13. **T+1 execution:** protects against close-price lookahead.
14. **Purging + simple embargo:** keep the core leakage principle; defer CPCV/PBO/DSR/HAC machinery.
15. **Limited tuning:** <= 8–12 settings per family, avoiding validation overfit and run explosion.
16. **Filesystem tracking before MLflow:** artefact reliability matters more than tooling polish.
17. **Profitability is not a pass/fail gate:** a valid negative result still demonstrates the required ML reasoning.
18. **One enhancement after stage gate:** NLP/events, sequences, MPT, RL or Australia—selected from evidence, not all implemented automatically.
