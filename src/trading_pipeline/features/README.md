# Features package

Owns deterministic predictors available after daily session T. `market_features(bars)` builds F0 and `join_fundamentals(frame, facts)` adds the two F1 fields; `build(bars, facts)` composes them.

Input is validated daily market bars plus irregular SEC fact records. Output retains source columns and adds F0: 5/10/20-session returns, 5/20-session volatility, 20-session average dollar volume, volume ratio, moving-average distance and 252-session-high distance. F1 adds latest basic EPS and Net Income plus their filed dates.

Rolling windows are per security and use data through T. Fundamental as-of joins disallow exact matches, enforcing `filed_date < session_date`. Non-finite market features become null for downstream train-fitted imputation. Fiscal end is never treated as public availability.

The package does not create targets, assign splits, impute values, fit models or choose features based on results. Adding a feature is a research change requiring a new registered definition and, after the observed test, a new holdout/data vintage.
