# Glossary

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
| Semantic label | Human-readable experiment name joined onto, but never replacing, an immutable ID |
| Shadow/paper trading | Non-capital operational rehearsal; not implemented and not equivalent to validated live trading |
| Split | Train, validation, test or excluded chronological partition |
| T+1 close | Signal uses data through T; trade occurs at the next market session's close |
| Turnover | Absolute traded value divided by pre-trade equity; reported as a fraction |
| Universe | Fixed checked-in list of approximately 100 current large-cap US equities; survivorship bias is accepted in Slice 1 |
