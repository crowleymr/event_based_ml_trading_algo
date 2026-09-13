# Backlog and stage gates

This backlog separates required close-out from possible research expansion. Priority does not constitute approval: every item must pass its gate, and research-policy changes require the human decisions identified in [FSD v2](FSD_v2.md).

## Stage gates

| Gate | Entry evidence | Exit criteria | Stop condition |
|---|---|---|---|
| G0 Slice 1 close-out | Completed immutable reference run and green tests | Reproducible reporting, reviewed limitations, assignment-support materials and preserved run | Any provenance, reconciliation, PIT or selection-isolation failure |
| G1 New research protocol | G0 complete | Human approves hypothesis, PIT-safe sources, new validation/final holdout and evaluation budget | Reuse of the observed final test for selection |
| G2 Model/engine expansion | G1 complete | New candidates run through registered common interfaces and validation-only selection | A bespoke path that weakens comparability |
| G3 Realism and robustness | Stable G2 candidate | Costs/execution/universe policies documented and sensitivity evidence generated | Unavailable data silently approximated |
| G4 Shadow/paper readiness | Positive research decision, operational controls and fresh evidence | Read-only monitoring, reconciliation, kill switch, compliance/data licences and review | Any direct capital deployment or uncontrolled signal-to-order path |

## Prioritised backlog

| Priority | Item | Dependencies | Acceptance criteria | Main risks |
|---|---|---|---|---|
| MUST | Coded Slice 1 reporting | Immutable run contract, registry | Deterministic CSV/Parquet/Markdown outputs; source reconciliation; hashes and code revision; tests | Accidental prose-sourced numbers or source-run mutation |
| MUST | Assignment evidence refinement | Reporting outputs | All tables/figures generated; claims linked to definitions and limitations; student reflection remains student-authored | Rubric padding, fabricated reflection |
| SHOULD | Public-notebook preparation | Stable report schema | Self-contained path/configuration, read-only generated tables, no local absolute paths or publication claim | Data size/licensing, duplicate calculations |
| SHOULD | New holdout/data-vintage protocol | Human approval, PIT-safe source plan | Written freeze dates, candidate budget, selection rule and untouched final period | Post-test overfitting |
| SHOULD | Stronger backtest realism | G1; spreads/corporate-action policy | Parameterized spread/impact/terminal liquidation; reconciliation tests; sensitivity reports | False precision, unavailable PIT inputs |
| SHOULD | Multiple feature/predictive models | G1, model registry extension | Stable model IDs, common prediction schema, train-only transforms, validation budget | Multiple testing and scope inflation |
| SHOULD | Multiple risk/return engines | G1, common prediction freeze | Same frozen predictions, registered portfolio IDs, constraints and cost accounting | Confounding signal and portfolio effects |
| COULD | Paper-outline automation | Stable reports | Figures/tables cited by generated IDs; no hand-entered metrics; bibliography/reflection prompts | Mistaking generated prose for student evidence |
| COULD | Alternative strategies/assets | PIT-safe data, explicit research question, new holdout | Separate strategy/asset IDs, currency/calendar/cost policy and baseline | Data snooping, incomparable assumptions |
| RESEARCH | Real-data Gymnasium RL environment | G1-G3, approved MDP | PIT-safe observation/action/reward definitions; episode boundaries; realistic costs; baseline parity; deterministic tests | Leakage, reward hacking, non-stationarity, exaggerated complexity |
| LATER | Shadow/paper trading | G4 only | Broker sandbox, read-only review first, reconciled market data, alerts, kill switch, audit trail | Operational loss, licences, unreliable live data |

## Roadmap detail

### Predictive layer

Support multiple feature and estimator registrations only after a new research protocol exists. Candidates could include carefully justified nonlinear or event-text models, but every candidate must emit the current prediction contract and remain within an explicit trial budget. Do not add a model merely to make the project appear complex.

### Portfolio and risk layer

Add engines behind the same frozen-prediction boundary so portfolio comparisons do not alter model selection. Possible research includes constraints, volatility targeting or mean-variance variants after covariance and turnover assumptions are justified. Keep the current equal-weight and inverse-volatility engines as auditable baselines.

### RL environment

A future Gymnasium environment must use real, point-in-time observations and an explicit Markov decision process: observation time, allowed actions, fill timing, transaction costs, reward, termination and train/validation/test episodes. Benchmark it against non-RL engines using the same data and constraints. Synthetic smoke data may test software but cannot establish trading evidence.

### Backtest realism

Priorities are PIT-safe universe membership/delistings, explicit corporate-action handling, spreads/impact, capacity, benchmark alignment and terminal liquidation policy. Historical market cap, industry, P/E and similar fields remain blocked until a licensed PIT-safe source and transformation policy are approved.

### Communication and eventual operations

Paper and five-minute presentation support should consume generated evidence. Alternative strategies or assets require a separate hypothesis and calendar/currency/cost policy. Shadow/paper trading is a later operational experiment, not a shortcut to production.

## Scope-creep warnings

- Do not expand to 500 securities, new countries, intraday data or new providers inside a reporting or documentation change.
- Do not add NLP, RL, neural networks or optimisation because they are fashionable.
- Do not create FastAPI, Power BI, a production UI or microservices before a read-only reporting need is demonstrated.
- Do not mix a data-source change, model change and portfolio change in one experiment.
- Do not promote smoke-run results as research evidence.
- Do not tune against the completed reference test under any label such as “sanity check” or “presentation improvement.”
