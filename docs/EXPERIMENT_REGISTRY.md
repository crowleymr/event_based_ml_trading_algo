# Experiment and model registry

This is the semantic registry for stable machine IDs. Existing artefacts are immutable: E0-E5 and B0 must never be renamed or rewritten. Display labels may improve without altering IDs.

## Completed Slice 1 registry

| ID | Semantic display label | Feature set | Estimator/signal | Portfolio engine | Selection role | Parameters/provenance |
|---|---|---|---|---|---|---|
| E0 | Momentum · Equal Weight | Momentum: 20-session return | Deterministic momentum score | P1 equal-weight top 10 | Non-ML strategy baseline | Locked config; weekly T+1 close; 10 bps one-way |
| E1 | Market Elastic Net · Equal Weight | F0 market | M1 Elastic Net | P1 equal-weight top 10 | Candidate; validation-tuned | Selected candidate stored at `selection.json.models.E1` |
| E2 | Market Histogram GBT · Equal Weight | F0 market | M2 histogram GBT | P1 equal-weight top 10 | Candidate; validation-tuned | Selected candidate stored at `selection.json.models.E2` |
| E3 | Market + SEC Elastic Net · Equal Weight | F1 market + PIT SEC | M1 Elastic Net | P1 equal-weight top 10 | Candidate; validation-tuned | Selected candidate stored at `selection.json.models.E3` |
| E4 | Market + SEC Histogram GBT · Equal Weight | F1 market + PIT SEC | M2 histogram GBT | P1 equal-weight top 10 | Candidate; validation-tuned | Selected candidate stored at `selection.json.models.E4` |
| E5 | Validation Winner · Inverse Volatility | Source candidate's features | Reuses frozen winner predictions | P2 inverse-volatility top 10 | Risk-engine comparison; no new predictive fit | Source stored at `selection.json.e5_source`; reference run source is E3 |
| B0 | SPY Buy and Hold | None | Broad-market price baseline | Buy once and hold | Descriptive benchmark, never an ML candidate | Yahoo adjusted SPY, T+1 close and same cost convention |

### Feature and estimator IDs

| ID | Meaning | Locked Slice 1 definition |
|---|---|---|
| F0 | Market features | returns 5/10/20, volatility 5/20, 20-day average dollar volume, volume ratio, 20-day moving-average distance, 252-session-high distance |
| F1 | Market + SEC | F0 plus latest strictly available basic EPS and Net Income |
| M1 | Elastic Net regressor | Median imputation, standardization, Elastic Net; four-candidate fixed grid |
| M2 | Histogram GBT regressor | Median imputation, sklearn histogram gradient boosting; four-candidate fixed grid |
| P1 | Equal-weight top-K | Long-only, top 10, weights 1/K |
| P2 | Inverse-volatility top-K | Same top 10, normalized inverse trailing 20-session volatility |

The reference run parameters are authoritative in its immutable `selection.json`; documentation must not transcribe numeric choices as a substitute for reading that file.

## Phase 2 exploratory selector registry

These IDs are append-only policy identifiers and do not rename or replace E0-E5/B0.
They first apply to the frozen Phase 2 protocol in `configs/rl_pilot.yaml`.

| ID | Definition | Selection eligibility | Provenance |
|---|---|---|---|
| RL0_RANDOM_SELECTOR | Seeded uniform choice among cash and frozen E0-E5 sleeves | Baseline only | Seed and actions in the RL run |
| RL1_DQN_SELECTOR | Fixed-budget DQN over the registered discrete action space; protocol v1 CPU, v2 explicit cpu/cuda/auto | Exploratory/diagnostic only | Frozen config, all seeds, requested/actual device, model and schema-versioned telemetry in the RL run |
| RL_B0_ALWAYS_E0 | Always select E0 | Fixed baseline | Frozen E0 weights from the reference run |
| RL_B1_ALWAYS_E1 | Always select E1 | Fixed baseline | Frozen E1 weights from the reference run |
| RL_B2_ALWAYS_E2 | Always select E2 | Fixed baseline | Frozen E2 weights from the reference run |
| RL_B3_ALWAYS_E3 | Always select E3 | Fixed baseline | Frozen E3 weights from the reference run |
| RL_B4_ALWAYS_E4 | Always select E4 | Fixed baseline | Frozen E4 weights from the reference run |
| RL_B5_ALWAYS_E5 | Always select E5 | Fixed baseline | Frozen E5 weights from the reference run |

The completed Slice 1 test is already observed. Phase 2 comparisons on it are
descriptive, never a new selection result. Confirmatory work requires a fresh vintage
and a predeclared walk-forward protocol.

Protocol v2 reruns retain the same `RL1_DQN_SELECTOR` policy because the model and
research role are unchanged; execution device is provenance, not a new candidate.
They carry `diagnostic_reproduction_not_model_selection` and cannot be used to prefer
GPU, CPU, a seed, or any model setting.

## ID formation rules

- A **run ID** is generated once as `YYYYMMDDTHHMMSSZ-<8 lowercase hex>`; it names one immutable execution contract.
- A new **experiment ID** uses `E<number>`, taking the next never-used integer. IDs are permanent and never recycled. A materially different feature/model/portfolio combination receives a new ID.
- A new **benchmark ID** uses `B<number>` and the same append-only rule.
- Reusable **feature**, **model** and **portfolio** definitions use the next `F<number>`, `M<number>` and `P<number>`. Hyperparameter candidates do not receive a new model-family ID; their exact parameters live in selection provenance.
- A display label is descriptive metadata, not an identifier. Reports join IDs to this registry and retain both columns.
- Never retrospectively rename fields or IDs inside a completed run. If a legacy contract needs interpretation, add a versioned adapter/report and record its source hashes.
- Every future registry entry must state definition, owner module, input/output schema, selection eligibility, parameter provenance and first applicable run.
