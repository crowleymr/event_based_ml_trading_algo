# Phase 2 real-data Gymnasium pilot

## Status and research question

This is an exploratory, not confirmatory, extension of completed Slice 1. It asks:

> Can an RL policy that dynamically selects among frozen, pre-existing portfolio
> strategies improve after-cost risk-adjusted performance relative to always using
> any single constituent strategy?

It does not create a new 100-security alpha model. E0-E5, their predictions and their
target security weights are frozen inputs from reference run
`20260912T071137Z-9899fd9a`. The completed final test has already been observed and is
used only for descriptive evaluation. No pilot choice may be tuned from it.

## Frozen Markov decision process

One episode is one existing chronological source split. Validation is the DQN training
episode; test is the descriptive evaluation episode. Timesteps are never shuffled and
an episode cannot cross a split boundary.

At the first observed session T of each ISO week, the agent sees:

- cross-sectional mean five-session return, mean trailing volatility and return dispersion;
- each frozen sleeve's recent realised return and volatility through T;
- contemporaneously available cross-sectional mean and dispersion of the frozen E1-E4 predictions;
- current selected action, cash weight, gross exposure and previous turnover.

The discrete action is cash or one of E0-E5. An action at T maps to that sleeve's frozen
underlying target security weights and executes at the next market session's close.
Existing holdings earn the T-to-T+1 move before the rebalance. New holdings earn returns
only after the T+1 fill. The portfolio then holds until the next weekly decision.

Reward is the weekly net log return:

```text
log(equity_at_next_decision / equity_at_current_decision)
```

The Slice 1 self-financing rebalance solver charges 10 bps one-way on actual traded
dollars. It is imported by the environment rather than reimplemented. Security weights
must be finite, non-negative and sum to no more than one. Missing held or execution bars,
non-finite observations, invalid actions and invalid weights fail immediately.

Gymnasium `terminated` is true after the last complete weekly transition in the source
split. `truncated` remains false because the fixed historical episode has no external
time limit. `reset(seed=...)` follows Gymnasium seeding and resets portfolio state to cash.

## Data and leakage controls

The source feature file lives outside the reference run. Before use, its SHA-256 is
matched independently to both the frozen config and `dataset_manifest.json`; the runner
then copies it into the new RL run and verifies the copy again. Relevant reference
manifests are also snapshotted and hashed. The reference run and source features are
never changed.

Observation builders filter every history through the current signal date. Frozen
predictions are read only at T. Sleeve weights are read for the already-defined T+1
execution date, but they are not included in the observation; they are the action's
execution payload. Fixed-action policies reproduce the corresponding E0-E5 equity
paths over the common episode interval, which tests the timing and accounting bridge.

## Policies and evaluation

- `RL_B0_ALWAYS_E0` through `RL_B5_ALWAYS_E5`: fixed sleeve baselines;
- `RL0_RANDOM_SELECTOR`: seeded uniform random selector;
- `RL1_DQN_SELECTOR`: one CPU MLP DQN configuration, 20,000 steps, seeds 41/42/43.

There is no parameter or reward search. Evaluation actions for DQN are deterministic.
All seeds are retained and the median seed is reported; the best seed is never selected.
Metrics use the existing financial definitions with 52 weekly periods per year, and add
reward sum, action frequencies and across-seed evidence.

## Limitations and confirmatory next step

The sleeves are products of the earlier research process and the evaluation interval is
already observed. The small validation history is reused as a repeated training episode,
regime coverage is limited, and action choice can overfit the frozen sleeve set. The
weekly state is an engineering approximation to a Markov state. Fixed 10 bps costs omit
spread and impact, and current-survivor/universe, Yahoo-adjustment and SEC limitations
carry forward from Slice 1.

Consequently, comparisons are descriptive and cannot establish RL superiority. A
confirmatory study requires a fresh data vintage, walk-forward frozen base predictions,
predeclared training windows/budget and an untouched final holdout. Continuous security
control, recurrent policies, GPU tuning, reward search and live/paper trading remain
explicitly deferred.
