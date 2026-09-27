# Expanded exploratory close-out study approval record

**Approval date:** 27 September 2026  
**Study ID:** `expanded_closeout_exploratory_v1`  
**Claim status:** exploratory close-out, not confirmatory

The human explicitly approved extending the fresh-vintage universe up to 500
securities, adding PPO, LSTM and causal Transformer families, treating deep/RL
architecture as a hyperparameter, applying HPO before comparison, reporting all
three risk appetites, and resolving remaining implementation questions analytically
or through predeclared bounded comparisons rather than pausing execution.

This approval is subject to the repository evidence policy and preserves Slice 1.
The already observed Slice 1 final test cannot select any expanded-study choice.
The expanded study must use its separately sealed fresh-vintage holdout exactly once
and only after complete family locks exist. Reporting must retain every declared risk
scenario and may not choose a preferred model or appetite from holdout outcomes.

Protocol decisions frozen before score-bearing execution are recorded in
`docs/DECISIONS.md` and the self-hashed approved study YAML. In particular:

- seven required families: Elastic Net, HistGBT, XGBoost, LSTM, causal Transformer,
  DQN and categorical PPO;
- shared `F1_CAUSAL_STACK_V1` information contract with fixed causal upstream E1-E4
  helper outputs, never candidate or holdout-selected outputs;
- nested purged expanding walk-forward development evaluation;
- conservative, balanced and aggressive scenarios with T+1 close execution and
  10 bps one-way transaction costs;
- score-blind runtime calibration selecting one complete budget tier before outcome
  rankings are opened; and
- descriptive holdout evaluation only, followed by generated read-only reporting.

This file records research authority, not successful execution. Capability evidence,
calibration, immutable locks, completion manifests and generated results remain
separate machine-verifiable artefacts.
