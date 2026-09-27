# Expanded exploratory close-out study approval record — v4 amendment

**Approval date:** 27 September 2026  
**Study ID:** `expanded_closeout_exploratory_v1`  
**Claim status:** exploratory close-out, not confirmatory  
**Parent authority:** `docs/planning/EXPANDED_STUDY_APPROVAL_V3.md`

The human explicitly required the evidence to support questions of the form:
“holding other aspects constant, what is the marginal effect of modulating a
given hyperparameter or architecture?” The approved-v3 study was stopped during
its first LSTM outer-fold work before family locks or holdout access after review
showed that two deterministic-random candidates per family could not identify
those controlled effects. Its partial artefacts and operational logs remain
historical failure/interruption evidence and are not research results.

## Controlled-sensitivity amendment

The replacement study retains approved-v3's universe, data vintage, causal
information contract, folds, seed, full-fidelity selection candidates, risk
scenarios, objectives, execution convention, costs and sealed descriptive
holdout. It adds a score-blind, selection-ineligible controlled sensitivity
screen before the holdout opens:

- candidate construction is generated from each registered component's
  declarative search-space metadata;
- one common reference and one valid single-factor contrast are retained for
  every active hyperparameter or architecture dimension;
- conditional parameters receive a conditional matched reference;
- comparison-group, reference/contrast indices, levels, changed fields and
  held-constant parameters are persisted before outcomes are evaluated;
- every proposed, completed and failed sensitivity candidate and every
  fold/seed/scenario cell is retained, including failure stage, exception,
  traceback, partial-cell count and available resource telemetry;
- the screen uses the first declared inner fold per outer fold and seed 41;
- deep sensitivity fits use two epochs with patience one; RL sensitivity fits
  use 1,000 environment steps; and
- sensitivity results cannot enter HPO ranking, family locks or final-holdout
  selection.

These are local matched contrasts at declared reduced fidelity. They support
descriptive marginal-effect and failure-probability questions at the reference
configuration; they are not global causal effects and must not be silently
generalised across interactions or fidelity levels. The final holdout remains
descriptive and cannot be used for sensitivity analysis or selection.

The amended matrix must receive a fresh score-blind runtime calibration and
capability/authority gate before execution. No model outcome or final-test result
may determine the contrast construction, fidelity or runtime budget.
