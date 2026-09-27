# Expanded exploratory close-out study approval record — v3 amendment

**Approval date:** 27 September 2026  
**Study ID:** `expanded_closeout_exploratory_v1`  
**Claim status:** exploratory close-out, not confirmatory  
**Parent authority:** `docs/planning/EXPANDED_STUDY_APPROVAL.md`

The original human authority, required seven model families, causal information
contract, nested purged walk-forward validation, three reported risk appetites,
score-blind deadline calibration, T+1 execution, transaction costs, and descriptive
holdout restrictions remain unchanged.

## Method correction

The approved-v2 attempt completed the supervised outer-fold work but failed closed
before an RL candidate completed and before the sealed holdout opened. The failure
exposed an engineering mismatch: a weekly RL episode required a held security to have
a price on every intervening daily session even though portfolio reward uses only the
exact observed T+1 execution and weekly endpoint prices. No score, ranking, complete
RL outcome or holdout result informed this correction.

The replacement protocol may value weekly returns only from exact observed execution
and endpoint prices; a missing endpoint remains a hard failure and prices are never
filled or substituted. At each signal, security eligibility additionally requires the
declared contiguous observed trailing-price history using only information then
available. A detected gap excludes that security from subsequent signals until the
trailing window recovers, with per-signal counts and reasons retained as audit
artefacts. Full-future-panel screening is forbidden because it would introduce
lookahead.

This method correction follows the human direction to resolve outstanding questions
analytically without allowing judgement calls to stall the final experiments. It
requires a newly self-hashed prepared protocol, fresh score-blind capability evidence
and a separately approved protocol. Approved v2, its original approval record and its
failed run remain immutable historical records.
