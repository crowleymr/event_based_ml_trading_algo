# AI use, challenges and verification register

This living register records factual implementation assistance. It is not the student's personal reflection.

## Assistance record

| Phase | Challenge | AI-assisted action | Independent verification/evidence | Status |
|---|---|---|---|---|
| Data | Preserve provenance while normalizing two heterogeneous sources | Implemented raw caching, identifier bridge, schemas and consumed-file hashes | Schema/idempotency tests; official mapping coverage; immutable manifests | Verified |
| PIT features | Prevent fiscal-period or filing-date look-ahead | Implemented strict as-of join with exact-date exclusion and filing provenance columns | Synthetic temporal tests and persisted-run strict-availability audit | Verified |
| Temporal validation | Five-session labels can overlap later periods | Implemented label-end purge and five-session post-boundary embargo | Split tests and persisted boundary audit | Verified |
| Model selection | Keep final test isolated | Implemented fixed grids, validation IC/RMSE rule and pre-test `selection.json` lock | Test-label perturbation test; selection/prediction audit | Verified |
| Backtest | Avoid same-close fills and inconsistent cost accounting | Implemented T+1-close execution, drifted holdings and self-financing traded-dollar costs | Accounting/unit tests plus cost/equity reconciliation audit | Verified |
| Reproducibility | Capture source, data, environment and device evidence | Implemented run-local snapshots, recursive source hashes, seed/thread/device metadata | Two seeded smoke runs, environment tests and 17-check audit | Verified with documented cross-platform limits |
| Architecture | Broad modules obscured responsibility boundaries | Refactored into data/features/modelling/portfolio/tracking/validation packages | Full test suite, smoke run and separate architecture review | Verified |
| Documentation/reporting | Avoid manually transcribed or prose-derived evidence | Added evidence policy, registry and coded report generator | Determinism/schema/failure/reconciliation tests and reference-run read-only build | Verified |

## Human and AI boundary

AI may inspect code, implement scoped changes, generate repeatable evidence, explain algorithms and propose interpretations. The human owns research-question changes, data/split/cost policy, stage-gate decisions and all first-person assignment reflection. AI-generated prose must not claim what the student understood, found difficult or independently learned.

## Verification standard

Verification should be reproducible and independent of the prose being checked:

- unit/integration tests assert code behavior;
- immutable manifests and SHA256 values establish source identity;
- coded reports reconcile to run tables and JSON;
- visual inspection checks presentation but does not replace numeric reconciliation;
- any discrepancy is fixed upstream and regenerated;
- final-test metrics may be reported but never used as a debugging target for model quality.

## Student-owned critique prompts

- **STUDENT TO COMPLETE:** Which AI-assisted action did you verify most deeply, and how?
- **STUDENT TO COMPLETE:** Identify one limitation or mistaken proposal from AI output and explain your correction.
- **STUDENT TO COMPLETE:** What concept did you learn well enough to explain without AI assistance?
- **STUDENT TO COMPLETE:** Which knowledge gap remains, and what concrete verification will close it?
