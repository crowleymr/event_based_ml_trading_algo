# Implementation log

## 2026-09-26 — Registered deep/PPO fold execution bridge (WP3/WP4)

- Added `optimisation.model_execution.fit_predict_supervised` for evaluator-owned,
  disjoint chronological fit/stopping/score partitions. It fits the sequence scaler
  on fit rows only, builds causal per-security windows, trains a registered LSTM or
  causal Transformer with the trial's architecture and `FitContext`, then emits the
  canonical prediction fields and fit/resource telemetry. Scoring labels can be
  absent; fit/stopping label intervals must end before the next partition.
- The registered RL runner now accepts real-data DQN/PPO calls only with a resolved
  approved study and a matching authority object freshly reverified against its
  pinned manifests. Its existing synthetic path remains available. Fixed the
  Gymnasium `Discrete.n` NumPy-integer validation and added policy parameter count
  to resource telemetry.
- An actual Stable-Baselines3 categorical PPO synthetic fit verifies learning,
  deterministic action evaluation, resource counters and no learning-state change.
  It is software evidence, not a trading result. The current draft study remains
  fail-closed and no expanded-data score-bearing run was performed.
- Verification: `.venv/Scripts/python -m pytest -q -p no:cacheprovider
  tests/test_registered_rl_runner.py tests/test_model_execution_bridge.py
  tests/test_rl_policy_contracts.py tests/test_deep_sequence_models.py
  tests/test_study_authority.py` passed 27 tests.

## 2026-09-26 — Expanded-study authority gate and runner dispatch (WP0/WP2)

- Added a separate `--study` entry to the authoritative `trading_pipeline.run` CLI.
  The existing `--config` Slice 1 route and selection code were not changed. Draft
  studies fail before ingestion or run-directory creation; a fully verified study
  currently stops explicitly before scoring because study execution remains
  unintegrated.
- Approved study schema now requires a content-matching protocol SHA-256, explicit
  real-data enablement, unique integer seeds, exploratory claim status, and continued
  prohibition of final-holdout and legacy observed-final selection.
- Added a read-only authority verifier for six repository-relative, SHA-256-pinned
  inputs: approval record, vintage snapshot, exposure ledger, outer and inner fold
  manifests, and sealed descriptive holdout. It checks study/vintage identity,
  chronological fold separation, inner-fold containment and holdout disjointness;
  it returns a frozen authority record for downstream gates.
- The checked-in expanded study remains a draft with unresolved vintage and
  manifests. No real-data optimisation, observed-test selection, or source-run
  artefact mutation occurred. The manifest JSON shape required by the verifier is
  exercised in `tests/test_study_authority.py` and needs producers in the eventual
  expanded-data workflow.
- Verification: 14 focused study/experiment/evaluator tests passed with an approved
  pytest temporary-directory permission. The first two restricted attempts failed
  at pytest fixture setup because Windows denied its temporary directories; those
  failures did not exercise application logic.

## 2026-09-26 — Temporal evaluator and registered RL engineering lane (WP2/WP4)

- Added a synthetic-only nested temporal evaluator over predeclared outer/inner
  folds. It validates date coverage and label intervals, keeps outer score rows
  away from inner fit/score callbacks, and purges an optional stopping tail.
- Added append-only per-fit/fold/seed JSONL evidence, complete-matrix equal-fold
  aggregation, explicit primary/secondary objective directions, and an
  exclusive-create selection lock tied to an evidence hash.
- Added one runner entry point for the registered DQN and categorical PPO
  adapters. It checks observation/action/cost parity, learns with a FitContext,
  evaluates deterministic actions, checks learning counters and adapter
  telemetry are unchanged during evaluation, and returns comparable resource
  facts. The legacy pilot path was preserved.
- Both paths reject real-data execution. No score-bearing HPO, final-test
  selection, authoritative `run.py` change, legacy artefact edit, or model
  performance claim was made. The policy adapters remain research-disabled.
- Verification: focused synthetic and existing RL-contract tests passed 16/16
  using `.venv/Scripts/python -m pytest -q -p no:cacheprovider
  tests/test_optimisation_evaluator.py tests/test_registered_rl_runner.py
  tests/test_rl_policy_contracts.py` with normal temporary-file access.


## 2026-09-26 — Deep supervised engineering lane (WP3)

- Added a causal per-security sequence view with explicit real-session and
  observed-feature masks. Its scaler fits only caller-specified training row indices;
  a target sequence contains sessions through its own T and excludes later sessions.
- Registered research-disabled PyTorch LSTM and causal Transformer supervised
  adapters. Both require a sequence batch and a separate stopping partition, expose
  declarative architecture/search dimensions, record epoch losses and device/resource
  telemetry, and save/load model state without changing `trading_pipeline.run`.
- Added synthetic tests for causal isolation, train-only scaling, masks, parameter
  validation, registry gate, seeded one-epoch fits and checkpoint round trips. No
  score-bearing data, legacy run artefact or final-test result was used.
- Verification: `.venv/Scripts/python -m pytest -q tests/test_deep_sequence_models.py
  tests/test_supervised_model_contracts.py` passed 15 tests; `git diff --check`
  passed. The complete suite and runner integration remain with the integration lane.


## 2026-09-20 — Phase 3 engineering foundations without research-policy selection

- Proceeded with the approved model refactoring and multi-run reporting work while
  treating unresolved research choices as fail-closed gates. No real-data HPO, new
  family winner, E5 change or observed-final-test selection was performed.
- Added separate `SupervisedModel`, `UnsupervisedModel` and `RLPolicy` contracts,
  allowlisted component registration and strict draft/approved study configuration.
  Every new adapter is `research_enabled=False`; approved configs require explicit
  authority, data-vintage, exposure-ledger and sealed-holdout manifests.
- Added isolated adapters for legacy Elastic Net, histogram GBT and XGBoost; a
  no-regime baseline and train-only deterministic GMM scaffold; and lazy DQN and
  categorical PPO policy adapters. The existing supervised and frozen RL runners were
  not replaced and completed experiment meanings remain unchanged.
- Added deterministic grid/random proposal generation, purged nested expanding
  walk-forward manifests and immutable trial-ledger writing for synthetic engineering
  verification. Adaptive or real-data optimisation remains gated.
- Added a read-only multi-run catalogue over completed audited runs, compatibility keys,
  persisted metric indexing, versioned checksums and dashboard run/config/time
  selection with incompatible-run warnings. Reporting provenance now hashes optional
  training and device inputs when consumed.
- Verification: the final complete test suite passed 74 tests in 56.02 seconds. Focused
  contract, optimisation, catalogue, dashboard, reporting and RL tests also passed.
  Authoritative synthetic run `runs/smoke/20260920T121854Z-ac1ac5c4` completed and its
  persisted 17-check audit passed. Versioned multi-run catalogue
  `reports/catalog/20260920T121854Z-ac1ac5c4-v1` was generated from completed audited
  smoke/RL inputs. `git diff --check` passed; line-ending conversion warnings remain
  informational.

## 2026-09-20 — HistGBT preservation and separate XGBoost diagnostic cycle

- Audited the current repository first and retained the already-implemented common
  telemetry/staged HistGBT work. Preserved every existing immutable run and report; no
  remote operation was performed.
- Kept sklearn Histogram GBT as M2/E2/E4 and added XGBoost as append-only M3/E6/E7.
  XGBoost uses histogram trees, a four-candidate grid, train-only median imputation,
  validation-only early stopping, fixed seed/one thread and configurable
  `cpu`/`cuda`/`auto` with executed CUDA preflight, explicit fallback and actual-device
  evidence. E5 remains locked to E1-E4.
- Extended the common trace/summary contract with XGBoost train/validation RMSE by
  boosting round, one-based selected iteration, selected parameters, duration,
  requested/actual device, fallback and XGBoost CUDA build metadata. HistGBT retains all
  100 supported staged train/validation RMSE points and validation permutation
  importance. XGBoost records native gain plus the same fixed-seed three-repeat
  validation permutation importance. SHAP was not installed and no SHAP values were
  fabricated.
- Completed the immutable full frozen-vintage diagnostic run
  `runs/20260920T010939Z-1225374b`; its 17-check persisted audit passed. It is labelled
  `diagnostic_reproduction_not_model_selection`; E5 remained E3. E2/E4 each selected
  the fixed 100-stage budget. E6/E7 each selected boosting round 1 through validation
  early stopping; this is evidence from the frozen data, not a recommendation to change
  the predeclared configuration.
- Paired timing/parity evidence in the generated `device_benchmark.parquet` completed
  without fallback and records the exact CPU/CUDA durations, validation RMSE and
  prediction deltas. Timing includes harness/preflight overhead and these small,
  early-stopped fits are not a general GPU performance benchmark.
- Generated the versioned read-only combined report
  `reports/20260920T010939Z-1225374b/v1` with the audited DQN run
  `runs/rl/20260920T002714Z-9c9692fc`. Training Diagnostics now provides consistent
  family/metric filtering, run/protocol/split metadata, devices, summaries, availability
  notices and CPU/GPU evidence for HistGBT, XGBoost and DQN. Elastic Net explicitly
  states that no conventional epoch/boosting curve applies.
- Verification: 37 tests passed; the deterministic two-run XGBoost smoke contract and
  report/dashboard tests passed; final durable software-verification run
  `runs/smoke/20260920T012021Z-fbf2d306` passed its persisted audit; the full run audit
  passed; an in-process Streamlit page execution reported zero exceptions; temporary
  dashboard health and page endpoints both returned HTTP 200. Report output hashes were
  recorded in provenance.

Limitations remain: the final test was already observed and is descriptive only; CUDA
timing is hardware/run-specific; SHAP is unavailable; and confirmatory comparison still
requires a fresh vintage and predeclared walk-forward/holdout protocol.

## 2026-09-20 — Unified evidence dashboard and CUDA telemetry diagnostic

- Preserved the user-owned `docs/planning/prompt-scratch_pad.md` edit and all existing
  immutable data, run and report artefacts.
- Added the common schema-versioned training trace/summary contract and explicit
  `cpu`/`cuda`/`auto` resolution with recorded fallback, determinism, package/CUDA and
  peak-memory evidence. Added a standalone tensor-placement preflight.
- Installed the optional official CUDA 12.8 PyTorch wheel locally; CUDA preflight placed
  the tensor on the detected RTX 4060 Ti.
- Ran the final frozen-budget three-seed DQN diagnostic reproduction as
  `runs/rl/20260920T002714Z-9c9692fc`. Its persisted audit passed. Device timing and
  parity evidence is generated in `device_benchmark.parquet`; per-seed durations and
  devices are generated in `training_summary.parquet`. No timing or observed-test result
  was used for selection.
- Generated report schema v2 in versioned output
  `reports/20260912T071137Z-9899fd9a/v3` from the immutable Slice 1 run, its hash-verified
  external feature snapshot and the completed audited diagnostic RL run. The report
  emits paired CSV/Parquet evidence and field-level definitions, units, limitations and
  provenance without modifying source artefacts.
- Expanded the read-only Streamlit surface into seven evidence views. In-process page
  execution had no application exceptions; local health and page endpoints returned
  successfully. Streamlit did not invoke a training or report-generation path.
- Verification command: `.venv/Scripts/python -m pytest -q`; the complete suite passed.
  The canonical Slice 1 persisted-run audit passed, and the prior RL run and report v1
  matched their stored manifests/hashes.
- Extended future authoritative `trading_pipeline.run` executions to emit the same
  telemetry contract without altering the fixed selection rule. Elastic Net records
  solver/final diagnostics rather than an invented epoch curve; histogram GBT records
  supported staged train/validation RMSE and deterministic GBT permutation importance.
  The final upgraded CPU smoke run is `runs/smoke/20260920T002409Z-dde46671`; its
  persisted audit passed and its v2 report is
  `reports/smoke/20260920T002409Z-dde46671/v2`.

This is diagnostic reproduction evidence only. The observed final test remains
descriptive; confirmatory selection still requires a fresh data vintage and predeclared
walk-forward protocol.

## 2026-09-19 — Local Phase 2 integration verification

Preserved the local FSD and Slice 1 plan hardware/reproducibility edits in a dedicated
commit, then applied the six completed Phase 2 commits to local `main` in their original
order without conflicts. Copied the named ignored RL run into the local checkout and
verified all 24 copied files against the source by relative path, byte count and SHA-256;
the run's 17-entry artefact manifest and every persisted audit check also passed.

Installed the optional Phase 2 dependency group and reran the complete suite: 32 tests
passed. A focused RL, reproducibility and dashboard run passed 9 tests. The read-only
dashboard loaded the existing generated Slice 1 report and audited RL run, and its health
and page endpoints returned HTTP 200. No remote, stash, existing report or source run was
modified.

## 2026-09-19 — Phase 2 exploratory selector pilot

- Read the authoritative FSD, implementation plan, decisions, completion evidence,
  architecture, backlog, registry, reporting implementation/tests and preserved run
  manifests before changing research behavior.
- Preserved the user's existing uncommitted hardware-policy edits in the FSD and Slice 1
  plan; they were not staged with Phase 2 work.
- Froze action/observation/reward/timing, CPU DQN budget, seeds, dependency groups and
  stable RL policy IDs before evaluating the descriptive test episode.
- Added hash-verified dataset preparation, a real-data Gymnasium environment and shared
  the canonical Slice 1 self-financing rebalance solver.
- Verified Gymnasium/SB3 compatibility, deterministic seeded transitions, T+1 timing,
  transaction costs, long-only/no-leverage rules, missing/non-finite failure paths,
  split isolation, feature-hash enforcement and exact fixed-sleeve parity.
- Added fixed/random baselines, three fixed-budget CPU DQN seeds and an immutable RL run
  contract with input snapshots, manifests, models, actions, trades, curves, metrics,
  action frequencies, audit and generated summary.
- Added a schema-checked read-only Streamlit dashboard over generated Slice 1 reports and
  completed audited RL runs. Browser smoke verification rendered overview, equity,
  drawdown and RL evidence without visible, browser-console or corrected server-log errors.
- The generated evidence does not support an RL superiority claim. Confirmatory work
  remains blocked on a fresh data vintage and predeclared walk-forward protocol.

## 2026-09-12 — Bootstrap
Read FSD v2 and the complete implementation plan (located in docs/planning).
Repository initially contained only licence, ignore rules and untracked planning documents.
Created Python package, fixed 100-security universe, separate research/synthetic smoke configs.
Python 3.12 selected from installed runtimes. No research policy changed.

## 2026-09-12 — Data phase
Implemented official SEC mapping/Company Facts, Yahoo daily raw caching, retries,
canonical Parquet, DuckDB views, synthetic fixtures and schema/coverage checks.
Verification: 4 tests passed. Windows sandbox temp-file and Git-index permissions
required approved elevated commands; no data provider was changed.

## 2026-09-12 — Features, models and portfolio
Added Polars rolling features, strict backward filing-date as-of joins, labels,
purge/embargo splits, train-only Elastic Net/GBT pipelines and small fixed grids.
Added weekly T+1-close holdings accounting with drift and self-financing costs.
Verification: 13 tests passed. Full synthetic E0–E5 run completed at
runs/smoke/20260912T070326Z-3265a9d6. This is not empirical trading evidence.
Real SEC mapping fetched; corrected verified MMC -> MRSH rename for the same issuer.

## 2026-09-12 — Runner and verification
Implemented immutable run snapshots/hashes, fitted model persistence, selection lock,
E0–E5 metrics/plots and manual GitHub Actions smoke upload. Added SPY B0 benchmark
using the existing approved Yahoo provider. Verification: 15 tests passed before
benchmark addition, including two identical seeded complete smoke runs and a
final-test-label perturbation test confirming unchanged validation selection.

## 2026-09-12 — Real run and audit
Full live run completed: runs/20260912T071137Z-9899fd9a (core commit 9cadfe9).
100 equities, 294000 daily bars, 2015-01-02 through 2026-09-11; 51934 SEC facts.
SEC issuer coverage: basic EPS 99%, Net Income 100%. Visa lacks the exact basic EPS
tag; retained with train-only imputation. No company or fundamental tag substituted.
E0–E5 and SPY B0 completed. E5 selected E3 from validation IC, before final test.
Independent 17-check audit passed, including SHA256, PIT, purge, identical E5
predictions/selections, T+1, cost and equity reconciliation. Equity plot inspected.
No test performance was used to change settings. 16-test suite passed after audit addition.

## 2026-09-12 — Concurrent authoritative document update
Observed user changes to FSD/plan adding hardware/reproducibility metadata. Reviewed
the complete diff and implemented best-effort CPU/GPU and CUDA inventory, actual
CPU devices, deterministic settings and fallback explanation. The existing live
metadata was supplemented explicitly after execution on the same host, without
rerunning or selecting on test outcomes. Future runs record this at startup.
Detected Ryzen 5 3600 (6 physical/12 logical cores), RTX 4060 Ti (16380 MiB), installed
CUDA toolkit/runtime 11.8; driver CUDA compatibility was not discoverable and is
explicitly unknown. No GPU execution or scope expansion.

## 2026-09-12 — Final verification
17 tests passed in 17.77 seconds. Final synthetic single-command smoke completed at
runs/smoke/20260912T072009Z-573f1f57 with startup hardware metadata and automatic
17-check integrity audit. Live-run audit passed 17/17 separately. Repaired Markdown
table spacing in the persisted live summary only; no predictions/metrics changed.
Created docs/SLICE1_COMPLETION_REPORT.md covering every DoD item, commands, artefacts,
known limitations and stage-gate risks. Remote GitHub workflow not dispatched.

## 2026-09-13 — Layout and operator documentation
Audited the implementation against the planned package layout. The required behavior
existed, but data, features, models and portfolio logic were each concentrated in one
broad module. Refactored these into the planned data, features, modelling, portfolio,
tracking and validation packages while preserving research policies and public imports.
Added a descriptive operator README, folder diagram, smoke/main quickstart, linked
documentation table, architecture/component status, persisted-results notebook, and
layout import tests. The first refactor checkpoint retained all 17 existing tests.
Verification after the full package split: 19 tests passed in 18.00 seconds and the
single-command synthetic run completed at runs/smoke/20260913T013307Z-0d43406e,
including E0-E5 and the persisted-run audit.
An independent higher-model architecture review found no missing mandatory component
or locked-policy regression. It reproduced validation gaps for null market keys,
conflicting SEC facts and null audit values, plus broad shared-cache provenance.
These were fixed with negative tests, run-local feature snapshots, exact consumed-input
hashes and recursive package source hashing.
Final verification: 20 tests passed in 18.60 seconds. A clean post-review smoke run
completed at runs/smoke/20260913T014046Z-f9b25e3a; its automatic 17-check audit and
manual re-audit both passed.

## 2026-09-13 — Documentation close-out and assignment support

Expanded the architecture into a reviewer-oriented current/planned design with exact
training and inference contracts, application-service boundaries, dependency rules and
Mermaid workflow/sequence diagrams. Added the gated backlog, stable experiment/model
registry, glossary, package ownership READMEs and a root evidence policy. Created living
assignment-support documents for task definition, algorithms, evaluation, paper and
five-minute presentation structure. Student first-person reflection remains explicitly
unwritten and is represented only by prompts.

AI-assisted actions in this checkpoint were repository/code/run-contract inspection,
documentation structuring, code-to-document mapping and consistency/link checks.
Independent verification consists of direct comparison against source modules, tests,
the immutable reference metadata/selection/split manifests and automated repository
checks; no numeric result was inferred from prose and no reference artefact was changed.

### Knowledge-gap register

| Gap or unresolved question | Current evidence | Closure owner/action |
|---|---|---|
| Whether engineering evidence satisfies the assignment's research stage gate | Completion report and immutable audit exist | Student/human reviewer decides; AI must not declare success |
| Strength of conclusions after survivorship and retrospective-adjustment bias | Limitation documented; no PIT constituent/delisting source | Human approves source/policy and a new holdout protocol |
| Comparable SEC fiscal-period normalization | Exact concepts and filed dates are preserved; durations can differ | Research design before any new feature |
| Realistic spread, impact, capacity and terminal liquidation assumptions | Fixed 10 bps and marked terminal holdings only | Human-approved policy plus sensitivity implementation |
| Student's own understanding, critique and remaining gaps | Cannot be evidenced by agent output | Student completes the prompts after review |

## 2026-09-13 — Repeatable coded reporting baseline

Added a read-only reporting command that requires one completed immutable run directory
and writes a new versioned output directory. It emits paired CSV/Parquet comparison,
universe, holdings, trades, security contribution, equity/drawdown, turnover/cost, IC,
metric-definition and deferred-field tables, plus Markdown and provenance JSON. Semantic
labels are joined beside immutable IDs. Contribution attribution uses prior-day holdings
and execution-day costs and must reconcile to every source daily return.

Verification covered deterministic derivation, schemas/labels, required-input failure,
numeric reconciliation to source comparison artefacts, contribution reconciliation,
versioned-output refusal and the complete regression suite. A read-only build against
the preserved live reference contract also passed. The notebook now consumes generated
reports through an explicit portable path; it has not been published.

## 2026-09-26 — Bounded universe admission foundation

Added `trading_pipeline.data.universe_admission` as a separate, opt-in admission
preflight. It reads a dated candidate CSV with an exact schema (ticker, source,
source-as-of, stable source rank and common-stock type), caps work at 500, and records
per-ticker SEC mapping, SEC EPS/Net Income coverage and Yahoo adjusted-history status.
SEC fundamental coverage is diagnostic and does not exclude a candidate. Mapping or
market-history failures receive explicit exclusion reasons. The production CLI checks
that all tickers in `configs/universe.txt` are represented in the candidate snapshot;
it does not edit that file or invoke the standard research ingestion pipeline.

Each admission snapshot has a provenance/rules manifest, append-only event log,
immutable progress checkpoints and immutable completed JSON ledger. A matching
snapshot resumes successful ticker checks; changed candidate bytes or preflight rules
fail closed. The CLI stores SEC/Yahoo cache files under that snapshot's own `raw`
directory. No live preflight or bulk download was run in this implementation task.

Verification: added six synthetic/mock tests covering candidate schema and order,
mapping and market admission, optional SEC fact coverage, explicit mapping exclusions,
resume/idempotence, changed-snapshot rejection and the unchanged 100-name Slice 1
config. The default system Python lacked project dependencies. The repository venv
collected the tests, but pytest could not create/read its temporary directories under
the managed Windows filesystem (WinError 5), so the fixture-based tests could not
execute here. One no-fixture baseline-preservation test passed before the same temp
directory setup issue affected remaining cases. No research data, raw caches, runs or
`configs/universe.txt` were changed.

Next live command (after preparing and reviewing the dated candidate CSV; Yahoo's
`--end` follows its exclusive-end convention):

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m trading_pipeline.data.universe_admission `
  --candidates data\universe_candidates_2026-09-26.csv `
  --output data\universe_admissions `
  --snapshot-id candidates-2026-09-26-v1 `
  --start 2015-01-01 --end 2026-09-25 --min-sessions 260
```

The candidate source snapshot and its source/as-of provenance must be established
before this command is run. Admission output is saved under
`data/universe_admissions/candidates-2026-09-26-v1/`.

## 2026-09-26 — WP1 dated candidate freeze and admission hardening

Frozen a 503-row current S&P 500 constituent enumeration from the Wikipedia
component table retrieved on 26 September 2026, retaining displayed symbol,
CIK and order in `configs/sp500_constituents_2026-09-26.csv`. The repeatable
builder places the unchanged 100-name baseline first and appends unique
constituents in source order, capped at 500. Its generated CSV and ticker-only
mirror are `configs/universe_candidates_500.csv` and `.txt`; they are
**candidates, not admitted securities**. The source/candidate/baseline SHA256
values and reproduction command are recorded in
`configs/UNIVERSE_CANDIDATES_2026-09-26.md`. The 500 candidate source CIKs
matched the preserved SEC mapping in a read-only exact-match check. No large-cap
rank is inferred from SEC row order.

Admission now checks each candidate's source CIK against SEC, preserves the
baseline order, admits later-listed securities with sufficient sessions through
the required end date, validates the 20-session warm-up plus five-day label minimum, and records
preflight F0/F1 readiness separately. Yahoo request errors remain pending and
are retried on resume; a final immutable ledger is sealed only when every
candidate has a definitive admitted/excluded outcome. The CLI records source
and SEC mapping hashes in the immutable manifest, and caches each successful
Yahoo raw frame in the isolated expanded root without replacing an existing
cache. The manifest includes the admission-code hash and source/mapping hashes;
each ledger row includes paths and hashes for available raw SEC/Yahoo files.
Existing Slice 1 files and runs were not modified.

Verification: the candidate builder reproduced the checked-in CSV byte for
byte; the text mirror matched all 500 CSV tickers in order; all source CIKs
matched the preserved SEC mapping. Added tests for builder order/immutability,
CIK mismatch exclusion and retry of transient Yahoo failure. The initial
sandboxed pytest run could not access its temporary fixtures (WinError 5).
Running the same focused suite with the permitted fixture write access gave
**11 passed in 6.79 seconds** on the final code, including the checked-in
source/candidate mirror, SEC identity and later-listing regressions. Earlier
live preflight attempts under `data/universe_admissions/sp500-20260926-v1/`
and `sp500-20260926-v2/` are superseded engineering checkpoints. A fresh
snapshot ID is required after this rule change because the admission-code
hash is in the immutable manifest. No completed admission count or expanded
security master is claimed here. The dated source and deterministic live
command are recorded in the candidate provenance document.

Added `trading_pipeline.data.admitted_universe` as a separate post-admission
freeze step. It requires a sealed ledger, reconciles every candidate and all
requested/admitted/excluded/F0/F1 counts, checks source CIKs and available raw
artifact hashes, then writes an ordered admitted-universe text file, expanded
security-master Parquet and source manifest with input/output SHA256 values in a
fresh directory. It refuses to overwrite an existing derived snapshot. A
synthetic fixture test passed, and the combined WP1 focused suite passed
**12 tests in 7.14 seconds**. This derived step has not yet been run on live
data because the final admission ledger is still being built.

## 2026-09-26 — Cross-lane integration checkpoint

Reviewed the study authority, registered RL trial, deep supervised fold bridge,
universe admission, dashboard and notebook work against the locked Slice 1
evidence and the expanded close-out plan. The central `--study` route verifies
approval and pinned manifests, then stops before opening score-bearing data.
The deep supervised bridge can execute a supplied fold for synthetic software
verification, but the real-data evaluator, trial/fit ledger orchestration,
outer-fold procedure lock, immutable study run contract and audit are not yet
connected to `trading_pipeline.run`. No expanded research result is claimed.

Closed a direct-call authority gap in the registered RL trial runner: even a
valid approved-study object and verified input manifests cannot execute a
non-synthetic RL trial until the central study path owns those missing gates.
Synthetic DQN/PPO adapter verification and telemetry remain available. Added
negative coverage for that valid-authority direct call and CLI routing tests
for the unchanged `--config` run and ingestion-only paths.

Verification: `.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-integration-full-20260926`
passed 111 tests before the final gate correction and CLI additions;
`.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-boundary-20260926 tests/test_study_authority.py tests/test_registered_rl_runner.py tests/test_run_entrypoints.py`
passed 11 tests after those changes. These commands required working pytest
temporary-directory permissions under the managed Windows filesystem. The
staged dashboard feedback file and immutable source runs were not changed.
Final full regression command:
`.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-integration-final-20260926`;
116 tests passed in 55.58 seconds.

## 2026-09-26 — WP8 notebook documentation close-out

Clarified the executable evidence notebook's research-status and source boundaries:
its displayed values come from the selected generated report, optional expanded-study
tables are only shown when present, and declared report output hashes are checked by
the read-only report loader. Added explicit metric-direction notes, the T+1 execution
versus five-session label caveat, and a reminder that the completed Slice 1 final test
is descriptive and cannot guide selection. Added the repository-root dashboard launch
command and report path requirement, and expanded student-owned reflection prompts.
Updated the README and public-notebook preparation page to link the notebook workflow
and avoid claiming unavailable optional evidence or a published URL.

Verification: manually checked the notebook's repository-relative source links and
confirmed the referenced implementation paths exist. `jupyter nbconvert --version`
reported 7.17.1. One `jupyter nbconvert --to notebook --execute` attempt was blocked
before kernel startup with Windows `WinError 5` writing Jupyter's secure connection
file; the environment also lacks `win32api` (pywin32), and the fallback Windows ACL
operation was denied. No numeric result was added or recomputed, and no run/report
artefact was changed. The public URL and licensing decision remain pending human review.
