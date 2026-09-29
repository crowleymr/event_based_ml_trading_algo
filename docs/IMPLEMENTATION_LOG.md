# Implementation log

## 2026-09-28 — Assignment-specific evidence exporter

- Added the separate read-only `trading_pipeline.reporting.assignment_export` CLI. It
  requires one completed, audit-verified expanded run and its hash-verified WP7 report,
  refuses incomplete or mismatched inputs, and writes only to a new version beneath
  `reports/assignment/<run-id>/`.
- Generated reproducible CSV/Parquet evidence for controlled, selection-ineligible
  sensitivity contrasts; recomputed and ledger-reconciled outer/holdout IC, RMSE and
  MAE; recorded fit telemetry; the approved model-input feature dictionary; outer-fit,
  stopping, outer-score and descriptive-holdout feature statistics; explicit evidence
  availability; and provenance. Count units are retained and the single-seed limitation
  is emitted as unavailable rather than estimated.
- Generated `reports/assignment/20260928T005423Z-0634efaf/v1` from immutable run
  `20260928T005423Z-0634efaf` and its existing expanded report. Its manifest declares
  1,307 source-run hashed artefacts, 18 direct report/export inputs and 14 hashed output
  files across seven tables. No file under `runs/` or the WP7 report was modified.
- Verification: focused exporter plus expanded-report tests passed 11 tests. The
  generated assignment manifest's 14 output hashes and all Parquet row counts were
  independently rechecked after generation.

## 2026-09-27 — Safe temporary workspace cleanup

- Added a repository-derived cleanup command for disposable `.tmp/` contents. Its
  default mode only lists children; `--purge` validates repository and `.tmp` paths,
  removes only immediate `.tmp` children and retains the `.tmp/` directory.
- Updated the temporary-workspace guide and README with the required layout and
  review-then-purge workflow. Existing legacy scratch directories were not removed.
- Added focused tests for listing/purge boundaries, preserved `runs/` data, symlinked
  `.tmp` refusal and non-repository refusal.
- Verification: cleanup CLI inspection and `git diff --check` passed. The focused
  pytest run could not complete because Windows denied access to its designated
  `.tmp/pytest/temporary-cleanup-check` base directory, including pytest's session
  cleanup; application assertions did not run. That scratch directory was left intact
  and no purge was run.

## 2026-09-27 — Add dedicated Slice 1 data-download command

- Added `python -m trading_pipeline.data.download [--config ...]` as a clear
  ingestion-only operator command. It defaults to `configs/poc.yaml` and delegates
  to the existing config loader and ingestion service; `trading_pipeline.run` remains
  the authoritative training and backtest execution path.
- Updated the README and architecture description. The older `--ingest-only` route
  remains supported for compatibility.
- Verification: `.venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider
  tests/test_data_download_cli.py tests/test_run_entrypoints.py` passed (5 tests).

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

## 2026-09-26 — Expanded canonical vintage and feature snapshot

Added `trading_pipeline.data.expanded_vintage`, an offline derivation from the sealed
`sp500-20260926-v3` admission and its admitted-universe source manifest. It verifies
the admission, security master and every consumed Yahoo/SEC raw hash before producing
new canonical Parquet tables, F0/F1 features, five-session labels, point-in-time filing
dates, readiness masks, a per-security summary and hash-declared manifests. It does
not download, overwrite caches, train models or enter the completed Slice 1 run path.
Tabular, sequence and RL representation contracts refer to the same feature snapshot;
train-only transforms and any out-of-fold upstream predictions remain evaluator work.
No upstream prediction feature was asserted without a fitted, fold-scoped lineage.

The first bounded materialisation correctly stopped at canonical OHLC validation:
the pinned HUBB Yahoo history has an Open below Low on 2021-05-05. Rather than
substitute a price or leave a held-price gap, the derivation excludes the entire
HUBB security from canonical bars, facts and features. It retains HUBB's sealed
admission outcome, source hashes and an explicit `market_bar_rejections.parquet` row.
The source/data manifest distinguishes admitted and canonical-security counts.

Successful command:

```powershell
.venv\Scripts\python.exe -m trading_pipeline.data.expanded_vintage --admitted-root data/expanded_2026-09-26/admitted_v3 --output data/expanded_2026-09-26/vintage_v1
```

Generated evidence is in `data/expanded_2026-09-26/vintage_v1/`. Its
`data_vintage_manifest.json` has SHA256
`173073f1b7bb23d9021ba76916797b26418e6364fb0b085569b5b36fc1ef3157`;
`feature_manifest.json` has SHA256
`331b14709c9e25394e9b9508bb07ce15e2a2b4f76cf7b58222520e099887c687`.
The latter pins `features/features.parquet` SHA256
`9e62981ac19c5de29ffcbfec29409e577151728db2a3d1873acd92d3bc10819e`.
Manifest-generated counts: 500 candidates, 497 admitted, 496 canonical securities,
one quality exclusion, 1,416,531 feature rows and 1,414,051 labelled rows. These
are derivation counts, not model performance or evidence of an unbiased historical
universe. The current-survivor and retrospective Yahoo adjustment limitations remain.

Focused verification command:

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-expanded-vintage-allow tests/test_expanded_vintage.py
```

Result: three tests passed. The first sandboxed fixture attempt hit Windows
`WinError 5` on pytest's temporary directory; rerunning the same focused command
with fixture write permission passed. The interrupted build staging directory was
removed after its exact workspace path was checked; no sealed admission input,
legacy cache, prior run or report was modified.

### 26 September 2026 — central expanded study gate (engineering checkpoint)

`trading_pipeline.run --study` now passes verified authority to a central study
runner. The runner requires all seven declared families before opening the
feature snapshot. Because deep and RL score-bearing bridges are not yet
integrated in this runner, an approved seven-family protocol fails closed
without reading outcomes or creating a study run. A partial family protocol
also fails before data access. The existing `--config` execution remains on
the Slice 1 path.

The classical lane code accepts a hash-pinned canonical feature snapshot,
explicit chronological fit/stopping/scoring windows and an exact frozen
proposal budget. It records proposed/completed/failed trials and fold/seed fit
telemetry, writes outer selection locks before outer scores, aggregates complete
inner evidence for a family lock, and projects descriptive holdout labels only
after that lock. This lane has not executed a real expanded study and creates
no performance evidence. Full seven-family integration, unified risk-scenario
backtesting, independent audit integration and real-run verification remain
open work.

Verification: `py_compile` passed for the runner and CLI. Targeted authority
and evaluator tests passed (9 tests) with a separate pytest temporary directory
after the default Windows sandbox denied temporary-directory enumeration. The
negative tests confirm incomplete and unsupported family matrices fail before
feature data is opened or a run directory is created.

### 27 September 2026 — WP5 causal upstream-output feature-store contract

Added `trading_pipeline.features.model_outputs` as a versioned, immutable builder
for supplied inner out-of-fold predictions. It verifies the canonical feature
manifest and Parquet hashes, the source prediction hash, unique security/session/
component keys, complete fit/fold/seed/cutoff lineage, finite outputs, and a
strictly prior fit/label information cutoff. Outer, final-fit, holdout and legacy
observed-final roles are rejected as training features. The materialised table
contains every canonical key for each declared component, with unavailable
outputs represented by a null and explicit availability mask. Consumers must
fit any imputation on the training partition only. A parity audit checks exact
values and causal consumer cutoffs for tabular, sequence and RL views while
reporting their legitimate coverage differences.

The builder does not fit upstream models, generate missing predictions, or run
the expanded study. The authoritative runner still needs to produce a fold-
scoped source prediction artefact and bind this store to `F1_CAUSAL_STACK_V1`.
No real-data feature store or performance result is claimed. Verification:
`.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-model-outputs-allow5-20260927 tests/test_model_outputs.py`
passed 8 synthetic tests in 0.78 seconds, including forbidden-role, future-fit,
source/store/manifest tamper and representation mismatch cases. A first sandboxed pytest
attempt could not enumerate its temporary directory (WinError 5); the focused
suite passed with permitted fixture access. Legacy run artefacts were untouched.

### 27 September 2026 — deep supervised study adapter

Added a fold adapter for registered LSTM and causal Transformer candidates. It
requires every architecture and optimisation parameter to be explicit, receives
declared fit/stopping/scoring partitions from the central evaluator, returns
predictions in the evaluator's scoring-row order, and exposes training trace,
device/fallback, scaler and resource telemetry for the common fit ledger. It
does not authorise a study or open the final holdout itself. The sequence view
now indexes each security history once per transform rather than scanning the
whole feature pool for every target. The adapter trims each fold pool to the
declared securities and dates plus exact per-security lookback history. The
causal through-T semantics are unchanged.

Verification: `.venv\Scripts\python.exe -m pytest -q --basetemp=tmp_deep_study_agent_tests_escalated tests/test_deep_study_adapter.py tests/test_model_execution_bridge.py tests/test_deep_sequence_models.py`
passed 14 tests in 5.81 seconds. The initial sandboxed pytest run passed 12
cases but ended with two Windows temporary-directory permission errors; the
same focused suite passed with normal temporary-directory access; the final
focused rerun, including shuffled-pool causal ordering, passed 15 tests in
6.61 seconds. No real-data
deep HPO or expanded research result was produced.

### 27 September 2026 — expanded DQN/PPO study bridge

Added `rl/study_integration.py`, an explicit fold episode builder and a
verified-authority trial bridge for both registered DQN and categorical PPO.
The builder consumes central-runner-provided causal observations, frozen E0–E5
sleeves, market prices, calendar and source hashes; it does not read the legacy
run. The trial bridge validates all three declared risk coefficients, seeds,
architecture parameters, T+1 chronology, after-cost accounting and shared
train/score lineage. It applies the same scenario constraints to both policy
families, trains on a declared net-log-return certainty-equivalent reward, and
checks that deterministic evaluation leaves model, replay and normaliser state
unchanged. Started/completed/failed trial records, resource rows and hashed
cell outputs are persisted separately from source inputs. Direct real-data
calls to the older pilot runner remain forbidden.

Verification: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
--basetemp D:\repos\event_based_ml_trading_algo\tmp_rl_study_escalated2
tests/test_rl_study_integration.py tests/test_registered_rl_runner.py
tests/test_rl_policy_contracts.py` passed 23 tests in 7.14 seconds. Synthetic
checks include both algorithms under all three risk scenarios, forbidden
evaluation updates, invalid PPO rollout divisibility, missing sleeves and
overlapping fit/score episodes. No score-bearing expanded RL result exists yet;
the central runner still needs to generate causal observations/sleeves and
invoke this bridge across all required folds, seeds and proposals.

### 27 September 2026 — central supervised family dispatch

The authoritative `experiments/study_runner.py` now has one supervised fit/score
dispatch for Elastic Net, HistGBT, XGBoost, LSTM and causal Transformer. Deep
folds use the audited sequence adapter; all five families return the same
IC/RMSE, prediction and fit-telemetry contract for the existing inner, outer and
descriptive-holdout ledgers. Proposal generation uses the frozen family-specific
budget, deterministic deep random search, exact parameter validation and
duplicate rejection. The causal-stack view requires a supplied, hash-pinned
augmented feature table and provenance manifest bound to the canonical feature
snapshot; its keys and F1 values must match the canonical table, and target
values cannot enter the augmented table. The runner does not generate upstream
predictions or substitute missing stacked features.

The seven-family capability gate still rejects score-bearing execution before
feature reads because DQN/PPO observations, sleeves and trial handling have not
yet been connected to the central evaluator. The prepared draft also lacks the
supplied augmented feature contract and frozen `search.deep_training` epochs /
patience; current four-point classical grids cannot satisfy its eight-proposal
budget without explicit validated additional proposals or a newly frozen
search-space revision. No real study or run artefact was created.

Verification: `py_compile` passed for the runner and synthetic tests;
`.venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider
--basetemp=D:\\repos\\event_based_ml_trading_algo\\tmp_study_runner_supervised_approved2
tests/test_study_runner_supervised.py tests/test_study_authority.py
tests/test_deep_study_adapter.py` passed 14 tests in 6.78 seconds. The first
sandboxed pytest attempt hit Windows temporary-directory access errors; the
same focused suite passed with normal fixture access. Existing run artefacts
were not modified.
### 27 September 2026 — score-blind calibration and capability evidence contract

Added `trading_pipeline.experiments.calibration` to select a complete compute-budget
tier from resource-only timed bridge smokes. The writer requires all seven arms,
rejects outcome-, prediction-, loss- and ranking-like fields recursively, estimates
the complete fold/seed/risk-scenario workload, and fails if even the minimum tier
cannot meet the declared deadline. A separate capability writer requires hash-pinned
synthetic and real-data-shaped bridge-smoke evidence for every arm exactly. These
functions create approval inputs only; they do not approve a protocol, read model
scores, execute HPO or open the final holdout.
### 27 September 2026 — causal RL episode and sleeve producer (synthetic contract)

Added `trading_pipeline.experiments.rl_episode_producer.produce_selector_episode`
for the central study runner. It accepts verified in-memory canonical feature,
model-output and adjusted-close views with their five required source hashes;
derives first-session weekly signals, T+1 execution and next-signal ends; checks
model fit/label cutoffs; freezes momentum E0, model-output E1–E4 and a declared
validation-source inverse-volatility E5; and creates the existing RL bridge's
`PilotDataset`. A declared trailing adjusted-close lookback estimates each
sleeve's annualised volatility through T. The minimum of volatility-target,
gross-cap and max-position scaling is applied and recorded per signal/action,
with cash retained. Missing outputs, lookback bars and held-security valuation
bars fail closed. The producer does not read legacy final runs or score real data.

Verification: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
--basetemp tmp_rl_episode_producer_tests tests/test_rl_episode_producer.py`
passed 3 synthetic tests. The current prepared protocol does not declare a
volatility estimation lookback or an E5 source; central integration must supply
these explicitly from an approved pre-score protocol before a real run.

### 27 September 2026 — fold-scoped causal upstream producer

Added `experiments/causal_stack.py` to produce fixed F1-only base predictions
inside declared chronological expanding inner windows. The close-out declaration
contains Elastic Net (alpha 0.001, l1 ratio 0.5), HistGBT (15 leaves, L2 10),
XGBoost (depth 3, lambda 10), seed 41, and a derived equal-weight E4 mean.
Every source row includes component, fold, seed, model/derivation hash, fit
cutoff, latest fitted label-end date and `inner_oof` role. The producer requires
an explicit protected holdout start, fit-session minimum and score-block size;
the central runner must pass the declared 120 and 10 for real execution. It
rejects overlapping/noncausal folds and score dates reaching the holdout.

The source table can be bound through the existing immutable OOF store builder.
The shared augmented tabular view retains null outputs and availability masks
and returns median imputation metadata computed only from supplied training
partition keys. Sequence and RL consumers can derive their representations
from this same keyed view/store. The prepared study still lacks a frozen
`data.upstream_base_layers` and `validation.upstream_cross_fit` declaration,
and registered upstream components remain research-disabled. Real-data
execution therefore remains gated; no legacy or holdout artefacts were read.

Verification: focused synthetic producer and OOF-store suite passed 12 tests
in 1.10 seconds with normal temporary-directory access after the sandboxed
pytest fixture directory failed with WinError 5. Synthetic checks cover
lineage, masks, train-only imputation, protected boundary rejection, future
label perturbation and the derived ensemble.

### 27 September 2026 — score-blind prepared-draft bridge smoke

Added a standalone seven-arm engineering smoke CLI. It validates the prepared
draft, pinned feature and inner-window manifests, causal augmented table and
lineage, RL output paths, and explicit deep/RL protocol fields before projecting
any parquet rows. Supervised smoke fits on a fixed small inner-fold sample and
checks feature-only evaluation output shape; deep arms execute one training epoch.
RL smoke constructs the declared causal selector episode and both policy
networks, resets the environment and checks a deterministic action. It does not
step or learn in the environment, because either would calculate realised
returns and rewards. Only time, device, shape and source hashes enter immutable
per-arm bridge receipts and calibration records. This is engineering capability
evidence, not study execution or approval.

The prepared v3 draft currently lacks a resolved augmented-feature contract,
so the CLI fails before parquet reads and emits no receipts. Verification:
`python -m py_compile src/trading_pipeline/experiments/real_data_bridge_smoke.py`
passed; `pytest -q -p no:cacheprovider tests/test_real_data_bridge_smoke.py`
passed four focused tests. The first test run encountered a Windows pytest
temporary-directory permission error; rerun without `tmp_path` fixtures passed.

### 27 September 2026 — expanded reporting table schema contracts

Added fail-fast required-column checks for each of the eleven optional expanded
research Parquet tables at both report generation and dashboard loading. The
contracts cover fields used by the funnel, security drill-down, trial evidence,
frontier/risk charts and final test-bench notebook views. Audit and hash-manifest
checks still run, and optional files remain optional; malformed declared evidence
now fails before it is passed to consumers.

Verification: the dashboard-only overview test passed (1 passed). The focused
reporting/dashboard suite could not execute its fixtures because pytest's
`tmp_path`/`tmp_path_factory` setup and cleanup hit Windows `WinError 5` access
denials on basetemp directories, including the configured writable workspace.
No real run or generated research evidence was accessed or changed.

### 27 September 2026 — read-only expanded closeout evidence postprocessor

Added `trading_pipeline.reporting.expanded_closeout` as a separate CLI. It accepts
one completed, audited `runs/expanded_closeout/<run_id>` and verifies the full
run hash inventory, protocol authority, pinned canonical feature/price snapshot,
sealed holdout calendar and complete arm/seed/risk matrix before writing eleven
optional WP7 Parquet tables under `reports/expanded_closeout/<run_id>`. A separate
`research_evidence_manifest.json` records run ID, input/output SHA-256 values,
code version and generation time. The postprocessor uses the runner's holdout
predictions and hashed RL cells; supervised portfolios use first-session weekly
signals, next-session close fills, declared caps and trailing sample volatility,
and the canonical self-financing one-way cost solver. RL and supervised series
must align on identical weekly marks. SPY is included only if complete canonical
SPY bars are present in the protocol-pinned feature snapshot; otherwise the
benchmark is explicitly marked unavailable. Realised profiles retain every
model, seed and scenario, including dominated outcomes. Classical frontier
points are generated only for supervised models. The question register contains
generated protocol facts and availability limitations, not student reflection.

The frontier utility solver applies declared gross and position caps to a
trailing point-in-time sample covariance; the separate volatility target is
recorded but is not an active classical-frontier constraint. The realised
supervised portfolio does enforce that target through ex-ante scaling. No
expanded completed run currently exists, so no report was generated or outcome
interpreted. The CLI also refuses partial or calendar-misaligned RL holdout
cells; this is a source-contract gate, not a claim that WP6 is complete.

Verification: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
--basetemp D:\repos\event_based_ml_trading_algo\tmp_expanded_postprocess_escalated
tests/test_expanded_closeout.py` passed four synthetic tests with normal
temporary-directory access. These cover tampered audited artifacts, T+1 cost
and gross-to-net reconciliation, weekly aggregation, equity mismatch rejection,
and pipeline count reconciliation. `py_compile` passed for the new module.

### 27 September 2026 — trusted component discovery and declared study roster

Moved the built-in supervised, unsupervised and RL registrations into metadata
declared by each implementation module. `default_registry()` discovers only direct
modules under the three trusted installed model packages; study YAML component IDs
are registry lookups, never Python import paths. Duplicate component IDs fail closed.
The expanded runner no longer compares its roster against a fixed seven-family set.
It validates unique declared arm/component IDs, registered interfaces, supported
bridges and the exact arm/component/interface mapping in the hash-pinned approved
capability gate. The gate writer and finaliser now operate on the YAML-declared
roster, while retaining the fixed expanded-study identity and score-blind evidence
requirements. No Slice 1 implementation, immutable run or real research data was
changed or executed.

Data asset paths and feature-column declarations remain manifest/protocol inputs.
`docs/ARCHITECTURE.md` records the remaining coded F0/F1, deep-bridge, RL-sleeve,
objective and budget constraints so they cannot be mistaken for open-ended plugin
support. Focused discovery, component contract, study runner, calibration and
manifest-resolution and authority tests passed: 44 tests in 8.19 seconds. Pytest needed an
unsandboxed run because the Windows sandbox denied temporary fixture access and
worker pipes; the initial sandboxed run was inconclusive for those fixtures.

### 27 September 2026 — synthetic verification receipt CLI

Added a deterministic-input CLI that accepts a JUnit XML report and the prepared
study YAML, requires at least one test with zero JUnit errors and failures, and
writes one immutable `synthetic_verification` JSON receipt per declared arm plus
an arm-to-receipt JSON mapping. Receipts bind the exact YAML component/interface
to the absolute JUnit path and SHA-256, suite totals, UTC generation time and
available Git code version. This records synthetic test evidence only; it does
not run tests, read real data or change research behavior.

### 27 September 2026 — expanded study routes by trusted capabilities

Removed the remaining Python component-ID rosters from the central study runner,
deep fold adapter, RL trial bridge and score-blind bridge smoke. Trusted model
modules now declare `study_adapter` plus one supervised view capability
(`tabular_view` or `sequence_view`); RL policies declare `study_adapter` and
`discrete_actions`. The runner rejects absent or ambiguous bridge metadata before
score-bearing reads, chooses the proposal budget and fit adapter from the
registered `ComponentSpec`, and constructs candidates through `registry.create`.
The deep and RL bridges independently verify the same registered capabilities.
YAML still supplies experiment arms as opaque registry IDs and cannot supply
Python import paths. No research policy, protocol, real data or run artefact was
changed.

Verification: focused supervised, deep, RL, smoke, discovery and authority suite
passed 31 tests in 7.44 seconds.
The initial sandboxed runs failed during Windows pytest temporary-directory
fixture setup (`WinError 5`); the suite passed with approved filesystem
escalation and a repository-local base temp.

### 27 September 2026 — deadline-closeout protocol controls

Added a shared declared tier table for score-blind calibration and study
finalisation, including a complete two-proposal budget per classical, deep and
RL family. Calibration now records the explicit cell-count formula, fold/seed/
scenario factors, timing-scope requirement and excluded work. The manifest-freeze
CLI now accepts the fold counts and all fixed window sizes; the draft study seed
list is `[41]`. These changes prepare a separately hashed exploratory revision.
They do not modify the existing approved protocol, inspect model outcomes, run
the real study, or access the descriptive holdout. With one seed, seed dispersion
is unavailable; the two-outer/two-inner fold design is deadline-constrained
exploratory evidence, not confirmatory. The focused calibration, manifest and
study-runner suite passed 15 tests in 3.07 seconds. The first sandboxed attempts
could not create pytest temporary fixtures on Windows (`WinError 5`); the final
run used approved filesystem access and a repository-local base temp.
# 27 September 2026 — read-only expanded evidence consumer glue

Added a separate eleven-table expanded report loader that verifies the generated
manifest's run identity, exact output inventory, SHA-256 hashes, required columns and
row counts before reading any table. The existing dashboard now accepts
`--expanded-report reports/expanded_closeout/<run_id>` alongside its legacy Slice 1
`--report` entry point. Expanded tabs display generated pipeline/security,
architecture/HPO, risk/frontier, final test-bench and evidence-register tables.
Presentation filters retain the benchmark, and curves preserve model, seed and
scenario identity; no model selection or report generation occurs in the UI. The
final evidence notebook detects expanded manifests through
`TRADING_REPORT_DIR` and executes against the self-contained report without Slice 1
tables. Operator commands are documented in the dashboard README, repository README
and public-notebook preparation page. No run artefacts, source data or research
protocol was changed.

Verification: `tests/test_dashboard.py` passed 9 tests, including an AppTest of the
expanded controls, an executed final-notebook code-cell integration check, loader
checksum rejection and read-only byte comparison. The local Windows sandbox denied
pytest temporary fixtures on the first attempt; the focused suite passed with normal
temporary-directory access. Notebook execution emitted only expected non-interactive
Matplotlib display warnings under pytest.

## 27 September 2026 — zero-argument expanded evidence operator flow

The expanded report generator now discovers the repository and newest fully audited
completed expanded run when invoked with no arguments. An existing valid report is
verified and reused; an incomplete or tampered run is excluded. The generated
manifest records source completion time, source Git commit and vintage, plus a
protocol-derived summary of budget tier, proposal counts, outer/inner folds, seeds,
requested device and declared arm/component mappings. It also records the generated
evaluation calendar hash, execution, cost, annualisation and benchmark contract.
These labels come from the immutable source protocol and metadata.

The no-argument dashboard discovers complete locally bound eleven-table reports,
opens the latest by source completion time and displays reports with the same
evaluation contract in a Model monitoring / run comparisons tab. Different protocol
hashes are flagged and their budgets, folds, seeds, devices and arm mappings shown.
Original generated model/seed/risk metric rows are shown with provenance columns;
no values are pooled or selected.
Partial, smoke and tampered reports are skipped. Explicit report/run arguments
remain available for advanced use. Documentation now leads with the two-command
operator flow. Verification: `tests/test_dashboard.py` and
`tests/test_expanded_closeout.py` passed 19 synthetic tests, including zero-argument
AppTest, audited run discovery and damaged-report exclusion. Headless Matplotlib
emitted three expected display warnings.

### 27 September 2026 — development-label scan reuse in expanded runner

The authoritative study runner now caches labels by date for the exact union of
declared inner and outer fit, stopping and scoring partitions. It loads each
partition on demand and reuses materialised rows across candidates and arms.
The cache rejects dates outside that union and overlap with the sealed holdout.
Outer score labels are first requested after their arm/outer selection lock is
written. The descriptive holdout is loaded separately, once, only after a
nonempty `family_locks.json` with `holdout_selection: forbidden` exists. Final
training/stopping labels outside nested partitions are also loaded separately
after that lock. No research policy, protocol, data, model outcomes or run
artefacts were changed.

The stopped run's fit telemetry excludes repeated Parquet scans and matrix
construction, so its ~3.6-second fit time cannot estimate total cell time.
With the two-by-two fold and two-proposal revision, repeated supervised label
scans fall from one per fit/stopping/scoring request to at most one for each
newly encountered inner partition and locked outer partition. This is an I/O
reduction, not a measured full-study runtime guarantee. CPU candidate process
parallelism was deferred: the existing loop appends trial and fit evidence and
seals selection locks in order, while candidate failure handling and the
shared feature pool would require an explicit parent-owned worker/result
contract to preserve that order. Deep and RL fits also share one CUDA device.

Verification: focused runner, authority, deep and RL integration suites passed
27 tests in 7.74 seconds. The first sandboxed pytest attempt could not create
temporary fixtures on Windows (`WinError 5`); tests passed with normal Windows
temporary-directory access.

### 27 September 2026 — bounded deep sequence device transfers

The LSTM and causal Transformer adapters now transfer at most the declared
`batch_size` sequence rows to the selected device for optimizer updates, epoch
train/stopping loss and prediction. Epoch MSE sums squared error over all
minibatches and divides by the full row count, retaining sample weighting when
the last minibatch is smaller. Prediction concatenates outputs in input row
order. The change addresses the production CUDA smoke's full-batch allocation
failure on the 16 GB RTX 4060 Ti; it does not change model parameters, seeds,
validation windows, early-stopping comparison or the approved protocol.

Verification: the focused deep-model and study-adapter suite passed 13 tests in
8.52 seconds. LSTM and Transformer tests compare bounded-batch predictions
with full-batch CPU predictions, compare uneven-batch MSE with full-batch MSE,
and instrument every device transfer to require no more than `batch_size` rows.
Full-vintage CUDA runtime remains to be measured by the score-blind smoke.

### 27 September 2026 — production batch search and operator pause

A 400-security, full-fit-history, resource-only CUDA smoke confirmed bounded VRAM
(approximately 1.3–1.4 GiB) and sustained GPU execution, but the sampled batch size of
64 was transfer-bound at roughly 30–50% GPU utilisation. The operator stopped the smoke
before it wrote evidence because the GPU was needed for foreground use. The smoke did
not persist predictions, scores, rankings or a partial capability record and did not
open the descriptive holdout.

Before any outcome-bearing run, the metadata-declared LSTM/Transformer batch-size search
range was changed from `[64, 128]` to `[512, 1024]`. This retains batch size as an HPO
dimension while matching the expanded-universe production scale and 16 GiB device. No
model outcome informed the change. The deep-model and study-adapter suite passed 13
tests. A fresh all-arm timing smoke is still required before approval; the interrupted
attempt is not valid calibration evidence.

The zero-argument dashboard monitoring view was also given generated human-readable run
and implementation-scope labels derived from immutable protocol metadata. The focused
dashboard/report suite passed 19 tests. Documentation now leads with
`python -m trading_pipeline.dashboard`; explicit paths remain forensic overrides.

# 27 September 2026 — unattended study operational monitoring

Added `trading_pipeline.operations.study_supervisor`, a read-only wrapper around
the authoritative `trading_pipeline.run --study` command. It creates a distinct
operator log directory with timestamped stdout/stderr, JSONL status and diagnostic
events, elapsed-time heartbeats, process-tree RSS/CPU, host RAM/CPU, and physical
NVIDIA GPU/VRAM samples. Process CPU utilization is calculated from cumulative
CPU-time and wall-time deltas, with the first sample marked unavailable. Host
load averages are recorded when supported. Every child output line becomes a
structured info, warning or error event. Missing NVIDIA tooling is recorded as unavailable.
Failure and interruption logs remain on disk. This adds no score-bearing path and
does not read run outcomes. Focused verification: `tests/test_study_supervisor.py`
passed 3 tests in 0.12 seconds, including a nonzero CPU-delta check.

### 27 September 2026 — final deadline capability gate and approval

Fixed two score-blind production-smoke defects before approval. RL smoke dates now
begin only when causal sleeve outputs are available, matching the authoritative runner.
Sparse expanded rows are materialised with full-schema inference before selecting RL
features, preventing Polars from freezing a column as Null based on its first 100 rows.
The shared materialiser is used by both the smoke and authoritative runner and has a
regression test.

The final 400-security/full-history smoke passed all seven arms. Per-cell wall times were
8.55s Elastic Net, 19.87s HistGBT, 11.02s CUDA XGBoost, 279.00s CUDA LSTM, 285.06s CUDA
Transformer, 32.11s CPU DQN and 32.48s CPU PPO. The fresh suite passed 215 tests with zero
failures/errors and generated seven new synthetic receipts. Score-blind calibration under
a six-hour search ceiling selected `deadline_complete` at 6,378.42 estimated seconds;
outcomes and rankings remained unopened. Capability gate
`capability_gate_deadline_v2.json` passed every declared arm.

`configs/studies/expanded_closeout_approved_v2.yaml` is approved and authority-verified
with protocol SHA-256
`71e2126ac30cd500773a1ff590c59a5987ac44fc5652a88b64039828541b86f0`.
It retains all seven arms, two proposals per family, two outer folds, two inner folds,
one seed and all three RL risk scenarios. At the time of this gate record no
authoritative score-bearing run had been launched; the later failed attempt is recorded
below.

### 27 September 2026 — concise supervisor console status

The operational supervisor now echoes only study lifecycle and phase transitions,
warnings, errors and terminal status to its launching terminal. Complete child output
and resource telemetry remain in the versioned operational log files. This changes
operator presentation only and does not read outcomes or alter the authoritative child
process. Focused supervisor verification passed 4 tests.

### 27 September 2026 — failed-run diagnosis, immutable resume and RL price correction

Approved-v2 run `20260927T075513Z-3a5052c9` failed closed after supervised outer
evaluation and before a complete RL candidate or holdout access. Both aggressive DQN
candidates encountered the same absent FISV interior daily bar. The source run,
operational logs and failure marker remain unchanged.

The authoritative runner now checkpoints each complete supervised/RL inner, outer and
holdout cell, including prediction/episode artefacts, content hashes, exact protocol and
input authority, and a byte-level execution-code hash. `--resume-from` creates a new
derived run, validates source lineage before creating it, copies only verified complete
cells, and reruns incomplete work. Protocol/input/code drift, checkpoint tampering,
completed sources and live unmarked processes fail closed. The supervisor forwards the
same option and logs the parent path. The pre-checkpoint approved-v2 failure is
deliberately ineligible. Resume/runner/authority verification passed 32 focused tests;
the supervisor suite passed 5 tests.

The weekly RL bridge now uses exact observed execution and endpoint prices rather than
demanding unused intervening daily marks. It never fills a price and still fails on a
missing required endpoint. A point-in-time trailing 20-session observed-price screen
excludes a security after a gap becomes known until the window recovers, recording
per-signal counts, missing dates and reasons. A future-panel completeness screen was
rejected as lookahead. Twenty-nine focused RL/authority tests passed. The approval
record and decision log document that no complete RL candidate or holdout result
informed the correction. Prepared-v8/protocol-v9 were generated immutably over the
unchanged canonical vintage; the fresh score-blind capability evidence and approved-v3
required before relaunch are recorded below.

The final replacement gate used prepared-v8/protocol-v9 and the separate immutable v3
approval amendment. The 400-security/full-history smoke passed all seven arms at
1,112,000 fit rows: DQN 34.14s CPU, PPO 36.51s CPU, Elastic Net 10.01s CPU,
HistGBT 22.48s CPU, XGBoost 11.93s CUDA, LSTM 252.14s CUDA and Transformer
319.64s CUDA. The full suite passed 224 tests with three expected non-interactive
notebook display warnings. Score-blind calibration selected the complete deadline tier
at 6,625.14 estimated seconds under the six-hour search ceiling. Capability gate v3
passed every declared arm. `expanded_closeout_approved_v3.yaml` is authority-verified
with protocol SHA-256
`d0d4c9c0454924a342839d5c88ae67bd338003cc88f4e2155bb1886d4189e4b5`.
No replacement score-bearing run or holdout evaluation was launched during this gate.

### 27 September 2026 — explicit neutral IC for constant cross-sections

The first approved-v3 attempt was stopped during the first outer fold after SciPy
reported `ConstantInputWarning` for constant daily prediction cross-sections. The
existing calculation already converted the resulting undefined Spearman statistic to
neutral IC `0.0`; RMSE remained defined. The runner now detects constant actual or
predicted cross-sections before calling SciPy, records the same explicit neutral IC and
continues to calculate RMSE normally. This removes ambiguous runtime warnings without
changing the declared selection objective or numeric fallback. The interrupted run
`20260927T123550Z-a6e5c43f` remains immutable and did not open the holdout. It cannot be
resumed after this source change. Two constant-actual/prediction regression cases were
added; focused scoring/resume tests passed 13 tests and the full suite passed 226 tests
with only three expected non-interactive notebook display warnings.

### 27 September 2026 — new-user guide and temporary workspace policy

Reworked the repository README into an end-to-end fresh-clone guide covering environment
creation, dependency groups, CUDA verification, tests, offline smoke, the explicitly
labelled Slice 1 data-download/validation step, Slice 1 execution, the pinned-bundle
precondition for approved expanded execution, supervision/resume, zero-argument report
and dashboard generation, and the educational notebook. Concrete generated run/report
directory names were removed. The guide records the genuine distribution gap: the
Git-ignored hash-pinned expanded evidence bundle is not recreated by the Slice 1
downloader and currently has no public one-command downloader.

Established `.tmp/` as the only new repository-local disposable workspace, documented
safe and prohibited cleanup boundaries, and added matching repository-agent rules.
Legacy root-level pytest/tool scratch remains ignored only for a one-time audited cleanup;
new tools and tests must use `.tmp/<tool>/<task>/`. README generated-ID scans and all
Markdown documentation links passed; `git diff --check` reported no content errors.

### 27 September 2026 — controlled marginal-effect evidence amendment

Stopped approved-v3 attempt `20260927T124032Z-bff62712` during its second LSTM
candidate, before family locks or holdout access, after determining that two
simultaneous-random candidates per family could not support held-constant marginal-
effect questions. The supervisor recorded a terminal nonzero exit and the partial run
and logs remain unchanged.

Added a deterministic metadata-driven controlled one-factor design over every
registered component search dimension. The authoritative runner now persists a
protocol-bound sensitivity design plus separate proposal and fit ledgers, including
conditional matched references, parameter levels, held-constant fields, selection-
eligibility flags, reduced fidelity, fold/seed/scenario identity, complete/failed
status, partial-cell counts, exception details, traceback and available telemetry.
Sensitivity trials use only declared inner data and are structurally excluded from
HPO ranking and family locks. Checkpoint continuation covers completed supervised and
RL sensitivity cells.

Prepared-v9 and approval amendment v4 declare one inner fold per outer fold, seed 41,
two deep epochs/patience one and 1,000 RL steps. Score-blind calibration counts 56
unique sensitivity configurations and estimates 4,305.08 seconds of sensitivity work
plus 6,625.14 seconds for the deadline HPO matrix, or 10,930.23 seconds before final
fits, holdout, reporting and retry reserve. The fresh full suite passed 235 tests with
one Windows symlink skip and three expected non-interactive notebook warnings. The
final v4 capability gate uses those fresh synthetic receipts and the unchanged
production-shaped real-data bridge evidence. Approved-v4 authority verification passed
with protocol SHA-256
`2b6095fbd48cb93a99e270292292dca1ceaf60b3c4c288f6e714339a66532678`.

Before the final gate, ordinary HPO failure records were brought to parity with the
controlled-sensitivity contract: supervised and RL terminal failures now retain the
exact failed fold/seed/scenario, completed and expected cells, elapsed resource time,
exception type and traceback, while successful partial cells remain in the fit ledger.
The post-change full suite again passed 235 tests with one platform skip and three
expected non-interactive notebook warnings; fresh synthetic receipts were issued before
the final approved-v4 hash above.

### 27 September 2026 — local-time human-readable supervisor logs

Removed the supervisor's extra UTC prefix from human-readable `stdout.log`,
`stderr.log` and concise terminal messages. Existing child application timestamps are
retained in operator-local time and normalised to whole-second precision; lines without
a child timestamp receive the current operator-local timestamp. Structured
`events.jsonl` and `metrics.jsonl` continue to record `timestamp_utc` for unambiguous
machine correlation. This is an operational-presentation change only and does not read
outcomes or change research execution. The focused supervisor suite passed 6 tests,
including an exact regression for the reported duplicate-timestamp example. The full
suite then passed 235 tests with one platform skip and three expected non-interactive
notebook warnings. Fresh v9 synthetic receipts and capability gate
`capability_gate_sensitivity_v4_final3.json` were issued, and approved-v4 was
authority-verified at the protocol hash recorded above. No production study was
launched.

### 28 September 2026 — linear RL evidence capture and visible sensitivity progress

At the human's direction, stopped approved-v4 attempt
`20260927T192408Z-b32c71ba` after the live heartbeat and immutable ledgers established
that it was active but increasingly slow. It had completed 47 of 220 controlled-
sensitivity cells with no recorded cell failure, had not sealed family locks and had
not opened the holdout. The supervisor recorded the forced child exit; the partial run
and operational logs were not edited. Because execution code changed, its checkpoints
are deliberately not resumable.

Removed the quadratic RL checkpoint path that reread and decoded all accumulated
`rl_price_eligibility.jsonl` and `rl_price_exclusions.jsonl` rows after every cell.
`_rl_trial_cell` now returns the exact rows it has just appended and callers store those
rows directly in the cell checkpoint. Immutable causal episode inputs are cached by
fit dates, score dates, risk scenario and pinned source/protocol hashes so architecture
candidates train independently on the same materialised dataset without repeatedly
rebuilding it. Each candidate still writes its own eligibility/exclusion evidence.

Added score-free controlled-sensitivity start, completion and failure messages with
candidate/cell counters, arm, fold, risk scenario, trial ID and elapsed time. The
supervisor now includes these messages in its concise terminal filter. The expanded
focused suite passed 28 tests; the full suite passed 236 tests with one Windows skip
and three expected headless-notebook warnings. Fresh v10 synthetic receipts and
`capability_gate_sensitivity_v5.json` passed all seven arms. Approved-v5 was generated
separately from immutable approved-v4 and authority-verified at protocol SHA-256
`231a824aa1c1167f081408972772c587d60ccbd9b7e5688ee8d037869c2ef84b`.
The fresh supervised attempt launched as run `20260927T212513Z-49234dd3` with operational
label `expanded-closeout-v5-attempt-1`; the terminal and detailed log both displayed
the first score-free `candidate=1/220 cells=0/220` controlled-sensitivity status.

### 28 September 2026 — bounded RL episode reuse after measured outer-fold cycle

The lineage continuation `20260927T231630Z-67e4fafb` was deliberately paused after
124 of 220 controlled-sensitivity cells had completed. All 124 cells have immutable
checkpoints; the active 125th LSTM sensitivity candidate had not completed, and no
ordinary HPO estimate, family lock or holdout access had occurred. The supervisor
recorded the forced worker exit. The source run and its ledgers were not edited.

Measured evidence separated model work from orchestration: the DQN/PPO backends needed
about one second for their reduced-fidelity training, while uncached candidates needed
about one minute because the same verified fold/scenario episodes were reconstructed
from the expanded feature, causal-output and price views. The central runner now uses
a caller-scoped episode cache keyed by exact fit dates, score dates and risk scenario.
Controlled sensitivity shares it across the DQN and PPO arms for one outer fold;
ordinary RL HPO scopes it to one risk scenario; the descriptive holdout shares it
across policy families. Candidate policies, seeds, training state, metrics, artefacts
and evidence rows remain independent. The dominant fold price dictionaries and calendar
are interned across risk scenarios, so reuse does not recreate the previous unbounded
multi-fold memory retention.

The change does not alter the approved protocol, candidate identities, parameter
values, fidelity, model outputs or selection rules. Continuation must explicitly record
authorised code drift and reuse every hash-verified completed cell. Focused controlled-
sensitivity/RL/resume verification passed 20 tests; the cache-specific suite passed 6;
the full suite passed 240 tests with one Windows platform skip and three expected
headless-notebook display warnings.

### 28 September 2026 — read-only terminal study dashboard

Added `scripts/watch-study.ps1`, a read-only PowerShell dashboard outside the
code-state-bound Python execution path. It automatically selects the latest operational
attempt or accepts an explicit `-LogDir`, and presents an ASCII header, current
candidate/trial/cell state, live candidate elapsed time, local progress and heartbeat
timestamps, total run time, process/system CPU, process RSS, available RAM, GPU load
and VRAM. It reads only supervisor logs and does not signal or inspect research
outcomes. `Ctrl+C` therefore closes the dashboard without stopping the study.

One-shot rendering was verified against active continuation
`expanded-closeout-v5-resume-3`: it resolved the current Transformer sensitivity
candidate and rendered both progress and resource fields. Redirected-output screen
clearing and PowerShell's automatic ISO-date conversion were covered by the same live
snapshot check. The monitoring run remained active and its Python code-state hash was
unchanged.

### 28 September 2026 — terminal dashboard phase and progression refinement

Refined `scripts/watch-study.ps1` without changing the active study's Python execution
path. The summary now presents the broad study phase separately from its activity,
combines process RSS and available host memory on one RAM line, and renders the latest
event as vertical fields. A borderless progression table reads the current run's
protocol, controlled-sensitivity/HPO trial ledgers and metadata-driven arm registry to
show completed/planned candidates plus cumulative runtime for each outer fold.

The observed-duration ETA uses only same-arm, same-fold completed-candidate medians.
Remaining folds without an observation are excluded and explicitly make the estimate
a partial lower bound, preventing resumed near-zero checkpoint timings from being
extrapolated into untouched work. PowerShell parsing and a live one-shot render against
`expanded-closeout-v5-resume-3` passed; the snapshot showed phase `HPO`, activity
`Controlled sensitivity`, all seven registry arms, both outer-fold columns and a
vertically rendered current event. The research worker remained active throughout.

The operator layout was subsequently consolidated into four ordered sections: `Run
status`, `Observed-duration ETA`, `Progress by model arm`, and `System status`. Run
status now owns the latest event fields, eliminating duplicate trial and timestamp
lines; the internal heartbeat was removed from the display. Fold headers are explicit
`HPO_outer_1`/`HPO_outer_2`, and metadata-driven `Final_training` and `Holdout` columns
track expected cells from the immutable holdout-fit ledger without modifying the
runner. The live one-shot check passed after the active study transitioned from
controlled sensitivity into nested HPO.

After observing the nested-selection display, replaced the aggregate arm progress and
runtime-only fold cells with per-fold candidate progression. Each HPO fold now renders
`completed/planned` first and adds its cumulative runtime in parentheses only after
work completes, while untouched folds explicitly show `0/planned`. A live render
during the first LSTM outer-fold search verified classical arms at `2/2`, LSTM at
`1/2`, future supervised folds at `0/2`, and RL folds at `0/6`.

The fold display was then corrected to reflect the full nested structure without
creating an unreadably wide table. Two stacked outer-fold tables now show `Inner 1`,
`Inner 2`, and `Outer eval` independently, followed by a compact final-evaluation
table. Progress is derived from `fit_ledger.jsonl` and `outer_fit_ledger.jsonl`, not
from inferred terminal state. A live check showed completed classical inner and outer
cells, both completed LSTM inner cells, and the still-running LSTM outer evaluation
separately and correctly.

Restored the completed controlled-sensitivity phase to the persistent dashboard view.
The previous phase switch retained all 220 immutable records but stopped rendering
them once selection HPO began. A compact, explicitly selection-ineligible table now
shows its total completion and per-arm counts/runtimes split across outer-history inner
fold 1, followed by the ordinary nested-HPO and final-evaluation tables.

Added concise educational objective subtitles beneath every progress-phase heading.
They distinguish selection-ineligible controlled sensitivity, nested inner-fold tuning
plus outer-window assessment, and locked final refitting plus descriptive holdout
reporting without reopening selection.

Replaced qualitative `pending` values in the final-evaluation table with protocol-known
completed/expected counts. Supervised arms now begin at `0/1`; DQN and PPO begin at
`0/3` because each has one final cell per declared risk scenario. `running` and
`waiting` remain secondary labels while a final fit/score call is in flight.

Replaced the misleading partial `Observed-duration ETA` with an overall remaining-time
estimate. The monitor verifies the protocol-pinned capability gate and its declared
score-blind calibration hash, counts remaining nested-HPO, outer-evaluation and combined
final-fit/holdout cells, and uses the larger of each arm's live median or calibrated
production cell duration. It refuses an overall ETA if that evidence is unavailable.
The display is rounded to five minutes to avoid false precision; a live verification
produced an approximately two-hour overall planning estimate rather than the former
five-minute measured lower bound.

Added an additive ETA reconciliation showing remaining inner HPO, outer evaluations,
and final fit/holdout separately (plus sensitivity while active). The component values
are rounded to five minutes and summed to the displayed total, making clear why the
backward-looking completed-cell runtimes in the progress tables do not themselves add
up to the forecast. A later live snapshot reflected study progress with a reconciled
`45m + 15m + 15m = 1h 15m` planning estimate.

### 28 September 2026 — verified completion takes precedence over a late process exit

Corrected a terminal-dashboard status conflict for run
`20260928T005423Z-0634efaf`. The authoritative runner logged completion and produced
`completion.json`; its referenced `audit.json` had status `passed` and a matching
SHA-256. Eight seconds later the Windows process returned `3221226505` (`0xC0000409`),
so the operational supervisor appended a failure event even though the immutable run
had already been sealed.

`scripts/watch-study.ps1` now verifies the completion status, audit status, run ID and
sealed audit hash before resolving the displayed state. Verified evidence reports
`COMPLETED`; a subsequent non-zero process exit is retained as a separate yellow
operational warning. Failures before a verified completion seal still report
`FAILED`. PowerShell parsing and a one-shot render against the affected attempt passed,
showing all study cells complete and the exit anomaly without misclassifying the
research result.

### 28 September 2026 — expanded report Windows path normalisation

The first read-only WP7 generation attempt against completed audited run
`20260928T005423Z-0634efaf` failed closed before writing a report. The runner had
recorded the canonical feature path in Windows form with backslashes, while the pinned
snapshot manifest used repository-portable forward slashes. Both records named the
same in-repository file and carried the same SHA-256, but the postprocessor compared
their raw strings and rejected the contract.

The postprocessor now resolves both representations through its existing repository-
containment validator before comparing paths; all hash, protocol and source checks are
unchanged. A Windows-separator regression was added to the separately pinned benchmark
contract test. The complete `tests/test_expanded_closeout.py` suite passed 7 tests.
The immutable run and its source manifests were not modified.

### 28 September 2026 — hash-verified assignment release bundle

Added `scripts/build_assignment_release.py` as a read-only packager for the generated
WP7 report, assignment evidence export, final evidence notebook, verified citation
register/BibTeX, dependency lock and licence. The packager verifies both report
manifests and every declared output hash before creating the ZIP; it never reads or
copies raw run artefacts. It writes a generated entry manifest inside the archive and
separate checksum and GitHub handoff files beside it.

The first bundle, `release/assignment-evidence-20260928T005423Z-0634efaf.zip`, contains
32 hash-declared source entries plus its bundle manifest. Independent archive testing
re-read every entry and reconciled its SHA-256 with the embedded manifest. The release
packager contract suite passed 2 tests. GitHub publication remains a human action; no
tag, release or public-access claim was created by this packaging step.

### 28 September 2026 — final evidence notebook publication contract

Strengthened `notebooks/final_evidence.ipynb` as the local precursor to the required
public notebook without publishing it or inventing a public notebook/release URL. All
cells now have stable unique IDs. The notebook names canonical run
`20260928T005423Z-0634efaf`, report directory
`reports/expanded_closeout/20260928T005423Z-0634efaf`, source revision
`e9b3e0cdf57c63c87a9dc881d58b1696f0472ba5`, and report-manifest SHA-256
`4144c9d1bd0c1637eb83021edd7d5359255aa56e73e13e49c5cd1f1d5fd1190a`.
Its setup cell verifies those identifiers and the exact eleven-table contract before
calling the existing expanded-report loader, which independently checks declared
table hashes and schemas.

Added separate local and future clean-runtime instructions. The known repository
remote is recorded as a location, not as a claim of public accessibility. Submission
tag, release-archive URL and archive-hash values remain explicit `NOT_PUBLISHED`
placeholders; opting into public bootstrap fails closed until they are replaced by
verified publication values. An opt-in bounded-demonstration cell invokes only
`python -m trading_pipeline.run --config ...`. It refuses execution without an
explicit in-repository config and states that the existing synthetic smoke config is
software verification rather than real-data research evidence.

Focused tests now require stable unique cell IDs, canonical run/revision/manifest
identifiers, publication placeholders, authoritative runner usage and fail-closed
public bootstrap. The existing synthetic standalone-report execution test uses an
explicit test-only noncanonical override. Verification passed with `3 passed` from
the final-notebook subset of `tests/test_dashboard.py`. A separate code-cell execution
against the actual canonical report loaded all eleven hash-verified tables and printed
the expected run ID and manifest SHA-256. The first plain Windows-console attempt
encountered only a CP1252 display encoding error while printing Unicode table borders;
rerunning that script with Python UTF-8 mode completed. A headless `nbconvert --execute`
Run All was initially blocked when the sandbox prevented Jupyter from applying its
Windows connection-file ACL; the approved local-kernel retry completed in 12 seconds
and wrote a disposable executed copy under `.tmp/`, which was removed after the check.
No report evidence or raw run artefact was modified.

### 28 September 2026 — generated factual assignment-report draft

Added `trading_pipeline.reporting.assignment_report`, a read-only report builder that
verifies the WP7 and assignment-export manifests and their declared output hashes
before rendering prose, tables and figures. It consumes only the generated reports
and the bounded citation register/BibTeX; it does not read or modify model artefacts,
fit models, tune parameters or select from the final holdout.

The generated draft is under
`reports/assignment/20260928T005423Z-0634efaf/v1/report/`. It covers the task input and
output contract, all seven arms, feature and point-in-time methodology, HPO chronology,
statistical-versus-economic objective mismatch, every arm/scenario holdout cell,
limitations and provenance. The holdout is labelled descriptive throughout. Personal
reflection, financial implications and knowledge gaps remain explicit student-authored
prompts, and the public notebook URL remains a fail-closed placeholder.

The build emitted a 3,175-word Markdown draft, a copied hash-identical bibliography,
three print-resolution PNG figures and `report_manifest.json` with exact input/output
hashes. The focused report contract suite passed 3 tests; Python compilation also
passed. No DOCX or PDF was created.
