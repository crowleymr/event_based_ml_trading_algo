# Implementation log

## 2026-09-20 — Unified evidence dashboard and CUDA telemetry diagnostic

- Preserved the user-owned `docs/planning/prompt-scratch_pad.md` edit and all existing
  immutable data, run and report artefacts.
- Added the common schema-versioned training trace/summary contract and explicit
  `cpu`/`cuda`/`auto` resolution with recorded fallback, determinism, package/CUDA and
  peak-memory evidence. Added a standalone tensor-placement preflight.
- Installed the optional official CUDA 12.8 PyTorch wheel locally; CUDA preflight placed
  the tensor on the detected RTX 4060 Ti.
- Ran the frozen-budget three-seed DQN diagnostic reproduction as
  `runs/rl/20260920T000735Z-eb113246`. Its persisted audit passed. Device timing and
  parity evidence is generated in `device_benchmark.parquet`; per-seed durations and
  devices are generated in `training_summary.parquet`. No timing or observed-test result
  was used for selection.
- Generated report schema v2 at
  `reports/20260912T071137Z-9899fd9a/v2` from the immutable Slice 1 run, its hash-verified
  external feature snapshot and the completed audited diagnostic RL run. The report
  emits paired CSV/Parquet evidence and field-level definitions, units, limitations and
  provenance without modifying source artefacts.
- Expanded the read-only Streamlit surface into seven evidence views. In-process page
  execution had no application exceptions; local health and page endpoints returned
  successfully. Streamlit did not invoke a training or report-generation path.
- Verification command: `.venv/Scripts/python -m pytest -q`; the complete suite passed.
  The canonical Slice 1 persisted-run audit passed, and the prior RL run and report v1
  matched their stored manifests/hashes.

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
