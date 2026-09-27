# Final Expanded Research Close-Out Plan

**Planning date:** 26 September 2026  
**Target:** freeze the codebase, research results, charts, dashboard and notebook by 11:00 am on 27 September 2026  
**Repository checkpoint:** clean `main` at `1af688e`; 74 tests passed on 26 September 2026  
**Usage envelope:** dynamically governed by the live five-hour and weekly limits, with a protected checkpoint reserve  
**Authority:** repository `AGENTS.md`, `docs/FSD_v2.md`, `docs/DECISIONS.md`, `docs/SLICE1_COMPLETION_REPORT.md`, `docs/planning/PHASE3_ARCHITECTURE_HPO_REPORTING_IMPLEMENTATION_PLAN.md`, and the user's 26 September 2026 direction

## 1. Outcome

Close the project with one reproducible, read-only evidence experience covering:

1. the largest successfully admitted US equity universe up to 500 securities;
2. comparable classical, deep supervised and reinforcement-learning model families;
3. architecture search and hyperparameter optimisation (HPO) for every model with tunable parameters;
4. causal access to the same point-in-time information contract, including eligible upstream model outputs;
5. conservative, balanced and aggressive risk scenarios reported without post-test preference selection;
6. a unified T+1, after-cost backtest test bench against the market benchmark;
7. a dashboard with pipeline and security drill-down, architecture/HPO/training evidence, and final comparisons; and
8. an executable narrative notebook that links each displayed result to the relevant code and immutable artefacts.

This plan supersedes the scope-freeze priority in `docs/close_out/SUBMISSION_CLOSEOUT_PLAN.md` because the human has now explicitly requested universe and model-family expansion. It does **not** supersede evidence immutability, point-in-time controls, final-test restrictions, or the single authoritative execution path.

## 2. Current-state review

### Failed-run recovery checkpoint — 27 September 2026

Approved-v2 run `20260927T075513Z-3a5052c9` failed closed after completing the
supervised outer-fold work and before any complete RL candidate or holdout access.
The immediate cause was an RL bridge requirement for every intervening daily price,
although the declared reward is an exact weekly execution-to-endpoint return. The run
and its failure evidence remain immutable and are not eligible for resume because they
predate code-state binding and per-cell prediction checkpoints.

The acceptable unattended execution window is now **6–8 hours**. The initial
production-scale deep smoke exposed a score-blind engineering bottleneck: the sampled
64-row batches left the RTX 4060 Ti transfer-bound at roughly 30–50% utilisation. The
declared deep search space now tests 512- and 1,024-row batches; 13 focused deep tests
pass and bounded transfers remain well below 16 GiB VRAM. Before an evening launch,
run one fresh representative all-arm smoke using this final code, calibrate against an
eight-hour ceiling with explicit final-fit/holdout reserve, then approve and start the
authoritative run. Do not reuse the interrupted timing attempt.

The intended operator surface after completion remains:

```powershell
.venv\Scripts\python.exe -m trading_pipeline.reporting.expanded_closeout
.venv\Scripts\python.exe -m trading_pipeline.dashboard
```

The dashboard opens the latest verified run; historical compatible runs appear only in
the monitoring tab with generated implementation and provenance labels.

### Live critical-path handoff — 27 September 2026

This checklist is the current operator/alternative-harness handoff. Update it whenever
an acceptance gate changes; generated artefacts and tests remain the evidence authority.

- [x] Freeze 500 dated candidates while preserving the original 100.
- [x] Seal admission v3: 497 admitted, three explicitly excluded for fewer than 260 sessions.
- [x] Derive the immutable admitted universe and security master.
- [x] Materialise canonical expanded vintage v1: 496 canonical securities after one explicit
  raw-quality exclusion (HUBB), 1,416,531 feature rows and 1,414,051 labelled rows.
- [x] Generate pinned snapshot, exposure, nested inner/outer-window and sealed-holdout manifests.
- [x] Generate a self-hashed seven-arm prepared study that remains execution-forbidden.
- [x] Implement and test the causal OOF model-output store/parity API, including lineage, cutoff, coverage-mask and representation checks (WP5 contract only).
- [x] Implement and test the fold adapter for LSTM and causal Transformer candidates (synthetic contract only).
- [x] Implement and test the verified-authority DQN/PPO fold bridge across all three risk scenarios (synthetic contract only).
- [x] Integrate classical, deep and RL adapters into the authoritative central study runner with common ledgers and fail-closed input contracts; focused supervised/RL/authority tests pass. End-to-end real-data smoke remains part of the capability gate.
- [x] Acquire a complete immutable SPY benchmark vintage (2,946 daily rows) and implement protocol binding so reporting consumes the separate pinned benchmark rather than assuming SPY is in the investable universe.
- [x] Make experiment membership YAML-driven and component implementations discoverable only from trusted model-package folders through module metadata. Residual execution routing by hard-coded component ID is being removed before approval.
- [x] Implement the fail-closed read-only expanded evidence postprocessor for all eleven WP7 tables, including complete inner/outer/seed/risk matrices and security contribution reconciliation.
- [x] Generate real causal cross-fit predictions across 46 chronological windows, materialise the pinned store, bind its 19-column view to `F1_CAUSAL_STACK_V1`, and audit cross-view parity. The sealed manifest records `protected_labels_used_for_fit: false`.
- [x] Freeze prepared protocol v4 against protocol-manifest set v5, with the causal stack and separate SPY benchmark paths and hashes embedded in the self-hashed, execution-forbidden study.
- [x] Generate final score-blind calibration and the complete declared-arm capability
  gate. The 400-security/full-history smoke passed all seven arms; the final 215-test
  suite produced fresh per-arm receipts. A six-hour search ceiling selected the complete
  two-proposal deadline tier at 6,378 seconds without opening outcomes.
- [x] Finalise approved self-hashed protocol `expanded_closeout_approved_v2.yaml`
  (`71e2126ac30cd500773a1ff590c59a5987ac44fc5652a88b64039828541b86f0`);
  all seven arms, two outer folds, two inner folds, one seed and three RL risk scenarios
  remain mandatory. Partial-family approval remains impossible.
- [x] Freeze the RL method correction in prepared-v8/protocol-v9 and complete a fresh
  score-blind gate: all seven production-shaped arms passed at 1,112,000 fit rows,
  224 tests passed, the six-hour calibration selected the complete deadline tier at
  6,625 seconds, and approved-v3 was authority-verified at protocol SHA-256
  `d0d4c9c0454924a342839d5c88ae67bd338003cc88f4e2155bb1886d4189e4b5`.
- [ ] Execute the deadline-feasible inner-HPO and outer-evaluation matrix; seal family locks before holdout access. The deadline-tier approved-v2 attempt failed before a complete RL candidate and before holdout access. Its completed supervised evidence is historical only because it predates the resume contract. Approved-v3 binds point-in-time observed-history eligibility, exact weekly endpoint valuation and immutable per-cell continuation.
  The first approved-v3 attempt `20260927T123550Z-a6e5c43f` was operator-stopped in
  outer fold 1 to replace SciPy's constant-input warning with an explicit, numerically
  identical neutral-IC guard. It did not open holdout and is not resumable after the
  source change. The post-fix full suite passes 226 tests; relaunch must be fresh.
- [x] Stop approved-v3 attempt `20260927T124032Z-bff62712` before family locks or
  holdout access after identifying that its two simultaneous-random candidates per
  family preserve parameters but cannot answer controlled marginal-effect questions.
  The supervisor recorded the terminal failure and partial artefacts remain immutable.
- [x] Add protocol-declared, selection-ineligible controlled sensitivity collection:
  one reference plus one valid single-factor contrast for every registered search
  dimension, conditional matched references, one inner fold per outer fold, seed 41,
  two deep epochs and 1,000 RL steps. Complete and failed cells retain comparison,
  fidelity, exception, traceback, partial-completion and telemetry evidence.
- [x] Freeze prepared-v9 and approval amendment v4. Score-blind calibration covers
  56 unique sensitivity configurations and estimates 4,305 seconds for sensitivity
  plus 6,625 seconds for the existing deadline HPO matrix (10,930 seconds total before
  final fits/reporting), retaining the six-hour gate. The fresh full suite passed
  235 tests with one platform skip; approved-v4 is authority-verified at protocol
  SHA-256 `2b6095fbd48cb93a99e270292292dca1ceaf60b3c4c288f6e714339a66532678`.
- [ ] Execute the descriptive holdout once, followed by the unified T+1/cost-reconciled model × risk test bench.
- [ ] Generate supervised model-conditioned frontiers and all-model realised risk-return curves.
- [x] Implement the WP7 report/dashboard/notebook consumers. Both report generation and
  dashboard launch now work with zero arguments: the report generator selects the newest
  completed audited expanded run, and `python -m trading_pipeline.dashboard` opens the newest
  verified report. The main views show that run only; the monitoring tab compares compatible
  reports with generated run labels, implementation scope (arm/component/device), timestamps,
  protocol and code provenance. Focused verification: 19 tests passed on 27 September.
- [x] Add and verify the read-only supervisor/heartbeat required for the unattended
  study. It must wrap the authoritative `trading_pipeline.run` command without changing
  research behaviour; persist stdout/stderr, lifecycle/warning/error events and terminal
  status; sample process-tree RSS, RAM, CPU load/utilisation, GPU utilisation and VRAM;
  degrade explicitly on CPU-only systems; and retain logs after interruption/failure.
- [x] Normalise operator-facing supervisor output to a single local timestamp at
  whole-second precision. Human-readable stdout/stderr and concise terminal status no
  longer contain a duplicate GMT prefix; structured JSONL retains UTC for machine
  correlation. Exact-format regression coverage passes.
- [x] Add a fail-closed authoritative resume facility before the next launch. Resume
  must create an immutable derived attempt with explicit parent lineage; bind the exact
  approved protocol, inputs and code state; reuse only hash-verified complete cells and
  checkpointed predictions; retain every earlier failure; reject tampering and protocol
  drift; and preserve the family-lock/holdout boundary. The failed v2 attempt predates
  code-state and prediction checkpoints and is therefore not eligible for silent reuse.
  `trading_pipeline.run --study <approved> --resume-from <failed-run-dir>` and the
  supervisor equivalent now create a new lineage-linked run, verify every source
  checkpoint/artefact before reuse, persist per-cell predictions, reject protocol/input/
  code drift, and retain the source unchanged. Focused resume/RL/authority verification
  and the full replacement capability gate passed.
- [ ] Generate WP7 report tables from the completed authoritative run, execute the final
  notebook, and run the WP9 audit/completeness checks. Extend the read-only report and
  dashboard at that stage with controlled marginal-effect and candidate-failure views
  sourced from `sensitivity_design.json`, `sensitivity_trial_ledger.jsonl`,
  `sensitivity_fit_ledger.jsonl`, `trial_ledger.jsonl` and `fit_ledger.jsonl`; do not
  alter or recompute the raw run evidence.
- [ ] Freeze and commit the exact code revision, run/report IDs, hashes and operator manual.

Remaining critical-path order: launch approved-v4 through the supervisor; run controlled
sensitivity, inner HPO
and outer evaluation with family locks; open the descriptive holdout once; then generate
the report, execute the notebook and complete the final audit/freeze. If that new run
fails after a verified cell, continue it only through a new `--resume-from` attempt.

Real-data causal OOF materialisation is complete (46 chronological windows, with protected
labels excluded from fitting). Deep and RL bridge results remain engineering/capability
evidence only. No completed expanded HPO/outer-evaluation or holdout result exists. Prepared
protocol v2 cleared its representative smoke, calibration and capability gate,
then failed as recorded above. Holdout access remains closed. Approved-v4 now supersedes
v3 after its fresh verification gate; it retains the unchanged canonical
data/window/benchmark/causal-stack sources and adds the separately approved controlled-
sensitivity evidence contract. The transient prepared-v7/protocol-v8 staging pair is not an
execution authority and was superseded before capability evidence was accepted.

The immutable v2 and v3 research approval records are separate from this live checklist
at `docs/planning/EXPANDED_STUDY_APPROVAL.md` and
`docs/planning/EXPANDED_STUDY_APPROVAL_V3.md` and
`docs/planning/EXPANDED_STUDY_APPROVAL_V4.md`, so operational updates cannot invalidate
a frozen protocol. Data artefacts and feature columns are manifest-driven. Trusted
model-package discovery and YAML-driven arm membership now remove the Python roster;
the central runner now routes from trusted component capability metadata rather than a
hard-coded Python model roster.

### Execution checkpoint — 26 September 2026, evening

This checkpoint records implementation status, not research completion. It does not
replace generated run evidence or the acceptance criteria below.

| Work package | Executed state | Remaining gate |
|---|---|---|
| WP0/WP2 authority | `trading_pipeline.run --study` now validates an approved, self-hashed protocol and six pinned manifests before any research I/O. Legacy `--config` behaviour is regression-tested. | The checked-in expanded study is deliberately unresolved. Central score-bearing orchestration, append-only trial/fit ledgers, outer selection lock and immutable run/audit output are not yet implemented. |
| WP1 universe | Final-provenance admission `sp500-20260926-v3` is sealed: 497 of 500 candidates admitted, 3 explicitly excluded for fewer than 260 market sessions, 497 F0-ready and 487 with both raw F1 SEC concepts. The derived admitted-universe and security-master hashes are frozen under `data/expanded_2026-09-26/admitted_v3`. | Materialise the canonical expanded market/fundamental/feature vintage from the admission raw artefacts and pin it into the study authority. Stopped v1/v2 attempts remain preliminary only and must not be cited. |
| WP3 deep supervised | Causal LSTM/Transformer adapters and an evaluator-fold execution bridge have architecture parameters, train-only scaling, distinct stopping/scoring windows, canonical predictions and telemetry tests. | Invoke the bridge only from the future central study orchestrator; no real-data deep HPO result exists yet. |
| WP4 RL | Registered DQN/PPO backends, PPO parameter telemetry and an actual synthetic PPO fit are tested. | Direct real-data calls remain intentionally forbidden until the central runner owns authority, ledgers, persistence and audit. No real PPO comparison exists yet. |
| WP7 dashboard | The read-only dashboard now leads with an executive graphical story, shows validation/test side by side, uses a run dropdown and keeps evidence tables behind drill-down controls. It passes headless AppTest and is runnable on the existing audited report. | Expanded HPO/frontier/risk/test-bench views remain correctly empty until WP6 emits their hash-declared tables. |
| WP8 notebook | The educational notebook explains concepts, methodology, metric direction, evidence limits and code links; all relative links resolve and clean `nbconvert --execute` succeeds on the existing audited report. | Regenerate and re-execute after WP6/WP7 evidence exists; public URL/licensing remains a recorded human publication decision. |
| WP9 verification | Full automated suite passes 116 tests; the legacy route and fail-closed expanded authority boundaries are included. | Repeat full suite, run audit, report regeneration, notebook execution and matrix-completeness checks after the final expanded run. |

The immediate critical path is therefore: finish and seal v2 admission; generate the
fresh-vintage, exposure, inner/outer-window and holdout manifests; resolve and hash the
study protocol; implement the central study orchestrator; execute the frozen feasible
budget; then regenerate WP7/WP8 evidence and perform WP9. Missing score-bearing results
must continue to display as unavailable rather than being inferred from synthetic tests.

| Area | Verified state | Close-out gap |
|---|---|---|
| Baseline | Completed E0-E7/B0 evidence, fixed 100-stock universe, audited immutable run | Preserve it as the fallback and legacy comparison; never rewrite it |
| Data | 100 configured stocks; cached data for those stocks plus SPY; ingestion loops over the configured file | No 500-name candidate snapshot, admission ledger, resumable partial preflight, or deterministic exclusion record |
| Classical supervised | Elastic Net, HistGBT and XGBoost implementations exist | Phase 3 contracts are research-disabled; the authoritative runner still uses the legacy fixed-grid path |
| Deep supervised | None | LSTM and Transformer sequence views, adapters, architecture spaces, training telemetry and tests are absent |
| RL | DQN diagnostic runner exists; DQN and categorical PPO contracts expose architecture spaces | PPO is research-disabled and not executed; the live runner is DQN-specific |
| HPO | Deterministic grid/random proposals, nested temporal split primitives and immutable trial ledgers exist | No real-data evaluator, fit ledger, objective aggregation, promotion, family selection lock or runner integration |
| Information parity | Supervised models use F0/F1; RL uses aggregates and frozen E1-E4 sleeve/model outputs | No general causal model-output feature-store contract and no cross-family parity audit |
| Backtest | Canonical T+1, cost-reconciled supervised/portfolio path; separate RL selector path | No unified selected-family test bench or complete risk-scenario matrix |
| Reporting | Strong versioned reports: coverage, splits, features, predictions, holdings, trades, contributions, training traces and SPY-relative evidence | Missing per-stock raw-to-final pipeline funnel and completed architecture/HPO trial views |
| Dashboard | Read-only seven-view Streamlit app | No ticker drill-down, pipeline-stage contribution view, architecture/HPO leaderboard, or unified final-family chart |
| Notebook | Minimal three-cell report explorer; public status incomplete | No end-to-end story, source links, dashboard launch/inline views, clean execution proof or public URL |

## 3. Research-integrity boundary

### 3.1 Preserve completed Slice 1

- Do not modify `configs/universe.txt`, old data snapshots, completed runs, reports, E IDs, B0, the original E5 eligibility set, or its selection record.
- Treat the 21 May 2024 to 11 September 2026 legacy final test as observed and descriptive.
- Never use legacy final-test outcomes to define candidate families, architecture ranges, parameters, objectives, budgets, stopping rules or rejection decisions.
- Show legacy and expanded-study evidence side by side only when clearly labelled protocol-incompatible; never pool or rank across compatibility keys.

### 3.2 What can be claimed in five days

Only about two trading weeks follow the observed legacy test. That is not a meaningful new confirmatory period for a five-day horizon. The expanded study must therefore be labelled:

> exploratory nested walk-forward model comparison with a predeclared descriptive close-out holdout; not independent confirmation of persistent alpha or model superiority.

The expanded universe and model outputs will be new, but the historical market regimes overlap already-observed calendar periods. A genuinely untouched confirmatory claim is deferred until a sufficiently long future period has accrued.

### 3.3 Single execution and evidence path

- `python -m trading_pipeline.run --config <config>` remains the authoritative route.
- The study/evaluator may be modular internally but must be called by `trading_pipeline.run`, not by a second research script or notebook.
- Reports, dashboard and notebook read completed audited artefacts. They do not train, tune, select, repair or reconstruct results.
- Every reported number and chart must carry or inherit run ID, source paths/hashes, code revision, protocol/compatibility key and generation time.

## 4. Close-out protocol defaults

These defaults avoid taste questions blocking execution. They must be frozen in the study YAML before score-bearing runs. Changes after seeing outcomes create a new study ID.

### 4.1 Universe admission

1. Create a dated candidate snapshot of up to 500 current US large-cap common-stock listings from one documented source.
2. Retain stable source order and stable internal IDs; preserve all original 100 candidates.
3. Preflight each candidate independently for:
   - exact SEC ticker-to-CIK mapping;
   - sufficient Yahoo daily adjusted-price history;
   - required identifier and schema validity;
   - minimum feature warm-up and target availability;
   - EPS and Net Income coverage, recorded but not necessarily required for F0 admission.
4. Admit every candidate that passes the frozen rules until the 500-security cap is reached.
5. Record every requested, admitted and excluded security, the exact reason, source/as-of date and hashes in `universe_admission.parquet/json`.
6. Never silently substitute a ticker/issuer, alternate fundamental tag, price, estimator or data source.
7. Refresh and extend the market/fundamental history through the latest available date in the new vintage; if a source adds no valid rows or is unavailable, retain the last valid immutable snapshot and record the attempt.

The achieved universe is `N <= 500`; no plan or report may claim 500 until the admission artefact proves it.

Extend the existing data sources and framework only. More robust commercial or historical-membership sources are outside the current close-out scope. Retain and quantify the accepted current-survivor limitation.

### 4.2 Common information contract

“Same information” means the same causal, point-in-time source snapshot and cutoff, not an identical tensor shape for unlike algorithms.

Create one versioned feature-view contract containing:

- canonical F0/F1 columns, availability dates, eligibility masks and hashes;
- train-only imputation/normalisation metadata;
- a tabular view for Elastic Net/HistGBT/XGBoost;
- a causal sequence view for LSTM/Transformer with lookback, padding and masks;
- an RL state view derived from the same information set and frozen portfolio sleeves;
- optional upstream prediction features generated strictly out of fold by an earlier declared layer; and
- upstream component ID, fit hash, cutoff, fold, seed and coverage for every model-output feature.

Final/outer/test predictions may never become training features. If upstream model outputs are used by a downstream arm, their generation and selection cost counts inside that arm's declared budget. The parity audit must distinguish shared raw information from model-specific representation.

### 4.3 Candidate matrix

Run matched F0 and F1 arms where the model supports them:

| Class | Families | Tuned dimensions include |
|---|---|---|
| Classical supervised | Elastic Net, HistGBT, XGBoost | regularisation, capacity, learning rate, iteration budget and family-specific controls |
| Deep supervised | LSTM, causal Transformer encoder | lookback; depth; width/hidden size; heads where applicable; feed-forward size; dropout; activation; optimiser; learning rate; batch size |
| Reinforcement learning | DQN, categorical PPO sleeve selector | network depth/width; activation; learning rate; rollout/replay settings; discounting; exploration/entropy; PPO policy/value sharing |
| Fixed baselines | momentum E0, SPY B0 and other parameter-free declared baselines | no artificial HPO; frozen rules only |

Categorical PPO must use the same discrete sleeve-selector MDP as DQN. Continuous control of hundreds of stock weights is a different research programme and is out of scope for this close-out.

### 4.4 Temporal evaluation and HPO

1. Freeze a new data-vintage manifest and exposure ledger.
2. Generate chronological, session-grouped, purged and embargoed nested walk-forward manifests.
3. Within each inner fold keep fit, early-stopping and scoring windows distinct.
4. Use inner folds only for architecture/HPO decisions.
5. Evaluate the locked family procedure on outer folds.
6. After all family choices, seeds and risk definitions are locked, evaluate once on the descriptive close-out holdout.
7. Retain every proposed, failed, pruned and completed trial; do not omit bad outcomes.

Default objectives:

- supervised primary: mean daily cross-sectional Spearman IC;
- supervised tie-breaks: RMSE, then lower complexity/latency within practical equivalence;
- RL primary: after-cost certainty-equivalent weekly return for the predeclared risk coefficient;
- RL diagnostics: return, volatility, drawdown, Sharpe, turnover, action stability and seed dispersion;
- final comparison: report the complete metric vector and Pareto views; do not choose a headline winner on holdout Sharpe.

### 4.5 Bounded search and deadline control

First run one timed, score-blind calibration fit per family on representative folds. Select one of these complete budget tiers using wall-clock estimates **before** opening result rankings:

| Tier | Classical family | Deep supervised family | RL family | Final seeds |
|---|---:|---:|---:|---:|
| Full | 24 proposals | 18 multi-fidelity proposals | 12 proposals | 3 |
| Reduced | 16 proposals | 12 multi-fidelity proposals | 8 proposals | 3 |
| Minimum defensible | 8 proposals | 8 multi-fidelity proposals | 6 proposals | 3 |

- Architecture is part of each proposal, not a separate informal experiment.
- Use deterministic grid/random or successive halving supported by the audited evaluator.
- Apply the chosen tier uniformly within comparable families.
- Record parameter count, fit time, inference latency, peak memory, device, epochs/boosting rounds, RL environment steps and gradient updates.
- Do not reduce a single poor-performing family's budget after viewing scores.

### 4.6 Risk scenarios

Run all selected model outputs through three frozen downstream scenarios. Initial numeric defaults are:

| Scenario | Gross exposure cap | Annualised volatility target | Max position | Cash permitted |
|---|---:|---:|---:|---|
| Conservative | 0.60 | 8% | 5% | Yes |
| Balanced | 0.80 | 12% | 10% | Yes |
| Aggressive | 1.00 | 16% | 20% | Yes |

Use the same predictions, eligible dates, transaction costs and T+1 execution. If a target cannot be achieved without leverage, retain cash; do not add leverage during close-out. Report all scenarios for every relevant final model. The labels are communication aids, not claims that these thresholds match a particular investor.

For RL, use three predeclared risk-aversion coefficients aligned with these scenarios and train distinct arms. Do not infer a universally “best” risk appetite from the holdout.

### 4.7 Risk–return frontier and profile visualisations

Produce two related but explicitly different visualisations. They must not be presented as interchangeable.

#### Model-conditioned efficient frontiers

Generate a separate classical efficient frontier for each eligible supervised predictor that produces a cross-sectional expected-return estimate, including Elastic Net, HistGBT, XGBoost, LSTM and Transformer finalists.

For comparability:

- use each model's point-in-time predicted-return vector as the expected-return input;
- use one shared, declared, point-in-time covariance estimator and estimation window;
- use identical eligible securities, long-only constraints, position caps, cash treatment and transaction-cost assumptions;
- sweep a fixed target-return or risk-aversion grid rather than choosing grid values from holdout outcomes;
- identify the conservative, balanced and aggressive portfolio on each model's frontier;
- retain infeasible targets as explicit missing/infeasible results rather than silently changing constraints; and
- persist the rebalance date, input hashes, optimiser status, expected return, expected volatility and portfolio weights for every frontier point.

Because the frontier changes through time, the dashboard must support a rebalance-date view. Any aggregate frontier summary must use a declared repeatable aggregation rule and must not combine weights or expected-risk points from different dates as though they were one contemporaneous frontier.

DQN and categorical PPO do not directly produce security-level expected-return vectors in the frozen sleeve-selector MDP. They must therefore **not** be assigned a classical stock-level efficient frontier unless their task and output contract are materially changed under a new study. The market/universe frontier may be shown once as a reference opportunity set, but it is not a universal frontier determined by the universe alone: its estimates and constraints must be stated.

#### Realised model risk–return curves

For every finalist, including DQN and PPO, plot annualised realised return against annualised realised volatility for the conservative, balanced and aggressive scenarios on identical evaluation dates.

- Show one connected curve per model, with the three scenario points ordered by the frozen risk-control parameter.
- Call this a **realised model risk–return curve** or **risk-appetite profile**, never an efficient frontier.
- Connect points only when scenarios differ through the declared monotonic risk control; otherwise display unconnected points and explain why.
- Preserve dominated and non-monotonic points because they are evidence about a model's response to risk controls.
- Include Sharpe, maximum drawdown, turnover, cumulative cost, benchmark-relative return and seed/fold dispersion in tooltips or the adjacent table.
- Use the same date window, annualisation, cash-return convention, transaction costs and T+1 execution for every model.
- Display confidence/dispersion bands where fold/seed replicates permit them; do not infer statistical confidence from three appetite points alone.

The realised curves are the headline cross-family comparison. Model-conditioned efficient frontiers are a supervised-model diagnostic explaining the opportunity set implied by each predictor.

## 5. Work packages and acceptance criteria

### WP0 — Freeze the expanded-study protocol

**Owner:** integration/research-protocol agent  
**Depends on:** nothing  
**Blocks:** every score-bearing run

Deliver:

- approved study YAML with new study/model/feature/risk IDs;
- data vintage, universe source/as-of rule and exposure ledger;
- nested fold and descriptive holdout manifests;
- objectives, practical-equivalence rules, search spaces, budget tier and seeds;
- explicit `exploratory_closeout_not_confirmatory` claim status; and
- baseline immutability/hash check.

Accept when invalid/null authority fields and `real_data_execution: forbidden` are replaced by resolved, audited fields for the new study only, and the legacy engineering YAML remains unchanged.

### WP1 — Expand and freeze the admitted universe

**Owner:** data agent  
**Depends on:** WP0 universe rule  
**May run in parallel with:** WP2-WP4 engineering

Deliver:

- `configs/universe_candidates_500.txt` and a separate expanded data root;
- resumable per-security preflight/cache behaviour;
- immutable admission/exclusion ledger;
- expanded security master and source manifests; and
- requested/admitted/excluded/feature-ready counts by security.

Accept when the admitted `N <= 500` is deterministic, all original 100 candidates are accounted for, every exclusion has one explicit reason, no issuer substitution occurs, and ingestion can resume without replacing successful raw caches.

### WP2 — Complete the common evaluator and HPO path

**Owner:** optimisation/integration agent  
**Depends on:** WP0  
**May run in parallel with:** WP1, WP3 and WP4

Deliver:

- study-schema loader and fail-closed validation;
- nested temporal evaluator with distinct fit/stopping/score windows;
- trial, fit, fold, seed, fidelity and resource ledgers;
- objective aggregation, practical-equivalence selection and family lock;
- deterministic resume/failure handling and optional halving; and
- integration into `trading_pipeline.run` while preserving legacy behaviour.

Accept when real-data execution is impossible without complete manifests; label/holdout perturbation cannot affect proposals or selection; all trials/folds/seeds are retained; resume is append-only; and the authoritative runner emits a sealed selection manifest before final evaluation.

### WP3 — Add LSTM and Transformer supervised families

**Owner:** deep-supervised agent  
**Depends on:** WP0 contracts; uses WP2 evaluator  
**May run in parallel with:** WP1, WP2 and WP4

Deliver:

- causal sequence builder with masks and train-only scaling;
- registered LSTM and causal Transformer adapters;
- architecture/HPO spaces, deterministic seeds and CPU fallback;
- epoch-level training/validation traces, early stopping and persistence; and
- parity, leakage, shape, masking, save/load and telemetry tests.

Accept when both families consume only the declared feature view, architecture appears in immutable trial parameters, future/outer labels cannot alter training inputs or choices, and selected models emit the canonical prediction schema.

### WP4 — Integrate DQN and PPO into the common study

**Owner:** RL agent  
**Depends on:** WP0 contracts; uses WP2 evaluator  
**May run in parallel with:** WP1-WP3

Deliver:

- registered DQN and categorical PPO runner path;
- identical frozen observations, action space, sleeves, calendar, reward accounting, costs and evaluation mode;
- architecture/HPO, complete seed runs and risk-coefficient arms;
- no-learning evaluation mode and normaliser/replay freeze; and
- policy/value losses, entropy, rewards, steps, updates, actions and resource telemetry.

Accept when PPO genuinely trains through the authoritative path, DQN/PPO resource accounting is comparable, invalid rollout/batch/architecture combinations fail before fitting, and evaluation cannot update policy state.

### WP5 — Build causal model-output feature parity

**Owner:** feature/integration agent  
**Depends on:** WP1-WP4 interfaces  
**Blocks:** downstream stacking/RL score-bearing runs

Deliver:

- versioned out-of-fold prediction feature table;
- complete upstream fit/cutoff/fold/seed hashes;
- coverage masks and missing-output policy; and
- parity/leakage audit across tabular, sequence and RL views.

Accept when all eligible families can access the same declared upstream outputs at the same historical cutoff, no final-fit/test output is exposed during training, and the audit can explain every representation difference.

### WP6 — Execute HPO, lock finalists and run the unified test bench

**Owner:** integration/run agent  
**Depends on:** WP1-WP5  
**Sequential score-bearing work:** yes

Order:

1. score-blind runtime calibration and budget-tier freeze;
2. classical HPO;
3. LSTM/Transformer architecture-HPO;
4. DQN/PPO architecture-HPO for all risk coefficients;
5. outer-fold evaluation and family-selection locks;
6. final fit using only authorised pre-holdout data;
7. one descriptive holdout prediction/action pass;
8. canonical T+1, cost-reconciled backtests for all model × risk combinations; and
9. B0/common-baseline alignment on the identical dates.

Accept when every tunable model has an HPO ledger, fixed baselines are clearly marked non-tunable, every selected configuration traces to inner evidence, the final matrix has no missing/quietly discarded cells, and positions/trades/costs/contributions reconcile.

### WP7 — Extend generated evidence and dashboard

**Owner:** reporting/dashboard agent  
**Depends on:** schema from WP0/WP2; finalises after WP6  
**May scaffold in parallel with:** WP1-WP6

Add generated tables:

- `pipeline_stage_summary`;
- `pipeline_security_summary`;
- `architecture_trial_summary`;
- `hpo_trial_summary`;
- `controlled_sensitivity_summary`, with matched reference/contrast levels, paired
  deltas, fidelity and fold/seed/scenario coverage;
- `candidate_failure_summary`, retaining failed/pruned candidate parameters, failed
  cell/stage, completed/expected cells, runtime, exception category and diagnostic
  source reference;
- `training_trace` and `training_summary` for every family;
- `risk_scenario_summary`;
- `model_conditioned_frontier_points` and per-date frontier portfolio weights;
- `realised_risk_return_curve` with model, risk scenario, fold/seed and benchmark-alignment fields;
- `final_testbench_metrics` and aligned equity/relative-return series; and
- `question_and_assumption_register`.

Required dashboard views:

1. **Protocol and provenance:** run IDs, hashes, compatibility, claim status and warnings.
2. **Pipeline funnel:** requested → mapped → priced → fundamental-covered → feature-ready → labelled → predicted → selected/held, globally and by ticker.
3. **Security drill-down:** date spans, missingness, feature/prediction history, selections, trades, gross return, costs and net contribution.
4. **Architecture selection:** sortable lowest-to-highest/higher-is-better leaderboard, parameter count, runtime and fold/seed dispersion.
5. **HPO:** all trials including failed/pruned, filters by family/features/risk/fold and selected-trial trace.
   Present success and failure as separate evidence: performance rankings must not
   assign artificial scores to failures, while failure rates, recurring causes,
   time-to-failure and partial coverage remain visible.
6. **Controlled sensitivity:** select a model and factor; show only verified matched
   reference/contrast groups, paired metric/resource deltas and failure-probability
   changes. Label reduced-fidelity local effects explicitly and mark unsupported
   global or interaction claims as not identifiable.
7. **Training:** loss/score curves, stopping point, device/resource use and availability labels when a curve is not applicable.
8. **Risk scenarios:** conservative/balanced/aggressive metric matrix and Pareto plots.
9. **Model-conditioned frontiers:** one selectable frontier per eligible supervised model and rebalance date, with shared estimator/constraint metadata and the three risk-scenario locations highlighted. Clearly mark infeasible points and exclude DQN/PPO from the classical-frontier claim.
10. **Realised risk–return curves:** one curve per model, including DQN/PPO, connecting the three risk scenarios only where the declared risk control is monotonic; show dominated/non-monotonic outcomes and supporting risk/cost/dispersion metrics.
11. **Final test bench:** aligned equity curves for every compatible finalist and B0, plus return, risk, drawdown, turnover, cost and benchmark-relative metrics.
12. **Limitations/questions:** unresolved judgement, data and confirmation constraints.

Accept when aggregate pipeline counts reconcile to security rows; contributions reconcile to portfolio returns; incompatible protocols cannot be pooled; ticker filters work; every supervised frontier uses the same declared covariance/constraint contract; realised curves use aligned dates and conventions; classical and realised terminology is never conflated; output checksums are verified; and source runs remain byte-identical.

### WP8 — Replace the notebook with the end-to-end evidence story

**Owner:** notebook/assignment agent  
**Depends on:** WP7 table contract; finalises after WP6  
**May scaffold in parallel with:** WP7

Create a parameterised final notebook that:

1. explains the problem, research status and non-confirmatory boundary;
2. validates report/run provenance and hashes;
3. links in Markdown to the exact repository modules for configuration, extraction, security master, PIT joins, features, contracts, HPO, each model family, portfolio/backtest, reporting and dashboard;
4. shows pipeline summaries and interactive ticker drill-down;
5. shows architecture, HPO and training outcomes from lowest to highest with metric direction explicit;
6. shows every risk scenario, model-conditioned efficient frontiers for eligible supervised models, realised model risk–return curves for all models, and the unified benchmark comparison;
7. launches the local read-only Streamlit dashboard from a clearly marked optional cell and provides equivalent inline evidence if Streamlit is unavailable;
8. distinguishes archived full-run artefacts from code executed in the current notebook session; and
9. contains student prompts, not fabricated first-person reflection or knowledge claims;
10. calls repository components rather than duplicating their implementations;
11. is linked from the repository README; and
12. presents at least one committed compact experimental summary while excluding raw data, large model binaries and oversized run artefacts from Git.

Accept when it runs via `jupyter nbconvert --execute` from a clean environment, contains no machine-specific absolute paths, all links resolve, all numbers come from generated artefacts, and the intended public URL/licensing decision is recorded.

### WP9 — Final audit and submission freeze

**Owner:** independent verification/integration agent  
**Depends on:** WP1-WP8

Run:

- full automated tests and focussed negative leakage/tamper tests;
- immutable old-run hash comparison;
- completed-run audit and complete-matrix check;
- report/catalog regeneration to a new versioned directory;
- notebook clean execution;
- Streamlit AppTest and browser smoke;
- frontier feasibility, common-covariance/constraint and per-date lineage tests;
- realised-curve date alignment, scenario completeness and metric-recalculation tests;
- broken-link and machine-path scan; and
- exact command/run ID/commit/environment capture.

Also produce a factual AI-interaction evidence pack for student use. For each accessible ChatGPT, Codex or Perplexity interaction, record the source/link or supplied transcript, date, purpose, prompt/response summary, recommendation or implementation, decision accepted/changed/rejected, verification performed and resulting code/document references. Keep this evidence separate from student-authored first-person reflection. Missing external-thread exports are recorded as unavailable evidence, not reconstructed from memory.

Accept only if no correctness test fails, all evidence derives from immutable artefacts, and any incomplete expanded component is explicitly removed from headline claims rather than silently patched.

## 6. Agent execution map

Use one integration owner throughout. Parallel agents own disjoint modules and do not select models using another lane's observed results.

| Wave | Parallel lanes | Integration gate |
|---|---|---|
| 0 | Integration owner only: WP0 | Protocol, IDs, schemas, budgets and claim status frozen |
| 1 | Data: WP1; Optimisation: WP2; Deep supervised: WP3; RL: WP4 | Contract tests, registry merge and legacy parity |
| 2 | Feature parity: WP5; Reporting scaffold: WP7; Notebook scaffold: WP8 | Causal lineage and report schema freeze |
| 3 | Integration owner: WP6 sequential execution; reporting agent consumes completed ledgers | Selection locks exist before holdout access |
| 4 | Reporting/dashboard: WP7; notebook: WP8; verifier: WP9 | Final audit, evidence freeze and handoff |

Every agent handoff must state:

- commit/working-tree state and files changed;
- governing study ID and compatibility key;
- tests/commands run and exact outcomes;
- generated run/report IDs and hashes;
- unresolved failures or missing matrix cells; and
- confirmation that no legacy artefact was modified.

## 7. Accelerated delivery schedule and stop rules

### Before the 4:10 pm Codex reset — checkpoint only

- Preserve and verify the completed engineering tranche.
- Do not begin another large implementation branch inside the protected checkpoint reserve.
- Record the exact restart state and critical path.

### After reset through overnight — finish code and start experiments

- Wire the expanded study through the authoritative runner.
- Extend the existing universe/data framework up to 500 candidates and freeze the achieved admission count.
- Complete data-vintage, temporal-window, feature-parity and selection manifests.
- Run synthetic and small-real-data integration smokes.
- Perform score-blind runtime calibration, freeze the feasible HPO tier and launch the score-bearing runs.
- Build dashboard/notebook evidence concurrently from completed ledgers and artefacts.

### By 11:00 am, 27 September — code and research evidence freeze

- Finish viable HPO, outer evaluation, finalist locks, risk scenarios and the descriptive test bench.
- Generate all final tables, charts, reports and dashboard inputs.
- Execute and verify the educational final notebook.
- Run the full audit/test suite and freeze the exact commit/run/report IDs.
- Transparently exclude any family that cannot clear correctness gates; do not extend the deadline or weaken leakage controls for matrix completeness.

### Afternoon, 27 September — first paper draft

- Use the frozen notebook/report evidence to prepare the first complete paper draft.
- Mirror the notebook structure: problem, data/PIT controls, features, models, HPO, portfolio/risk methodology, results, interpretation, limitations and conclusions.
- Keep student-authored reflection separate from the factual AI-interaction evidence pack.

### Remaining two days — review and refinement

- Repair only evidenced correctness or presentation gaps.
- Reconcile every paper number and figure to immutable artefacts.
- Refine explanation, citations, reflection and submission packaging; do not reopen model selection without a demonstrated invalid result.

## 8. Five-hour checkpoint reserve

Live Codex five-hour and weekly usage readings are authoritative.

- Focus implementation strictly on the code/results critical path; writing support follows the 11:00 am evidence freeze.
- Protect at least 15 percentage points of the five-hour window and 10 percentage points of the weekly window for checkpoint, review, tests and handoff.
- Trigger the checkpoint when either window reaches its reserve threshold, or 45 minutes before the five-hour reset time, whichever occurs first.
- At checkpoint: stop starting new work; collect every agent; run targeted tests; record worktree status, commits/diffs, completed acceptance gates, running commands, artefact IDs, blockers and exact restart prompts; then yield a self-contained status report.
- Prefer code/test agents for bounded implementation and one integration agent for protocol/run ownership.
- HPO compute time is controlled by the frozen tier selected before viewing scores.
- Do not consume the available reset credit without explicit user confirmation.

## 9. Fallback hierarchy

The close-out must always leave a defensible submission.

1. **Preferred:** admitted expanded universe, all seven tunable families, complete HPO, three risk scenarios, unified dashboard/notebook.
2. **Reduced compute:** same families and protocol using the Reduced or Minimum defensible tier chosen before results.
3. **Partial expanded study:** only families that passed all gates; complete cells and transparent exclusions; no claim that HPO covered absent/failed families.
4. **Submission safety floor:** immutable completed 100-stock Slice 1/E0-E7/B0 evidence and existing audited dashboard/report, with expanded work described as incomplete future work.

Never trade research integrity for matrix completeness. Missing evidence is a disclosed limitation; reused test evidence presented as selection is invalid.

## 10. Resolved directions and research determinations

Outstanding choices do not block execution. Resolve them analytically where the evidence is sufficient; otherwise run a predeclared bounded comparison and report every outcome.

1. **History refresh:** attempt a new immutable vintage through the latest available date. If no valid extension is available, retain the existing end date and record the attempt. A short extension improves currency but is not relabelled as independent confirmation.
2. **Confirmation status:** use nested walk-forward evidence and a descriptive close-out holdout now. Persistent superiority remains a future-observation question and is not required for successful completion of the experimental methodology.
3. **Universe realism:** perform the historical-membership/delisting source study defined in Section 4.1. Use the most defensible source that can be obtained and validated in the window; otherwise retain and quantify the current-survivor limitation.
4. **Publication boundary:** use market/fundamental data privately for the assignment and trading research. Publish code, methodology and compact generated summaries only; do not commit or distribute raw caches, trained models or full large run artefacts.
5. **Notebook role:** the notebook is an executable showcase and orchestrator over repository components, not a duplicated implementation or a requirement to rerun the full 500-stock HPO study interactively. Link it from README and commit at least one compact experimental summary.
6. **Risk appetite:** the complete conservative/balanced/aggressive sensitivity grid is approved. Freeze numeric definitions before score-bearing runs and report all outcomes without selecting an appetite from holdout performance.
7. **Success criterion:** successfully demonstrate a reproducible advanced-ML experimental method that uses public point-in-time information for trading decisions, evaluates those decisions after realistic costs, and reports how risk-adjusted outcomes compare with the market across models and risk appetites. Beating the market is an observed outcome, not a prerequisite for methodological success.
8. **AI-use/reflection support:** produce the factual, referenced interaction evidence pack in WP9. It may support but must remain separate from the student's personal reflection, understanding, critique and knowledge-gap statements.

## 11. Final definition of done

The expanded close-out is complete when:

- admitted universe count and all exclusions are generated, reproducible and reported;
- all included tunable models have architecture/HPO ledgers and complete seeds/folds;
- common information and causal model-output parity audits pass;
- model selection is traceable to non-holdout evidence;
- all relevant finalist × risk-scenario cells complete under identical T+1/cost conventions;
- model-conditioned efficient frontiers exist for every eligible supervised finalist using common point-in-time risk estimates and constraints, with infeasible points retained explicitly;
- realised risk–return curves exist for every finalist, including DQN/PPO, on identical dates and with terminology that distinguishes them from efficient frontiers;
- final equity and benchmark series share the same dates;
- pipeline aggregates reconcile to per-stock rows and portfolio returns reconcile to contributions/costs;
- dashboard and notebook are read-only, executable and provenance-verified;
- legacy runs remain byte-identical;
- the full test/audit suite passes; and
- conclusions clearly separate observed facts, generated evidence, interpretation, assumptions, exploratory findings and questions requiring human/future evidence.
