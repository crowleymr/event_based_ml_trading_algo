# Phase 3 Architecture, HPO and Multi-Run Reporting Implementation Plan

**Planned implementation start:** 20 September 2026, 8:12 pm Australia/Sydney  
**Status:** engineering foundation implemented; adaptive/real-data research execution remains gated  
**Authority:** `docs/FSD_v2.md`, `docs/DECISIONS.md`, `docs/SLICE1_COMPLETION_REPORT.md`, `docs/BACKLOG.md`, repository `AGENTS.md`  
**Inputs:** completed Deep HPO research task, current working-tree review, and the user's approval of model refactoring and multi-run reporting

## Implementation checkpoint — 20 September 2026

Implemented without resolving research-policy choices: three model contracts and
research-disabled family adapters; allowlisted registries; strict draft-study YAML;
deterministic grid/random proposals; nested purged walk-forward manifests; immutable
trial ledgers; audited multi-run catalogue; dashboard run/config/time inspection;
optional report-input provenance and checksum verification. The existing authoritative
runner remains unchanged in research behaviour.

Still gated: real-data HPO, objective approval, new-vintage window manifests,
successive-halving/TPE execution, family-winner selection, risk preferences and sealed
confirmation. See `docs/IMPLEMENTATION_LOG.md` for generated verification evidence.

## 1. Objective

The next round will turn the current model-specific training paths into a registered,
configuration-driven research platform while preserving the completed Slice 1 evidence.
It will establish the engineering foundations needed to compare supervised,
unsupervised and reinforcement-learning approaches without allowing their different
tasks, objectives or implementation details to contaminate one another.

The round has four outcomes:

1. Replace hard-coded model-family branching with explicit `SupervisedModel`,
   `UnsupervisedModel` and `RLPolicy` contracts and allowlisted registries.
2. Add a leakage-safe optimisation harness capable of bounded grid, random,
   space-filling, successive-halving and Bayesian search under one temporal evaluator.
3. Add a read-only multi-run catalogue and dashboard comparison layer that makes
   protocol compatibility, execution time and configuration differences explicit.
4. Create isolated model-family extension points so later agents can develop and test
   one family without learning from, editing or implicitly adapting to another
   family's experimental results.

The defensible optimisation claim is:

> Best configuration found within a declared model family, search space, validation
> protocol and compute budget.

It is not “the best possible model”, and it is not evidence of sustainable alpha.

## 2. Boundaries and non-negotiable requirements

### 2.1 Preserve completed evidence

- E0–E7, B0, the original E5 eligibility set and all completed run artefacts remain
  immutable.
- The already-observed Slice 1 final test is descriptive only. It cannot determine
  candidate inclusion, parameter ranges, architectures, objectives, stopping rules,
  budgets or rejection decisions.
- `trading_pipeline.run` remains the authoritative end-to-end execution path.
- Reports, notebooks and dashboards remain read-only consumers of completed artefacts.
- Point-in-time SEC joins, information cut-offs, purge/embargo, T+1 execution,
  long-only constraints and transaction-cost reconciliation must not weaken.
- Synthetic data may verify software but cannot support research conclusions.

### 2.2 Separate engineering enablement from research approval

The 8:12 pm round may implement interfaces, adapters, registries, configuration
validation, synthetic fixtures, optimisation machinery, provenance, reporting and
disabled candidate implementations. It must not execute a new real-data selection
study until the human has approved:

- the new data vintage and genuinely untouched evaluation periods;
- the nested temporal split and final-confirmation protocol;
- permitted families, search spaces and compute budgets;
- supervised scoring conventions;
- unsupervised task definitions;
- RL utility, risk constraints, observations, actions, reward and evaluation mode;
- any change to the universe, target, execution timing or headline cost policy; and
- the statistical claim and success criteria.

### 2.3 Fail closed

Unknown registry IDs, unresolved protocol fields, incompatible objective/model pairs,
invalid conditional parameters, incomplete folds or seeds, forbidden data roles and
failed audits must stop execution. The runner must never silently substitute a model,
metric, data source, estimator, device or search method.

## 3. Mental models guiding the design

### 3.1 The research protocol is a firewall

The model never decides what data it may see. A shared evaluator supplies explicit
fit, stopping, inner-scoring and outer-evaluation partitions. The optimiser proposes
parameters; it does not own data access or scoring truth.

### 3.2 Component, experiment, trial and fit are different entities

- A **component registration** identifies an implementation and its semantics.
- An **experiment arm** combines a component, data, features, objective and portfolio.
- A **study** defines the approved protocol and comparison set.
- A **trial** is one resolved configuration proposed by a search method.
- A **fit** is one trial × fold × seed × fidelity execution.

Hyperparameter trials do not receive new E/M/RL family identifiers. A material change
to a target, reward, action space, estimator algorithm or representation does.

### 3.3 Search is not validation

Grid search, random search, halving, Bayesian optimisation and evolutionary search are
proposal strategies. Every proposal must pass through the same chronological,
purged, embargoed evaluator. Changing the search algorithm must not change the scoring
calendar, preprocessing boundary or final-test visibility.

### 3.4 Five folds means five chronological experiments

Ordinary shuffled `KFold`, `StratifiedKFold` and random five-fold cross-validation are
prohibited. Securities on the same session are not independent observations, and
overlapping five-session labels create temporal leakage.

The proposed replacement is five outer expanding walk-forward folds, each containing
purged inner walk-forward splits. Within an inner split, fitting, early stopping and
candidate scoring use distinct chronological regions. Window counts and dates remain
planning defaults until approved and frozen in a manifest.

### 3.5 Fairness is opportunity plus resource accounting

Equal trial counts alone are not fair. Families receive bounded, declared opportunity
to search and must report wall time, CPU core-seconds, GPU seconds, memory, parameter
count and inference latency. DQN and PPO also report environment steps and gradient
updates because equal environment steps do not imply equal compute.

### 3.6 Reporting is an evidence index, not a second research pipeline

The catalogue and dashboard expose immutable evidence. They do not train, tune,
reconstruct missing histories, change objectives or select a winner.

## 4. Agent isolation and integration model

Model-family exploration will use separate agents and worktrees to limit context leak,
anchoring and accidental shared-file edits.

### 4.1 Shared contract first

An integration agent must first freeze:

- interface signatures and typed result objects;
- registry and YAML schemas;
- temporal fold descriptors;
- objective and metric identifiers;
- trial/fit ledger schemas;
- telemetry and artefact contracts; and
- ownership boundaries.

The three model-family agents then receive only the authoritative research rules,
shared contracts, their owned package and relevant tests. They do not receive another
family's candidate rankings, selected parameters, performance results or implementation
notes.

### 4.2 Ownership

| Agent | Exclusive implementation scope | May read | Must not edit |
|---|---|---|---|
| Supervised agent | `modelling/supervised/**` and supervised tests/registry fragment | shared contracts, features, legacy supervised code | RL and unsupervised implementations, central runner |
| Unsupervised agent | `modelling/unsupervised/**` and unsupervised tests/registry fragment | shared contracts, causal feature schema | supervised/RL implementations, central runner |
| RL agent | `rl/policies/**` and RL-policy tests/registry fragment | shared contracts, frozen environment and accounting boundary | supervised/unsupervised implementations, central runner |
| Optimisation-platform agent | temporal evaluator, search adapters, ledgers | only public model interfaces | model-family internals |
| Reporting agent | catalogue, compatibility rules, report/dashboard schema | completed artefact contracts | training modules and model code |
| Integration agent | registries, config composition, `run.py`, final verification | all approved patches after family work completes | family internals except conflict resolution with evidence |

Each family agent returns a bounded patch, registry fragment, schema fixtures,
verification record and explicit unsupported capabilities. Integration happens only
after contract tests pass. Cross-family performance is revealed only by the generated
outer-evaluation report, never during implementation.

## 5. Component A — Baseline preservation and provenance closure

### Objective

Create a reproducible pre-refactor checkpoint and close known provenance gaps before
the architecture expands.

### Requirements and approach

- Preserve the current dirty working tree without discarding user changes.
- Record the exact source hashes, dependency lock, resolved configs, test result and
  audited smoke evidence associated with the checkpoint.
- Add every optional training/device table consumed by reporting to its input hash
  manifest.
- Require report and dashboard loaders to reject failed source audits, missing hashes
  or mismatched checksums.
- Never edit existing run artefacts to retrofit provenance.

### Blast radius

`tracking/`, `audit.py`, `reporting/generate.py`, `dashboard/loader.py`, metadata and
provenance tests, plus factual entries in `docs/IMPLEMENTATION_LOG.md`.

### Task breakdown

1. Inventory modified and untracked source files.
2. Extend source/report input manifests and checksum verification.
3. Add tamper, missing-optional-input and read-only regression tests.
4. Run the complete suite and a clean synthetic smoke execution.
5. Record generated evidence; do not transcribe performance numbers into this plan.

### Acceptance

Legacy runs remain readable, new reports hash every consumed input, tampering fails,
and the pre-refactor smoke contract is reproducible from an identified source state.

## 6. Component B — Shared model contracts, registries and YAML schema

### Objective

Decouple data, fitting, evaluation and model implementations without creating parallel
execution applications.

### Requirements and approach

Create distinct abstract contracts:

- `SupervisedModel`: capability declaration, search space, train-only preprocessing,
  fit with explicit stopping data, predict, save/load, complexity and telemetry.
- `UnsupervisedModel`: unit-of-analysis declaration, train-only fit, causal
  transform/assignment, save/load, native diagnostics and telemetry.
- `RLPolicy`: construct, learn, deterministic/stochastic act, continuation checkpoint,
  save/load, resource counters and telemetry.

The central registry maps allowlisted IDs to constructors and capabilities. YAML refers
to IDs only; it cannot import arbitrary Python. Separate registries describe models,
feature/representation sets, objectives, search spaces, portfolios, cost policies and
validation protocols.

The first study schema should contain authority, data vintage, exposure ledger,
validation manifests, experiment arms, objectives, selection rule, search budget,
seeds, device policy and output contract. The fully resolved YAML and registry hashes
are immutable run inputs.

### Blast radius

New `experiments/` and `optimisation/` packages; refactoring within `modelling/` and
`rl/`; `config.py`; `run.py`; registries; package READMEs; layout/config tests.

### Task breakdown

1. Define interfaces and typed fit/prediction/representation/policy result records.
2. Define capability flags, including early stopping, continuation and staged metrics.
3. Implement allowlisted registries and duplicate/unknown-ID validation.
4. Implement strict study-config parsing and resolved-config hashing.
5. Adapt legacy Elastic Net, HistGBT, XGBoost and DQN without changing behaviour.
6. Replace hard-coded family branches only after parity tests pass.

### Acceptance

Existing smoke predictions, selection rules and artefact semantics remain unchanged;
adding a registered implementation does not require another family-specific branch in
the authoritative runner.

## 7. Component C — Temporal evaluator and optimisation platform

### Objective

Provide one leakage-safe evaluator and an auditable search engine shared by all model
classes.

### Validation design

The proposed new protocol uses:

1. a sealed final confirmation period;
2. five chronological outer expanding walk-forward folds for procedure comparison;
3. purged and embargoed inner walk-forward splits for HPO;
4. a chronological fit tail used only for early stopping where required; and
5. candidate scoring dates fixed before the first proposal.

Actual dates, minimum window lengths and the number of inner splits require approval
and a new data-vintage manifest. All securities from one session remain together.
Purging uses actual label/reward information intervals, not row counts alone.

### Search approaches

| Method | Initial role |
|---|---|
| Explicit grid | Small interpretable spaces and factorial questions; default for Elastic Net and small regime models |
| Random or Sobol/Latin-hypercube screening | Broad mixed/continuous spaces; default first pass for trees, neural models and RL |
| Successive halving | Only when a valid resource axis exists, such as boosting rounds, epochs or environment steps |
| Bounded TPE/Bayesian refinement | Second stage after space-filling screening; all finalists reconfirmed at full fidelity |
| Genetic/evolutionary search | Deferred comparator for genuinely structured architecture or multi-objective spaces |

`HalvingGridSearchCV` must not use random row fractions as fidelity. The evaluator keeps
validation dates fixed while increasing a meaningful training resource. Low-fidelity
ranking reliability must be verified before promotion decisions are accepted.

The planning ceiling is approximately 16 screening plus 16 refinement proposals per
approved model-feature arm. This is not authorised research budget until the human
approves it. Reducing budget happens before results are inspected.

### Trial and fit ledgers

Persist proposal order, resolved conditional parameters, sampler state, fold, seed,
fidelity, stopping reason, status, failure/pruning reason, timestamps, resources,
predictions/representations/actions and hashes. Failed attempts remain part of the
evidence. Resume is append-only and cannot overwrite a completed fit.

### Selection

1. Reject leaking, invalid, incomplete or constraint-violating candidates.
2. Rank using the class-specific primary objective aggregated over complete inner folds.
3. Form a predeclared practical-equivalence set.
4. Prefer lower family-specific complexity, then the secondary criterion, then lower
   resource use, then stable candidate ID.

The rule may select a baseline or conclude that no candidate is preferable.

### Blast radius

New `optimisation/search.py`, `optimisation/evaluator.py`, `optimisation/objectives.py`,
`optimisation/ledgers.py`, nested split support, artefact writers, audits and extensive
synthetic tests.

### Acceptance

Synthetic future perturbations cannot change earlier preprocessing, proposals,
stopping or selection; outer/final labels are inaccessible to inner HPO; identical
seeds/configs reproduce proposal order and selection.

## 8. Component D — `SupervisedModel` lane

### Objective and objective function

Select predictors for cross-sectional ranking, not directly for attractive backtests.
The primary objective is equal-date, equal-fold mean ranking utility. Conventional
daily Spearman IC remains reported, but constant finite predictions receive a separately
named neutral `ranking_utility = 0` so candidates cannot benefit from dropped dates.
Nonfinite or missing required predictions fail the fit. RMSE is the secondary
tie-breaker; MAE, weekly-signal-date IC, coverage and constant-prediction frequency are
diagnostics.

This revised utility convention requires human approval before real-data use.

### Initial family spaces

- Elastic Net: training-derived log regularisation ratios and a small `l1_ratio` grid;
  require convergence and avoid redundant stochastic repeats.
- HistGBT: leaves, minimum leaf samples, L2, learning rate and boosting budget, with
  explicit capacity × regularisation and learning-rate × budget interactions.
- XGBoost: depth, child weight, row/column fractions, L2, learning rate and boosting
  budget; keep tree method and growth policy fixed in the first study.
- Optional shallow MLP: depth, width, activation, dropout, AdamW learning rate, weight
  decay and batch size under a parameter/memory ceiling. It is excluded unless approved.

Use balanced categorical strata and logarithmic sampling where scales span orders of
magnitude. Main effects and prespecified two-way interactions are estimated from the
screening design; adaptive-search history alone is not treated as a clean factorial
experiment.

### Agent tasks

1. Implement legacy adapters and parity fixtures.
2. Declare conditional spaces and capability metadata per family.
3. Separate fit, chronological stopping and scoring partitions.
4. Emit complexity, convergence, importance and resource telemetry.
5. Add interaction-design metadata and family-specific invalid-combination checks.
6. Produce only registry fragments; do not edit the central runner.

### Acceptance

Train-only transformations are proved, final/outer perturbations leave selection
unchanged, early stopping cannot score the same dates used for candidate selection,
and legacy Slice 1 paths retain their frozen behaviour.

## 9. Component E — `UnsupervisedModel` lane

### Objective

Introduce unsupervised learning as a registered representation or regime treatment,
not as a way to search future returns indirectly.

There is no universal unsupervised score. Objectives depend on the declared task:

- PCA/autoencoder: forward reconstruction error in a common original feature space at
  a declared complexity budget.
- KMeans/security clustering: forward assignment geometry and permutation-invariant
  stability subject to occupancy constraints.
- GMM regimes: forward log predictive density in a fixed representation.
- HMM regimes: causal one-step predictive log density using filtered probabilities,
  never full-sequence smoothing or future-informed Viterbi states.
- Downstream value: evaluated only as a nested experiment arm, with representation
  selection counted inside the supervised or RL search budget.

### Initial scope

Implement the interface and synthetic tests first. The recommended first real candidate
is one simple, low-dimensional GMM regime model with a no-regime baseline. PCA may be a
separately registered preprocessing treatment. KMeans is an interpretable engineering
fixture; HMM and autoencoder work remain later extensions until causal handling and data
volume justify them.

Small explicit spaces should cover component count, covariance form, regularisation,
representation rank and a fixed restart procedure. Restart count is part of fitting,
not a parameter selected on trading returns. Regime labels are arbitrary and must use
permutation-invariant comparisons; semantic names cannot be inferred from future returns.

### Agent tasks

1. Define observation unit, transform schema and information-cutoff fields.
2. Implement no-transform/no-regime baselines and a disabled GMM adapter.
3. Implement train-only scaling, forward assignment and deterministic restart policy.
4. Emit occupancy, convergence, stability and forward-density diagnostics.
5. Add future-perturbation, label-permutation and empty/degenerate-state tests.
6. Produce registry fragments without reading supervised or RL results.

### Acceptance

Changing observations after time T cannot change the representation emitted at T;
state-label permutations do not change evaluation; downstream experiments identify the
exact upstream transform fit and hash.

## 10. Component F — `RLPolicy` lane

### Objective and controlled comparison

Refactor the current DQN into `RLPolicy` and add categorical PPO behind the same frozen
discrete sleeve-selector environment. DQN versus PPO must hold observations, actions,
reward, sleeves, execution, costs and evaluation calendars fixed. Continuous 100-stock
weight control is a different research programme and remains deferred.

The proposed primary selection objective is equal-fold mean after-cost certainty-
equivalent weekly return, averaging all registered training seeds within each fold,
subject to predeclared risk constraints. Risk aversion, return convention, variance
estimator, cash yield, annualisation and drawdown constraint must be approved and frozen.
Secondary reporting includes net growth, Sharpe, drawdown, tail loss, turnover, costs,
cash exposure and action switching. TD loss, PPO surrogate loss, entropy and training
episode reward are diagnostics only.

### Search and fairness

- DQN factors: network depth/width, learning rate, replay/batch ratio, target update,
  exploration schedule, gamma and training steps.
- PPO factors: separate/shared policy-value architecture, depth/width, learning rate,
  rollout length, batch size, epochs, gamma, GAE lambda, clip range and entropy/value
  coefficients.
- Invalid divisibility or rollout combinations fail config validation.
- Use balanced random screening and conservative full-fidelity refinement.
- Retain `[41, 42, 43]` as proposed screening seeds and use a disjoint, predeclared
  finalist seed set. These are planning defaults, not approved research settings.
- Report environment steps, gradient updates and actual resource use; include a
  separately labelled equal-wall-time view if required.

### Causal upstream lineage

Every policy observation must use base predictions produced by models whose fitting and
selection information was available at that historical time. The first affordable
protocol should choose base specifications in an initial calibration period, freeze
them, then refit causally. Do not retroactively insert an eventually winning sleeve
into earlier policy history.

Transitions whose reward interval crosses a protected boundary are purged. Evaluation
starts in cash, applies identical entry costs, and cannot update weights, replay,
normalisers or checkpoints. Cash-reset folds are reported separately rather than
stitched into a fictitious continuous equity curve.

### Agent tasks

1. Adapt DQN to the shared policy interface without changing the frozen pilot.
2. Implement disabled categorical PPO with configurable policy/value architectures.
3. Move family-specific telemetry, checkpointing and resource counters into adapters.
4. Validate conditional spaces and continuation semantics.
5. Add evaluation-freeze, boundary, seed-completeness and DQN/PPO parity tests.
6. Emit registry fragments; do not edit upstream model selection or reporting code.

### Acceptance

Fixed sleeves still reconcile with canonical accounting, evaluation cannot learn, all
registered seeds are retained, upstream lineage is auditable, and existing diagnostic
DQN evidence remains unchanged.

## 11. Component G — Multi-run reporting and dashboard

### Objective

Make completed results selectable and comparable by execution time and configuration
without implying comparability where protocols differ.

### Requirements and approach

Build a versioned, read-only catalogue of completed audited runs. Index:

- run/study/trial/fit IDs and execution timestamps;
- resolved config and protocol hashes;
- data vintage and exposure role;
- model, feature/representation, portfolio and objective IDs;
- fold, seed, fidelity and search method;
- requested/actual device, duration and resource measures;
- research status, audit status and report checksum; and
- predictive, representation, policy and economic metrics in their native units.

A compatibility key includes data vintage, universe, target, outer windows, execution,
costs, objective version, baselines and budget tier. Incompatible runs may be inspected
side by side but cannot be pooled or ranked as a fair comparison.

Dashboard views will include run/config/time selectors, inner-search histories,
outer-evaluation summaries, final-confirmation separation, fold/seed distributions,
paired deltas, trial failures/pruning, compute-versus-quality curves, Pareto frontiers,
and declared main-effect/interaction plots. Missing telemetry is displayed as
unavailable, never reconstructed.

### Blast radius

New `reporting/catalog.py` and compatibility schemas; report schema v3 adapters;
`dashboard/loader.py`, `dashboard/app.py`, CLIs, documentation and reporting/dashboard
tests. Existing v1/v2 reports remain readable through adapters.

### Task breakdown

1. Define catalogue row and compatibility-key schemas.
2. Scan only completed, audited runs without modifying them.
3. Hash every source and generated catalogue output.
4. Add schema adapters and reject failed/tampered runs.
5. Add multi-select filters and matched-comparison views.
6. Test deterministic catalogue generation, mixed schemas and incompatibility warnings.

### Acceptance

Users can select runs by time and config, compare only matched evidence, inspect why
runs are incompatible, and reproduce every table from hashed source artefacts.

## 12. Component H — Audit, tests and documentation

### Required tests

- Future data cannot change earlier transforms, proposals, stopping or selection.
- All same-date securities remain in one fold; irregular calendars and actual label
  ends are handled correctly.
- Outer/final perturbations leave inner selection unchanged.
- Conditional parameter errors fail before training.
- Search resume preserves sampler state and does not duplicate completed fits.
- Halving continuation respects the declared resource axis and fixed scoring calendar.
- HMM/GMM causal outputs and label-invariant tests pass where those models exist.
- RL reward intervals and upstream selection cut-offs precede protected boundaries.
- RL evaluation cannot update learned state.
- Required folds/seeds cannot be silently dropped.
- Optional artefact tampering is detected and reporting leaves source bytes unchanged.
- Legacy E5 eligibility and all existing Slice 1 regression tests remain intact.

### Documentation

Update architecture/package READMEs, experiment registry, backlog, decisions and the
implementation log only with factual implemented behaviour and generated verification.
Student reflection and interpretation remain student-authored.

## 13. Program blast radius summary

| Area | Change level | Primary risk | Control |
|---|---:|---|---|
| Immutable runs/data | None | Accidental mutation | Read-only loaders, hashes, byte-preservation tests |
| `run.py` | High | Second orchestration path or changed legacy selection | Integration-agent ownership and legacy parity tests |
| Config/registries | High | Silent defaults or arbitrary imports | Strict schema, allowlists, resolved-config hashes |
| Supervised modelling | High | Changed Slice 1 behaviour or stopping leakage | Legacy adapter mode, separate stopping/scoring partitions |
| Unsupervised modelling | New | Future-aware regimes or meaningless objective | Causal transforms, native task objectives, downstream nesting |
| RL | High | Policy leakage, unfair DQN/PPO budget, reward hacking | Frozen MDP, causal lineage, complete seeds, risk constraints |
| Optimisation | New/high | Search overfitting and irreproducible proposals | Nested evaluator, bounded budgets, append-only ledgers |
| Reporting/dashboard | High | Pooling incompatible evidence | Compatibility keys, audit/checksum gating, read-only design |
| Tests/audit | High | False confidence from synthetic-only success | Negative leakage/tamper tests plus approved fresh-vintage gate |

## 14. Execution waves for the 8:12 pm round

### Wave 0 — Baseline and contract freeze

1. Preserve the working-tree checkpoint and run baseline verification.
2. Freeze shared interfaces, schemas and ownership boundaries.
3. Close report provenance/checksum gaps.

No model-family work begins until these contracts pass review.

### Wave 1 — Parallel isolated implementation

- Supervised agent: legacy adapters, family spaces and supervised tests.
- Unsupervised agent: interface, baseline/GMM scaffold and causal tests.
- RL agent: DQN adapter, disabled PPO scaffold and RL tests.
- Optimisation agent: nested evaluator, grid/random search and ledgers.
- Reporting agent: catalogue, compatibility schema and loader tests.

Agents use separate worktrees and do not edit central integration files.

### Wave 2 — Integration

1. Merge contract-conforming patches one family at a time.
2. Register components and compose the authoritative runner.
3. Run interface, parity, leakage, audit and full regression suites after each merge.
4. Add halving and bounded TPE only after grid/random evidence contracts are stable.

### Wave 3 — Engineering evidence only

1. Run synthetic multi-fold smoke studies.
2. Generate a multi-run catalogue and dashboard from audited synthetic/diagnostic runs.
3. Record resource measurements to estimate the proposed research budget.
4. Produce an approval packet containing unresolved research choices.

No new real-data HPO or family winner is selected in this wave.

### Wave 4 — Later, after explicit approval

Freeze the new data vintage, exposure ledger, temporal manifests, objectives, spaces,
budgets and claim policy. Then execute inner HPO, lock family winners, evaluate outer
folds, select the complete pipeline if authorised, and finally evaluate the sealed
confirmation period once.

## 15. Definition of done for this implementation round

- The current baseline is preserved and reproducibly identified.
- Three model contracts and allowlisted registries exist.
- Legacy Elastic Net, HistGBT, XGBoost and DQN pass behavioural parity tests.
- GMM and categorical PPO scaffolds are disabled for real research until approval.
- Strict YAML can describe a study, arm, trial and fit and rejects unresolved policy.
- Nested purged walk-forward split generation is tested on synthetic data.
- Grid and random/space-filling search produce complete append-only ledgers.
- Halving/TPE are either implemented behind validated capability gates or explicitly
  deferred without weakening the basic harness.
- The multi-run catalogue supports selection by execution time and configuration.
- Compatibility rules prevent invalid pooled comparisons.
- Every consumed report input is hashed and failed audits are rejected.
- The complete test suite and synthetic smoke study pass.
- `docs/IMPLEMENTATION_LOG.md` records factual actions and verification.
- No observed final-test result influences any new choice.

## 16. Decisions required before real HPO begins

The implementation round should end with a concise approval request covering:

1. data vintage, historical exposure ledger and sealed confirmation dates;
2. exact outer/inner windows, purge and embargo definitions;
3. approved family/feature/representation arms;
4. ranking-utility convention and eligibility rules;
5. unsupervised task and first candidate;
6. RL risk aversion, constraints, seed sets and DQN/PPO scope;
7. search and compute ceilings;
8. multiplicity, bootstrap and practical-equivalence rules; and
9. criteria for academic conclusions versus profitability claims.

Until those decisions are recorded, the new system is an engineering capability—not
authority to optimise against the completed test or claim an improved trading system.
