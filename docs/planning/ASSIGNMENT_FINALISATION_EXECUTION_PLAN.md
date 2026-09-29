# Assignment Finalisation Execution Plan

**Purpose:** plan the remaining work for the A2 submission without writing the final report, notebook narrative, or student reflection in this document.  
**Prepared:** 28 September 2026 (Australia/Sydney)  
**Scope:** project report, public Python notebook, implementation log, reproducible tables/figures, final assembly and verification.  
**Research authority:** `docs/FSD_v2.md`, `docs/DECISIONS.md`, approved expanded-study protocol, immutable run artefacts, and the assignment specification.  
**Planning rule:** comments added as `<!-- ... -->` are review instructions. Resolve each comment in place, record the resolution in Section 14, and do not silently delete unresolved comments.

## 1. Executive decision

The submission should be built as **one master DOCX working document exported to the single required PDF**. The PDF must contain:

1. a plain-text public notebook URL;
2. the project report;
3. the implementation log; and
4. optional appendices needed to support reproducibility and AI-use disclosure.

The recommended notebook route is a **public Colab orchestration notebook backed by the public repository and immutable evidence**, not a disconnected toy implementation and not a duplicated monolithic rewrite of the pipeline. From a clean runtime it should clone a pinned repository revision, install pinned dependencies, expose and explain the complete implementation, run a bounded real-data demonstration through the authoritative `trading_pipeline.run` path, and load the immutable full-study report for the submitted results. It must state clearly which cells execute a bounded demonstration and which tables/figures come from the full canonical run.

This route best reconciles the assignment's “complete implementation” and “self-contained” wording with the repository's tested modular architecture. A toy-only notebook is not sufficient. Copying the full repository into notebook cells is high-risk, likely to drift from the tested code, and should not be attempted tonight.

No new research selection or experimental rerun is in scope. The completed final holdout is descriptive only and must not be used to choose, tune, reject, or prefer a feature set, model, risk scenario, portfolio method, or parameter.

## 2. Evidence snapshot and immediate implications

### 2.1 Observed facts

| Item | Observed state | Implication |
|---|---|---|
| Assignment specification | 23 pages reviewed; A2 deliverables and Full Mark descriptors confirmed | Plan must optimise Criteria A, B and C, not add cosmetic EDA |
| Authoritative expanded run | `20260928T005423Z-0634efaf` | Use this run as the final expanded evidence source |
| Run status | `complete`; audit `passed` | Read-only reporting can proceed |
| Source Git revision | `e9b3e0cdf57c63c87a9dc881d58b1696f0472ba5` | Record in every generated evidence manifest and final submission |
| Claim status | `exploratory_closeout_not_confirmatory` | Do not write confirmatory or causal claims |
| Holdout role | `descriptive_only` | Report all predeclared arms/seeds/scenarios; no post-holdout winner selection |
| Expanded report | Generated under `reports/expanded_closeout/20260928T005423Z-0634efaf`; 11 hash-declared tables; focused reporting/dashboard suite passed 19 tests | Use only through the verified read-only loader and manifest |
| Assignment evidence | Generated under `reports/assignment/20260928T005423Z-0634efaf/v1`; 7 tables in CSV and Parquet plus manifest | Use these generated tables for sensitivity, predictive diagnostics, telemetry, features, availability and provenance |
| Existing final notebook | `notebooks/final_evidence.ipynb`, 21 stable-ID cells; canonical local Run All passed | Publication fields deliberately fail closed until the student creates the GitHub Release and supplies its URL/tag/hash |
| Generated factual report draft | `reports/assignment/20260928T005423Z-0634efaf/v1/report/PROJECT_REPORT_DRAFT.md`; 3,175 words, 3 generated figures, hash manifest | No numeric placeholders remain; finalisation waits for the public notebook URL and three student-authored prompt responses |
| Existing implementation log | Detailed engineering chronology through 28 September | Reconcile with Git, run logs and agent threads; add final completion and student-authored reflection |
| Citation library | 13 verified report-relevant records in Zotero collection `GCZBX6BQ`, with item-key register and BibTeX export; one malformed Harvey record quarantined | Cite only records permitted by `docs/assignment/CITATION_REGISTER.md`; do not cite the quarantined item until corrected |
| Release bundle | `release/assignment-evidence-20260928T005423Z-0634efaf.zip`, 32 hash-declared entries plus bundle manifest | Student will create the GitHub Release and upload the ZIP and checksum sidecar |

### 2.2 Evidence classes to keep separate

- **Observed fact:** directly read from canonical data, immutable runs, generated reports, Git or the assignment specification.
- **Generated evidence:** table, statistic or chart created repeatably from canonical inputs, with provenance.
- **Interpretation:** explanation of what an observed result may mean; must be phrased cautiously.
- **Assumption/limitation:** unverified mechanism, simplified execution model, data constraint or boundary.
- **Student evidence:** first-person understanding, critique, learning or knowledge gap; must be written by the student.

## 3. Deadline-critical execution order

The order below is dependency-driven. Steps may overlap only where they read immutable artefacts and cannot change research outcomes.

| Gate | Task | Exit evidence | Owner/service | Model power |
|---|---|---|---|---|
| G0 | Freeze run identity and research rules | Run ID, source commit, protocol hash, claim status and holdout role copied from machine artefacts | Codex | Luna/low for extraction; Sol/high for verification |
| G1 | Generate and validate the expanded report | Eleven hash-declared Parquet tables plus `research_evidence_manifest.json`; loader passes | Codex local | Sol/high |
| G2 | Generate assignment tables and figures from G1 only | Versioned output directory with source hashes, code version and timestamps | Codex local | Sol/high; Astra/high reviews analytical completeness |
| G3 | Reconcile citations and literature claims | Verified Zotero library or citation ledger; no Perplexity citation tokens or placeholder DOIs | Perplexity research + student/Codex verification | Highest-accuracy research mode; Astra/high synthesis |
| G4 | Draft the report against the evidence matrix | Complete draft with no numeric placeholders; interpretations labelled | ChatGPT/Codex + student | Astra/high or xhigh |
| G5 | Build and execute the public notebook | Public URL; clean-runtime execution receipt; links resolve | Codex + Colab + student publication | Sol/high for code; Luna/medium for link checks |
| G6 | Reconstruct the implementation log | Evidence-backed chronology plus explicit student prompts | Codex extraction + student authorship | Sol/high extraction; Astra/high critical review |
| G7 | Assemble DOCX and export the single PDF | Required filename, plain-text notebook URL, APA references, readable figures/tables | Codex document workflow + Word/Zotero | Sol/high |
| G8 | Rubric red-team and submission check | Criterion A/B/C evidence checklist; no unsupported claims; PDF visually inspected | Independent Codex/ChatGPT review + student | Astra/xhigh |

**Stop rule:** if time compresses, preserve G0-G1-G4-G5-G6-G7-G8. Drop decorative EDA, extra sensitivity graphics and optional appendices before compromising notebook accessibility, evidence provenance, theory-to-code explanation, loss-versus-objective analysis, or the student's ability to defend the work.

### 3.1 Usage-budget rule

At execution start the account had used 53% of the weekly allowance, leaving 47%. The student requires a hard buffer of at least **17 percentage points** for refinements and follow-ups. Therefore this execution may consume at most 30 additional percentage points and must stop before weekly usage reaches 83%. The final execution check showed 63% used and 37% remaining, preserving the full 17-point buffer plus 20 additional points.

Use the allowance in three passes:

1. **Production pass - 60% of this task allocation:** evidence generation, notebook/publication work and the first complete draft.
2. **Verification pass - 25%:** numeric/citation reconciliation and document rendering checks.
3. **Reserved red-team pass - 15%:** final rubric review and only blocker-level corrections.

Default to Luna for exact extraction/link checks, Sol for implementation and assembly, and Astra only for high-value synthesis or the final independent review. Avoid repeated whole-repository reads, duplicate literature searches and multiple stylistic rewrites. Cache evidence matrices and give later reviewers the manifest plus the current draft rather than full raw history.

<!-- RESOLVED C-001: Added a quantified usage ceiling, model-routing rule and protected review reserve. -->
## 4. Full Mark rubric plan

### 4.1 Criterion A - Task Definition

**Full Mark target:** a clear, technically testable, well-motivated and feasible research question supported by practical or theoretical evidence and critical thinking.

Required report evidence:

1. Define the business problem as **cross-sectional security ranking and risk-controlled allocation**, not generic “stock-price prediction”.
2. State the primary research question and bounded sub-questions before discussing results.
3. Specify training input precisely:
   - one row or sequence indexed by security and information date;
   - feature names, units/transforms, lookback rules and missingness handling;
   - point-in-time SEC availability rule;
   - eligibility and sequence construction rules.
4. Specify training target precisely: five-session forward adjusted return, including formula, time index, and why it is label-only.
5. Specify model outputs per interface:
   - supervised models: real-valued return score used for cross-sectional ranking;
   - sequence models: same economic score from a bounded historical tensor;
   - RL selectors: action/policy output and risk-scenario-conditioned reward interface.
6. Specify deployment interface:
   - information available through session T;
   - signal after T;
   - execution at T+1 under the exact implemented convention;
   - portfolio weights, constraints, transaction costs and reporting outputs.
7. Explain why rule-only approaches are insufficient but remain necessary baselines.
8. Support feasibility with observed dataset coverage, the canonical architecture and completed run evidence.
9. Distinguish the locked Slice 1 question from the expanded exploratory closeout. Do not rewrite history as though deep/RL models were part of the original confirmatory design.
10. End the section with explicit hypotheses or evaluative expectations that were fixed before descriptive holdout access.

### 4.2 Criterion B - Model, Algorithm, Components and System Structure

**Full Mark target:** seamless theory-to-code mapping, critical structural choices, limitations of standard implementations, and technically sound customisation for the task.

Required report/notebook evidence:

1. One end-to-end architecture diagram showing the authoritative path:

   `config -> ingest -> validate -> curate -> features -> temporal split -> HPO -> family locks -> final fit -> prediction -> portfolio/RL -> T+1 backtest -> audit -> report`

2. One ETL/point-in-time diagram showing ticker/CIK/security ID, SEC filing availability, as-of joining, target construction and temporal boundaries.
3. A compact theory-to-code table for every reported family:

   | Family | Hypothesis/function family | Training objective/algorithm | Key hyperparameters | Repository implementation | Task-specific limitation |
   |---|---|---|---|---|---|
   | Elastic Net | Linear return function with L1/L2 regularisation | Penalised regression | `alpha`, `l1_ratio` | supervised model and optimisation modules | Bias/linearity; constant cross-sections can imply neutral IC |
   | HistGBT/XGBoost | Additive boosted decision trees | Sequential residual/gradient fitting | depth/leaves, learning rate, L2 | registered supervised components | flexible capacity, instability/overfit risk, importance is not causal |
   | LSTM | Recurrent sequence hypothesis | Backpropagation through time with declared optimizer/loss | lookback, width, depth, dropout, learning rate | deep-sequence component | overlapping sequences and non-stationarity reduce effective sample size |
   | Transformer | Attention-based sequence hypothesis | Gradient optimisation of the declared supervised loss | heads, width, depth, FFN, dropout | deep-sequence component | data/compute burden and weak inductive bias for noisy short panels |
   | DQN | Discrete action-value approximation | Bootstrapped temporal-difference objective | gamma, replay, exploration, target update | registered RL policy | off-policy instability and reward/simulator dependence |
   | PPO | Stochastic actor-critic policy | Clipped policy objective plus value/entropy terms | clip, GAE, gamma, network, learning rate | registered RL policy | on-policy sample cost and reward sensitivity |

4. For each family actually discussed, include:
   - equations in readable notation;
   - exact input/output shapes;
   - how the library implementation realises the algorithm;
   - the model-specific training trace available in artefacts;
   - why its search space and selected parameters are reasonable;
   - an observed failure mode or limitation.
5. Explain the custom task-specific engineering with the strongest research value:
   - point-in-time joins and availability dates;
   - label-aware purge/embargo and nested temporal evaluation;
   - immutable family locks before final holdout;
   - causal upstream prediction features;
   - exact observed T+1/weekly endpoint handling for RL;
   - controlled sensitivity evidence that is explicitly selection-ineligible;
   - transaction-cost and contribution reconciliation.
6. Connect every diagram and formula to a repository module and every claimed behaviour to a test or immutable artefact.
7. Keep breadth under control. The report may list all seven model arms, but the student must choose two or three families for deep viva-ready explanation and label the others as bounded exploratory comparisons.

### 4.3 Criterion C - Model Evaluation and Refinement

**Full Mark target:** rigorous and insightful evaluation, explicit treatment of loss-versus-business-objective mismatch, sophisticated task-aligned metrics, researcher-level reflection, and evidence addressing the research question.

Required report/notebook evidence:

1. Define expected behaviour before results: scores should rank securities by realised five-session return; portfolio conversion should retain useful rank signal after costs and risk controls.
2. Separate three evaluation layers:
   - predictive loss: MAE/RMSE and train/validation generalisation;
   - cross-sectional decision quality: Spearman IC/rank behaviour;
   - economic outcome: after-cost return, volatility, Sharpe, drawdown, turnover, costs, concentration and benchmark-relative return.
3. Make the mismatch central:

   `minimise point forecast error != maximise rank quality != maximise after-cost risk-adjusted return`

4. Give at least one coded, observed example where two layers disagree. Do not invent the example before G1/G2 tables exist.
5. Explain HPO and selection chronology, including candidate budgets, inner/outer folds, seeds, failed trials and family locks.
6. Report training versus validation diagnostics for every family using the metrics the code actually records. Do not label validation metrics as training “loss” where no equivalent loss trace exists.
7. Report all predeclared seeds and risk scenarios. Highlight best values within a table only descriptively; never convert holdout performance into a new chosen winner.
8. Use the predeclared certainty-equivalent/risk-scenario evaluation and realised risk-return curve as the sophisticated task-aligned layer. Explain its assumptions rather than presenting it as a universal custom metric.
9. Analyse generalisation gaps, seed dispersion and risk-scenario sensitivity. Connect empirical gaps to model capacity, regularisation, non-stationarity and effective sample size.
10. Tie future work to observed failure modes. Avoid generic “more data/more models” recommendations.

## 5. Project report production plan

### 5.1 Proposed structure and objective per section

| Section | Objective | Mandatory evidence/output |
|---|---|---|
| Abstract | Summarise question, method, principal descriptive result and limitations without overclaiming | Generated headline values plus explicit exploratory status |
| 1. Problem definition | Translate business need into precise train/deploy I/O and research question | I/O schema, target formula, execution convention, research questions |
| 2. Literature review | Critically synthesise methods, validation and economic evaluation relevant to the implemented system | Verified APA citations; contradiction/gap matrix |
| 3. Data and ETL | Justify sources/universe; document cleaning, PIT joins, coverage and bias | Source table, ETL diagram, before/after counts, missingness/coverage tables |
| 4. Method | Explain architecture, feature/target construction, models, HPO, temporal validation, portfolio/RL and audit | Architecture diagram, theory-to-code table, split timeline, equations |
| 5. Evaluation design | State metrics, loss mismatch, baselines, costs, risk scenarios and descriptive holdout role | Metric dictionary, selection chronology, benchmark contract |
| 6. Results | Present generated evidence without interpretation drift | Data tables, HPO/training diagnostics, performance matrix, curves/frontiers |
| 7. Discussion | Explain implications and competing interpretations | Observed-vs-interpretation table; academic and personal-finance implications |
| 8. Limitations | State data, modelling, validation, execution and inference constraints | Evidence-linked limitation register |
| 9. Future work/deployment | Derive improvements from observed failures and separate research from production | Prioritised roadmap; paper-trading and monitoring gates |
| 10. Conclusion | Answer the research question within the allowed claim status | Bounded conclusions; no promises of live profitability |
| Implementation log | Provide chronological, critical AI-use and knowledge-gap record | See Section 7 |
| References/appendices | Enable checking and reproducibility | APA list, artefact manifest, full metric definitions, public URL |

### 5.2 Literature review work package

The literature review should be a critical synthesis, not a catalogue of papers. Organise it around:

1. cross-sectional return prediction and the economic weakness/noise of short-horizon labels;
2. linear regularisation versus nonlinear trees;
3. sequence models for financial panels;
4. RL portfolio selection and simulator/reward limitations;
5. PIT fundamentals and event availability;
6. temporal validation, leakage and backtest overfitting;
7. prediction metrics versus portfolio outcomes; and
8. risk allocation, costs and benchmark design.

For each theme record: claim, supporting paper, contradictory paper, dataset/horizon, validation design, cost assumptions, relevance to this implementation, and a limitation on comparability.

Do not cite the archived Perplexity review directly. Import only sources whose title, authors, year, venue and DOI/URL have been independently verified. Remove all `[cite:n]` tokens and placeholder-looking identifiers.

### 5.3 Citation workflow

Recommended workflow: **Zotero + Word plugin**, because it is easier to audit and reformat than hand-entered Word citations.

The approved destination is the existing Zotero collection **`ML-Driven Event-Based Systematic Trading Strategy`**.

1. The 28 September readiness re-check confirmed Zotero 7.0.27, the local API and the connector are all running and returning HTTP 200. The relevant read endpoints are operational.
2. Use collection key **`GCZBX6BQ`** for `ML-Driven Event-Based Systematic Trading Strategy`, and search that collection for duplicates before import. Use JSON output for scripted inventory because the plain-text helper can fail when Windows console encoding encounters some existing Unicode metadata; this is a display issue, not an API failure.
3. Import verified papers by DOI, publisher metadata, BibTeX or RIS into that collection only.
4. Reject or quarantine records with placeholder metadata or only secondary-source support.
5. Verify metadata against the primary paper.
6. Attach accessible PDFs where permitted.
7. Tag each item by report section and claim.
8. Insert citations through Zotero in the DOCX and generate APA 7 references.
9. Before export, run a citation audit: every in-text citation in bibliography; every bibliography item cited; no unverified source.

The student's comment authorises this collection as the destination for verified project references. It does not relax the source-verification or duplicate-control requirements.

<!-- RESOLVED C-002: Adopted the named Zotero collection and its verification workflow. -->

<!-- RESOLVED C-016: Re-probed Zotero after the student enabled the API. The API and connector return HTTP 200, and collection GCZBX6BQ is available. No records were imported during this readiness check. -->

### 5.4 Word budget

No formal limit was found in the assignment PDF or reported from Canvas. Use a target of **3,500 words total** (acceptable working range 3,150-3,850), excluding references and generated table/figure contents unless Canvas states otherwise:

| Component | Target words |
|---|---:|
| Abstract and problem definition | 400 |
| Literature review | 650 |
| Data and method | 900 |
| Evaluation design and results | 700 |
| Discussion, limitations, future work and conclusion | 500 |
| Implementation log narrative | 350 |

The compact implementation-log allocation means its detailed chronology should be a generated appendix table; the prose should concentrate on challenges, AI-use critique, verification and student knowledge gaps. If the evidence cannot be explained defensibly in this space, exceed the target rather than delete a rubric-critical explanation.

## 6. Notebook plan and defensibility decision

### 6.1 Chosen design: repository-backed, self-contained Colab

The notebook should be an executable learning document and public entry point to the complete implementation:

1. environment and reproducibility notice;
2. pin/clone exact public repository revision;
3. install pinned dependencies;
4. verify repository and expected artefact/report hashes;
5. explain task I/O and end-to-end architecture;
6. download a declared bounded real-data/artefact bundle from a stable public release;
7. run a bounded demonstration using `python -m trading_pipeline.run` or an explicitly supported smoke configuration;
8. inspect data acquisition, cleaning, PIT features and temporal splits;
9. link theory cells to the actual model classes/functions;
10. show HPO scheduling and evaluation contracts without rerunning the full study;
11. load the immutable expanded report and reproduce all submitted tables/figures;
12. explain final results, limitations and provenance; and
13. link each section to the exact repository files and source revision.

This satisfies self-containment by making a fresh runtime obtain the complete implementation and all required inputs itself. The notebook must not depend on `D:\` paths, hidden credentials, local caches or a private repository.

### 6.2 Notebook acceptance tests

- Public in an incognito browser without requesting access.
- Plain-text URL copied into the master DOCX/PDF.
- “Run all” completes in a clean hosted runtime within a declared practical duration.
- Exact repository commit and dependency lock are pinned.
- All data/download URLs are stable and licence-compatible.
- No API key or private path is required.
- Bounded demonstration is labelled as such.
- Full-study result tables are loaded from immutable artefacts, never implied to have been trained in the short Colab session.
- Every displayed number is generated from downloaded artefacts by code.
- Relative GitHub/repository links resolve.
- Notebook and offline demo use the same archived version.

### 6.3 Rejected routes

- **Toy-only notebook:** fails to demonstrate the complete practical system.
- **Notebook that merely imports local modules:** not self-contained in a cloud runtime.
- **Full pipeline copied into notebook cells:** duplicates authority, is hard to test, and risks inconsistency.
- **Full expanded research rerun in Colab:** unnecessary, slow and inconsistent with the frozen evidence policy.

## 7. Implementation log plan

The implementation log belongs in the final PDF but should be built from evidence before the student writes reflection.

### 7.1 Evidence-backed chronology

For every material stage, generate or verify:

- date/time and source timezone;
- goal and initiating request;
- human decision/approval;
- AI service and model where known;
- files or artefacts changed;
- technical challenge and failure evidence;
- resolution and tests/checks;
- Git commit/run/report identifiers;
- remaining limitation or uncertainty.

Primary sources, in precedence order:

1. Git history and diffs;
2. immutable run, supervisor, failure and report artefacts;
3. `docs/IMPLEMENTATION_LOG.md` and `docs/DECISIONS.md`;
4. Codex thread histories;
5. original ChatGPT thread `Assignment Project Direction`;
6. Perplexity research thread and archived outputs; and
7. student recollection, clearly labelled retrospective.

Do not describe reconstructed entries as contemporaneous. Do not include every conversational turn verbatim; summarise only decisions, actions, failures, verification and learning relevance, while preserving a reference to the source conversation.

### 7.2 AI-use critical review

The log must distinguish:

- research discovery versus source verification;
- planning recommendations versus human approvals;
- code authored/edited by agents versus student review;
- automated tests versus evidence that the student understands the mechanism;
- useful AI acceleration versus risks introduced by scope expansion, unsupported claims, citation errors or leakage.

### 7.3 Student-authored material

Insert prompts, not fabricated first-person prose, for:

- what the student understood before and after each major issue;
- which recommendation they rejected and why;
- which result changed their prior belief;
- remaining knowledge gaps;
- what they did to verify a mechanism;
- what they can explain without reading notes; and
- what would make them distrust the backtest.

## 8. Presentational coding work package

Create a **read-only assignment evidence exporter** that consumes only the verified expanded report and writes a new versioned directory outside `runs/`. It must record source run ID, input paths/hashes, source Git revision, exporter revision and generation timestamp. It must never edit raw run artefacts or silently recompute model outcomes.

### 8.1 Required tables

1. Dataset and cleaning funnel: raw candidates, admitted universe, observations before/after validation, feature-ready rows, split counts.
2. Feature dictionary: definition, lookback, transform, missingness, eligible models.
3. Descriptive feature statistics by split: count, missingness, mean, standard deviation, percentiles, min/max; flag heavy tails and avoid implying normality.
4. Model and search-space table: family, interface, parameters, candidate count, folds, seeds, device.
5. HPO table: candidate, mean/dispersion across inner folds, selection metric and lock result.
6. Training-versus-validation diagnostics matrix using metrics actually recorded for each model.
7. Final descriptive performance table: all arms, seeds and scenarios; predictive and economic measures; SPY benchmark.
8. Winner-format table: best value per metric may be visually highlighted, but footnote that holdout highlights are descriptive and did not select a model.
9. Assumption/limitation register.
10. Provenance table: run/report IDs, hashes, code revision, timestamps and artefact locations.

### 8.2 Required figures

| Figure | Purpose | Source/derivation requirement |
|---|---|---|
| End-to-end system architecture | Criterion B system overview | Diagram source versioned in repository; labels match modules |
| ETL and PIT timeline | Show download/transform/load, filing availability, target and T+1 execution | No numeric values unless generated |
| Temporal split timeline | Show outer/inner folds, purge/embargo, sealed descriptive holdout | Generated from split/protocol manifests |
| Data cleaning/admission funnel | Show before/after observations and exclusions | Generated counts only |
| Feature missingness/coverage | Support data quality discussion | Generated from report/canonical feature summary |
| HPO candidate performance | Show candidate/fold dispersion and selected lock | `hpo_trial_summary` |
| Training diagnostics | Compare generalisation behaviour per family | Recorded traces/summaries only |
| Predictive-metric heatmap | Compare RMSE/MAE/IC without pretending common direction | Normalise or facet carefully; retain raw values in table |
| Long-run equity curves | Compare all eligible model/risk curves with benchmark | `final_testbench_equity_curve`; facet or reduce clutter using predeclared grouping |
| Drawdown curves | Show path risk hidden by terminal returns | Derive deterministically from the same equity table |
| Realised return-volatility chart | Compare all model/seed/scenario observations | `realised_risk_return_curve`; label scenarios and benchmark |
| Model-conditioned frontier | Show supervised expected-return/volatility frontier | `model_conditioned_frontier_points`; bold/annotate the left-most feasible point per curve, not a model “winner” |
| Turnover and cumulative costs | Explain loss-to-business-objective slippage | Generated from reconciled portfolio tables where available |
| Security contribution distribution | Diagnose concentration and outliers | `pipeline_security_summary` |

### 8.3 Terminology and mathematical glossary

Use plain language at first occurrence and retain the academic term in parentheses:

- **Temporally ordered evaluation window (fold):** one declared train/validation partition used to estimate how a model behaves across time. An **inner fold** is used for hyperparameter comparison; an **outer fold** estimates performance after the inner choice has been fixed. Unlike random cross-validation, these windows preserve chronological order and apply the declared purge/embargo rules.
- **Drawdown:** the percentage decline in portfolio value from the highest value previously reached. If equity is \(V_t\), then \(D_t = V_t / \max_{s \leq t} V_s - 1\). **Maximum drawdown** is the most negative \(D_t\) in the evaluation period. It measures path-dependent loss severity, not ordinary return volatility.

Add an appendix glossary containing symbols, formulas, units and plain-language meanings for: feature, target/label, fold, purge, embargo, hyperparameter, information coefficient/Spearman IC, MAE, RMSE, Sharpe ratio, volatility, drawdown, turnover, transaction cost, certainty equivalent, frontier, benchmark-relative return and T+1 execution. The first occurrence in the body should still be understandable without consulting the appendix.

<!-- RESOLVED C-003: Added first-use language and a required technical glossary with fold and drawdown definitions. -->

### 8.4 Rendering standards

- Use a colour-blind-safe palette and consistent model/scenario colours across all figures.
- Export report graphics at print resolution and retain a vector format where practical.
- Put units, sample periods, annualisation basis, transaction-cost convention and descriptive-holdout status in captions or footnotes.
- Avoid dual axes unless essential.
- Do not truncate axes in ways that exaggerate differences.
- Every table/figure gets an ID, title, source line and generated manifest entry.
- Keep dense diagnostic tables in appendices; the report body should show the evidence needed for the argument.

## 9. Delegation and quality-control model

Model recommendations follow the current OpenAI model-selection principle: use lower-power models for well-scoped extraction, a workhorse model for coordinated coding/document work, and the frontier model for complex synthesis and red-teaming. Current official guidance: <https://developers.openai.com/api/docs/guides/model-selection>.

| Work package | Primary executor | Reviewer | Recommended power | Guardrail |
|---|---|---|---|---|
| Run/report identity extraction | Codex | Codex independent pass | Luna/low | Exact values only; hash/source recorded |
| Evidence exporter and notebook engineering | Codex local | Codex code review | Sol/high | Read-only consumers; tests; no result mutation |
| Rubric/evidence matrix | ChatGPT or Codex | Student | Astra/high | Assessor wording mapped line by line |
| Literature discovery | Perplexity research | Student/Codex against primary sources | Highest-accuracy research mode | No source enters paper before metadata and claim verification |
| Literature synthesis | ChatGPT/Codex | Student | Astra/high or xhigh | Contradictions and comparability limits required |
| Report draft | ChatGPT/Codex | Student rewrite and factual audit | Astra/high | No fabricated personal claims; numbers injected from generated tables |
| Implementation chronology | Codex | Student | Sol/high | Evidence precedence; reconstructed entries labelled |
| Student reflection/knowledge gaps | Student | Optional AI clarity edit after authorship | Human first; Luna/medium edit | Preserve meaning and first-person ownership |
| DOCX/PDF assembly | Codex document workflow | Student visual review | Sol/high | APA, page flow, captions, URL, filename |
| Final rubric red-team | Independent ChatGPT/Codex session | Student | Astra/xhigh | Read-only; list unsupported/weak claims rather than rewriting evidence |

## 10. External research prompts

### 10.1 Perplexity prompt - literature refresh and citation audit

```text
Act as a research librarian and critical reviewer for a master's-level machine-learning assignment on short-horizon cross-sectional US equity selection.

Research question:
Can machine-learning models using daily market features and point-in-time public SEC fundamentals improve five-trading-day cross-sectional equity ranking, and does any improvement survive T+1 execution, transaction costs, portfolio construction and risk controls?

Implemented comparison context:
- Elastic Net, HistGradientBoosting/XGBoost, LSTM, Transformer;
- exploratory DQN and PPO portfolio selectors;
- nested chronological validation with purge/embargo;
- final holdout is descriptive only;
- long-only weekly decisions, three predeclared risk scenarios and an SPY benchmark.

Produce a source-verification pack, not report prose. Use primary peer-reviewed papers, authoritative working papers, books from recognised academic publishers, and official documentation only. Prefer papers whose full text you can inspect. Do not use blogs or secondary summaries as evidence.

For each of these themes find the strongest relevant sources and at least one credible contradictory or limiting source where available:
1. cross-sectional equity return prediction with linear, tree and neural models;
2. short-horizon momentum/reversal and publicly available fundamentals;
3. LSTM/Transformer evidence for tabular or panel equity prediction;
4. RL portfolio selection, sample efficiency, reward design and simulator realism;
5. purged/embargoed and nested temporal validation, backtest overfitting and multiple testing;
6. prediction loss/rank correlation versus after-cost portfolio objectives;
7. risk-aware allocation, transaction costs and benchmark design.

Return:
A. a claim-evidence matrix with claim, source, contradiction, dataset, horizon, universe, validation, costs, key quantitative finding and comparability limitation;
B. complete APA 7 metadata, DOI and stable publisher/author URL for every source;
C. an explicit verification status: full text inspected, abstract only, or metadata only;
D. a list of any claims from the attached archived review that cannot be verified or contain suspicious/placeholder bibliographic details;
E. no fabricated identifiers, no citation tokens, and no prose that implies this project's empirical results.

Treat all project descriptions as context, not as evidence. Flag uncertainty rather than filling gaps.
```

### 10.2 Response to the completed Perplexity audit

The downloaded audit was reviewed. The best response is **not** to request all three broad extensions. That would consume time and usage while adding sources that are peripheral to the 3,500-word report.

Use the current output as a discovery pack with three evidence tiers:

- **Core or useful foundations:** Gu, Kelly and Xiu (2020); Fieberg et al. (2022); Novy-Marx and Velikov (2016); DeMiguel et al. (2020); Harvey, Liu and Zhu (2016); McLean and Pontiff (2016); López de Prado (2018). Verify/import only those actually cited, and do not describe Fieberg et al. as US-equity evidence.
- **Contextual but non-comparable:** Wang et al. (2026) is Chinese A-share research with a different evaluation design; Jiang, Xu and Liang (2017) studies cryptocurrency. They may motivate sequence/RL interest but cannot validate this project's US-equity results.
- **Cautious survey evidence:** the 2026 RL systematic review aggregates heterogeneous markets, costs and validation practices. Do not restate its phase-average Sharpe ratios as expected DQN/PPO performance or as a direct comparison with this study.

Primary publisher checks support the central Gu et al. claim that nonlinear trees/neural networks can outperform linear methods in their monthly US-equity setting, while the paper itself warns that deeper networks can lose performance in low-signal financial data. This supports broad academic motivation but not an expectation that this project's Transformer or policy models must win.

The requested bounded corrective follow-up is complete and preserved with the full thread history at `C:\Users\crowl\Downloads\Act as a research librarian and critical reviewer(1).md`. No further Perplexity query is required. Apply its findings as follows:

- Remove or quarantine unresolved `[cite:n]` markers, placeholder DOIs `10.1016/j.frl.2024.105xxx` and `10.1016/j.frl.2025.107xxx`, the incorrect Feng DOI, low-authority substitutes and pooled RL Sharpe ranges.
- Keep claims narrower than the sources: Fieberg et al. is DAX/German rather than direct US-equity evidence; classic momentum horizons are measured in months rather than five days; and purge/embargo requirements must be stated conditionally for overlapping labels rather than as a universal rule.
- Use Kirtac and Germano (2024), DOI `10.1016/j.frl.2024.105227`, and Feng et al. (2020), DOI `10.1111/jofi.12883`, only for the claims their primary records support.
- Jensen et al. (2026), *Machine Learning and the Implementable Efficient Frontier*, DOI `10.1093/rfs/hhag022`, is strong adjacent evidence for a cost-aware portfolio objective, but not a direct five-day design match.
- Hambly, Xu and Yang (2023) supports cautious discussion of RL non-stationarity, simulator and transaction-cost limitations. Jiang, Xu and Liang (2017) remains cryptocurrency evidence and is not a direct US-equity benchmark.
- The audit found no strong directly comparable peer-reviewed US-equity source for either five-day rank-correlation versus after-cost performance or five-day LSTM/Transformer prediction under this project's validation design. State that gap explicitly rather than substituting Chinese-equity, cryptocurrency, monthly-horizon or metadata-only evidence.

Treat the follow-up's verified APA list as an import candidate list, not automatic authority. Before Zotero import and report citation, retain only records actually used, deduplicate them in collection `GCZBX6BQ`, and confirm each report-relevant claim against its primary paper or publisher record.

<!-- RESOLVED C-004: Reviewed the initial audit, rejected broad extensions and limited additional research to one bounded corrective audit. -->

<!-- RESOLVED C-015: Reviewed the completed Perplexity follow-up and converted its corrections, verified identifiers, evidence limits and quarantine list into the execution plan. -->

### 10.3 Independent rubric red-team prompt

```text
Review the attached final PDF, public notebook and evidence manifest as a strict assessor against A2 Criteria A, B and C. Do not rewrite the submission. For every Full Mark descriptor, identify the exact page/cell that supports it, then list missing evidence, unsupported claims, ambiguous task I/O, weak theory-to-code links, loss-versus-objective gaps, methodological leakage risks and questions likely in the viva. Check that every numeric claim is traceable to the evidence manifest and that the final descriptive holdout was not used for selection. Rank findings as submission blocker, high value, or optional polish.
```

## 11. Claims and wording policy

Use wording proportional to evidence:

- Prefer “observed in the sealed descriptive holdout” over “proves”.
- Prefer “was associated with” over “caused” unless the controlled design supports the stated contrast.
- Do not call the left-most frontier point or best holdout Sharpe the selected model.
- Do not imply future profit, production readiness or persistent alpha.
- Separate personal financial implications from financial advice and from scientific evidence.
- Report negative and null results as valid evidence.
- State survivorship/selection bias, free-data limitations, adjusted-price assumptions, simplified costs/fills, multiple comparisons, limited holdout history and non-stationarity.

## 12. Final assembly and verification checklist

### Submission contract

- [ ] One PDF named `STUDENTNAME_STUDENTID_YEAR_UTS_ML_Journal.pdf`.
- [ ] Plain-text public notebook URL appears near the front, not only as an embedded hyperlink.
- [ ] Project report and implementation log are both inside the PDF.
- [ ] Notebook is public and clean-runtime executable.
- [ ] Offline demonstration matches the archived online revision.

### Evidence integrity

- [x] Expanded report generated from `20260928T005423Z-0634efaf` and validated.
- [x] Source commit, protocol hash, report manifest and generation time recorded.
- [ ] Every number/table/chart comes from repeatable code and declared inputs.
- [x] No raw run artefact edited.
- [x] No final holdout result used for selection or report scope decisions.
- [ ] All seeds/scenarios and failed trials reported as required.

### Report quality

- [ ] Training and deployment I/O are testably precise.
- [ ] Research question is feasible, motivated and answered within evidence limits.
- [ ] Architecture and ETL diagrams match the code.
- [ ] Theory-to-code links are exact and viva-ready.
- [ ] Loss-versus-task-objective mismatch has a concrete observed example.
- [ ] Results and interpretation are separated.
- [x] Literature claims and APA metadata are verified for the permitted citation register; one malformed record is explicitly quarantined.
- [ ] No `[VERIFY]`, `[cite:n]`, placeholder DOI, TODO or unresolved review comment remains.

### Implementation log integrity

- [ ] Timeline reconciled to Git, runs, reports and agent threads.
- [ ] Reconstructed entries labelled.
- [ ] AI services, purposes, risks and verification are critically reviewed.
- [ ] Student reflection and knowledge gaps are student-authored.

### Visual and document QA

- [ ] Figures legible at PDF page size and use consistent colours/labels.
- [ ] Tables do not overflow margins or split incorrectly.
- [ ] Captions include sample/period/units/source where relevant.
- [ ] Cross-references and page numbers work.
- [ ] Notebook URL, repository links and DOI links work.
- [ ] PDF opened and visually inspected page by page after export.

## 13. Resolved planning decisions

The student's comments have been reviewed. The resulting decisions are below.

1. **Report emphasis:** Present the full breadth of the predeclared comparison, including LSTM, Transformer, DQN and PPO, because the comparison is academically relevant. However, do not decide narrative depth from descriptive holdout success. Deep theory-to-code treatment should cover Elastic Net, boosted trees and **one sequence or policy family selected for structural relevance and the student's ability to defend it**, not because it won the final holdout. If a deep/RL approach performs notably, report and discuss that observed result without converting it into post-holdout model selection.  
   <!-- RESOLVED C-005: Preserved the student's broad academic interest while preventing holdout-dependent scope selection. -->

2. **Repository publication:** Publish a clean, tagged snapshot in the student's public GitHub repository after secret, licence and large-file checks.  
   <!-- RESOLVED C-006: Public tagged snapshot approved. -->

3. **Public artefact hosting:** Use a GitHub Release attached to the submission tag, subject to size and licensing checks.  
   <!-- RESOLVED C-007: GitHub Release approved. -->

4. **Notebook runtime:** Target 10-15 minutes for the bounded demonstration; load full-study results from immutable release artefacts.  
   <!-- RESOLVED C-008: Runtime target approved. -->

5. **Document constraints:** No additional constraint is known. Target 3,500 words under the allocation in Section 5.4.  
   <!-- RESOLVED C-009: Added the 3,500-word target and allocation. -->

6. **Identity and filename:** Matthew Crowley, student ID 10869473, Spring 2026, subject 32513 Advanced Data Analytics Algorithms (Machine Learning). Use `Matthew_Crowley_10869473_2026_UTS_ML_Journal.pdf` unless Canvas imposes a different filename validator.  
   <!-- RESOLVED C-010: Recorded identity and derived the specification-compliant filename. -->

7. **Literature refresh:** Both the initial Perplexity audit and its completed bounded follow-up have been reviewed. Apply the corrections and evidence limits in Section 10.2; no further Perplexity query is required.  
   <!-- RESOLVED C-011: Incorporated the initial audit and bounded the only justified follow-up. -->
   <!-- RESOLVED C-015 (duplicate response): Incorporated the completed follow-up and closed the external-research action. -->

8. **Zotero availability:** Use the installed Zotero plugin and named collection. The re-check confirms that the local API and connector work; the destination collection key is `GCZBX6BQ`. Import remains a separate controlled step after source selection and duplicate checks.  
   <!-- RESOLVED C-012: Zotero approved and the readiness gate defined. -->
   <!-- RESOLVED C-016 (duplicate response): Verified HTTP 200 API/connector access and resolved the destination collection key. -->

9. **Personal financial implications:** Limit this section to the student's own deployment criteria/risk tolerance and include a clear no-financial-advice boundary.  
   <!-- RESOLVED C-013: Scope approved. -->

10. **Historical conversations:** Use `C:\Users\crowl\Downloads\Act as a research librarian and critical reviewer.md` as evidence for the completed literature-audit interaction. Record any other unavailable Perplexity threads as missing external evidence rather than inferring their contents.  
    <!-- RESOLVED C-014: One exported Perplexity interaction is available; missing conversations remain explicit gaps. -->

## 14. Comment resolution log

When reviewing this plan, add comments at the relevant location. On the next revision, record each resolved comment here.

| Comment ID | Location | Decision/answer | Plan change | Status |
|---|---|---|---|---|
| C-001 | Section 3 | Cap this phase at about 30% of the 49% remaining allowance | Added usage routing and a protected red-team reserve | Resolved |
| C-002 | Section 5.3 | Use the named Zotero collection | Added collection workflow and API verification gate | Resolved |
| C-003 | Section 8 | Explain fold/drawdown and add a glossary | Added definitions, formula and appendix vocabulary | Resolved |
| C-004 | Section 10 | Review initial Perplexity output and its proposed follow-ups | Tiered sources; rejected broad extensions; bounded the corrective audit | Resolved |
| C-005 | Section 13.1 | Preserve academically interesting broad comparison | Broad reporting retained; deep-dive scope cannot be chosen from holdout results | Resolved |
| C-006 | Section 13.2 | Public GitHub snapshot approved | Recorded publication decision | Resolved |
| C-007 | Section 13.3 | GitHub Release approved | Recorded artefact-hosting decision | Resolved |
| C-008 | Section 13.4 | 10-15 minute notebook approved | Recorded runtime target | Resolved |
| C-009 | Section 13.5 | Target 3,500 words | Added section-level allocation | Resolved |
| C-010 | Section 13.6 | Identity supplied | Added final filename | Resolved |
| C-011 | Section 13.7 | Initial literature audit already executed | Incorporated audit assessment and narrow follow-up | Resolved |
| C-012 | Section 13.8 | Zotero available | Added API verification prerequisite | Resolved |
| C-013 | Section 13.9 | Personal-finance scope approved | Recorded boundary | Resolved |
| C-014 | Section 13.10 | One Perplexity thread exported | Added source path and missing-evidence rule | Resolved |
| C-015 | Sections 10.2 and 13.7 | Bounded Perplexity follow-up completed with full history | Reviewed corrections; closed further research; added verified IDs, evidence limits and quarantine rules | Resolved |
| C-016 | Sections 5.3 and 13.8 | Zotero local API enabled | Verified API/connector HTTP 200 and destination collection key `GCZBX6BQ`; no import performed | Resolved |

## 15. Definition of done

Planning review is complete: all current comments have been resolved. Assignment finalisation is complete only when:

1. the single PDF and public notebook meet the explicit submission contract;
2. every Full Mark requirement has a specific evidence location or an acknowledged gap;
3. every numeric or visual claim is generated and traceable;
4. the report distinguishes locked Slice 1, expanded exploratory work and descriptive holdout evidence;
5. the student has authored the personal reflection and can defend the chosen deep-dive models, validation logic, loss mismatch and backtest limitations; and
6. an independent red-team finds no submission-blocking issue.
