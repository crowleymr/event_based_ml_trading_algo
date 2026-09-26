# AI interaction evidence pack

**Prepared:** 2026-09-26  
**Purpose:** Factual repository-backed support for the student's review and reflection. This record does not represent the student's personal reflection, understanding, critique, or knowledge gaps.

## Evidence boundary

The repository contains an AI assistance register, dated implementation records, design decisions, source code, tests, immutable-run references, and Git history. These establish documented work and verification. They generally do **not** preserve the originating assistant, conversation URL, exact prompt, or full response. Unless a primary transcript or commit attribution establishes the source, the interaction source is recorded as **unavailable**; it is not inferred to be Codex, ChatGPT, or Perplexity.

The entries below summarize repository evidence rather than reconstructing missing conversations. Verification is reported as recorded in the repository, not independently re-run for this pack. Numeric experiment results are intentionally omitted; consult the cited generated artefacts and logs for those.

## Dated assistance and verification evidence

| Date / phase | Challenge and documented AI-assisted work | Decision or outcome evidenced in repository | Verification recorded | Source attribution and references |
|---|---|---|---|---|
| 2026-09-12, data and PIT features | Implemented cached SEC/Yahoo ingestion, identifier mapping, canonical schemas and consumed-file hashes; strict as-of fundamentals joins with filing provenance; temporal splits with label-end purge and post-boundary embargo. | Decisions document ticker correction MMC→MRSH for the same issuer using official mapping/announcement evidence; split and execution conventions are recorded. | `docs/AI_USE_AND_VERIFICATION.md` records schema/idempotency, mapping, synthetic temporal tests and persisted-run availability audit as verified. Git history includes `e159ceb`, `c9dea18`, `04a4113`. | Assistant/vendor, thread, prompts and responses: unavailable. `docs/AI_USE_AND_VERIFICATION.md`; `docs/DECISIONS.md`; Git commits `e159ceb`, `c9dea18`, `04a4113`. |
| 2026-09-12, model selection and backtest | Implemented validation-only model selection with a pre-test selection lock; T+1 close execution, drifted holdings and self-financing traded-dollar cost accounting. | The test is reserved for descriptive evaluation; selection uses validation evidence. Execution timing, costs and split policy are documented in `docs/DECISIONS.md`. | The register records test-label perturbation, selection/prediction audits, accounting/unit tests, and cost/equity reconciliation as verified. Git history includes `659aa92`, `9cadfe9`. | Assistant/vendor, thread, prompts and responses: unavailable. `docs/AI_USE_AND_VERIFICATION.md`; `docs/DECISIONS.md`; Git commits `659aa92`, `9cadfe9`. |
| 2026-09-12 to 2026-09-13, reproducibility and architecture | Implemented run-local snapshots, source hashes, seed/thread/device metadata and modular package boundaries; structured architecture and assignment support documentation. | Kept a layered local application with `run.py` as orchestrator; reports and notebooks are read-only derivatives. Student-authored reflection remains separate. | Register/log record seeded smoke runs, environment tests, persisted audits, full tests and architecture review. The 2026-09-13 log describes source-to-document checks and immutable reference-manifest comparisons. | Assistant/vendor, thread, prompts and responses: unavailable. `docs/AI_USE_AND_VERIFICATION.md`; `docs/IMPLEMENTATION_LOG.md` (2026-09-13 entries); `docs/DECISIONS.md`; commits `50db641`, `cd728f8`, `0d0bc29`. |
| 2026-09-13, reporting | Added deterministic, read-only reports derived from an immutable run, with tables, reconciliation and provenance. | Reports are versioned derivatives; source run artefacts remain immutable. | Log records deterministic derivation, schema/label checks, required-input failure, numeric reconciliation, version refusal and regression verification. | Assistant/vendor, thread, prompts and responses: unavailable. `docs/IMPLEMENTATION_LOG.md` (2026-09-13 “Repeatable coded reporting baseline”); commit `a902f97`. |
| 2026-09-19, RL pilot | Added a discrete selector over frozen strategy sleeves and a separate exploratory RL path. | Frozen MDP, timing, reward, costs, seeds and descriptive test boundary are documented; continuous control and tuning are deferred. | Phase 2 integration verification is recorded in the implementation log and commit history. | Assistant/vendor, thread, prompts and responses: unavailable. `docs/DECISIONS.md` (2026-09-19 decision); `docs/IMPLEMENTATION_LOG.md`; commits `3a45b2d`, `9b28ed2`, `60d213e`. |
| 2026-09-20, diagnostics and reporting | Added XGBoost separately from the frozen HistGBT selection set, model-training telemetry, CUDA diagnostics and read-only multi-run reporting/dashboard work. | E5 eligibility remains unchanged; use of the already-observed final test is marked diagnostic reproduction, not model selection. Missing SHAP output was not fabricated. | Log records tests, smoke runs, persisted audits, report provenance, dashboard checks, and CUDA parity/timing diagnostics. | Assistant/vendor, thread, prompts and responses: unavailable. `docs/IMPLEMENTATION_LOG.md` (2026-09-20 entries); `docs/DECISIONS.md`; commits `7be36cc`, `4d9b11e`, `e038888`, `0992aa3`. |
| 2026-09-20, planning record | Scratchpad planning material was added. | The commit proves a scratchpad change only; it does not establish the identity or contents of any external interaction. | Git commit exists; no transcript is included in the reviewed repository records. | Source of scratchpad ideas: unavailable. `docs/planning/prompt-scratch_pad.md`; commit `0284bb1`. |
| 2026-09-20 to 2026-09-26, expanded research engineering | Added research-disabled Phase 3 model/evaluator foundations, deep sequence models, registered RL runner, and bounded universe-admission preflight. | New adapters remain gated; no score-bearing HPO or policy selection is recorded in the 2026-09-26 work. Legacy configuration and immutable evidence are protected. | The 2026-09-26 log records focused test outcomes for the evaluator/RL and deep sequence lanes; universe preflight tests were added, but execution was blocked by managed Windows temporary-directory access. The log explicitly records that no live preflight ran. | Assistant/vendor, thread, prompts and responses: unavailable. `docs/IMPLEMENTATION_LOG.md` (2026-09-20 and 2026-09-26 entries); `docs/planning/FINAL_5_DAY_EXPANDED_RESEARCH_CLOSEOUT_PLAN.md`; commit `1af688e`. |

## Current close-out decisions evidenced in the plan

The 2026-09-26 final five-day plan records these resolved directions: attempt a new immutable data vintage through the latest available date; treat the close-out holdout as descriptive and persistent superiority as requiring future observations; study survivorship/delisting sources and disclose limits if a defensible historical source cannot be obtained; keep raw caches and large run/model artefacts private; make the notebook an executable read-only showcase over repository components; freeze and report the full conservative/balanced/aggressive sensitivity grid; and judge methodological success by reproducible point-in-time, after-cost evaluation, not by requiring market outperformance. These are plan statements, not evidence that their future work packages have completed.

Reference: `docs/planning/FINAL_5_DAY_EXPANDED_RESEARCH_CLOSEOUT_PLAN.md`, especially sections 4, 5, 8, 10 and 11.

## Interaction transcript and link register

No Codex, ChatGPT, or Perplexity transcript/export or conversation link was found in the materials reviewed for this pack. Fill a row only from a source the student can provide or an accessible conversation record. Do not recreate prompt wording or response details from memory.

| Product/source | Date | Link or supplied export path | Purpose / prompt summary | Advice or work produced | Accepted, changed, or rejected decision | Verification and repository references |
|---|---|---|---|---|---|---|
| Codex / ChatGPT / Perplexity / other (student to identify) | Student to supply | Student to supply | Student to supply from transcript | Student to supply from transcript | Student to complete | Student to link evidence |
| Codex / ChatGPT / Perplexity / other (student to identify) | Student to supply | Student to supply | Student to supply from transcript | Student to supply from transcript | Student to complete | Student to link evidence |

## Student reflection prompts

The prompts below are copied from the repository's AI-use register. They are questions for the student, not claims about the student's experience. The student should write their own responses after reviewing the work and evidence.

- Which AI-assisted action did you verify most deeply, and how?
- Identify one limitation or mistaken proposal from AI output and explain your correction.
- What concept did you learn well enough to explain without AI assistance?
- Which knowledge gap remains, and what concrete verification will close it?

## Evidence currently unavailable

- Conversation URLs, transcript exports, exact prompt text, complete assistant responses, and per-interaction product/model identity for the implementation history above.
- Any evidence that the 2026-09-20 scratchpad content came from a particular AI product or conversation.
- Student-authored reflection, personal understanding, critique, and knowledge-gap answers. These must come from the student.
- Transcript-derived records for any Perplexity or ChatGPT conversations not exported into or linked from the repository.
- Completion evidence for future close-out work packages that the plan marks as planned rather than complete.

