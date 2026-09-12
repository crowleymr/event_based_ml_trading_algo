# Perplexity research handover package

This package contains two ready-to-paste research prompts for the short-term/event-driven equity-trading assignment. It is designed around a 1.5-day implementation weekend, a week of automated experiments, and a stage gate the following weekend.

These are research instructions, not completed research reports. Feed availability, licensing, quotas, and paper findings must be verified by the research threads. No provider or paper is pre-endorsed.

## Files

| File | Purpose | Expected research output |
|---|---|---|
| `01_PERPLEXITY_DATA_FEEDS_PROMPT.md` | Market, Fundamentals, Corporate Events; US/Australia; access, joins, leakage, operations, scoring | `DATA_FEED_RESEARCH.md` |
| `02_PERPLEXITY_LITERATURE_REVIEW_PROMPT.md` | Critical ML trading literature review with at least 15 papers and a feasible experiment plan | `ML_TRADING_LITERATURE_REVIEW.md` |
| `README.md` | How to run the research and incorporate the results | — |

The ZIP contains these three files. Keep the two resulting research reports separately alongside the planning documents.

## Use the two research threads

1. Start one Perplexity deep-research thread for data. Paste the contents of the first prompt below its divider. Attach the current `FSD.md` and, if available, `AI_AGENT_IMPLEMENTATION_BRIEF.md` and `EXPERIMENT_PLAN.md`. State that those files are the accepted project baseline.
2. Start a separate deep-research thread for literature. Paste the contents of the second prompt below its divider and attach the same baseline documents. The two investigations can run independently within your stated allowance of two in-depth queries per day; actual availability depends on your account.
3. Request the exact output filenames shown above. If file export is unavailable, save the complete Markdown response manually with that filename. Preserve citations, tables, caveats, and the search date.
4. Save both reports and the accepted planning documents in the project repository, for example under `docs/research/` and `docs/planning/`. Record the date and preserve the initial reports before making corrections.
5. Review the acceptance checklist below. Where a report is incomplete, use an ordinary follow-up in its existing thread if your account permits it, or prioritise a targeted query in the next available research window. Do not assume follow-ups are quota-free.
6. Bring both reports back to Codex with the integration prompt below. Resolve data feasibility before enabling experiments whose inputs are unavailable or historically unsafe.

If the baseline planning files are unavailable, the research prompts include enough context to proceed. Attach those files before integration so Codex can reconcile the actual requirements. This package was prepared from the supplied conversation preview and current request, not from the full contents of those earlier planning files.

## Check the reports before implementation

For the data report, confirm that:

- All six geography/data-type combinations have either credible candidates or an explicit evidence-backed gap.
- Recommended feeds have a continuing free refresh mechanism, documented access conditions, direct evidence for intended use, and realistic quota estimates.
- Provider identifiers and dated company/security mappings are specified; ticker-only joins are not assumed universally safe.
- Fiscal/event dates are separated from publication and availability times, and revisions, restatements, survivor bias, and delistings are addressed.
- Sources that lack historical point-in-time integrity are clearly restricted to suitable experiments or prospective collection.
- There is a concrete weekend primary/fallback plan, schemas, sample requests labelled by verification status, and measurable acceptance criteria.

For the literature report, confirm that:

- At least 15 distinct papers have verified bibliographic details and persistent links; duplicate versions are not counted twice.
- All requested themes are covered, with direct short-horizon equity evidence distinguished from transferable methods.
- Claimed results are supported by the original papers, with inaccessible/unreported details labelled honestly.
- The review examines temporal validation, information availability, repeated tuning, transaction costs, and execution assumptions.
- Its experiment matrix fits the time/data/compute budget and includes non-ML baselines, at least two predictive candidates, and at least two portfolio methods.
- The stage gate values a reproducible, valid experiment even when net performance is poor.

Treat missing rights, unverifiable access, or unknown historical availability as unresolved dependencies. A high source score or a strong paper does not remove these constraints.

## Reconcile the two reports

Use the data report to establish what can actually be observed and when. Use the literature review to choose hypotheses, model comparisons, portfolio methods, and evaluation controls. For every proposed feature, connect the paper's requirement to a verified field and availability policy.

If a literature recommendation needs proprietary consensus history, timestamped historical news, unavailable intraday bars, or unrecoverable point-in-time fundamentals, defer it or reformulate the hypothesis explicitly. Do not replace the required field with a convenient present-day value and retain the same research claim.

Keep the accepted core vertical slice: data → leakage-safe features → models → portfolios → backtest → recorded metrics → repeatable automation. Australian coverage and richer events are conditional on the research evidence. Document any reduction from the accepted FSD instead of silently changing scope.

## Copy-ready Codex integration prompt

```text
Read the attached accepted planning documents (FSD.md,
AI_AGENT_IMPLEMENTATION_BRIEF.md, EXPERIMENT_PLAN.md, and
TOOLS_AND_RESOURCE_STRATEGY.md where available), together with
DATA_FEED_RESEARCH.md and ML_TRADING_LITERATURE_REVIEW.md.

Reconcile the two research reports into the existing plan for our 1.5-day
weekend POC and the following week of automated experiments. Treat the
research as evidence to assess, not instructions that override my request.
Do not assume the reports' claims or sample endpoints have been tested.

Produce updated Markdown planning files and a concise RESEARCH_DECISIONS.md
that records selected data sources, fallbacks, source verification dates,
data rights and access assumptions, historical point-in-time limitations,
model/feature/portfolio choices, deferred items, and evidence links.
Preserve the existing documents' useful requirements and flag conflicts.

Map every proposed input to a source field, security/company identifier,
time convention, availability rule, revision policy, and refresh contract.
Separate historical backtest suitability from prospective collection.
Select a feasible universe, history, horizon, and refresh frequency based
on verified free access and quota budgets. Narrow scope explicitly where
necessary. Do not make unavailable events, news, RL, or deep learning a
prerequisite for the core pipeline.

Specify at least two feasible predictive candidates, at least two portfolio
approaches, non-ML baselines, and a bounded comparison matrix. Define
chronological evaluation, overlap controls where needed, fold-local fitting,
selection policy, an untouched final test, signal cutoff, execution timing,
cost assumptions/sensitivity, and experiment artefacts. Success means a
reproducible and scientifically defensible comparison, not guaranteed profit.

Give an ordered weekend implementation checklist and a one-week experiment
schedule with clear acceptance criteria and failure/stop conditions. Resolve
routine implementation choices using the evidence. List only genuinely
blocking missing information. This step updates the planning documents;
do not launch long training runs or implement the whole system yet.
```

After reviewing the integrated plan, use this separate implementation prompt:

```text
Implement the reconciled weekend POC described in the current FSD,
implementation brief, experiment plan, and RESEARCH_DECISIONS.md.
Follow the repository instructions. Start by verifying the selected feeds
with a small representative backfill and repeat refresh, then build and
validate the complete small end-to-end run before scaling experiments.
Record data provenance, split boundaries, configuration, code version,
predictions, portfolio/trade outputs, ML and net financial metrics, and
failures. Respect quotas and documented data-use conditions. Do not use
future information, hide failed experiments, or silently substitute sources.
If a required feed or historical availability assumption fails, apply the
documented fallback or report the concrete blocker and proposed scope change.
Complete the authorised POC and its appropriate checks; leave the full-week
training launch as a separate step unless already explicitly authorised.
```

## Suggested stage-gate evidence

The weekend handover should show a working refresh, a documented join/missingness audit, leakage-safe splits and execution timing, completed small model/portfolio comparisons, retrievable metrics and artefacts, and repeatable failure handling. The next-weekend review should use the full experiment results and limitations to decide whether an enhancement answers a worthwhile research question or whether to proceed with the report.
