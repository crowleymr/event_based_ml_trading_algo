# Perplexity thread 1: Free, refreshable equity data feeds

Copy everything below the divider into one Perplexity deep-research thread. The requested deliverable is `DATA_FEED_RESEARCH.md`.

---

## Mission and project context

Act as a research data architect for an academic proof of concept (POC) in short-term/event-driven systematic equity trading. Research and verify current sources; do not simply list familiar vendors.

The implementation window is one weekend of approximately 1.5 days. The following week is available for automated model training and experiments, followed by a stage gate to choose enhancements or proceed with the report. The mandatory vertical slice is data → leakage-safe features → at least two ML models → at least two portfolio approaches → backtest → financial and ML metrics → persisted experiment artefacts → repeatable automated execution.

Find candidates for all three data types—**Market, Fundamentals, and Corporate Events**—for **US and Australian equities where feasible**. A narrower US-first POC is acceptable if evidence shows it is materially more achievable. Daily data and holding periods of days to weeks are a pragmatic default; investigate intraday availability separately without making it a prerequisite. News/NLP is an optional extension, not a requirement for a working first weekend.

## Hard requirements and evidence rules

- Sources must be publicly obtainable, free for the proposed use, and refreshable programmatically. Identify registration/API-key requirements. Distinguish anonymous public access, free registered access, and access restricted to students or institutions.
- Exclude paid trials, card-required trials, expiring promotional credits, single downloadable snapshots without an update mechanism, and datasets whose only access depends on a paid terminal. A periodically updated bulk file can qualify if it has a dependable refresh mechanism.
- No brokerage subscription, funded account, or institutional entitlement should be assumed. Report such sources only as conditional alternatives, outside the default eligible set.
- “Free to access” is not equivalent to permission to automate, store, process with ML, or redistribute. Verify each separately. Do not assume public-domain status or permission from an open-source client library.
- Use current primary documentation, official pricing/free-tier pages, API references, terms, exchange/regulator documentation, and official repositories where available. Secondary evidence can supplement, but label it.
- Date-stamp the report and each source's verification. Link directly to supporting pages. Separate verified facts, reasonable inferences, unknowns, and untested claims.
- Do not fabricate endpoints, limits, schemas, prices, licence rights, or successful tests. Label illustrative requests and inferred schemas as such. If an endpoint can be tested within your environment, report exactly what was tested and its result; otherwise say “not tested.”
- Do not evade access controls, robots restrictions, rate limits, or terms. Treat prohibited or unstable scraping as unsuitable for the default POC.
- If no source meets a requirement, report the gap explicitly and propose a scientifically defensible scope reduction. Never silently substitute a paid feed or snapshot.

## Research questions

### 1. Source coverage by data type and geography

Investigate at least three plausible candidates per data type overall, and at least one plausible Australian candidate per type, if available. Do not pad the results: document unsuccessful searches and disqualifications where the market does not offer a credible free option.

For **Market** data, establish availability of OHLCV, raw versus adjusted prices, adjustment factors, dividends/splits, trading calendars, identifiers, exchange/currency, historical coverage, delisted securities, benchmarks, and ongoing updates. Separate daily from intraday access, historical from live/delayed access, and quote availability from trade/bar availability. Explain whether corrections rewrite historical records.

For **Fundamentals**, investigate periodic financial statements, accounting facts, ratios if available, fiscal periods, units/currencies, publication/filing timestamps, revisions/restatements, and identifier mapping. Distinguish fiscal period end from first public availability. Determine whether historical versions are preserved or only the latest/restated values are returned. Note filing-format and taxonomy differences across US and Australian issuers.

For **Corporate Events**, distinguish:

- Scheduled events such as earnings dates, with original schedule publication and subsequent schedule changes where available.
- Actual releases/filings and announcements, with publication timestamps and event types.
- Corporate actions such as dividends, splits, mergers, ticker changes, and delistings, with announcement, ex, record, payment, and effective dates as applicable.

Assess earnings surprises only if both the forecast and realised value can be obtained with defensible historical availability. Distinguish an event calendar reconstructed today from the calendar known at the historical decision time. Do not treat corporate actions alone as adequate coverage of all event-driven hypotheses.

### 2. Joinability and identifier lifecycle

Design a practical mapping between provider ticker codes and stable company/security identifiers. Investigate available identifiers such as company registration/regulator IDs, exchange-qualified tickers, and security IDs; do not assume a company ID uniquely identifies a share class.

Address exchange suffixes, share classes, dual listings, ticker reuse, renamed companies, ticker changes, mergers, delisted securities, and provider-specific symbols. Require mappings with valid-from/valid-to intervals and provenance when obtainable. Quantify mapping coverage where evidence permits and distinguish measured coverage from estimates.

Specify joins using a stable internal `security_id` plus exchange and effective date, with company-level fundamentals mapped to securities separately. Define an as-of join on information availability; a ticker-plus-calendar-date equality join is insufficient for many records.

### 3. Point-in-time integrity and leakage

For every candidate, answer:

1. What time does the record describe, when did it become public, when did the vendor expose it, and when would our pipeline have received it?
2. Are original versions and subsequent corrections available? Can the source support a genuine historical point-in-time reconstruction?
3. What timestamps, precision, timezone, daylight-saving rules, and market-session conventions are supplied?
4. How should after-hours announcements, weekends, holidays, pre-open releases, missing times, and cross-market timestamps map to the first eligible trade?
5. What survivorship, universe-selection, delisting, corporate-action adjustment, restatement, and future-calendar biases remain?
6. What conservative lag or exclusion policy is defensible, and what bias does it fail to cure?

Explain that collecting ingestion timestamps from now onward does not establish historical availability. Do not claim that a fixed lag repairs unknown revision histories. Label sources as historically point-in-time capable, conditionally usable with documented limitations, prospective-only, or unsuitable for the proposed backtest.

### 4. Operational and legal feasibility

For each source, document:

- Provider, official URL, endpoint/download mechanism, authentication, eligibility, free-tier permanence and restrictions.
- Asset/geography coverage, fields, granularity, history, freshness, delay, expected update schedule, and correction behaviour.
- Requests per second/minute/day/month, concurrent-call restrictions, rows per response, pagination, batch support, bulk downloads, and historical-window restrictions. Use “not documented” where necessary.
- A request-budget estimate for initial backfill and daily refresh at 50, 100, and 500 securities. State assumptions, formula, and whether quotas make the scope practical; avoid false precision about runtime.
- Automation, local caching, retention, academic use, ML processing, attribution, and redistribution rights. Cite relevant terms and flag ambiguity without presenting a legal opinion as fact.
- Reliability evidence, versioning/deprecations, unofficial dependencies, missing-data behaviour, and provider outage risks.
- Retry/backoff and throttling guidance, checkpointed pagination, idempotency/deduplication keys, incremental update strategy, correction windows, caching, raw immutable snapshots, and secrets handling.

### 5. Implementation contracts

Provide a proposed normalised schema and example records for these logical tables; label examples as synthetic unless observed:

| Table | Required concepts |
|---|---|
| `security_master` | Internal security and company IDs, exchange, currency, security type, listing/delisting dates |
| `identifier_map` | Provider, provider identifier, identifier type, company/security link, valid-from/to, source |
| `market_bars` | Security ID, interval/session, raw OHLCV, adjustment data or explicit separate adjusted fields, source |
| `fundamental_facts` | Company ID, fact/taxonomy, value, unit, fiscal period, filing/release ID, publication and revision times |
| `corporate_events` | Company/security link, event ID/type, scheduled/actual status, publication time, event/effective dates, revision |
| `ingestion_manifest` | Source, request/run ID, retrieval time, coverage, checksum, raw path, schema version, status |

Specify types, nullability, unique keys, timezone conventions, provenance, and applicable `event_time`, `published_at`, `available_at`, `ingested_at`, and revision fields. Do not invent provider timestamps to fill the schema. Explain how unknown availability is represented and handled.

Include one documented minimal request or bulk-download example per shortlisted source, with credentials represented only by placeholders. Include mapping and as-of-join pseudocode, and one worked US and one Australian example if feasible. Show how information arriving after the decision cutoff is excluded and how execution occurs only at a subsequent tradable time.

## Required comparison and scoring

Produce a coverage matrix spanning all six data-type/geography combinations and a source-level comparison table. Include eligibility, point-in-time classification, fields/history, joinability, quotas, licence status, implementation effort, confidence, and citations. Split wide tables if needed for readability.

Apply eligibility gates before ranking: continuing free access, permitted refresh mechanism, adequate fields, and access available to this project. Mark uncertain gates as conditional, not passed. Point-in-time adequacy is a separate gate for use in historical predictive experiments; a source may remain useful for prospective collection.

For eligible candidates, score 0–5 against the following weights and calculate `sum(weight × score / 5)` out of 100:

| Criterion | Weight |
|---|---:|
| Historical point-in-time integrity | 25 |
| Coverage and history for the proposed scope | 20 |
| Joinability and identifier quality | 15 |
| Refresh reliability and quota feasibility | 15 |
| Clarity of rights for intended use | 15 |
| Weekend implementation simplicity | 10 |

Explain each score briefly. Unknown evidence should reduce confidence and prevent a firm recommendation where it concerns a gate; do not turn a high weighted score into proof of suitability. Report Australian coverage separately so US breadth cannot hide an Australian gap.

## Deliverable: DATA_FEED_RESEARCH.md

Return a self-contained Markdown report, downloadable if supported, with this structure:

1. Executive recommendation and dated evidence summary.
2. Requirements, assumptions, eligibility gates, and unresolved questions.
3. Coverage matrix and candidate comparison/scoring tables.
4. Source dossiers and verified access examples for the shortlist.
5. Identifier mapping and normalised schemas.
6. Point-in-time policy, trading-time alignment, and remaining biases.
7. Initial backfill and refresh budgets; operational design and failure recovery.
8. **Weekend POC recommendation:** primary and fallback source per type, geography, feasible universe size, history, frequency, refresh schedule, concrete first-day/second-day sequence, and what is deferred. Explicitly distinguish the core market-data vertical slice from event/fundamental extensions that fail historical integrity checks.
9. Acceptance checks: a representative backfill, repeat refresh without duplicate rows, quota compliance, identifier-match report, missingness report, timestamp/leakage audit, corporate-action check, outage recovery, and provenance artefacts. Give measurable proposed pass criteria without pretending they were tested.
10. **FSD INSERT:** concise paste-ready data requirements, selected source contracts, eligibility/limitations, refresh cadence, and acceptance criteria.
11. **CODEX HANDOVER:** implementable tasks, dependencies, proposed configuration keys, artefact locations, test fixtures, stop conditions, and decisions requiring evidence. Instruct the implementation agent to read the existing FSD and reconcile changes instead of replacing it blindly.
12. Sources with direct links and access/verification dates; rejected candidates with reasons.

Optimise for a defensible, repeatable academic POC. If a requested free dataset does not exist or cannot be verified, a clear limitation and executable narrower design is a better result than an unsupported recommendation.
