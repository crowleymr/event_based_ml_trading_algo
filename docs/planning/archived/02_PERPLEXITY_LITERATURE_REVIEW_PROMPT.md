# Perplexity thread 2: ML for short-term and event-driven equity trading

Copy everything below the divider into a second Perplexity deep-research thread. The requested deliverable is `ML_TRADING_LITERATURE_REVIEW.md`.

---

## Mission and constraints

Act as an academic research assistant and empirical-finance reviewer. Produce a critical literature review to guide a short-term/event-driven systematic equity-trading POC and its subsequent paper.

We have approximately 1.5 days to build a repeatable end-to-end pipeline and the following week to train and compare models. The first stage gate occurs next weekend. The core pipeline is data → leakage-safe features → at least two predictive models → at least two portfolio approaches → backtest → recorded ML/financial metrics → autonomous repeatable runs. Daily data and holding periods of days to weeks are the initial default. US equities are the likely initial universe; Australian applicability is valuable. Free, public, refreshable data is required for implementation.

The aim is credible comparative evidence, not a promise of profitability. News, deep networks, reinforcement learning, and complex infrastructure must not block the weekend vertical slice. A parallel research thread is evaluating Market, Fundamentals, and Corporate Events feeds; do not assume its findings or invent available data.

## Evidence and search protocol

- Find **at least 15 strong, distinct papers**. Target approximately 18–25 if this adds meaningful coverage, combining seminal methods, strong empirical studies, and recent work through the date of the search.
- Prefer peer-reviewed research and authoritative working papers with clear methodology. Identify preprints, publication status, revised versions, and retractions/corrections where relevant. Count a working paper and its published version only once.
- Use original papers and official publisher, DOI, journal, author, institutional, or research-repository pages. Do not base technical conclusions on vendor blogs, search snippets, or AI-generated summaries.
- Verify titles, authors, year, venue/status, DOI or persistent URL, and links to available full text. Never invent bibliographic details, metrics, datasets, code links, or quotations. If only an abstract is accessible, label the evidence abstract-only and narrow the conclusions accordingly.
- State search date, queries, search sources, inclusion/exclusion criteria, and limitations. Call this a structured review unless the search and screening actually meet the requirements of a systematic review. Report real screening counts only if recorded.
- Distinguish direct evidence about tradable short-horizon equities from transferable methods evaluated on other assets, long horizons, synthetic data, or prediction-only tasks.
- Cite every substantive empirical claim near the claim. Quote sparingly; prefer precise paraphrases. Include null/negative findings and methodological critiques.

## Required thematic coverage

Choose papers so that each theme below has substantive coverage; one paper may cover multiple themes. Aim for at least three directly relevant empirical short-horizon/event-equity papers, at least three validation or backtest-methodology papers, and at least two strong sources each on text/NLP, portfolio construction, and RL, within the overall set. If evidence is thin, say so rather than lowering quality silently.

### A. Market and event features

Review short-horizon momentum/reversal, volatility, volume/liquidity, cross-sectional and sector context, market regimes, earnings/filing/announcement reactions, post-event drift, fundamental changes, and interaction features. Distinguish tradable signals from retrospective explanations.

Discuss label choice: next-period return, multi-day forward return, benchmark/sector-relative return, direction, ranking, and event-window returns. Address overlapping labels, event clustering, selection of event windows, and scheduled versus unscheduled events. Explain when consensus-surprise features require unavailable proprietary history.

### B. Linear and tree-based models

Compare naive and regularised linear/logistic baselines with random forests and gradient-boosted trees. Evaluate data efficiency, nonlinear interactions, missing values, calibration, cross-sectional ranking, feature importance stability, and the validity of explainability claims. Separate prediction improvement from after-cost portfolio improvement.

### C. Deep learning

Review relevant MLP, temporal convolution, recurrent/LSTM, attention/transformer, or related models. Identify which evidence supports their use for the proposed horizon and data volume. Assess sequence construction, sample dependence, retraining cost, tuning burden, compute, and reproducibility. Do not equate parameter count or recent publication with superior performance.

### D. Financial text, FinBERT, and NLP

Review financial-domain language modelling, sentiment and event extraction, including FinBERT-family work where relevant. Distinguish text classification benchmark gains from evidence of incremental tradable signal.

Cover news and filing availability, ingestion delay, duplicate/syndicated articles, revised text, entity linking, timestamp accuracy, pretraining-corpus overlap with evaluation periods, historical availability of model weights, and text-label leakage. Explain the distinction between a retrospective research experiment using a modern pretrained model and a historically deployable strategy.

### E. Portfolio construction and risk

Review score-based top-k/rank portfolios, equal weighting, inverse-volatility weighting, volatility targeting, constrained mean-variance optimisation, covariance shrinkage, turnover penalties, and relevant alternative risk objectives. Separate forecasting from allocation and document estimation risk.

For the POC, assess a long-only baseline with explicit cash treatment and risk caps. Treat shorting as an extension requiring defensible borrow availability/cost assumptions. Compare portfolio methods on identical predictions and execution assumptions.

### F. Reinforcement learning

Review strong RL trading/portfolio papers and critical evidence concerning sample efficiency, non-stationarity, reward design, action spaces, transaction-cost modelling, and simulator realism. Distinguish offline/backtest performance from prospective validation. Assess whether RL is a useful later experiment or unjustified for this weekend, with specific prerequisites rather than hype.

### G. Validation, leakage, and backtest overfitting

Cover chronological train/validation/test design, expanding/rolling walk-forward evaluation, nested tuning, purging and embargo where label/information intervals overlap, time-aware cross-sectional splits, fold-local preprocessing and feature selection, and an untouched final test interval.

Explain which split scheme fits each proposed label and why. Do not recommend random row splits for temporally dependent returns. Embargo duration must follow the label/information structure, not a universal arbitrary number.

Review multiple testing, repeated strategy selection, backtest overfitting, selection-adjusted risk measures where appropriate, dependent-observation uncertainty, effective sample size, and robustness across periods/universes. Account for the experiment search budget and avoid selecting a strategy based on its final test performance.

### H. Transaction costs and execution

Cover commissions/fees, bid–ask spread, slippage, turnover, liquidity/participation limits, market impact, execution delays, delistings, dividends/corporate actions, financing and borrow costs where applicable. Distinguish daily-bar assumptions from evidence requiring quotes or order-book data.

Define signal cutoff and first eligible execution time explicitly. A feature containing a day's closing price cannot assume an executable fill at that same close without a defensible order-timing model. Propose after-cost sensitivity analysis and break-even cost estimates, with assumed costs clearly separated from measured costs.

## Paper evidence matrix

Provide a compact overview table and detailed per-paper notes. For every included paper record:

- Verified citation, persistent link, year, venue/status, and full-text access status.
- Theme and relevance: direct equity/horizon evidence or transferable method.
- Research question, market/universe, sample dates, data source, frequency, and prediction/holding horizon.
- Features, target, model, baselines, and portfolio/execution construction.
- Validation/tuning design; leakage and survivorship controls actually documented.
- Costs, turnover, risk metrics, uncertainty and statistical tests actually reported.
- Principal result with exact table/page/section reference where accessible; indicate “not reported” instead of inferring.
- Data/code availability, reproduction obstacles, and compatibility with public free data.
- Main threats to validity and what the POC should adopt, test, or avoid.

Do not compare published Sharpe ratios or accuracies as if they were measured on the same task. Explain differences in horizons, leverage, universes, costs, and validation that prevent direct ranking. Identify evidence gaps rather than filling them with assumptions.

## Translate the literature into a feasible experiment plan

Recommend a small pre-specified comparison ladder:

1. Non-ML investment and simple signal baselines, with benchmark choice aligned to the universe and risk exposure.
2. At least two feasible predictive candidates; explain whether a regularised linear model plus a boosted-tree model is the strongest first comparison, and nominate up to two additional candidates if time permits.
3. At least two feature sets, such as market-only and market plus point-in-time events/fundamentals; separate candidate feature transformations from predictive algorithms. Mark any feature set conditional on the data research results.
4. At least two portfolio/risk approaches evaluated using identical forecasts, universe, rebalance schedule, risk limits, and cost assumptions.
5. A bounded experiment matrix with named hypotheses, run count, seeds where relevant, tuning budget, selection metric, and an untouched final test policy. Avoid a combinatorial grid that cannot complete in a week.

Specify proposed train/validation/test windows as relative rules until dataset history is verified. Explain how to choose horizon and rebalance frequency before final-test inspection. Propose ML metrics appropriate to the target (for example rank correlation for rankings and calibration for probabilities), plus net return, Sharpe with stated annualisation assumptions, drawdown, volatility, turnover, exposure, and concentration. Discuss dependence-aware uncertainty instead of assuming independent daily returns.

Require cost stress tests, feature ablations, period robustness, and an explicit record of failed/negative experiments. Include a low-cost smoke run followed by full runs. Define artefacts such as dataset/version manifest, feature/label configuration, split boundaries, code version, seed, model, predictions, trades/weights, costs, equity curve, metrics, logs, and failure status.

Set a stage gate based on reproducibility, leakage control, completed comparisons, and evidence quality—not mandatory profitability. Distinguish a valid negative result from a broken experiment. Identify which later enhancements would answer a specific unresolved question.

## Deliverable: ML_TRADING_LITERATURE_REVIEW.md

Return a self-contained Markdown report, downloadable if supported, with:

1. Executive synthesis: supported findings, uncertain claims, and practical implications.
2. Search protocol and inclusion/exclusion criteria.
3. Thematic critical review covering A–H.
4. Evidence matrix of at least 15 verified distinct strong papers, plus detailed notes.
5. Cross-paper agreement, contradictions, comparability limits, and research gaps.
6. Weekend POC versus later-work recommendation table, with data/compute dependencies and evidence links.
7. Pre-specified experiment matrix, leakage-safe evaluation and execution policy, cost sensitivity, and stage-gate criteria.
8. **FSD INSERT:** paste-ready model, features, portfolio, validation, metrics, and scope requirements.
9. **CODEX HANDOVER:** ordered implementation tasks, configuration/artefact contracts, meaningful validation checks, and stop conditions. Reconcile with the existing FSD; do not replace its scope blindly.
10. **DATA THREAD DEPENDENCIES:** exact fields, identifiers, timestamps, history, and versioning required by each recommended experiment; alternatives where free data is unavailable.
11. Verified references with DOI/persistent links and a reading order of five highest-priority papers.

Make the result rigorous enough to support an academic report and concrete enough for an implementation agent to act on. Never infer economic usefulness from predictive accuracy alone.
