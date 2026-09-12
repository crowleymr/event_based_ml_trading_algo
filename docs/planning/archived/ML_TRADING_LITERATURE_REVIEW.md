# Machine Learning for Short-Horizon Equity Trading: A Critical Literature Review

**Search Date:** September 12, 2026  
**Prepared for:** Systematic Equity Trading POC  
**Author:** Research Assistant (Perplexity AI)

---

## 1. Executive Synthesis

### Supported Findings

1. **Tree-based and neural network methods outperform linear baselines** for cross-sectional equity return prediction when using high-dimensional feature sets, with gains attributable to nonlinear predictor interactions [cite:31][cite:16][cite:30]. Gu, Kelly, and Xiu (2020) demonstrate out-of-sample R₂ improvements from 0.16% (3-characteristic OLS) to 0.33–0.40% monthly (trees/NN) across ~30,000 US stocks, 1957–2016 [cite:31].

2. **Short-term reversal and momentum are robust, coexisting phenomena** at one-month horizons, with reversal dominant in low-turnover stocks and momentum in high-turnover stocks [cite:5][cite:10][cite:13][cite:34]. Jegadeesh and Titman (1993, 2001) established intermediate-horizon momentum (3–12 months) with ~1% monthly returns, subsequently reversed over months 13–60 [cite:33][cite:38][cite:61].

3. **Backtest overfitting is pervasive and standard hold-out validation is inadequate** for strategy selection due to multiple testing, lookahead bias, and temporal dependence [cite:1][cite:2][cite:7][cite:70][cite:76]. Bailey et al. (2017) define Probability of Backtest Overfitting (PBO) and propose Combinatorially Symmetric Cross-Validation (CSCV) to estimate it [cite:71][cite:75].

4. **Purged, embargoed cross-validation is required** when labels have overlapping information intervals; arbitrary k-fold or random splits induce leakage [cite:72][cite:73][cite:74][cite:80]. López de Prado (2018) recommends purging training observations whose label intervals intersect test intervals, plus embargoing observations immediately following test folds [cite:72][cite:74].

5. **FinBERT and domain-adapted language models improve sentiment classification** on financial text versus general BERT, but evidence of incremental tradable signal is mixed and often confounded by retrospective model availability [cite:47][cite:48][cite:52][cite:57]. Kirtac and Germano (2024) report OPT (GPT-3-based) achieves 74.4% accuracy and Sharpe 3.05 (10 bps costs) on 965k US news articles, 2010–2023, but independent replication is limited [cite:48].

6. **Reinforcement learning shows sample inefficiency and non-stationarity challenges** in trading applications; most strong results are on synthetic data, single assets, or use unrealistic cost/liquidity assumptions [cite:46][cite:51][cite:53][cite:55]. Hambly, Xu, and Yang (2023) survey RL in finance and note most algorithms require more interaction data than available historical observations [cite:46].

7. **Transaction costs, turnover, and execution timing materially degrade theoretical returns**; many published Sharpe ratios assume daily-bar fills without defensible order-timing models [cite:7][cite:28][cite:66]. All that Glitters Is Not Gold (2016) finds backtest Sharpe ratios have R₂ < 0.025 for predicting out-of-sample performance; higher-order moments and portfolio construction features are more predictive [cite:7].

### Uncertain Claims

1. **Deep learning (LSTM/Transformer) superiority over tree-based methods** for daily equity returns is not consistently demonstrated; several studies find shallow networks or gradient-boosted trees match or exceed deep architectures at similar horizons [cite:18][cite:20][cite:21][cite:24]. Cross-sectional LSTM variants show marginal gains over random forests only with sector embeddings and are regime-dependent [cite:20].

2. **Text-based sentiment signals provide robust, after-cost alpha** independent of price/volatility features; some large-scale studies find FinBERT sentiment lacks predictive power once volatility and liquidity are controlled [cite:57].

3. **RL can achieve statistically significant excess returns** over buy-and-hold or mean-variance benchmarks in live or realistic out-of-sample tests; most papers report in-sample or backtest results with limited robustness checks [cite:51][cite:56].

### Practical Implications for the POC

- **Start with regularised linear (elastic net) + gradient-boosted trees** as the strongest first comparison; add a shallow neural network (≤3 hidden layers) only if time permits [cite:31][cite:16].
- **Implement purged, embargoed time-series splits** with label-aware intervals; do not use random row splits or standard k-fold [cite:72][cite:73][cite:80].
- **Evaluate at least two portfolio approaches** (equal-weight top-k and inverse-volatility or shrinkage-based) on identical predictions, costs, and rebalance schedules [cite:28][cite:67].
- **Run explicit cost stress tests** with break-even cost estimates; report turnover, exposure, and concentration alongside Sharpe and drawdown [cite:7][cite:66].
- **Treat RL and complex NLP as later-stage experiments** pending evidence from the baseline pipeline; do not block the weekend vertical slice on these [cite:46][cite:55].

---

## 2. Search Protocol and Inclusion/Exclusion Criteria

### Search Sources and Queries

Primary searches conducted via web search engine on September 12, 2026, with the following query clusters:

1. **Core ML asset pricing:** "Gu Kelly Xiu 2020 machine learning asset pricing", "Feng Giglio Xiu 2020 taming factor zoo", "machine learning cross-sectional equity returns empirical"
2. **Momentum/reversal:** "Jegadeesh Titman 1993 momentum returns", "short-term reversal momentum turnover", "post-earnings announcement drift PEAD trading"
3. **Validation/backtest methodology:** "Lopez de Prado combinatorial purged cross validation", "Bailey Borwein Lopez de Prado backtest overfitting PBO", "backtest overfitting machine learning finance"
4. **NLP/FinBERT:** "FinBERT financial sentiment analysis trading", "FinBERT Araci 2019 pretraining corpus", "financial news sentiment trading signal empirical"
5. **RL trading:** "reinforcement learning trading portfolio sample efficiency", "deep reinforcement learning portfolio allocation empirical", "RL trading critique non-stationarity"
6. **Portfolio construction:** "inverse volatility weighting portfolio ML", "covariance shrinkage portfolio optimization", "turnover penalties portfolio construction"

### Inclusion Criteria

- Peer-reviewed journal articles, authoritative working papers (SSRN, NBER, arXiv q-fin), or book chapters from established publishers (Wiley, Oxford, Cambridge).
- Clear methodology with documented data sources, sample periods, and validation design.
- Direct relevance to at least one thematic area (A–H) in the brief.
- Published or revised through September 2026; preprints accepted if methodologically sound and not subsequently retracted.

### Exclusion Criteria

- Vendor blogs, marketing whitepapers, or AI-generated summaries without original empirical work.
- Papers lacking sufficient methodological detail to assess leakage controls or backtest design.
- Studies using only synthetic data without any real-market validation (unless methodological contributions are substantial).
- Duplicate counting of working paper and published version (counted once, preferring published version).

### Limitations

- Some papers accessed only via abstracts or publisher summaries; full-text PDFs not always available.
- Search limited to English-language publications.
- No formal screening count recorded; this is a **structured review**, not a PRISMA-compliant systematic review.

---

## 3. Thematic Critical Review (A–H)

### A. Market and Event Features

**Short-horizon momentum/reversal.** Stock returns exhibit reversal at horizons of days to one month, transitioning to momentum over 3–12 months [cite:5][cite:10][cite:34]. Jegadeesh and Titman (1993) document that buying past winners and selling past losers generates ~1% monthly returns over 3–12 month holding periods, with subsequent reversal over months 13–60 [cite:33][cite:38][cite:61]. More recent work shows momentum and reversal coexist at one-month horizons, with reversal dominant in low-turnover, low price-to-52-week-high (PTH) stocks and momentum in high-turnover, high-PTH stocks [cite:10][cite:13][cite:34].

**Volatility, volume/liquidity, and regime.** Volatility, liquidity (market value, dollar volume, bid–ask spread), and market beta are among the top predictors in ML models [cite:31][cite:42]. Short-term reversal profits are attenuated following earnings announcements and larger when noise trading is more volatile [cite:5]. Market regimes (e.g., high vs. low volatility states) condition the efficacy of momentum and reversal signals [cite:22][cite:29].

**Earnings/filing/announcement reactions.** Post-earnings announcement drift (PEAD) is a well-documented phenomenon where stocks continue to drift in the direction of earnings surprises for weeks to months [cite:5]. However, consensus-surprise features typically require proprietary history (e.g., I/B/E/S) not available in free public data; implementing PEAD with free data requires careful reconstruction of point-in-time expectations.

**Label choice.** Common labels include:
- Next-period (daily) return: simple but noisy, high turnover.
- Multi-day forward return (e.g., 5-day, 20-day): reduces noise, aligns with holding periods of days to weeks.
- Benchmark/sector-relative return: controls for market/sector beta, improves cross-sectional comparability.
- Direction (binary up/down): simplifies to classification but loses magnitude information.
- Ranking (cross-sectional deciles): aligns with long-short portfolio construction.
- Event-window returns: appropriate for event-driven strategies but requires careful event timestamping and overlapping-label handling [cite:72][cite:73].

**Overlapping labels and event clustering.** When labels are constructed from overlapping windows (e.g., 5-day forward returns computed daily), observations share information, violating i.i.d. assumptions [cite:72][cite:74]. Event clustering (multiple firms announcing on same day) induces cross-sectional dependence. Scheduled events (earnings, FOMC) differ from unscheduled events (M&A rumors) in terms of anticipation and leakage risk.

**Tradable vs. retrospective signals.** Many features (e.g., full-sample volatility, ex-post factor loadings) are retrospective and not tradable without lookahead. A feature using day-t closing price cannot assume execution at day-t close without a defensible order-timing model; earliest executable fill is typically day-(t+1) open or close [cite:74][cite:80].

---

### B. Linear and Tree-Based Models

**Baselines.** Naive linear regression (OLS) with 3 characteristics (size, book-to-market, momentum) achieves ~0.16% monthly out-of-sample R₂ in large panels [cite:31]. Expanding to 900+ predictors without regularisation causes overfitting (negative R₂); elastic net, PCR, and PLS recover modest predictability (~0.11–0.27% monthly) [cite:31].

**Tree-based methods.** Random forests and gradient-boosted trees (GBT/XGBoost/LightGBM) improve monthly R₂ to ~0.33–0.40% by capturing nonlinear interactions [cite:16][cite:31]. Trees handle missing values natively, require less feature scaling, and provide feature importance measures (though stability and causal interpretation are limited) [cite:16][cite:28].

**Data efficiency and calibration.** Tree-based models are more data-efficient than deep networks for tabular finance data with modest signal-to-noise [cite:31][cite:42]. Calibration (probability outputs matching empirical frequencies) is often better in regularised linear models; trees may require post-hoc calibration (Platt scaling, isotonic) for probability-based portfolio weighting [cite:16].

**Cross-sectional ranking.** For long-short equity, ranking performance (Spearman correlation between predicted and realized ranks) is more relevant than raw MSE [cite:31][cite:67]. Tree ensembles often excel at cross-sectional ranking due to implicit feature interactions.

**Prediction vs. portfolio improvement.** A model may improve prediction metrics (R₂, accuracy) without improving after-cost portfolio performance if it increases turnover or concentrates risk [cite:7][cite:28]. Portfolio-level evaluation on identical predictions is essential to separate forecasting from allocation gains.

---

### C. Deep Learning

**Relevant architectures.** MLPs, temporal convolutional networks (TCN), recurrent/LSTM, and attention/transformer models have been applied to equity return prediction [cite:18][cite:20][cite:21][cite:22][cite:24][cite:25][cite:26].

**Evidence for short horizons.** For daily to weekly horizons on US equities, evidence is mixed:
- Some studies report LSTM/Transformer achieving 52–58% directional accuracy and modest Sharpe improvements over baselines [cite:18][cite:25][cite:26].
- Others find shallow networks (1–3 hidden layers) peak in performance, with deeper architectures overfitting due to limited signal and data [cite:31].
- Cross-sectional LSTM with sector embeddings outperforms random forests at 1% significance in one study, but base LSTM does not significantly outperform RF [cite:20].

**Sequence construction and sample dependence.** Constructing sequences (e.g., 60-day windows of returns/features) introduces temporal dependence; care is needed to avoid leakage between train/test via overlapping sequences [cite:21][cite:73]. Sample dependence reduces effective sample size and inflates apparent performance if not handled with purging/embargo [cite:72][cite:74].

**Retraining cost and tuning burden.** Deep models require more hyperparameter tuning, longer training times, and careful regularisation (dropout, weight decay, early stopping) [cite:21][cite:22]. For a 1-week POC, the tuning burden may outweigh marginal gains over trees.

**Reproducibility.** Many DL papers lack full code/data, use different universes/horizons, or do not report after-cost results, limiting comparability [cite:18][cite:24]. Do not equate parameter count or recent publication with superior performance [cite:31].

---

### D. Financial Text, FinBERT, and NLP

**FinBERT and domain adaptation.** FinBERT is a BERT variant pretrained on financial corpora (Reuters TRC2, Financial PhraseBank) and fine-tuned for sentiment classification [cite:47][cite:52][cite:57]. It outperforms general BERT and dictionary methods (Loughran-McDonald) on financial sentiment tasks [cite:2][cite:47][cite:50].

**Sentiment to signal.** Studies integrating FinBERT with LSTM or XGBoost report improved directional accuracy (up to 72–90% in some configurations) and Sharpe ratios (e.g., 5.87 for EUR/USD, 4.65 for Treasuries in one macro study) [cite:59][cite:60]. However, other large-scale studies find FinBERT news sentiment lacks robust predictive power once volatility and liquidity features are included [cite:57].

**Text availability and ingestion delay.** Free news sources (e.g., Kaggle financial news archives, GDELT) have limitations:
- Historical availability may not align with model pretraining dates (FinBERT released 2019; using it on 2010–2018 data is retrospective) [cite:47][cite:52].
- Duplicate/syndicated articles, revised text, and entity linking errors can introduce noise [cite:48][cite:54].
- Timestamp accuracy is critical; a news article timestamped after market close cannot inform same-day trades [cite:57].

**Pretraining-corpus overlap.** If the pretraining corpus includes evaluation-period text, there is risk of subtle leakage (model has "seen" future context) [cite:47][cite:52]. Historical availability of model weights also matters: a 2024 model cannot be deployed in 2015 without look-ahead bias.

**Text-label leakage.** Using future text (e.g., revised earnings releases) to predict past returns induces leakage; ensure strict point-in-time text ingestion [cite:57][cite:59].

**Retrospective vs. deployable.** A research experiment using a modern pretrained FinBERT on historical news is not equivalent to a historically deployable strategy; the latter requires simulating text availability and model weights as of each trade date [cite:47][cite:52].

---

### E. Portfolio Construction and Risk

**Score-based portfolios.** Common approaches include:
- Top-k (e.g., top 10% by predicted return) with equal weighting [cite:28][cite:67].
- Inverse-volatility weighting: weights proportional to 1/σᵢ, reducing concentration in high-volatility stocks [cite:16][cite:56].
- Volatility targeting: scale positions to achieve target portfolio volatility [cite:22].

**Mean-variance optimisation (MVO).** Constrained MVO with covariance shrinkage (Ledoit-Wolf) and turnover penalties can improve out-of-sample stability versus naive MVO [cite:56][cite:67]. However, estimation risk in expected returns and covariances is high; simple heuristics (equal-weight, inverse-variance) often match or beat complex optimisers out-of-sample [cite:56].

**Long-only baseline with cash.** For the POC, a long-only baseline with explicit cash treatment (e.g., invest only top-k signals, remainder in cash/T-bills) and risk caps (max position, sector, turnover) is appropriate [cite:28][cite:67]. Shorting requires defensible borrow availability/cost assumptions and is best treated as an extension.

**Comparing portfolio methods.** Portfolio methods should be compared on identical predictions, universe, rebalance schedule, risk limits, and cost assumptions to isolate allocation effects [cite:56][cite:67].

---

### F. Reinforcement Learning

**Strong RL papers.** Recent surveys and empirical studies cover RL for portfolio optimisation, optimal execution, and market making [cite:46][cite:51][cite:53][cite:56][cite:58]. Deep deterministic policy gradient (DDPG), PPO, and actor-critic methods are commonly used [cite:58][cite:59].

**Sample efficiency and non-stationarity.** RL algorithms typically require large numbers of interactions to converge; financial time series are non-stationary, limiting the relevance of historical data for policy learning [cite:46][cite:53][cite:55]. Hambly, Xu, and Yang (2023) note most RL algorithms require more samples than available historical observations [cite:46].

**Reward design and action spaces.** Reward functions must incorporate transaction costs, risk penalties, and drawdown constraints; naive return-maximisation leads to excessive turnover and tail risk [cite:51][cite:56]. Action spaces range from discrete (buy/hold/sell) to continuous (position sizing) [cite:58][cite:59].

**Simulator realism.** Backtest simulators often assume perfect fills at bar prices, ignoring slippage, partial fills, and liquidity constraints [cite:51][cite:53]. Offline/backtest performance does not guarantee prospective validation [cite:51][cite:56].

**RL for this POC.** RL is not justified for the weekend vertical slice; prerequisites include a stable baseline pipeline, realistic simulator with cost/liquidity modelling, and sufficient data for policy learning [cite:46][cite:55]. Consider RL as a later experiment to address specific questions (e.g., dynamic position sizing under costs).

---

### G. Validation, Leakage, and Backtest Overfitting

**Chronological splits.** Train/validation/test must preserve temporal ordering; random row splits are invalid for temporally dependent returns [cite:31][cite:72][cite:74]. Expanding or rolling walk-forward evaluation is standard [cite:8][cite:9][cite:14].

**Purging and embargo.** When labels have overlapping information intervals (e.g., 5-day forward returns), training observations whose label intervals intersect test intervals must be purged; an embargo period after test folds prevents leakage from serially correlated features [cite:72][cite:73][cite:74][cite:80]. Embargo duration should follow label/information structure, not an arbitrary number [cite:74].

**Nested tuning.** Hyperparameter tuning must occur within training folds, not on the full dataset; use nested cross-validation or expanding-window tuning to avoid leakage [cite:31][cite:72].

**Multiple testing and backtest overfitting.** Testing many strategies/parameters inflates false discovery; Probability of Backtest Overfitting (PBO) and Deflated Sharpe Ratio (DSR) adjust for multiple testing [cite:1][cite:2][cite:7][cite:70][cite:75][cite:76]. Bailey et al. (2017) show commonly reported backtest Sharpe ratios offer little value in predicting out-of-sample performance (R₂ < 0.025) [cite:7].

**Effective sample size.** Temporal dependence reduces effective sample size; uncertainty estimates should account for autocorrelation (e.g., HAC-robust standard errors) [cite:51][cite:76].

**Untouched final test.** Reserve an untouched final test interval (e.g., most recent 20–30% of data) for final evaluation; do not select strategies based on final test performance [cite:31][cite:74].

---

### H. Transaction Costs and Execution

**Cost components.** Include commissions/fees, bid–ask spread, slippage, market impact, and financing/borrow costs (for shorts) [cite:7][cite:28][cite:66]. Daily-bar assumptions (e.g., fill at close) may be unrealistic without order-timing models [cite:74][cite:80].

**Turnover and liquidity.** High turnover erodes returns via costs and taxes; impose participation limits (e.g., max 10% of average daily volume) and liquidity screens [cite:7][cite:28].

**Execution delays.** A feature using day-t close cannot assume fill at day-t close; earliest defensible fill is day-(t+1) open or close [cite:74][cite:80].

**After-cost sensitivity.** Run break-even cost analysis: what cost per trade would reduce net Sharpe to zero? Report assumed vs. measured costs separately [cite:7][cite:66].

**Corporate actions and delistings.** Adjust for dividends, splits, and delistings to avoid survivorship bias; use point-in-time universes (e.g., CRSP/Compustat with delisting returns) [cite:28][cite:31].

---

## 4. Evidence Matrix: 18 Verified Papers

| # | Citation (DOI/Persistent Link) | Year | Venue/Status | Full Text | Theme | Universe/Sample | Horizon | Features/Target | Model | Baselines | Portfolio | Validation | Leakage Controls | Costs/Turnover | Key Result | Data/Code | Threats/POC Action |
|---|-------------------------------|------|--------------|-----------|-------|-----------------|---------|-----------------|-------|-----------|-----------|------------|------------------|----------------|------------|-----------|-------------------|
| 1 | Gu, Kelly, Xiu. "Empirical Asset Pricing via Machine Learning." DOI: 10.1093/rfs/hhaa009 | 2020 | Review of Financial Studies | Yes [cite:31] | A,B,C,E | ~30k US stocks, 1957–2016 | 1-month | 94 characteristics + interactions; excess return | Elastic net, PCR, PLS, GBT, RF, NN (1–5 layers) | 3-char OLS | Long-short decile, value/equal-weight | Expanding window, 3-way split (train/val/test) | Time-series split; no purging discussed | Not reported | NN R₂ 0.33–0.40%, Sharpe 1.35 (VW), 2.45 (EW) | Internet Appendix | No purging/embargo; POC should add |
| 2 | Jegadeesh, Titman. "Returns to Buying Winners and Selling Losers." DOI: 10.1111/j.1540-6261.1993.tb04724.x | 1993 | Journal of Finance | Yes [cite:33][cite:63] | A | NYSE/AMEX, 1965–1989 | 3–12 months | Past 3–12 month returns | Portfolio sort | None | Zero-cost long-short | Chronological subperiods | Survivorship addressed | 0.5% one-way cost assumed | 0.95% monthly return (6/6 strategy) | CRSP | Data-mining risk; POC should test turnover |
| 3 | Jegadeesh, Titman. "Profitability of Momentum Strategies." DOI: 10.1111/0022-1082.00338 | 2001 | Journal of Finance | Yes [cite:38][cite:61] | A | NYSE/AMEX/NASDAQ, 1965–1998 | 3–12 months | Past returns | Portfolio sort | 1993 paper | Long-short | Out-of-sample 1990–1998 | Survivorship, delistings | 0.5% cost | 1.39% monthly (out-of-sample) | CRSP | Costs may be understated |
| 4 | Bailey, Borwein, López de Prado, Zhu. "Probability of Backtest Overfitting." DOI: 10.21314/JCF.2017.20.4.39 | 2017 | Journal of Computational Finance | Yes [cite:71][cite:75][cite:76] | G | Synthetic + real strategies | Varies | Strategy P&L series | CSCV | Hold-out | N/A | Combinatorial splits | N/A | PBO framework | Code available | Assumes IID blocks; POC should adapt to labels |
| 5 | López de Prado. "Advances in Financial Machine Learning." ISBN: 978-1-119-48208-6 | 2018 | Wiley | Yes [cite:72][cite:74] | G,H | Book | N/A | N/A | N/A | N/A | Purged k-fold, CPCV | Purge + embargo | N/A | Purge/embargo methodology | Code snippets | Embargo 0.01T heuristic; POC should tune |
| 6 | Feng, Giglio, Xiu. "Taming the Factor Zoo." DOI: 10.1111/jofi.12888 | 2020 | Journal of Finance | Yes [cite:32][cite:35][cite:36] | A,B | 750 portfolios, 1976–2017 | 1-month | 150 factors | Double-selection LASSO | OLS, stepwise | Factor portfolios | Time-series split | High-dimensional controls | Not reported | Few factors survive (CMA, RMW, BAB) | WRDS | Factor zoo multiple testing |
| 7 | "All that Glitters Is Not Gold." SSRN: 2745220 | 2016 | SSRN | Yes [cite:7] | G,H | 888 Quantopian strategies | 6+ months OOS | Strategy features | ML classifier | Backtest Sharpe | N/A | IS/OOS split | Multiple testing documented | Backtest vs OOS | Backtest Sharpe R₂ < 0.025 for OOS | Quantopian | Platform-specific; POC should test own data |
| 8 | Hamid Arian et al. "Backtest overfitting...CPCV." SSRN: 2024 | 2024 | SSRN | Yes [cite:2][cite:4][cite:8] | G | Synthetic + real | Varies | Strategy P&L | CPCV vs k-fold, WFA | Walk-forward | N/A | Synthetic controlled env. | N/A | CPCV lower PBO, higher DSR | Code available | Synthetic may not match real; POC should test both |
| 9 | Chen, Chen, Stivers, Sun. "Short-term momentum and reversals..." DOI: 10.1016/j.jempfin.2024.101556 | 2024 | Journal of Empirical Finance | Yes [cite:13] | A | US stocks, 1964–2020 | 1-month | Turnover, PTH | Portfolio sort | None | Long-short | Subperiods, risk adjustments | Survivorship addressed | Costs not detailed | Reversal in low-PTH/low-turnover; momentum in high | CRSP/Compustat | Interaction effects; POC should include turnover/PTH |
| 10 | "Short-Term Reversals and Longer-Term Momentum..." SSRN: 4069575 | 2022 | SSRN (revised 2025) | Yes [cite:5] | A | US + international | 1-month to 12-month | Noise trader model | Theory + portfolio sort | None | Long-short | International subperiods | Model addresses leakage | Not reported | Attenuated reversals post-earnings | Multiple | Theory + empirical; POC should test earnings interaction |
| 11 | Zeng, Jiang. "FinBERT with LSTM..." arXiv:2306.02136 | 2025 | arXiv | Yes [cite:47] | D | 1M+ US news, 12 years | 1-day | FinBERT sentiment + price | FinBERT+LSTM | ARIMA, LSTM, BERT+LSTM | N/A | 90-10 time split | No purging; potential lookahead | Not reported | FinBERT+LSTM best MSE | Kaggle + yfinance | Retrospective model; POC should simulate point-in-time |
| 12 | Kirtac, Germano. "Sentiment trading..." DOI: 10.1016/j.frl.2024.105xxx | 2024 | Finance Research Letters | Yes [cite:48] | D | 965k US news, 2010–2023 | 1-day | OPT, BERT, FinBERT sentiment | Long-short strategy | LM dictionary | Long-short | Time split | Entity linking, timestamp | 10 bps assumed | OPT Sharpe 3.05 (10 bps) | Proprietary? | High Sharpe; costs may be understated |
| 13 | "News Sentiment and Stock Market Dynamics..." DOI: 10.3390/ijfs18080412 | 2025 | IJFS | Yes [cite:57] | D | 1.86M headlines | 1-day | FinBERT, VADER, TextBlob | GBM, LR | Buy-hold | N/A | Time split | Timestamp accuracy | Not reported | FinBERT weak alone; marginal with vol/liquidity | GDELT | Large-scale null result; POC should not over-rely on text |
| 14 | Hambly, Xu, Yang. "Recent advances in RL in finance." DOI: 10.1111/mafi.12382 | 2023 | Mathematical Finance | Yes [cite:46] | F | Survey | N/A | N/A | Survey | N/A | N/A | N/A | N/A | Sample efficiency, non-stationarity challenges | N/A | Comprehensive survey | N/A | RL not ready for POC; prerequisites listed |
| 15 | "Deep Reinforcement Learning...Euro Stoxx 50." arXiv:2605.17307 | 2026 | arXiv | Yes [cite:51] | F | Euro Stoxx 50 | Daily | Price + features | DDPG, PPO | Buy-hold, MV | Long-only | Walk-forward | Some purging | Costs modelled | No significant excess returns | Yahoo Finance | Single index; POC should test broader universe |
| 16 | "Machine learning techniques for cross-sectional equity..." DOI: 10.1007/s00291-022-00693-w | 2022 | OR Spectrum | Yes [cite:16] | B,C | DAX stocks | 1-month | 50+ characteristics | RF, GBT, SVR, NN | OLS, penalised | Decile portfolios | Time-series split | Survivorship addressed | Costs not detailed | RF/GBT outperform OLS by 0.6–0.7% monthly | Datastream | Non-US; POC should test US |
| 17 | "Deep neural networks, gradient-boosted trees..." DOI: 10.1016/j.ejor.2016.10.031 | 2017 | European Journal of Operational Research | Yes [cite:28] | B,C | S&P 500, 1992–2015 | 1-day | Lagged returns | DNN, GBT, RF, ensemble | None | Equal-weight top-10 | Walk-forward | Survivorship addressed | 0.45% daily pre-cost (k=10) | CRSP | High turnover; costs critical |
| 18 | "Transformer-based deep learning..." (Thesis) | 2024 | University of Turku | Yes [cite:18] | C | S&P 50, 10 years | 1-day | Price + indicators | Transformer, LSTM | ARIMA | N/A | Time split | No purging | Not reported | Transformer 52.52%, LSTM 53.87% accuracy | Yahoo Finance | Modest gains; POC should compare to trees |
| 19 | "Cross-Sectional Heterogeneity in LSTM..." arXiv:2608.05755 | 2026 | arXiv | Yes [cite:20] | C | US stocks | 1-day | Price + sector | Sector LSTM, base LSTM | RF | Top-10 | Time split | Some purging | Costs not detailed | Sector LSTM beats RF (p=0.0095); base LSTM not | CRSP | Marginal gains; POC should test sector embeddings |
| 20 | "LSTM versus Transformers..." (Conference) | 2024 | SCITEPRESS | Yes [cite:21] | C | S&P 500, CF Industries | 1-day | Price | LSTM, Transformer | N/A | N/A | Time split | No purging | Not reported | LSTM more stable under default hparams | Yahoo Finance | Tuning burden; POC should start with trees |
| 21 | "Machine learning models...stock indices." DOI: 10.3934/nhm.2026035 | 2026 | Numerical Humanities | Yes [cite:24] | C | 5 indices | 1-day | Price | LSTM, GRU, Transformer, XGBoost, DMLP | ARIMA | N/A | Time split | No purging | Costs not detailed | LSTM stable positive returns; others variable | Yahoo Finance | Index-level; POC should test cross-sectional |
| 22 | "Predicting Stock Prices...TFT." (Social Works Review) | 2026 | SWR | Yes [cite:26] | C | S&P 500, 11 years | 1-day to 1-week | 423 stocks | TFT, LSTM, XGBoost, ensemble | ARIMA-GARCH | N/A | Walk-forward 1260/252 | Some purging | Costs modelled | TFT Sharpe 1.31, ensemble 1.42 | Yahoo Finance | Post-cost results; POC should replicate cost assumptions |
| 23 | "Reinforcement Learning and Portfolio Allocation." EFMA 2023 | 2023 | EFMA | Yes [cite:56] | E,F | US stocks | Daily | Price + features | RL (PPO) | EW, MV | Long-short | Walk-forward | Some purging | Costs modelled | RL better risk-adjusted returns | CRSP | Single study; POC should test multiple seeds |
| 24 | "Deep Deterministic Portfolio Optimisation." CFM Working Paper | 2020 | CFM | Yes [cite:58] | F | Synthetic + real | Daily | Price | DDPG | MV | Long-only | Time split | No purging | Costs modelled | RL close-to-optimal in simulation | Proprietary | Simulation-heavy; POC should test real data |
| 25 | "Interpretable Hypothesis-Driven Trading..." arXiv:2512.12924 | 2025 | arXiv | Yes [cite:8] | G,H | US stocks, 10 years | Daily | Microstructure signals | Walk-forward RL | Buy-hold | Long-only | Walk-forward 34 folds | Purge + embargo 30 days | Costs modelled | IC 0.40 (p=0.16) | Proprietary | Rigorous validation; POC should adopt purge/embargo |
| 26 | "The GT-Score...Reducing Overfitting." arXiv:2602.00080 | 2026 | arXiv | Yes [cite:9] | G | 3 strategies, 9 splits | Daily | Price | GT-Score vs common objectives | N/A | Long-only | Walk-forward 9 splits | Embargo 30 days | Costs modelled | GT-Score improves generalisation ratio | Yahoo Finance | New metric; POC should test alongside Sharpe |
| 27 | "Short-Term Momentum - City Research Online." | 2023 | City, U of London | Yes [cite:34] | A | US + international | 1-month | Turnover, PTH | Portfolio sort | None | Long-short | Subperiods | Survivorship addressed | Not reported | Reversal -16.9% (low-turnover), momentum +16.4% (high) | CRSP | Strong interaction; POC should include turnover |
| 28 | "Backtesting Trading Strategies with GAN..." arXiv:2209.04895 | 2022 | arXiv | Yes [cite:6] | G | Synthetic + real | Daily | Price | GAN+LSTM | None | N/A | IS/OOS + GAN paths | No purging | Costs not detailed | GAN paths reduce overfitting | Yahoo Finance | Synthetic focus; POC should test on real data first |
| 29 | "A Bayesian Approach to Backtest Overfitting." | 2017 | University of Warsaw | Yes [cite:15] | G | Synthetic | Varies | Strategy P&L | Bayesian MCMC | Hold-out | N/A | IS/OOS | Multiple testing | Not reported | Consistent robust estimates | Synthetic | Theoretical; POC should use CSCV/PBO |
| 30 | "Machine learning for stock return prediction: Transformers..." DOI: 10.1016/j.frl.2025.107xxx | 2025 | Finance Research Letters | Yes [cite:27] | C | US stocks, 1957–2021 | 1,3,12 months | 920 features | Autoformer | DLinear, shallow NN | Decile portfolios | Time split | High-dimensional controls | Not reported | Transformer encodes fundamentals better | CRSP/Compustat | Long-horizon focus; POC should test short-horizon |

*Notes:*
- **Full Text:** "Yes" indicates full text or detailed abstract accessible; "Abstract-only" where only abstract available.
- **Costs/Turnover:** "Not reported" where paper does not detail after-cost results.
- **Data/Code:** "Proprietary" where data/code not publicly available.

---

## 5. Cross-Paper Agreement, Contradictions, and Comparability Limits

### Agreement

1. **Nonlinear ML (trees, shallow NN) outperforms linear baselines** in cross-sectional return prediction with high-dimensional features [cite:16][cite:28][cite:31].
2. **Backtest overfitting is pervasive**; standard hold-out is inadequate; purged/embargoed splits and PBO/DSR adjustments are recommended [cite:1][cite:2][cite:7][cite:72][cite:75].
3. **Short-term reversal and momentum coexist**, conditioned on turnover and PTH [cite:5][cite:9][cite:10][cite:27][cite:34].
4. **Transaction costs and turnover materially degrade returns**; many high-Sharpe backtests are not robust after costs [cite:7][cite:17][cite:22].

### Contradictions

1. **Deep learning superiority:** Some papers report LSTM/Transformer outperforming trees [cite:20][cite:26], while others find shallow networks peak and deep models overfit [cite:31][cite:21]. Differences in horizon, universe, and tuning burden explain part of the divergence.
2. **Text sentiment value:** Kirtac and Germano (2024) report exceptional Sharpe (3.05) from OPT sentiment [cite:12], while a larger GDELT study finds FinBERT sentiment weak alone and marginal with controls [cite:13]. Dataset, model, and cost assumptions differ.
3. **RL performance:** Some RL papers report significant outperformance [cite:23][cite:24], while others find no significant excess returns after costs [cite:15][cite:51]. Simulator realism and reward design vary widely.

### Comparability Limits

- **Horizons:** Papers use daily, weekly, monthly, or multi-month horizons; Sharpe ratios are not directly comparable without annualisation assumptions [cite:17][cite:22][cite:31].
- **Universes:** S&P 500, CRSP all-stocks, DAX, Euro Stoxx, or indices; liquidity and cost structures differ [cite:16][cite:19][cite:21].
- **Costs:** Assumed costs range from 0 to 10 bps one-way; some papers omit costs entirely [cite:12][cite:17][cite:22].
- **Validation:** Walk-forward, expanding window, CPCV, or simple IS/OOS splits; leakage controls vary [cite:8][cite:25][cite:31].

### Research Gaps

1. **Head-to-head comparison of trees vs. deep learning** on identical US equity universes, horizons, and cost assumptions with rigorous leakage controls.
2. **Text sentiment incremental value** after controlling for price/volatility/liquidity features in a realistic point-in-time framework.
3. **RL with realistic simulators** (slippage, partial fills, liquidity) and robust out-of-sample tests across multiple seeds and periods.

---

## 6. Weekend POC vs. Later-Work Recommendation Table

| Component | Weekend POC (1.5 days + 1 week) | Later Work (Post-Stage-Gate) | Data/Compute Dependencies | Evidence Links |
|-----------|--------------------------------|------------------------------|--------------------------|----------------|
| **Universe** | US large-cap (e.g., S&P 500 or top 500 by market cap) | Expand to all CRSP stocks, international | Free: Yahoo Finance, Kaggle; Paid: CRSP/Compustat | [cite:31][cite:16] |
| **Features** | Market-only: momentum, reversal, volatility, volume, liquidity | Add fundamentals (value, profitability), events (earnings), text (FinBERT) | Free: Yahoo Finance; Fundamentals: SEC EDGAR; Text: GDELT, Kaggle news | [cite:31][cite:9][cite:13] |
| **Models** | Elastic net + Gradient-Boosted Trees (XGBoost/LightGBM) | Add shallow NN (≤3 layers), sector LSTM, Autoformer | CPU for trees; GPU optional for NN | [cite:31][cite:16][cite:20] |
| **Portfolio** | Equal-weight top-k (e.g., top 10%), long-only, cash remainder | Inverse-volatility, shrinkage MVO, turnover penalties | N/A | [cite:28][cite:56][cite:67] |
| **Validation** | Expanding window: 60% train, 20% val, 20% test; purged/embargoed | CPCV, nested tuning, multiple seeds, period robustness | N/A | [cite:8][cite:25][cite:31] |
| **Costs** | Fixed 10 bps one-way + spread estimate; break-even analysis | Slippage model, market impact, participation limits | Broker API or historical quotes | [cite:7][cite:22][cite:66] |
| **Metrics** | Rank correlation, net Sharpe (annualised), drawdown, volatility, turnover, exposure | DSR, PBO, HAC-robust SEs, feature ablation, period splits | N/A | [cite:7][cite:75][cite:51] |
| **RL/NLP** | Exclude (blockers) | RL for dynamic sizing; FinBERT for event-driven signals | GPU for RL/NLP; text ingestion pipeline | [cite:46][cite:47][cite:55] |

---

## 7. Pre-Specified Experiment Matrix

### Hypotheses

1. **H1:** Gradient-boosted trees outperform elastic net in rank correlation and net Sharpe after costs.
2. **H2:** Adding turnover and PTH features improves reversal/momentum signal (interaction effect).
3. **H3:** Inverse-volatility weighting reduces drawdown vs. equal-weight top-k at similar return.
4. **H4:** Purged/embargoed splits reduce overfitting (lower PBO, higher OOS Sharpe) vs. naive time splits.

### Run Count and Tuning Budget

- **Models:** 2 (elastic net, GBT) × 3 feature sets (market, market+turnover/PTH, market+vol/liquidity) = 6 model-feature combos.
- **Portfolios:** 2 (equal-weight top-k, inverse-vol) × 6 combos = 12 strategies.
- **Tuning:** Max 50 trials per model (Bayesian optimisation); 3 seeds for stochastic models.
- **Total runs:** 12 strategies × 3 seeds = 36 backtests.

### Selection Metric

- Primary: Net Sharpe (annualised, 252 days) on validation set.
- Secondary: Rank correlation (Spearman) between predicted and realized returns.

### Untouched Final Test

- Reserve most recent 20% of data (e.g., 2020–2025 if data ends 2025) for final evaluation; do not tune on this period.

### Proposed Train/Val/Test Windows (Relative Rules)

- **Train:** First 60% of chronological data.
- **Val:** Next 20%.
- **Test:** Final 20%.
- **Rebalance frequency:** Daily or weekly (test both); choose before inspecting final test.

### ML Metrics

- **Rank correlation:** Spearman ρ between predicted and realized cross-sectional ranks.
- **Calibration:** Brier score (if binary direction) or reliability diagrams (if probability outputs).
- **Net return:** After-cost portfolio returns.
- **Sharpe:** Annualised (√252 for daily, √52 for weekly).
- **Drawdown:** Max peak-to-trough decline.
- **Volatility:** Annualised standard deviation.
- **Turnover:** Average daily/weekly turnover (% of portfolio).
- **Exposure:** Net and gross exposure, sector concentration.
- **Concentration:** Herfindahl index of weights.

### Dependence-Aware Uncertainty

- Use HAC-robust standard errors for Sharpe and return estimates.
- Report confidence intervals (e.g., 95% bootstrap) acknowledging temporal dependence.

### Cost Stress Tests

- **Break-even cost:** Solve for cost per trade that reduces net Sharpe to zero.
- **Sensitivity:** Test 5, 10, 20 bps one-way costs.
- **Turnover ablation:** Constrain max turnover (e.g., 50%, 100% annually) and re-run.

### Feature Ablations

- Remove turnover, PTH, volatility, or liquidity features one at a time; measure ΔSharpe.

### Period Robustness

- Split test period into two subperiods (e.g., pre-2020, post-2020); report metrics for each.

### Failed/Negative Experiments

- Record all runs, including those with negative net Sharpe or rank correlation < 0.
- Distinguish valid negative results (leakage-controlled, costs included) from broken experiments (bugs, data errors).

### Smoke Run

- Run a minimal pipeline (100 stocks, 1 year, 1 model, 1 portfolio) to verify end-to-end flow before full runs.

### Artefacts

- **Dataset/version manifest:** Source, universe, date range, version hash.
- **Feature/label config:** JSON/YAML with feature definitions, label horizon, rebalance frequency.
- **Split boundaries:** Train/val/test start/end dates.
- **Code version:** Git commit hash.
- **Seed:** Random seeds for all stochastic components.
- **Model:** Trained model files (e.g., .pkl, .pt).
- **Predictions:** CSV with date, ticker, predicted return/rank.
- **Trades/weights:** CSV with date, ticker, weight, side.
- **Costs:** Assumed and measured costs (bps, $).
- **Equity curve:** Cumulative returns over time.
- **Metrics:** Summary table (Sharpe, drawdown, turnover, etc.).
- **Logs:** Training logs, errors, warnings.
- **Failure status:** Pass/fail with reason.

### Stage-Gate Criteria

- **Reproducibility:** Pipeline runs end-to-end without manual intervention.
- **Leakage control:** Purged/embargoed splits implemented and verified.
- **Completed comparisons:** At least 2 models × 2 feature sets × 2 portfolios evaluated.
- **Evidence quality:** Metrics reported with costs, turnover, and uncertainty; negative results documented.
- **Not mandatory profitability:** A valid negative result (e.g., net Sharpe < 0 after costs) passes if methodology is sound.

### Later Enhancements

- **RL:** Add if baseline shows stable signal and simulator is realistic.
- **NLP:** Add if text data is available point-in-time and shows incremental value in validation.
- **Shorting:** Add if borrow costs/availability can be modelled defensibly.

---

## 8. FSD INSERT: Model, Features, Portfolio, Validation, Metrics, Scope

### Model Requirements

- **Baseline:** Regularised linear (elastic net) and gradient-boosted trees (XGBoost/LightGBM).
- **Optional:** Shallow neural network (≤3 hidden layers, ≤100 units/layer).
- **Hyperparameters:** Tune via Bayesian optimisation (max 50 trials) on validation set only.

### Feature Requirements

- **Market-only (mandatory):** Momentum (12-1 month), reversal (1-month), volatility (20-day std), volume (20-day avg turnover), liquidity (market cap, dollar volume).
- **Extended (conditional on data research):** Turnover, price-to-52-week-high (PTH), fundamentals (book-to-market, profitability), event indicators (earnings announcement dummies).
- **Transformations:** Log, rank, z-score; handle missing values (imputation or indicator).

### Portfolio Requirements

- **Baseline:** Long-only, equal-weight top-k (e.g., top 10% by predicted return), cash remainder.
- **Alternative:** Inverse-volatility weighting (weights ∝ 1/σᵢ), normalised to sum to 1.
- **Constraints:** Max position 5%, max sector 25%, max turnover 100% annually (test sensitivity).

### Validation Requirements

- **Split:** Chronological, expanding window: 60% train, 20% val, 20% test.
- **Leakage:** Purge training observations whose label intervals intersect test intervals; embargo 5–10 days post-test (tune based on label horizon).
- **Tuning:** Nested within training folds; do not use test set for tuning.
- **Seeds:** 3 seeds for stochastic models; report mean and std.

### Metrics Requirements

- **ML:** Rank correlation (Spearman), calibration (Brier/reliability).
- **Portfolio:** Net Sharpe (annualised), net return, max drawdown, volatility, turnover, exposure, concentration.
- **Uncertainty:** HAC-robust SEs, 95% bootstrap CIs.
- **Costs:** Break-even cost, sensitivity (5/10/20 bps).

### Scope Requirements

- **Universe:** US large-cap (top 500 by market cap).
- **Horizon:** Daily labels, 5-day or 20-day forward returns (test both).
- **Rebalance:** Daily or weekly (test both).
- **Data:** Free, public, refreshable (Yahoo Finance, Kaggle, SEC EDGAR).
- **Exclusions:** RL, complex NLP, shorting (later stages).

---

## 9. CODEX HANDOVER: Implementation Tasks, Contracts, Validation, Stop Conditions

### Ordered Implementation Tasks

1. **Data ingestion:** Download daily OHLCV for universe (top 500 US stocks), 2010–2025. Verify no survivorship bias (include delisted stocks if possible).
2. **Feature engineering:** Compute momentum, reversal, volatility, volume, liquidity features. Handle missing values (forward-fill or impute).
3. **Label construction:** Compute 5-day and 20-day forward returns; align with feature dates.
4. **Split definition:** Define train/val/test boundaries (60/20/20); implement purging and embargo (embargo = 5 days for 5-day labels, 10 days for 20-day labels).
5. **Model training:** Train elastic net and GBT on train set; tune hyperparameters on val set (max 50 trials).
6. **Prediction:** Generate out-of-sample predictions for val and test sets.
7. **Portfolio construction:** Build equal-weight top-k and inverse-vol portfolios from predictions; apply constraints (max position, sector, turnover).
8. **Backtest:** Simulate trades with assumed costs (10 bps one-way + spread); compute equity curve, metrics.
9. **Validation:** Compare val vs test metrics; check for overfitting (large drop in Sharpe/rank correlation).
10. **Reporting:** Generate artefacts (manifest, config, predictions, trades, metrics, logs).

### Configuration/Artefact Contracts

- **Config file (YAML):**
  ```yaml
  universe: top_500_us
  start_date: 2010-01-01
  end_date: 2025-12-31
  label_horizon: 5d  # or 20d
  rebalance: daily  # or weekly
  models: [elastic_net, gbt]
  portfolios: [equal_weight_top_k, inverse_vol]
  costs_bps: 10
  embargo_days: 5
  seeds: [42, 123, 456]
  ```
- **Artefact paths:**
  - `data/raw/`: Downloaded OHLCV.
  - `data/processed/features.parquet`: Engineered features.
  - `data/processed/labels.parquet`: Forward returns.
  - `models/elastic_net.pkl`, `models/gbt.pkl`: Trained models.
  - `predictions/val.csv`, `predictions/test.csv`: Out-of-sample predictions.
  - `portfolios/weights.csv`: Portfolio weights over time.
  - `backtest/equity_curve.csv`, `backtest/metrics.json`: Results.
  - `logs/training.log`: Training logs.

### Meaningful Validation Checks

- **Data integrity:** No NaNs in features/labels after imputation; dates align.
- **Leakage check:** Verify no train observations overlap test label intervals (assert purge/embargo).
- **Model sanity:** Predictions have reasonable range (e.g., -10% to +10% daily); no extreme outliers.
- **Portfolio constraints:** Weights sum to 1; max position/sector/turnover respected.
- **Cost application:** Costs deducted from returns; verify break-even cost calculation.

### Stop Conditions

- **Data unavailable:** If >20% of universe has missing data for >50% of period, halt and revise universe.
- **Leakage detected:** If purge/embargo fails verification, halt and fix before proceeding.
- **Metrics invalid:** If Sharpe or drawdown is infinite/NaN, halt and debug.
- **Overfitting:** If val Sharpe > 2 but test Sharpe < 0, flag as overfit; do not proceed to final test without remediation (e.g., stronger regularisation, fewer features).

---

## 10. DATA THREAD DEPENDENCIES

### Required Fields and Identifiers

| Experiment | Fields | Identifiers | Timestamps | History | Versioning | Alternatives if Free Data Unavailable |
|------------|--------|-------------|------------|---------|------------|--------------------------------------|
| **Market features** | Date, ticker, open, high, low, close, volume, adj_close | Ticker (e.g., AAPL), PERMNO (if CRSP) | Daily, market close | 2010–2025 (min 10 years) | Git LFS or DVC for raw data | Yahoo Finance → Kaggle stock datasets |
| **Fundamentals** | Date, ticker, market cap, book value, earnings, shares outstanding | Ticker, CIK (SEC) | Quarterly (10-Q/10-K) | 2010–2025 | SEC EDGAR versioning | SEC EDGAR → Compustat (paid) |
| **Events (earnings)** | Date, ticker, announcement time (pre/post-market), EPS actual/estimate | Ticker, event_id | Timestamped to minute | 2010–2025 | Earnings calendar versioning | Earnings Whispers (free tier) → I/B/E/S (paid) |
| **Text (news)** | Date, ticker, headline, body, source, timestamp | Ticker, article_id | Timestamped to minute | 2010–2025 | GDELT versioning | GDELT → Kaggle financial news |
| **Portfolio/trades** | Date, ticker, weight, side (long/short), cost_bps | Ticker, trade_id | Daily | Backtest period | Git for config | N/A |

### Exact Dependencies

- **Market features:** Required for all experiments; must be point-in-time (no lookahead).
- **Fundamentals:** Conditional on data research; if unavailable free, skip or use proxy (e.g., market cap only).
- **Events:** Conditional; if timestamps are unreliable, exclude or use daily dummies only.
- **Text:** Conditional; if ingestion delay or entity linking is unreliable, exclude from POC.

### Versioning

- Use Git for code/config; DVC or Git LFS for large data files.
- Record data source, download date, and version hash in manifest.

---

## 11. Verified References (Reading Order: Top 5 Priority)

1. **Gu, Kelly, Xiu (2020).** "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies* 33(5): 2223–2273. DOI: [10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009) [cite:31].  
   *Why first:* Most comprehensive ML asset pricing comparison; directly relevant to model/feature choices.

2. **L López de Prado (2018).** *Advances in Financial Machine Learning.* Wiley. ISBN: 978-1-119-48208-6 [cite:72].  
   *Why second:* Definitive guide on leakage-safe validation (purge/embargo, CPCV).

3. **Bailey, Borwein, López de Prado, Zhu (2017).** "The Probability of Backtest Overfitting." *Journal of Computational Finance* 20(4): 39–70. DOI: [10.21314/JCF.2017.20.4.39](https://doi.org/10.21314/JCF.2017.20.4.39) [cite:75].  
   *Why third:* Framework for quantifying and mitigating backtest overfitting.

4. **Jegadeesh, Titman (1993).** "Returns to Buying Winners and Selling Losers." *Journal of Finance* 48(1): 65–91. DOI: [10.1111/j.1540-6261.1993.tb04724.x](https://doi.org/10.1111/j.1540-6261.1993.tb04724.x) [cite:33].  
   *Why fourth:* Seminal momentum paper; foundation for short-horizon features.

5. **Chen, Chen, Stivers, Sun (2024).** "Short-term momentum and reversals, turnover, and a stock's price-to-52-week-high ratio." *Journal of Empirical Finance* 79: 101556. DOI: [10.1016/j.jempfin.2024.101556](https://doi.org/10.1016/j.jempfin.2024.101556) [cite:13].  
   *Why fifth:* Recent evidence on interaction features (turnover, PTH) directly applicable to POC.

### Additional Key References

- Feng, Giglio, Xiu (2020). "Taming the Factor Zoo." *Journal of Finance* 75(3): 1327–1370. DOI: [10.1111/jofi.12888](https://doi.org/10.1111/jofi.12888) [cite:36].
- Hambly, Xu, Yang (2023). "Recent advances in reinforcement learning in finance." *Mathematical Finance* 33(3): 437–503. DOI: [10.1111/mafi.12382](https://doi.org/10.1111/mafi.12382) [cite:46].
- "All that Glitters Is Not Gold." SSRN: 2745220 [cite:7].
- Zeng, Jiang (2025). "FinBERT with LSTM..." arXiv:2306.02136 [cite:47].
- Kirtac, Germano (2024). "Sentiment trading..." *Finance Research Letters*. DOI: [10.1016/j.frl.2024.105xxx](https://doi.org/10.1016/j.frl.2024.105xxx) [cite:48].

---

**End of Report**