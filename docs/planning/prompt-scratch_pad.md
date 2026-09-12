# Missing component list
By the end of the weekend (1 and a half days) I would like a POC that could serve as a basis for the paper, this includes:
- A dataset or data pipeline
- One or model candidates for "Feature Engineering" and "Portfolio / Risk Engine" (min 2, ideally 4 to show that I had compared several approaches to find the "best" candidate)
- A robust ML pipeline that can train in the coming week
- A robust MLOps process to automatically:
    - Train a model
    - Backtest it
    - Ensure training and backtesting performance are recorded and readily available to be used in a report
    - All steps executed autonomously without error

Next weekend will be a stage gate where I decide on enhancements or simply progress with the report.

# Additional considerations 
I have some additional considerations I have been musing about. However, thier scope is likely too broad for our 1-weekend POC stage-gate.

## Resources:
- I am willing to start a low level CLAUDE, CODEX, and/or Cursor subscription
- I have 2x free perplexity.ai accounts
- I have a GitHub education pack upgrade
- I have a free Google Antigravity account

Please make recommendations that will enable us to realistically accomplish our POC given usage windows and token limits.

## Data pipeline
- Types of data collected
    - Financial
    - News
- Universe of stocks
    - US Equities
    - AUS Equities
- Investment horizon
    - Intra-day: Capitalising on short term fluctuations, either from momentum or information releases
    - Short term: Bets on securities made over days or weeks
    - Medium term: Portfolios actively managed over months or years
    - Long term: Passive portfolios managed over years
- Feeds and their considerations
    - Online sources
    - Update frequency based on horizon
    - How much history based on horizon
- Database design
    - Local Vs remote (e.g. Supabase, Azure, or AWS?)
    - File based (e.g. collections of parquet files) vs DBMS (e.g. SQLite)

## Models and components
I really like the proposed "ML-Driven Event-Based Systematic Trading Strategy" architecture. In this framework, or in a simplified POC (that would be suitable to present for the university project) I would like to also explore comparing several component models:
### Feature engineering
- News / Corporate Events: Agentic evaluation of news data Vs simpler, faster, cheaper sentiment analysis models based on BERT-style architectures
- Market Data: Automatically generated features from RNN model(s) vs rules-based features selected by traditional feature selection methods
- Fundamentals: Agentic process to scrape and extract data directly from annual reports, vs data pre-collected and aggregated from available sources 
### Portfolio / Risk Engine
There are two types of risk engine approaches that I am interested in:
- Traditional model based on Modern Portfolio Theory that suggests a combination that has the best risk optimised return
- A reinforcement learning model that recommends bets

## MLOps Strategy
- I have made use of both Weights and Biases and ClearML in the past. I am fine with any framework that can be verified locally before deployed remotely via github actions.
- I am also fine with local implementations if they can be executed quickly and simply
