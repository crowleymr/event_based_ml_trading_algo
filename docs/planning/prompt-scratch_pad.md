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



I am still proceeding on my review of the build work done by codex://threads/01a0945b-a038-7972-bbeb-96b8bace5c00 and initially planned by https://chatgpt.com/share/6aa5f2e9-ca08-83ec-99ba-50c41d93fe73. However, I find some concerns that I'd like to plan out a solution to that should be triaged to a model that is effective in preparing documentation:
1. I need a clearer more traditional project README.md that contains the following information:
   - Project purpose and overview
   - Folder structure diagram to help users navigate where everything is
   - Quick start guide, which includes how to install and run both the smoke test & a more complete run
   - Table with intra-repo relative links to other markdown files that describe key pieces of project documentation (e.g. ARCHITECTURE.md, AGENTS.md)
2. We need to enrich project documentation so that each folder contains a README.md describing the programs inside and their key business considerations. For instance:
    - 

    that makes it easy for a user to understand the project at a glance, how to start it,



Help me plan out the remaining parts of the POC that was initially planned by initially planned by https://chatgpt.com/share/6aa5f2e9-ca08-83ec-99ba-50c41d93fe73. 





Thread codex://threads/01a0945b-a038-7972-bbeb-96b8bace5c00 has successfully finished build that aligns with our slice 1 POC plan. However, I you to help me plan out some additional features before we can consider our baseline work in a complete enough state that I could deliver for the assignment and achieve a PASS or CREDIT result:
1. Better documentation including:
    - ARCHITECTURE.md written out in a way that describes the problem being solved and the solution we have designed. It needs to include the overall workflow diagram that you had previously described. Relating to the current state components built, and planned final state, calling out parts that are either explicitly known, or still pending decisions. We also need to provide any additional architecture guidelines regarding our architectural approach like Service Layer Architecture (if appropriate), and architectural principles like YAGNI ()"You Aren't Gonna Need It,")
    - A backlog document (BACKLOG.md?), describing planned features we intend to build out in slices 2+
    - README.md files in each subfolder within src/ contains a README.md describing the programs inside and their key business considerations. For instance:
        - src/trading_pipeline/ describes the overall pipeline and their phases of execution
        - src/trading_pipeline/data/ describes the data collected, final data model diagram in mermaid, their schemas through each stage of transformation, how they are extracted, how much history they have, and their frequencies. Please order this information logically
        - etc
2. Expanded reporting surfaces:
    - run logging
    - xlsx/csv/parquet reports to analyse run results
    - FastAPI or Streamlit UI to execute and navigate run results
    - If deemed necessary, I am also fine with dashboards generated in a powerbi report or a simple XLSX pivot
3. Expanded results reporting (using additional reporting surfaces mentioned in point 2.)
    - Results need to be able to be presented at all levels of (diss)aggregation:
        - Security level
        - Industrial classification
        - Model level (by feature model and final risk-performance model)
        - Universe level
    - Summary / descriptive statistics (at each level of aggregation)
        - Number, Frequency of observations
        - Statistical returns performance metrics of securities including, but not limited to:
            - Return, daily, continuously (exponential), annualised, total, etc
            - Standard Deviation and/or variance of return
            - Market covariance, Beta, Alpha, etc
            - Performance across 
        - Share price and market capitalization
        - Fundamentals performance, including but not limited to if available:
            - Earnings
            - Price-earnings ratio
        - Industrial classification
        - Listed Exchange
    - Backtesting runs performance including diagrams
    - Improve labelling of experiments and models, as original experiment labels were not clear
4. Expand the universe to 500 as was suggested by the perplexity.ai analysis
5. Richer baselines: Identify better baselines to compare against, including:
    - Major US Indices: Dow Jones Industrial Average, S&P 500 Index, NASDAQ Composite Index
    - Bond yield, representing the price of borrowing
    - Our own market-cap weighted index across our universe
    - A dart board (it is always funny to see when it wins)
6. Any other features you consider relevant to close out the first POC

# Phase 2+ objectives:
- Implement multiple models for both feature generation AND risk/return evaluation
- Build an "RL gym" that samples real world data to simulate situations to train RL algorithms to make investment decisions then compare against other more traditional approaches
- Cross-validate methodology with other models and harnesses to be as accurate as possible
    - Buy/sell costs and market frictions
    - Information delay
    - Etc
- Draft out the paper I need to write (I prefer to do this part myself with a writing outline, and you proofread it)
- Trial alternative trading strategies
    - Longer or shorter term
    - Different information streams
    - Different security/asset classes
    - Etc
- Find a trading platform to deploy the algorithm on a trial basis with limited seed money once back-testing and simulated live results are solid enough

    




We've finally finished initial build and refactor that aligns with the plan




**Constraint:** My 5hr limit appears to be running out after this morning's refactor. 


Pretending we're an Agile team for a moment, and our 

We need to plan out the remaining work that would allow myself to directly use what we



Please enrich the documentation of the repository so that 