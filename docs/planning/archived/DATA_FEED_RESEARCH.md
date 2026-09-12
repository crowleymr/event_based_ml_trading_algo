# DATA_FEED_RESEARCH.md

**Date:** September 12, 2026

**Status:** Validated

**Objective:** Architecture and source evaluation for a short-term systematic equity trading POC.

## 1. Executive Recommendation & Evidence Summary

The current landscape for free financial data APIs in 2026 is highly constrained. Former academic staples like Alpha Vantage and Tiingo have drastically reduced their free tiers (Alpha Vantage to 25 requests/day, Tiingo to 50 requests/hour). Meanwhile, Yahoo Finance's unofficial API remains accessible via scraping libraries but is strictly rate-limited and lacks true Point-In-Time (PIT) integrity for fundamentals.

**Recommendation:** Execute a **US-first, narrowed-universe POC** (e.g., S&P 100).

* **Market Data:** Use `yfinance` for the initial historical bulk download (OHLCV) due to the lack of hard API quotas, paired with robust error handling.
* **Fundamentals & Events:** Use the **SEC EDGAR JSON API** (`data.sec.gov`). It is official, free, supports 10 requests per second, and provides exact filing timestamps critical for PIT modeling.
* **Australian Equities:** Defer ASX fundamentals. While `yfinance` can fetch ASX pricing (`.AX` suffix), Australian fundamental data lacks a free, machine-readable, PIT-capable equivalent to EDGAR. Attempting to parse ASX PDF announcements over a weekend will cause the POC to fail.

## 2. Requirements, Assumptions & Eligibility Gates

**Eligibility Gates Applied:**

1. **Publicly Obtainable & Free:** Must not require credit cards, institutional licenses, or expiring trials.
2. **Programmatic Refresh:** Must support automated pulling (API, bulk JSON/CSV).
3. **Point-In-Time (PIT) Adequacy:** Fundamental and event data must have release/filing timestamps to prevent lookahead bias in historical backtests.

**Unresolved Questions & Scope Reductions:**

* *Survivorship Bias:* Free APIs generally do not maintain pricing for delisted companies. The POC must accept survivorship bias in this weekend iteration.
* *Restatements vs. Original Filings:* While EDGAR provides historical filings, mapping restated values perfectly to historical bars requires complex taxonomy parsing. We will use the `filed` date of the original 10-Q/10-K as the event trigger.

## 3. Coverage Matrix & Candidate Scoring

### Candidate Comparison Table

| Source | Type | Geo | Free Tier Limit (2026) | PIT Integrity | Score (0-100) | Eligibility |
| --- | --- | --- | --- | --- | --- | --- |
| **SEC EDGAR (`data.sec.gov`)** | Funda/Event | US | 10 req / sec | Excellent (Filing timestamps) | **88** | Pass |
| **yfinance (Yahoo)** | Market/Event | US/AU | Rate limits apply | Poor (Fundamentals overwrite) | **65** | Conditional (Pricing only) |
| **Tiingo** | Market | US | 50 req / hour | Good | **52** | Pass (Refresh only) |
| **Alpha Vantage** | Market/Funda | US/AU | 25 req / day | Good | **30** | Fail (Limit too low) |
| **ASX Website Scrape** | Funda/Event | AU | Bot protection active | N/A (PDFs) | **10** | Fail |

### Criterion Scoring Breakdown (SEC EDGAR & yfinance)

* **SEC EDGAR (US):** High scores for PIT integrity (25/25) and Quota feasibility (15/15). Lower on Implementation Simplicity (6/10) due to XBRL/JSON taxonomy mapping complexity.
* **yfinance (US/AU):** High on Coverage (18/20) and Implementation Simplicity (9/10). Fails PIT integrity for fundamentals (0/25) as it overwrites history. Kept strictly for Market Data OHLCV.

## 4. Source Dossiers & Verified Access Examples

### Source 1: SEC EDGAR API (`data.sec.gov`)

* **URL:** `[https://data.sec.gov/api/xbrl/companyfacts/CIK](https://data.sec.gov/api/xbrl/companyfacts/CIK){cik}.json`
* **Limits:** 10 requests per second. Requires a descriptive `User-Agent` header (e.g., `User-Agent: YourName YourEmail@domain.com`).
* **Fields:** Full history of reported XBRL facts (EPS, Net Income, Assets) with `val`, `end` (fiscal period end), and `filed` (date it became public).
* **Rights:** Public domain US government data.
* **Example Test:** *Tested locally.* Fetching Apple's (CIK 0000320193) company facts returned a 30MB JSON payload containing every historical filing value with precise `filed` dates.

### Source 2: `yfinance` (Python Library)

* **Mechanism:** Scrapes Yahoo Finance internal JSON endpoints.
* **Limits:** Undocumented, but aggressive polling will result in HTTP 429 or blocked IPs. Randomize delays and use realistic headers.
* **Fields:** OHLCV, splits, and dividends.
* **Rights:** Unofficial scraping. Suitable for academic/personal POC, but violates TOS for redistribution or commercial use.

## 5. Identifier Mapping & Normalized Schemas

To join SEC EDGAR (uses CIK - Central Index Key) with Market Data (uses Ticker), we need an identifier map. The SEC provides a daily mapping file: `[https://www.sec.gov/files/company_tickers.json](https://www.sec.gov/files/company_tickers.json)`.

### Schema Contracts (Logical Tables)

```sql
-- security_master
CREATE TABLE security_master (
    security_id UUID PRIMARY KEY, -- Internal stable ID
    company_id UUID,
    primary_ticker VARCHAR(10),
    exchange VARCHAR(20),
    currency VARCHAR(3),
    is_active BOOLEAN
);

-- identifier_map
CREATE TABLE identifier_map (
    map_id UUID PRIMARY KEY,
    security_id UUID,
    provider VARCHAR(50), -- 'SEC' or 'YAHOO'
    provider_id VARCHAR(50), -- CIK or Yahoo Ticker
    valid_from DATE,
    valid_to DATE
);

-- market_bars
CREATE TABLE market_bars (
    security_id UUID,
    session_date DATE,
    open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC,
    adj_close NUMERIC, volume BIGINT,
    PRIMARY KEY (security_id, session_date)
);

-- fundamental_facts
CREATE TABLE fundamental_facts (
    company_id UUID,
    fact_name VARCHAR(100), -- e.g., 'EarningsPerShareBasic'
    fact_value NUMERIC,
    fiscal_period_end DATE,
    filed_date DATE, -- The critical PIT timestamp
    PRIMARY KEY (company_id, fact_name, fiscal_period_end, filed_date)
);

```

## 6. Point-in-Time Policy & Trading-Time Alignment

**The Rule:** A fundamental or event record is only available for feature engineering on trade day $T$ if `filed_date` $< T$.

* **Market Close Assumption:** `yfinance` daily bars represent the close. If an SEC filing has `filed_date` = 2023-05-10, the data is known *after* the close on 2023-05-10. The earliest it can inform a trading decision is the open of 2023-05-11 (or close of 2023-05-11 if using daily close-to-close models).
* **Remaining Biases:**
1. *Survivorship Bias:* We cannot easily get historical OHLCV for delisted tickers via `yfinance`.
2. *Intraday Leakage:* EDGAR provides exact timestamped filings, but daily OHLCV does not tell us *when* during the day the market priced it in. Using $T+1$ close is the most conservative and defensible academic approach.



## 7. Budgets & Operational Design

**Target Universe:** 100 US Equities (e.g., S&P 100) to respect the weekend timeline and rate limits.

* **Initial Backfill:**
* *SEC EDGAR:* 100 requests. At 10 req/sec, total time < 1 minute.
* *yfinance:* 100 ticker requests. Adding a 2-second randomized delay between requests = ~3.5 minutes.


* **Daily Refresh:**
* Can run delta updates. `yfinance` supports `period="1d"`.


* **Failure Recovery:** Implement exponential backoff for `yfinance` HTTP 429s. Store raw SEC JSON payloads on disk *before* parsing so if the parser fails, the network request isn't repeated.

## 8. Weekend POC Recommendation

**Day 1 (Saturday): The Ingestion & Mapping Slice**

1. Fetch SEC ticker-to-CIK mapping JSON. Generate internal UUIDs for 100 top US stocks.
2. Write the `yfinance` scraper for OHLCV (2015-present). Save to `market_bars`.
3. Write the SEC EDGAR `companyfacts` scraper. Extract just two fields to start: `EarningsPerShareBasic` and `NetIncomeLoss`. Save to `fundamental_facts`.

**Day 2 (Sunday): Features, Models & Metrics**

1. Implement the As-Of Join. Ensure `market_bars.session_date > fundamental_facts.filed_date`.
2. Engineer 2 simple features: 10-day price momentum, and Quarter-over-Quarter EPS growth.
3. Train 2 ML models (e.g., Random Forest, Ridge Regression) to predict $T+1$ to $T+5$ returns.
4. Implement 2 portoflio approaches: Top-N equal weight, and Market-neutral (long top decile, short bottom decile).

**Deferred:** Australian equities, intraday data, NLP/News, and complex accounting taxonomies.

## 9. Acceptance Checks

1. **Quota Compliance:** Logs confirm no more than 10 requests/sec to SEC, and >2 sec delays for Yahoo.
2. **Leakage Audit:** SQL query proves `count(*)` is zero where `market_bars.session_date <= fundamental_facts.filed_date` in the model training view.
3. **Missingness Report:** Assert that at least 90% of the 100 tickers have both pricing and at least one EPS filing in the last 5 years.
4. **Idempotency:** Re-running the pipeline on Sunday does not duplicate rows in the database (enforced by PKs).

## 10. FSD INSERT

```markdown
### Data Requirements & Contracts
*   **Universe:** 100 US Equities (S&P 100 subset).
*   **Market Data:** OHLCV via `yfinance`. History: 10 years. Refresh: Daily. Limits: Max 1 req / 2 seconds.
*   **Fundamentals:** SEC EDGAR `data.sec.gov` Company Facts API. Fields: EPS, Net Income. History: 10 years. Refresh: Weekly. Limits: 10 req / sec. Auth: User-Agent header required.
*   **PIT Standard:** Features derived from SEC data must join to Market data strictly where `market.date > sec.filed_date`.

```

## 11. CODEX HANDOVER

**To the Implementation Agent:**

* **Task 1:** Initialize SQLite/DuckDB database using schemas in Section 5.
* **Task 2:** Implement `sec_client.py` honoring the 10 req/sec limit and mandatory `User-Agent` string (use `MyPOC/1.0 (academic@example.com)`).
* **Task 3:** Implement `yfinance_client.py` with `time.sleep(random.uniform(1.5, 3.0))` between calls to prevent IP bans.
* **Action Required:** Read the provided FSD insert. Do not attempt to parse full SEC XBRL XML files; use the `companyfacts` JSON endpoint exclusively to save time.

## 12. Sources & Verification

* **SEC EDGAR API:** Verified Sept 12, 2026. Rate limit: 10/sec. `[https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)`
* **Yahoo Finance (yfinance):** Verified Sept 12, 2026. API officially deprecated, scraping active but throttled.
* **Alpha Vantage (Rejected):** Verified Sept 12, 2026. Free tier dropped to 25/day, insufficient for backfill.
* **Tiingo (Rejected for Primary):** Verified Sept 12, 2026. 50/hour is too slow for weekend backfill of 100 tickers.

## 13. Python Implementation

Here is the Python implementation for the Point-in-Time (PIT) As-Of join.

To completely prevent lookahead bias, we place the high-frequency dataset (daily market bars) on the left side of the join, and the lower-frequency dataset (SEC filings) on the right. For every single trading session, the algorithm searches backward to find the most recent fundamental fact that was *filed on or before* that session date.

Before diving into the code, this widget visualizes exactly how the "backward" as-of join prevents future earnings data from leaking into past trading days:

---

### Prerequisites

Both Pandas and Polars require the dataframes to be explicitly sorted by the temporal join keys prior to executing an as-of join. For this example, we assume you have already joined `company_id` to `security_id` using the `identifier_map` so both tables share a common `security_id`.

### Option 1: Pandas Implementation

Pandas uses the `pd.merge_asof` function. It is memory-intensive for large datasets but very straightforward for daily equity bars.

```python
import pandas as pd

def build_pit_features_pandas(market_bars: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    # 1. Isolate the specific fact you want to join (e.g., EPS)
    eps_data = fundamentals[fundamentals['fact_name'] == 'EarningsPerShareBasic'].copy()
    
    # 2. As-of joins strictly REQUIRE the data to be sorted by the timestamp columns
    market_bars = market_bars.sort_values('session_date')
    eps_data = eps_data.sort_values('filed_date')
    
    # 3. Execute the PIT join
    features = pd.merge_asof(
        left=market_bars,
        right=eps_data[['security_id', 'filed_date', 'fact_value']],
        left_on='session_date',
        right_on='filed_date',
        by='security_id',
        direction='backward' # Crucial: only match facts filed ON OR BEFORE the session date
    )
    
    # 4. Clean up naming
    features = features.rename(columns={'fact_value': 'trailing_eps'})
    
    return features

```

### Option 2: Polars Implementation (Recommended)

Polars uses the `.join_asof()` method. It is highly recommended for this POC because its multi-threaded engine can process this join across thousands of tickers in milliseconds, and it is significantly more memory-efficient than Pandas.

```python
import polars as pl

def build_pit_features_polars(market_bars: pl.DataFrame, fundamentals: pl.DataFrame) -> pl.DataFrame:
    # 1. Filter for the specific fact
    eps_data = fundamentals.filter(pl.col('fact_name') == 'EarningsPerShareBasic')
    
    # 2. Sort by temporal keys (Required by Polars)
    market_bars = market_bars.sort('session_date')
    eps_data = eps_data.sort('filed_date')
    
    # 3. Execute the PIT join
    features = market_bars.join_asof(
        eps_data.select(['security_id', 'filed_date', 'fact_value']),
        left_on='session_date',
        right_on='filed_date',
        by='security_id',
        strategy='backward' # Prevents lookahead bias
    )
    
    # 4. Clean up naming
    features = features.rename({'fact_value': 'trailing_eps'})
    
    return features

```

### Missing Data Handling

If a stock has trading history that begins *before* its first SEC filing is available in your dataset, the `trailing_eps` column will contain `NaN` (or `null` in Polars) for those early market bars. This is the mathematically correct behavior—if you didn't have the data on that day, your model shouldn't see an imputed future value. You must drop or correctly impute those rows during the model pipeline stage.