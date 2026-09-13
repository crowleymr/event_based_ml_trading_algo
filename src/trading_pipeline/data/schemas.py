"""Canonical schemas, file hashing, atomic Parquet writes, and data validation."""

from pathlib import Path
import hashlib
import polars as pl

FACTS = {"EarningsPerShareBasic": "USD/shares", "NetIncomeLoss": "USD"}
FACT_SCHEMA = {
    "company_id": pl.String, "security_id": pl.String, "cik": pl.String,
    "fact_name": pl.String, "fact_value": pl.Float64, "unit": pl.String,
    "fiscal_period_start": pl.Date, "fiscal_period_end": pl.Date,
    "filed_date": pl.Date, "form": pl.String, "accession_number": pl.String,
    "source": pl.String, "ingested_at": pl.String,
}


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_parquet(frame: pl.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.write_parquet(temporary)
    temporary.replace(path)


def validate(master: pl.DataFrame, bars: pl.DataFrame, facts: pl.DataFrame) -> None:
    required = {"security_id", "ticker", "session_date", "open", "high", "low", "close", "adjusted_close", "volume", "source", "ingested_at"}
    if not required <= set(bars.columns):
        raise ValueError("Invalid market schema")
    if not set(FACT_SCHEMA) <= set(facts.columns):
        raise ValueError("Invalid SEC schema")
    if master["security_id"].n_unique() != master.height or master["ticker"].n_unique() != master.height:
        raise ValueError("Duplicate security mapping")
    if master.select("security_id", "company_id", "ticker", "cik").null_count().sum_horizontal()[0] > 0:
        raise ValueError("Null security-master key")
    if bars.select("security_id", "ticker", "session_date").null_count().sum_horizontal()[0] > 0:
        raise ValueError("Null market key")
    if bars.select("security_id", "session_date").unique().height != bars.height:
        raise ValueError("Duplicate market key")
    price_columns = pl.col("open", "high", "low", "close", "adjusted_close")
    if bars.filter(pl.any_horizontal(price_columns.is_null() | ~price_columns.is_finite() | (price_columns <= 0)) |
                   pl.col("volume").is_null() | ~pl.col("volume").is_finite() | (pl.col("volume") < 0)).height:
        raise ValueError("Invalid market values")
    if bars.filter((pl.col("high") < pl.max_horizontal("open", "close", "low")) |
                   (pl.col("low") > pl.min_horizontal("open", "close", "high"))).height:
        raise ValueError("Invalid OHLC range")
    if set(bars["security_id"]) != set(master["security_id"]):
        raise ValueError("Market mapping coverage incomplete")
    if not set(facts["security_id"]) <= set(master["security_id"]):
        raise ValueError("Unmapped SEC security")
    fact_keys = [
        "company_id", "security_id", "cik", "fact_name", "unit",
        "fiscal_period_start", "fiscal_period_end", "filed_date", "form",
        "accession_number",
    ]
    conflicts = facts.group_by(fact_keys, maintain_order=True).agg(
        pl.col("fact_value").n_unique().alias("distinct_values")
    ).filter(pl.col("distinct_values") > 1)
    if conflicts.height:
        raise ValueError("Conflicting SEC fact values for the same fact identity")
    if facts.filter(pl.col("fact_value").is_null() | ~pl.col("fact_value").is_finite() |
                    pl.col("filed_date").is_null() | pl.col("fiscal_period_end").is_null()).height:
        raise ValueError("Invalid SEC values")
    for group in bars.partition_by("security_id"):
        if not group["session_date"].is_sorted():
            raise ValueError("Market dates not monotonic")
