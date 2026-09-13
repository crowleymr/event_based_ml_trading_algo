"""Stable internal identifiers built from the official SEC ticker/CIK mapping."""

import polars as pl


def build_security_master(tickers: list[str], mapping: dict) -> pl.DataFrame:
    entries = mapping["data"] if "data" in mapping else list(mapping.values())
    if "fields" in mapping:
        entries = [dict(zip(mapping["fields"], row)) for row in entries]
    lookup = {row["ticker"].replace(".", "-"): row for row in entries}
    missing = sorted(set(tickers) - lookup.keys())
    if missing:
        raise ValueError(f"SEC mapping missing tickers: {missing}; do not silently change universe")
    rows = []
    for ticker in tickers:
        item = lookup[ticker]
        cik = str(item.get("cik", item.get("cik_str"))).zfill(10)
        rows.append({
            "security_id": f"US-{cik}-{ticker}", "company_id": f"CIK-{cik}",
            "ticker": ticker, "exchange": item.get("exchange", "unknown"),
            "currency": "USD", "cik": cik, "is_active": True,
        })
    return pl.DataFrame(rows)
