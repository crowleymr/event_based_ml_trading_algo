"""Official SEC JSON cache and Company Facts normalization."""

from datetime import date
from pathlib import Path
import json
import os
import time
import polars as pl
import requests

from .schemas import FACTS, FACT_SCHEMA


def cached_json(url: str, path: str | Path, cfg: dict) -> dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    contact_file = Path(".sec-user-agent")
    user_agent = (
        os.environ.get("SEC_USER_AGENT") or cfg.get("sec_user_agent") or
        (contact_file.read_text(encoding="utf-8").strip() if contact_file.exists() else None)
    )
    if not user_agent or "@" not in user_agent:
        raise ValueError("Set SEC_USER_AGENT to a real name/application and contact email")
    path.parent.mkdir(parents=True, exist_ok=True)
    retries = cfg.get("retries", 4)
    for attempt in range(retries):
        try:
            time.sleep(cfg.get("throttle_seconds", 0.3))
            response = requests.get(
                url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=60
            )
            response.raise_for_status()
            payload = response.json()
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(path)
            return payload
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("SEC retry loop terminated unexpectedly")


def normalize_facts(payload: dict, security: dict, ingested_at: str) -> pl.DataFrame:
    rows = []
    concepts = payload.get("facts", {}).get("us-gaap", {})
    for name in FACTS:
        for unit, items in concepts.get(name, {}).get("units", {}).items():
            for item in items:
                if not all(key in item for key in ("val", "end", "filed")):
                    continue
                rows.append({
                    **{key: security[key] for key in ("security_id", "company_id", "cik")},
                    "fact_name": name, "fact_value": float(item["val"]), "unit": unit,
                    "fiscal_period_start": date.fromisoformat(item["start"]) if item.get("start") else None,
                    "fiscal_period_end": date.fromisoformat(item["end"]),
                    "filed_date": date.fromisoformat(item["filed"]), "form": item.get("form"),
                    "accession_number": item.get("accn"), "source": "sec", "ingested_at": ingested_at,
                })
    return pl.DataFrame(rows, schema=FACT_SCHEMA).unique().sort(["security_id", "filed_date", "fact_name"])


def events_from_facts(facts: pl.DataFrame) -> pl.DataFrame:
    return facts.select(
        "company_id", "security_id", "cik", pl.col("filed_date").alias("event_date"),
        pl.lit(None, dtype=pl.Datetime).alias("event_datetime"),
        pl.lit("sec_filing").alias("event_type"), "form", "accession_number", "source", "ingested_at",
    ).unique()
