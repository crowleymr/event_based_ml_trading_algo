"""Materialise one immutable expanded vintage from a sealed admission snapshot.

This is an offline derivation. It never downloads data or mutates admission caches.
The study runner remains the sole authority for score-bearing execution.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import polars as pl

from trading_pipeline.data.schemas import digest, validate
from trading_pipeline.data.sec_client import events_from_facts, normalize_facts
from trading_pipeline.features.build import F0, F1, build
from trading_pipeline.modelling.targets import add_target


SCHEMA_VERSION = 1


def _read_verified(path: Path, expected: str) -> None:
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Missing or changed sealed input: {path}")


def _write(frame: pl.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        frame.write_parquet(stream)
    return {"path": str(path.resolve()), "sha256": digest(path), "rows": frame.height}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _market(raw_path: Path, security: dict, stamp: str) -> pl.DataFrame:
    raw = pl.read_parquet(raw_path)
    required = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    if not required <= set(raw.columns) or raw["Date"].dtype.base_type() != pl.Datetime:
        raise ValueError(f"Invalid pinned Yahoo schema: {raw_path}")
    return raw.select(
        pl.lit(security["security_id"]).alias("security_id"),
        pl.lit(security["ticker"]).alias("ticker"),
        pl.col("Date").dt.date().alias("session_date"),
        *[pl.col(source).cast(pl.Float64).alias(target) for source, target in (
            ("Open", "open"), ("High", "high"), ("Low", "low"),
            ("Close", "close"), ("Adj Close", "adjusted_close"),
            ("Volume", "volume"),
        )],
        pl.lit("yahoo").alias("source"),
        pl.lit(stamp).alias("ingested_at"),
    )


def materialise_expanded_vintage(admitted_root: str | Path, output_dir: str | Path) -> dict:
    """Verify every pinned input and write a new, complete, hash-declared vintage.

    Existing output directories are refused. A failed derivation removes only its own
    newly created staging directory; all sealed inputs remain untouched.
    """
    source_root = Path(admitted_root).resolve()
    source = _json(source_root / "source_manifest.json")
    admission_path = Path(source["admission_ledger"])
    _read_verified(admission_path, source["admission_ledger_sha256"])
    _read_verified(Path(source["admission_manifest"]), source["admission_manifest_sha256"])
    _read_verified(source_root / "security_master.parquet", source["security_master_sha256"])
    _read_verified(source_root / "admitted_universe.txt", source["admitted_universe_sha256"])
    ledger = _json(admission_path)
    admitted = [row for row in ledger["ledger"] if row["outcome"] == "admitted"]
    if len(admitted) != ledger["admitted_count"] or len(admitted) != source["admitted_count"]:
        raise ValueError("Sealed admission counts differ")
    master = pl.read_parquet(source_root / "security_master.parquet")
    if master["ticker"].to_list() != [row["ticker"] for row in admitted]:
        raise ValueError("Master order differs from admitted universe")
    if master["cik"].to_list() != [row["source_cik"] for row in admitted]:
        raise ValueError("Master CIK differs from sealed source identity")

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Immutable vintage already exists: {destination}")
    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Incomplete build needs operator inspection: {staging}")
    staging.mkdir(parents=True)
    try:
        markets, fundamentals, raw_rows, rejected_bars, quality_excluded = [], [], [], [], []
        for row, security in zip(admitted, master.to_dicts()):
            raw = row["raw_artifacts"]
            market_artifact = raw["yahoo_daily_history"]
            market_path = Path(market_artifact["path"])
            _read_verified(market_path, market_artifact["sha256"])
            # The acquisition timestamp is fixed by the sealed admission record.
            normalized = _market(market_path, security, row["checked_at_utc"])
            invalid_ohlc = ((pl.col("high") < pl.max_horizontal("open", "close", "low")) |
                            (pl.col("low") > pl.min_horizontal("open", "close", "high")))
            rejected = normalized.filter(invalid_ohlc)
            for bad in rejected.select("security_id", "ticker", "session_date", "open", "high", "low", "close").to_dicts():
                rejected_bars.append(bad | {"reason": "raw_ohlc_range_invalid",
                                             "raw_path": str(market_path),
                                             "raw_sha256": market_artifact["sha256"]})
            if rejected.height:
                quality_excluded.append(security["security_id"])
            else:
                markets.append(normalized)
            raw_rows.append({"ticker": row["ticker"], "security_id": security["security_id"],
                             "source_kind": "yahoo_daily_history", "path": str(market_path),
                             "sha256": market_artifact["sha256"]})
            sec_artifact = raw.get("sec_company_facts")
            if sec_artifact is not None:
                sec_path = Path(sec_artifact["path"])
                _read_verified(sec_path, sec_artifact["sha256"])
                if not rejected.height:
                    fundamentals.append(normalize_facts(_json(sec_path), security, row["checked_at_utc"]))
                raw_rows.append({"ticker": row["ticker"], "security_id": security["security_id"],
                                 "source_kind": "sec_company_facts", "path": str(sec_path),
                                 "sha256": sec_artifact["sha256"]})
        master = master.filter(~pl.col("security_id").is_in(quality_excluded))
        bars = pl.concat(markets).sort(["security_id", "session_date"])
        facts = pl.concat(fundamentals).sort(["security_id", "filed_date", "fact_name"])
        validate(master, bars, facts)
        tables = {
            "security_master": master,
            "identifier_map": master.select("security_id", "company_id", "ticker", "cik"),
            "market_bars": bars,
            "fundamental_facts": facts,
            "corporate_events": events_from_facts(facts),
        }
        artifacts = {}
        for name, frame in tables.items():
            artifacts[name] = _write(frame, staging / "curated" / name / "part.parquet")
        raw_manifest = pl.DataFrame(raw_rows).sort(["security_id", "source_kind"])
        artifacts["raw_input_manifest"] = _write(raw_manifest, staging / "raw_input_manifest.parquet")
        if rejected_bars:
            rejection_frame = pl.DataFrame(rejected_bars).sort(["security_id", "session_date"])
        else:
            rejection_frame = pl.DataFrame(schema={
                "security_id": pl.String, "ticker": pl.String, "session_date": pl.Date,
                "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
                "reason": pl.String, "raw_path": pl.String, "raw_sha256": pl.String,
            })
        artifacts["market_bar_rejections"] = _write(rejection_frame, staging / "market_bar_rejections.parquet")

        features = add_target(build(bars, facts)).sort(["security_id", "session_date"])
        if features.select("security_id", "session_date").unique().height != features.height:
            raise ValueError("Feature keys are not unique")
        if features.filter(
            (pl.col("latest_eps_filed_date") >= pl.col("session_date")) |
            (pl.col("latest_net_income_filed_date") >= pl.col("session_date"))
        ).height:
            raise ValueError("Feature PIT date is not strictly earlier than session")
        features = features.with_columns(
            pl.all_horizontal(pl.col(column).is_not_null() for column in F0[:-1]).alias("f0_core_ready"),
            pl.all_horizontal(pl.col(column).is_not_null() for column in F0).alias("f0_full_ready"),
            (pl.col("latest_eps").is_not_null() & pl.col("latest_net_income").is_not_null()).alias("sec_ready"),
            (pl.col("forward_return_5d").is_not_null() & pl.col("label_end_date").is_not_null()).alias("label_ready"),
        ).with_columns(
            (pl.col("f0_core_ready") & pl.col("sec_ready")).alias("f1_core_ready"),
            (pl.col("f0_full_ready") & pl.col("sec_ready")).alias("f1_full_ready"),
        )
        eligibility_cols = ["security_id", "ticker", "session_date", "label_end_date",
                            "f0_core_ready", "f0_full_ready", "sec_ready", "f1_core_ready",
                            "f1_full_ready", "label_ready"]
        artifacts["features"] = _write(features, staging / "features" / "features.parquet")
        artifacts["eligibility"] = _write(features.select(eligibility_cols),
                                           staging / "features" / "eligibility.parquet")
        summary = features.group_by("security_id", "ticker").agg(
            pl.len().alias("market_rows"), pl.col("session_date").min().alias("first_session"),
            pl.col("session_date").max().alias("last_session"),
            *[pl.col(column).sum().alias(f"{column}_rows") for column in
              ("f0_core_ready", "f0_full_ready", "sec_ready", "f1_core_ready", "f1_full_ready", "label_ready")],
        ).sort("security_id")
        artifacts["security_summary"] = _write(summary, staging / "security_summary.parquet")
        created = datetime.now(timezone.utc).isoformat()
        source_hashes = {"admission_ledger": source["admission_ledger_sha256"],
                         "admission_manifest": source["admission_manifest_sha256"],
                         "admitted_master": source["security_master_sha256"],
                         "admitted_source_manifest": digest(source_root / "source_manifest.json")}
        # Manifest paths refer to the final immutable directory, not the staging path.
        repo_root = Path(__file__).resolve().parents[3]
        for value in artifacts.values():
            final_path = destination / Path(value["path"]).relative_to(staging)
            value["path"] = str(final_path)
            value["relative_path"] = final_path.relative_to(repo_root).as_posix()
        common = {"schema_version": SCHEMA_VERSION, "snapshot_id": destination.name,
                  "admission_snapshot_id": source["admission_snapshot_id"],
                  "generated_at_utc": created, "generator_sha256": digest(__file__),
                  "source_hashes": source_hashes}
        data_manifest = common | {
            "source_manifest_path": str(source_root / "source_manifest.json"),
            "source_manifest_sha256": source_hashes["admitted_source_manifest"],
            "admission_ledger_path": str(admission_path),
            "candidate_count": ledger["candidate_count"], "admitted_count": len(admitted),
            "canonical_security_count": master.height,
            "quality_excluded_security_count": len(quality_excluded),
            "excluded_count": ledger["excluded_count"], "raw_input_count": raw_manifest.height,
            "rejected_market_bar_count": rejection_frame.height,
            "quality_exclusion_policy": "Exclude entire admitted security from canonical market/fact/feature views if any raw OHLC row violates range consistency; no prices are substituted and no held-price gap is introduced",
            "artifacts": {key: artifacts[key] for key in (*tables, "raw_input_manifest", "market_bar_rejections", "security_summary")},
            "exposure_note": "Current 2026 S&P 500 survivor snapshot; retrospective Yahoo adjustments; no historical membership or delisting source",
        }
        _manifest(staging / "data_vintage_manifest.json", data_manifest)
        feature_manifest = common | {
            "data_vintage_manifest_path": str(destination / "data_vintage_manifest.json"),
            "data_vintage_manifest_sha256": digest(staging / "data_vintage_manifest.json"),
            "f0_columns": F0, "f1_columns": F1,
            "feature_columns": {"F0": F0, "F1": F1},
            "label_column": "forward_return_5d", "label_end_column": "label_end_date",
            "signal_cutoff": "after session_date close; SEC filed_date strictly before session_date",
            "representation_contract": {
                "tabular": "Canonical scalar F0/F1 columns; train-fold-only imputation and scaling by evaluator",
                "sequence": "Same F0/F1 history through T; causal lookback, padding and masks by CausalSequenceView",
                "rl": "Same source snapshot and T cutoff; frozen sleeve state and prior outputs only through declared upstream feature layer",
            },
            "upstream_prediction_features": "none; require a separately versioned out-of-fold lineage table before use",
            "artifacts": {key: artifacts[key] for key in ("features", "eligibility")},
            "feature_rows": features.height,
            "label_ready_rows": int(features["label_ready"].sum()),
            "f0_core_ready_rows": int(features["f0_core_ready"].sum()),
            "f1_core_ready_rows": int(features["f1_core_ready"].sum()),
        }
        _manifest(staging / "feature_manifest.json", feature_manifest)
        staging.rename(destination)
        return {"data_vintage_manifest": str(destination / "data_vintage_manifest.json"),
                "feature_manifest": str(destination / "feature_manifest.json"),
                "feature_manifest_sha256": digest(destination / "feature_manifest.json"),
                "data_vintage_manifest_sha256": digest(destination / "data_vintage_manifest.json"),
                "admitted_count": len(admitted), "canonical_security_count": master.height,
                "feature_rows": features.height}
    except Exception:
        shutil.rmtree(staging)
        raise


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admitted-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(materialise_expanded_vintage(args.admitted_root, args.output), indent=2))


if __name__ == "__main__":
    _cli()
