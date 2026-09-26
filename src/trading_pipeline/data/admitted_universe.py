"""Derive immutable expanded-universe inputs from a sealed admission ledger."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .security_master import build_security_master
from .universe_admission import load_candidates, sha256_file


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_admitted_universe(
    admission_root: str | Path, candidate_path: str | Path,
    sec_mapping_path: str | Path, output_dir: str | Path,
) -> dict:
    """Validate admission provenance and materialize a separate derived snapshot."""
    root = Path(admission_root)
    candidates = load_candidates(candidate_path)
    manifest = _json(root / "manifest.json")
    final = _json(root / "admission.json")
    if manifest["candidate_sha256"] != sha256_file(candidate_path):
        raise ValueError("Candidate hash differs from the sealed admission manifest")
    if manifest["sec_mapping_sha256"] != sha256_file(sec_mapping_path):
        raise ValueError("SEC mapping hash differs from the sealed admission manifest")
    if final["candidate_sha256"] != manifest["candidate_sha256"]:
        raise ValueError("Final admission references another candidate snapshot")
    ledger = final["ledger"]
    if len(ledger) != len(candidates) or final["candidate_count"] != len(candidates):
        raise ValueError("Sealed ledger does not account for every candidate")
    if [row["ticker"] for row in ledger] != [row["ticker"] for row in candidates]:
        raise ValueError("Sealed ledger changed candidate order")
    if any(row["outcome"] not in ("admitted", "excluded") for row in ledger):
        raise ValueError("Sealed ledger contains pending or unknown outcomes")
    if sum(row["outcome"] == "admitted" for row in ledger) != final["admitted_count"]:
        raise ValueError("Admitted count does not reconcile to ledger")
    if sum(row["outcome"] == "excluded" for row in ledger) != final["excluded_count"]:
        raise ValueError("Excluded count does not reconcile to ledger")
    for key, field in (("f0_preflight_ready_count", "f0_preflight_ready"),
                       ("f1_preflight_ready_count", "f1_preflight_ready")):
        if sum(bool(row[field]) for row in ledger) != final[key]:
            raise ValueError(f"{key} does not reconcile to ledger")
    for candidate, row in zip(candidates, ledger):
        if row["source_cik"] != candidate["source_cik"]:
            raise ValueError(f"Source CIK changed for {row['ticker']}")
        if row["outcome"] == "admitted" and (
            row["sec_mapping"]["cik"] != row["source_cik"] or row["exclusion_reasons"]
        ):
            raise ValueError(f"Admitted identifier or reason invalid: {row['ticker']}")
        if row["outcome"] == "excluded" and not row["exclusion_reasons"]:
            raise ValueError(f"Exclusion lacks reason: {row['ticker']}")
        for artifact in row.get("raw_artifacts", {}).values():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Raw artifact hash differs for {row['ticker']}")
    admitted = [row for row in ledger if row["outcome"] == "admitted"]
    mapping = _json(Path(sec_mapping_path))
    master = build_security_master([row["ticker"] for row in admitted], mapping)
    if master["cik"].to_list() != [row["source_cik"] for row in admitted]:
        raise ValueError("Security master CIK differs from source/admission")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    universe_path = destination / "admitted_universe.txt"
    master_path = destination / "security_master.parquet"
    source_path = destination / "source_manifest.json"
    if any(path.exists() for path in (universe_path, master_path, source_path)):
        raise ValueError("Derived snapshot already exists; use a fresh output directory")
    with universe_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(row["ticker"] for row in admitted) + "\n")
    with master_path.open("xb") as stream:
        master.write_parquet(stream)
    payload = {
        "schema_version": 1,
        "admission_snapshot_id": final["snapshot_id"],
        "admission_manifest": str((root / "manifest.json").resolve()),
        "admission_manifest_sha256": sha256_file(root / "manifest.json"),
        "admission_ledger": str((root / "admission.json").resolve()),
        "admission_ledger_sha256": sha256_file(root / "admission.json"),
        "candidate_path": str(Path(candidate_path).resolve()),
        "candidate_sha256": sha256_file(candidate_path),
        "sec_mapping_path": str(Path(sec_mapping_path).resolve()),
        "sec_mapping_sha256": sha256_file(sec_mapping_path),
        "generator_sha256": sha256_file(__file__),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_count": len(candidates),
        "admitted_count": len(admitted),
        "excluded_count": len(ledger) - len(admitted),
        "f0_preflight_ready_count": sum(row["f0_preflight_ready"] for row in ledger),
        "f1_preflight_ready_count": sum(row["f1_preflight_ready"] for row in ledger),
        "admitted_universe_sha256": sha256_file(universe_path),
        "security_master_sha256": sha256_file(master_path),
    }
    with source_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return payload


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-root", required=True)
    parser.add_argument("--candidates", default="configs/universe_candidates_500.csv")
    parser.add_argument("--sec-mapping", default="data/raw/sec/company_tickers_exchange.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_admitted_universe(args.admission_root, args.candidates,
                                               args.sec_mapping, args.output), indent=2))


if __name__ == "__main__":
    _cli()
