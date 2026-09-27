"""Create per-arm synthetic verification receipts from a passing JUnit report.

This is an engineering evidence writer only. It does not execute tests, inspect
research data, or make claims about real-data capability.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from .schema import load_study


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_version() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=2,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _junit_totals(path: Path) -> tuple[int, int, int]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Cannot read JUnit XML: {path}") from exc
    suites = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "testsuite"]
    if root.tag.rsplit("}", 1)[-1] == "testsuite" and root not in suites:
        suites.insert(0, root)
    if not suites:
        raise ValueError("JUnit XML contains no testsuite elements")
    try:
        return tuple(sum(int(suite.attrib.get(key, "0")) for suite in suites)
                     for key in ("tests", "errors", "failures"))
    except ValueError as exc:
        raise ValueError("JUnit XML has non-integer testsuite totals") from exc


def write_synthetic_verification_receipts(
    *, junit_xml: str | Path, prepared_study: str | Path,
    output_dir: str | Path, mapping_path: str | Path,
) -> dict[str, str]:
    """Write one immutable receipt per YAML arm and an arm-to-path JSON map."""
    junit_path = Path(junit_xml).resolve()
    study_path = Path(prepared_study).resolve()
    output = Path(output_dir)
    mapping = Path(mapping_path)
    if not junit_path.is_file():
        raise ValueError(f"JUnit XML does not exist: {junit_path}")
    tests, errors, failures = _junit_totals(junit_path)
    if tests <= 0 or errors != 0 or failures != 0:
        raise ValueError(
            f"JUnit report must contain tests and have zero errors/failures "
            f"(tests={tests}, errors={errors}, failures={failures})"
        )
    study = load_study(study_path, allow_engineering_draft=True)
    arms = study.config["experiment_arms"]
    output.mkdir(parents=True, exist_ok=True)
    junit_hash = _sha256(junit_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    version = _code_version()
    result: dict[str, str] = {}
    for arm in arms:
        arm_id = arm["id"]
        receipt_path = output / f"{arm_id}.synthetic_verification.json"
        payload = {
            "schema_version": 1,
            "study_id": study.config["study_id"],
            "evidence_kind": "synthetic_verification",
            "status": "passed",
            "arm_id": arm_id,
            "component_id": arm["component_id"],
            "interface": arm["interface"],
            "junit_xml_path": junit_path.as_posix(),
            "junit_xml_sha256": junit_hash,
            "junit_totals": {"tests": tests, "errors": errors, "failures": failures},
            "generated_at_utc": timestamp,
        }
        if version is not None:
            payload["code_version"] = version
        with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        result[arm_id] = receipt_path.resolve().as_posix()
    mapping.parent.mkdir(parents=True, exist_ok=True)
    with mapping.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit-xml", required=True)
    parser.add_argument("--prepared-study", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mapping", required=True,
                        help="Output JSON path for arm ID to receipt path mapping")
    args = parser.parse_args()
    result = write_synthetic_verification_receipts(
        junit_xml=args.junit_xml, prepared_study=args.prepared_study,
        output_dir=args.output_dir, mapping_path=args.mapping,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
