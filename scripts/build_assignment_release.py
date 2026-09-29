"""Build a hash-verified release bundle from generated assignment evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


RUN_ID = "20260928T005423Z-0634efaf"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _verify_outputs(directory: Path, manifest: dict) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"Manifest has no declared outputs: {directory}")
    for item in outputs:
        relative = item.get("relative_path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"Malformed output declaration in {directory}")
        path = (directory / relative).resolve()
        if directory.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"Declared output is missing or escapes its directory: {relative}")
        if _sha256(path) != expected:
            raise ValueError(f"Declared output hash mismatch: {path}")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 9, 28, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build(repository_root: Path, output: Path) -> tuple[Path, Path, Path]:
    root = repository_root.resolve()
    expanded = root / "reports" / "expanded_closeout" / RUN_ID
    assignment = root / "reports" / "assignment" / RUN_ID / "v1"
    expanded_manifest_path = expanded / "research_evidence_manifest.json"
    assignment_manifest_path = assignment / "assignment_evidence_manifest.json"
    expanded_manifest = _json(expanded_manifest_path)
    assignment_manifest = _json(assignment_manifest_path)

    if expanded_manifest.get("source_run_id") != RUN_ID:
        raise ValueError("Expanded report does not identify the canonical run")
    if assignment_manifest.get("source_run_id") != RUN_ID:
        raise ValueError("Assignment export does not identify the canonical run")
    if assignment_manifest.get("source_wp7_manifest_sha256") != _sha256(expanded_manifest_path):
        raise ValueError("Assignment export is not bound to the supplied expanded report")
    _verify_outputs(expanded, expanded_manifest)
    _verify_outputs(assignment, assignment_manifest)

    fixed_files = [
        root / "notebooks" / "final_evidence.ipynb",
        root / "docs" / "assignment" / "CITATION_REGISTER.md",
        root / "docs" / "assignment" / "references.bib",
        root / "requirements-lock.txt",
        root / "LICENSE",
    ]
    sources = sorted(
        [path for directory in (expanded, assignment) for path in directory.iterdir() if path.is_file()]
        + fixed_files,
        key=lambda path: path.relative_to(root).as_posix(),
    )
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Release input is missing: {missing[0]}")

    entries = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sources
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": RUN_ID,
        "claim_status": assignment_manifest.get("claim_status"),
        "holdout_role": assignment_manifest.get("holdout_role"),
        "protocol_sha256": assignment_manifest.get("protocol_sha256"),
        "expanded_report_manifest_sha256": _sha256(expanded_manifest_path),
        "assignment_evidence_manifest_sha256": _sha256(assignment_manifest_path),
        "entries": entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Release bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sources:
            name = path.relative_to(root).as_posix()
            archive.writestr(_zip_info(name), path.read_bytes())
        archive.writestr(_zip_info("bundle_manifest.json"), manifest_bytes)

    bundle_sha = _sha256(output)
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{bundle_sha}  {output.name}\n", encoding="ascii")
    handoff = output.with_suffix(".md")
    handoff.write_text(
        "# GitHub Release Handoff\n\n"
        f"Release asset: `{output.name}`  \n"
        f"SHA-256: `{bundle_sha}`  \n"
        f"Canonical run: `{RUN_ID}`  \n"
        f"Protocol SHA-256: `{manifest['protocol_sha256']}`  \n\n"
        "Upload both the ZIP and its `.sha256` sidecar to the GitHub Release. "
        "After publication, replace the notebook's fail-closed release placeholders "
        "with the final tag, asset URL and this asset hash, then execute the notebook "
        "in a clean public runtime before inserting its plain-text URL into the submission.\n",
        encoding="utf-8",
    )
    return output, checksum, handoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--output",
        default=f"release/assignment-evidence-{RUN_ID}.zip",
    )
    args = parser.parse_args()
    outputs = build(Path(args.repository_root), Path(args.output))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
