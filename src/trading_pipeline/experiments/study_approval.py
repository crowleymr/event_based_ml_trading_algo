"""Command-line approval gate for a fully evidenced expanded study."""

from __future__ import annotations

import argparse
import json

from .manifest_resolution import finalise_approved_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--prepared-study", required=True)
    parser.add_argument("--capability-gate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = finalise_approved_study(repository_root=args.repository_root,
        prepared_study=args.prepared_study, capability_gate=args.capability_gate,
        approved_study=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
