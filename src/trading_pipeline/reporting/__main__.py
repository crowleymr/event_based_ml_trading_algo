"""Command-line entry point for immutable-run reporting."""

import argparse
from pathlib import Path

from .generate import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate versioned read-only reports from one immutable run directory."
    )
    parser.add_argument("--run", required=True, help="Exact runs/<run_id> directory")
    parser.add_argument(
        "--output",
        help="New output directory; defaults to reports/<run_id>/v1",
    )
    args = parser.parse_args()
    run = Path(args.run)
    output = Path(args.output) if args.output else Path("reports") / run.name / "v1"
    print(generate_report(run, output))


if __name__ == "__main__":
    main()
