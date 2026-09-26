"""Generate a versioned read-only catalogue from run roots."""

import argparse

from .catalog import build_metric_catalog, build_run_catalog, write_run_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Index completed audited run artefacts")
    parser.add_argument("--runs", action="append", required=True, help="Run root; repeatable")
    parser.add_argument("--output", required=True, help="New versioned catalogue directory")
    args = parser.parse_args()
    catalog = build_run_catalog(args.runs)
    metrics = build_metric_catalog(args.runs)
    print(write_run_catalog(catalog, args.output, metrics))


if __name__ == "__main__":
    main()
