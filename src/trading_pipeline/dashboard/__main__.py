"""Deterministic dashboard start command."""

import argparse
from pathlib import Path
import subprocess
import sys

from trading_pipeline.dashboard.loader import discover_expanded_reports, repository_root


def main():
    parser = argparse.ArgumentParser(description="Start the read-only research dashboard")
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--report")
    evidence.add_argument("--expanded-report")
    parser.add_argument("--rl-run")
    parser.add_argument("--catalog")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not args.report and not args.expanded_report:
        reports = discover_expanded_reports(repository_root())
        if not reports:
            parser.error("No completed expanded report is available. Generate one with: "
                         "python -m trading_pipeline.reporting.expanded_closeout")
        args.expanded_report = str(reports[0]["root"])
    app = Path(__file__).with_name("app.py")
    command = [sys.executable, "-m", "streamlit", "run", str(app),
               "--server.port", str(args.port),
               "--browser.gatherUsageStats", "false",
               "--"]
    command += ["--expanded-report", args.expanded_report] if args.expanded_report else ["--report", args.report]
    if args.rl_run:
        command += ["--rl-run", args.rl_run]
    if args.catalog:
        command += ["--catalog", args.catalog]
    try:
        raise SystemExit(subprocess.call(command))
    except KeyboardInterrupt:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
