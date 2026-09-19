"""Deterministic dashboard start command."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Start the read-only research dashboard")
    parser.add_argument("--report", required=True)
    parser.add_argument("--rl-run")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    app = Path(__file__).with_name("app.py")
    command = [sys.executable, "-m", "streamlit", "run", str(app),
               "--server.port", str(args.port), "--", "--report", args.report]
    if args.rl_run:
        command += ["--rl-run", args.rl_run]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
