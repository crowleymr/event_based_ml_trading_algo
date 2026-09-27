"""Download and validate configured source data without running a study."""

from __future__ import annotations

import argparse
import logging

from trading_pipeline.config import load_config
from trading_pipeline.data import ingest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/reuse and validate source data for a Slice 1 config."
    )
    parser.add_argument(
        "--config", default="configs/poc.yaml",
        help="Slice 1 YAML config (default: configs/poc.yaml)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest(load_config(args.config))


if __name__ == "__main__":
    main()
