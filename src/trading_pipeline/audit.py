"""Command-line facade for the persisted-run leakage and integrity audit."""

import argparse
import json
from trading_pipeline.validation.leakage import audit

__all__ = ["audit"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    print(json.dumps(audit(parser.parse_args().run), indent=2))
