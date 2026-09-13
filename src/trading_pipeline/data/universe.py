"""Fixed, checked-in Slice 1 equity universe."""

from pathlib import Path


def load_universe(path: str | Path) -> list[str]:
    """Load a unique, ordered ticker universe from a checked-in text file."""
    tickers = Path(path).read_text(encoding="utf-8").split()
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError("Universe must contain unique tickers")
    return tickers
