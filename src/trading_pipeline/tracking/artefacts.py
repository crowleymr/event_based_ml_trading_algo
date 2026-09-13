"""Persist JSON and research plots under the immutable run contract."""

import json
from pathlib import Path
import matplotlib
import numpy as np
import polars as pl
from trading_pipeline.data.schemas import digest

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def input_vintage(cfg: dict, master: pl.DataFrame, include_spy: bool = False) -> list[dict]:
    """Hash only raw files consumed by this run, rather than every shared cache file."""
    data_root = Path(cfg["data_dir"])
    if cfg["mode"] == "synthetic":
        candidates = [data_root / "raw/synthetic/mapping.json"]
        candidates += [data_root / "raw/synthetic" / f"{row['ticker']}.parquet"
                       for row in master.select("ticker").to_dicts()]
        candidates += [data_root / "raw/synthetic" / f"{row['cik']}.json"
                       for row in master.select("cik").to_dicts()]
    else:
        candidates = [data_root / "raw/sec/company_tickers_exchange.json"]
        candidates += [data_root / "raw/sec" / f"CIK{row['cik']}.json"
                       for row in master.select("cik").to_dicts()]
        tickers = master["ticker"].to_list() + (["SPY"] if include_spy else [])
        for ticker in tickers:
            base = data_root / "raw/yahoo" / f"{ticker}_{cfg['start']}_{cfg['end']}.parquet"
            candidates.extend([base, base.with_suffix(".json")])
    return [{
        "cache_path": str(path.resolve()),
        "relative_cache_path": path.relative_to(data_root).as_posix(),
        "sha256": digest(path),
    } for path in candidates if path.is_file()]


def plots(curves: pl.DataFrame, comparison: pl.DataFrame, root: Path) -> None:
    root.mkdir(exist_ok=True)
    for metric, filename in (("equity", "equity_curve.png"), ("drawdown", "drawdown.png")):
        figure, axis = plt.subplots(figsize=(10, 5))
        for group in curves.filter(pl.col("split") == "test").partition_by("experiment"):
            values = group["equity"].to_numpy()
            if metric == "drawdown":
                values = values / np.maximum.accumulate(np.r_[1.0, values])[1:] - 1
            axis.plot(group["session_date"].to_list(), values, label=group["experiment"][0])
        axis.set(title=f"Held-out test {metric}", ylabel=metric)
        axis.legend(ncol=3)
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(root / filename, dpi=140)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 5))
    test = comparison.filter(pl.col("split") == "test")
    axis.bar(test["experiment"].to_list(), test["total_return"].to_list())
    axis.set(title="Held-out test return after costs (not a selection criterion)", ylabel="Total return")
    figure.tight_layout()
    figure.savefig(root / "model_comparison.png", dpi=140)
    plt.close(figure)
