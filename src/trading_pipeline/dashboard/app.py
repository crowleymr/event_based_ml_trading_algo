"""Streamlit review surface over immutable generated evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import streamlit as st

from trading_pipeline.dashboard.loader import load_report, load_rl_run


def _arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", required=True)
    parser.add_argument("--rl-run")
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def _report(path: str):
    value = load_report(path)
    return value["provenance"], value["tables"]


@st.cache_data(show_spinner=False)
def _rl(path: str):
    value = load_rl_run(path)
    return value["metadata"], value["tables"]


def main():
    args = _arguments()
    provenance, tables = _report(str(Path(args.report).resolve()))
    st.set_page_config(page_title="Trading research evidence", layout="wide")
    st.title("Trading research evidence")
    st.warning("Research POC only. Slice 1 final-test results are descriptive; RL evidence is exploratory and not confirmatory.")
    overview, performance, prediction, trading, securities, rl_tab = st.tabs([
        "Overview", "Equity & drawdown", "IC & predictions", "Turnover & costs",
        "Securities", "RL pilot",
    ])
    with overview:
        st.subheader("Provenance and limitations")
        st.json({
            "source_run_id": provenance.get("source_run_id"),
            "source_run_git_commit": provenance.get("source_run_git_commit"),
            "report_code_revision": provenance.get("report_code_revision"),
            "generated_at": provenance.get("generated_at"),
            "input_hashes": provenance.get("inputs"),
        }, expanded=False)
        st.dataframe(tables["experiment_comparison"].to_pandas(), width="stretch")
    with performance:
        series = tables["equity_drawdown_series"].to_pandas()
        split = st.selectbox("Split", sorted(series["split"].unique()), key="performance_split")
        chosen = series[series["split"] == split]
        st.line_chart(chosen, x="session_date", y="equity", color="display_label")
        st.line_chart(chosen, x="session_date", y="drawdown", color="display_label")
    with prediction:
        ic = tables["ic_series"].to_pandas()
        st.line_chart(ic, x="session_date", y="ic", color="display_label")
        st.dataframe(tables["experiment_comparison"].select(
            "split", "experiment_id", "display_label", "mae", "rmse", "mean_ic"
        ).to_pandas(), width="stretch")
    with trading:
        costs = tables["turnover_cost_series"].to_pandas()
        st.line_chart(costs, x="session_date", y="cumulative_turnover", color="display_label")
        st.line_chart(costs, x="session_date", y="cumulative_cost", color="display_label")
    with securities:
        st.caption("Generated holdings, trades and reconciled return contributions")
        st.dataframe(tables["security_holdings_summary"].to_pandas(), width="stretch")
        st.dataframe(tables["security_trades_summary"].to_pandas(), width="stretch")
        st.dataframe(tables["security_contribution_summary"].to_pandas(), width="stretch")
    with rl_tab:
        st.error("Exploratory pilot — not confirmatory. No RL superiority claim is supported without a fresh vintage and predeclared walk-forward protocol.")
        if not args.rl_run:
            st.info("Start the dashboard with --rl-run to display an audited Phase 2 run.")
        else:
            metadata, rl = _rl(str(Path(args.rl_run).resolve()))
            st.json({key: metadata.get(key) for key in (
                "run_id", "reference_run_id", "code_revision", "generated_at",
                "seeds", "actual_device", "research_status",
            )}, expanded=False)
            st.dataframe(rl["policy_summary"].to_pandas(), width="stretch")
            curve = rl["equity_curve"].to_pandas()
            curve["policy_seed"] = curve["policy_id"] + " · " + curve["seed"].astype(str)
            st.line_chart(curve, x="session_date", y="equity", color="policy_seed")
            st.dataframe(rl["action_frequencies"].to_pandas(), width="stretch")
            st.dataframe(rl["metrics"].to_pandas(), width="stretch")


if __name__ == "__main__":
    main()
