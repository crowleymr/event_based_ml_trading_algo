"""Read-only Streamlit review surface over immutable generated evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st

from trading_pipeline.dashboard.loader import load_catalog, load_report, load_rl_run


def _arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", required=True)
    parser.add_argument("--rl-run")
    parser.add_argument("--catalog")
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


@st.cache_data(show_spinner=False)
def _catalog(path: str):
    value = load_catalog(path)
    return value["provenance"], value["tables"]


def _filtered(frame, split=None, experiments=None):
    value = frame
    if split is not None and "split" in value.columns:
        value = value.filter(value["split"] == split)
    if experiments and "experiment_id" in value.columns:
        value = value.filter(value["experiment_id"].is_in(experiments))
    return value


def main():
    args = _arguments()
    provenance, tables = _report(str(Path(args.report).resolve()))
    st.set_page_config(page_title="Trading research evidence", layout="wide")
    st.title("Trading research evidence")
    st.warning("Research POC only. Final-test and RL rerun evidence are descriptive diagnostics, never selection evidence.")

    splits = sorted(tables["experiment_comparison"]["split"].unique().to_list())
    experiments = sorted(tables["experiment_comparison"]["experiment_id"].unique().to_list())
    with st.sidebar:
        st.header("Evidence filters")
        split = st.selectbox("Split", splits, index=splits.index("test") if "test" in splits else 0)
        selected = st.multiselect("Experiments", experiments, default=experiments)
        st.caption(f"Report schema v{provenance.get('report_schema_version')}")

    tabs = st.tabs([
        "Overview / Evidence", "Data & Splits", "Model Diagnostics",
        "Training Diagnostics", "RL Gym", "Backtest & Benchmark", "Securities",
    ])
    with tabs[0]:
        st.subheader("Immutable evidence and research status")
        st.json({
            "source_run_id": provenance.get("source_run_id"),
            "source_run_git_commit": provenance.get("source_run_git_commit"),
            "report_code_revision": provenance.get("report_code_revision"),
            "generated_at": provenance.get("generated_at"),
            "input_hashes": provenance.get("inputs"),
        }, expanded=False)
        st.dataframe(_filtered(tables["experiment_comparison"], split, selected).to_pandas(), width="stretch")
        if args.catalog:
            catalog_provenance, catalog_tables = _catalog(str(Path(args.catalog).resolve()))
            st.subheader("Audited multi-run catalogue")
            runs = catalog_tables["runs"].to_pandas()
            run_ids = runs["run_id"].tolist()
            chosen_runs = st.multiselect(
                "Runs", run_ids, default=run_ids, key="catalog_run_ids"
            )
            selected_runs = runs[runs["run_id"].isin(chosen_runs)]
            st.dataframe(selected_runs, width="stretch")
            if selected_runs["compatibility_key"].nunique() > 1:
                st.warning(
                    "Selected runs have different compatibility keys and must not be pooled "
                    "or ranked as a fair comparison."
                )
            metrics = catalog_tables.get("metrics")
            if metrics is not None and not metrics.is_empty():
                selected_metrics = metrics.filter(pl.col("run_id").is_in(chosen_runs))
                st.dataframe(selected_metrics.to_pandas(), width="stretch")
            st.caption(
                f"Catalogue schema v{catalog_provenance.get('schema_version')}; "
                "only complete audited runs are indexed."
            )
        if "metric_definitions" in tables:
            st.dataframe(tables["metric_definitions"].to_pandas(), width="stretch")

    with tabs[1]:
        if "split_profile" not in tables:
            st.info("Data and split diagnostics were not recorded in this report version.")
        else:
            st.subheader("Coverage and target availability")
            st.dataframe(tables["split_profile"].to_pandas(), width="stretch")
            st.subheader("Feature missingness and distributions")
            st.dataframe(_filtered(tables["feature_summary"], split).to_pandas(), width="stretch")
            st.subheader("Target distribution")
            st.dataframe(_filtered(tables["target_summary"], split).to_pandas(), width="stretch")

    with tabs[2]:
        if "prediction_diagnostics" not in tables:
            st.info("Prediction diagnostics were not recorded in this report version.")
        else:
            st.subheader("Prediction error and direction")
            st.dataframe(_filtered(tables["prediction_diagnostics"], split, selected).to_pandas(), width="stretch")
            deciles = _filtered(tables["prediction_deciles"], split, selected).to_pandas()
            if not deciles.empty:
                st.line_chart(deciles, x="prediction_decile", y="mean_realised_return", color="display_label")
            ic = _filtered(tables["ic_series"], split, selected).to_pandas()
            if not ic.empty:
                st.line_chart(ic, x="session_date", y="ic", color="display_label")
            st.subheader("Feature importance")
            st.dataframe(_filtered(tables["feature_importance"], experiments=selected).to_pandas(), width="stretch")
            if not any("shap" in str(value).lower() for value in tables["feature_importance"]["method"].unique()):
                st.info("SHAP importance was not recorded. Native gain/coefficient and validation permutation evidence are shown where available.")

    with tabs[3]:
        st.subheader("Common training diagnostics")
        st.caption(
            "Axes use the recorded training step and the selected metric's native semantics. "
            "Train and validation RMSE are decimal-return errors; DQN reward/loss metrics are not directly comparable to RMSE."
        )
        st.json({
            "supervised_run_id": provenance.get("source_run_id"),
            "supervised_protocol_status": provenance.get("source_run_research_status", "not recorded"),
            "split_semantics": "training = fitted rows; validation = tuning/early stopping only; test = descriptive and absent from training curves",
        }, expanded=False)
        if "training_availability" in tables:
            st.dataframe(tables["training_availability"].to_pandas(), width="stretch")
        trace_frames = []
        trace = tables.get("training_trace")
        if trace is not None and not trace.is_empty():
            supervised_trace = trace.to_pandas()
            supervised_trace["run_id"] = provenance.get("source_run_id")
            trace_frames.append(supervised_trace)
        else:
            st.info("Supervised staged telemetry was not recorded for this run.")
        if args.rl_run:
            metadata, rl = _rl(str(Path(args.rl_run).resolve()))
            st.json({key: metadata.get(key) for key in (
                "run_id", "requested_device", "actual_device", "device_fallback_reason",
                "versions", "telemetry_schema_version", "research_status",
            )}, expanded=False)
            if "training_trace" in rl and not rl["training_trace"].is_empty():
                dqn_trace = rl["training_trace"].to_pandas()
                dqn_trace["run_id"] = metadata.get("run_id")
                trace_frames.append(dqn_trace)
        else:
            st.info("DQN training telemetry is unavailable because no completed audited RL run was supplied.")
        if trace_frames:
            combined = pd.concat(trace_frames, ignore_index=True, sort=False)
            combined["trace_id"] = (
                combined["model_family"].fillna("Unknown") + " · "
                + combined["experiment_id"].fillna(combined["policy_id"]).fillna("unregistered")
                + " · " + combined["phase"].fillna("unspecified")
            )
            families = sorted(combined["model_family"].dropna().unique().tolist())
            selected_families = st.multiselect(
                "Model families", families, default=families, key="training_model_families"
            )
            available_metrics = sorted(
                combined.loc[combined["model_family"].isin(selected_families), "metric_name"]
                .dropna().unique().tolist()
            )
            if available_metrics:
                default_metric = "validation/rmse" if "validation/rmse" in available_metrics else available_metrics[0]
                metric_name = st.selectbox(
                    "Metric", available_metrics,
                    index=available_metrics.index(default_metric), key="training_metric",
                )
                chart = combined[
                    combined["model_family"].isin(selected_families)
                    & (combined["metric_name"] == metric_name)
                ]
                st.line_chart(chart, x="step", y="metric_value", color="trace_id")
            else:
                st.info("Select at least one model family to display a recorded metric.")
        st.info(
            "Elastic Net has solver iterations, convergence and final objective/RMSE diagnostics, "
            "but no conventional epoch or boosting learning curve applies; none is fabricated."
        )
        st.subheader("Training summaries and device evidence")
        if "training_summary" in tables:
            st.dataframe(tables["training_summary"].to_pandas(), width="stretch")
        if "device_benchmark" in tables and not tables["device_benchmark"].is_empty():
            st.dataframe(tables["device_benchmark"].to_pandas(), width="stretch")
        else:
            st.info("Supervised CPU/GPU benchmark not recorded for this run.")
        if args.rl_run:
            if "training_summary" in rl:
                st.dataframe(rl["training_summary"].to_pandas(), width="stretch")
            if "device_benchmark" in rl:
                st.dataframe(rl["device_benchmark"].to_pandas(), width="stretch")

    with tabs[4]:
        st.error("Diagnostic reproduction only. No RL superiority claim is supported without a fresh vintage and predeclared walk-forward protocol.")
        if not args.rl_run:
            st.info("Start with --rl-run to display audited RL evidence.")
        else:
            _, rl = _rl(str(Path(args.rl_run).resolve()))
            st.dataframe(rl["policy_summary"].to_pandas(), width="stretch")
            curve = rl["equity_curve"].to_pandas()
            curve["policy_seed"] = curve["policy_id"] + " · " + curve["seed"].astype(str)
            st.line_chart(curve, x="session_date", y="equity", color="policy_seed")
            st.line_chart(curve, x="session_date", y="daily_return", color="policy_seed")
            st.dataframe(rl["action_frequencies"].to_pandas(), width="stretch")
            st.dataframe(rl["actions"].to_pandas(), width="stretch")

    with tabs[5]:
        series = _filtered(tables["equity_drawdown_series"], split, selected).to_pandas()
        if not series.empty:
            st.line_chart(series, x="session_date", y="equity", color="display_label")
            st.line_chart(series, x="session_date", y="drawdown", color="display_label")
        if "benchmark_relative_series" in tables:
            relative = _filtered(tables["benchmark_relative_series"], split, selected).to_pandas()
            if not relative.empty:
                st.line_chart(relative, x="session_date", y="relative_return", color="display_label")
                st.line_chart(relative, x="session_date", y="rolling_63d_tracking_error", color="display_label")
            st.dataframe(_filtered(tables["benchmark_relative_metrics"], split, selected).to_pandas(), width="stretch")
        costs = _filtered(tables["turnover_cost_series"], split, selected).to_pandas()
        st.line_chart(costs, x="session_date", y="cumulative_turnover", color="display_label")
        st.line_chart(costs, x="session_date", y="cumulative_cost", color="display_label")

    with tabs[6]:
        st.caption("Generated holdings, trades and reconciled security contributions")
        st.dataframe(_filtered(tables["security_holdings_summary"], split, selected).to_pandas(), width="stretch")
        st.dataframe(_filtered(tables["security_trades_summary"], split, selected).to_pandas(), width="stretch")
        st.dataframe(_filtered(tables["security_contribution_summary"], split, selected).to_pandas(), width="stretch")


if __name__ == "__main__":
    main()
