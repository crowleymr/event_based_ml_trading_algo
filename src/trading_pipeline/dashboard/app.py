"""Read-only Streamlit review surface over immutable generated evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st

from trading_pipeline.dashboard.loader import (
    discover_expanded_reports, load_catalog, load_expanded_report, load_report,
    load_rl_run, repository_root,
)


def _arguments():
    parser = argparse.ArgumentParser(add_help=False)
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--report")
    evidence.add_argument("--expanded-report")
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


@st.cache_data(show_spinner=False, max_entries=4)
def _expanded(path: str):
    value = load_expanded_report(path)
    return value["provenance"], value["tables"]


def _expanded_view(path: str) -> None:
    provenance, tables = _expanded(str(Path(path).resolve()))
    st.set_page_config(page_title="Expanded research evidence", layout="wide")
    st.title("Expanded research evidence")
    st.warning("Completed holdout evidence is descriptive. Do not select a model or risk scenario from these results.")
    st.caption(f"Source run: {provenance['source_run_id']} · Claim status: {provenance.get('claim_status', 'not recorded')}")
    with st.expander("Evidence provenance"):
        st.json(provenance, expanded=False)

    metrics = tables["final_testbench_metrics"]
    model_ids = sorted(value for value in metrics["model_id"].drop_nulls().unique().to_list()
                       if value != "B0_SPY")
    with st.sidebar:
        st.header("Evidence controls")
        selected_models = st.multiselect("Models", model_ids, default=model_ids, key="expanded_models")
        scenarios = sorted(metrics["risk_scenario"].drop_nulls().unique().to_list())
        selected_scenarios = st.multiselect(
            "Risk scenarios", scenarios, default=scenarios, key="expanded_scenarios"
        )
        st.caption("The SPY benchmark is retained in the test bench regardless of model filter.")

    def selected(frame: pl.DataFrame) -> pl.DataFrame:
        value = frame
        if "model_id" in value.columns:
            base_model = pl.col("model_id").str.split("|").list.first()
            value = value.filter(base_model.is_in(selected_models) | (pl.col("model_id") == "B0_SPY"))
        if "risk_scenario" in value.columns:
            value = value.filter(pl.col("risk_scenario").is_in(selected_scenarios) |
                                 (pl.col("model_id") == "B0_SPY") if "model_id" in value.columns
                                 else pl.col("risk_scenario").is_in(selected_scenarios))
        return value

    tabs = st.tabs(["Pipeline and securities", "Architecture and HPO", "Risk and frontiers",
                    "Final test bench", "Model monitoring / run comparisons", "Evidence register"])
    with tabs[0]:
        stage = tables["pipeline_stage_summary"]
        if "count" in stage.columns:
            st.bar_chart(stage.to_pandas(), x="stage", y="count")
        st.dataframe(stage.to_pandas(), hide_index=True)
        st.subheader("Security drill-down")
        security = tables["pipeline_security_summary"]
        tickers = sorted(security["ticker"].drop_nulls().unique().to_list()) if "ticker" in security.columns else []
        ticker = st.selectbox("Ticker", ["All", *tickers], key="expanded_ticker")
        if ticker != "All":
            security = security.filter(pl.col("ticker") == ticker)
        st.dataframe(security.to_pandas(), hide_index=True)
    with tabs[1]:
        for name in ("architecture_trial_summary", "hpo_trial_summary"):
            st.subheader(name.replace("_", " ").capitalize())
            st.dataframe(selected(tables[name]).to_pandas(), hide_index=True)
    with tabs[2]:
        st.subheader("Declared risk scenarios")
        st.dataframe(selected(tables["risk_scenario_summary"]).to_pandas(), hide_index=True)
        st.subheader("Model-conditioned efficient frontiers")
        frontier = selected(tables["model_conditioned_frontier_points"])
        dates = sorted(frontier["rebalance_date"].unique().to_list())
        if dates:
            day = st.selectbox("Rebalance date", dates, index=len(dates) - 1, key="expanded_rebalance")
            dated = frontier.filter(pl.col("rebalance_date") == day)
            completed = dated.filter(pl.col("status") == "complete")
            if not completed.is_empty():
                st.line_chart(completed.to_pandas(), x="expected_volatility", y="expected_return", color="model_id")
            st.dataframe(dated.to_pandas(), hide_index=True)
            weights = selected(tables["model_conditioned_frontier_weights"])
            st.dataframe(weights.filter(pl.col("rebalance_date") == day).to_pandas(), hide_index=True)
        st.subheader("Realised model risk-return curves")
        realised = selected(tables["realised_risk_return_curve"])
        if not realised.is_empty():
            if "seed" in realised.columns:
                realised = realised.with_columns(
                    (pl.col("model_id") + " · seed " + pl.col("seed").cast(pl.String)).alias("series_label")
                )
            else:
                realised = realised.with_columns(pl.col("model_id").alias("series_label"))
            connected = realised.filter(pl.col("risk_control_monotonic"))
            unconnected = realised.filter(~pl.col("risk_control_monotonic"))
            if not connected.is_empty():
                st.line_chart(connected.sort(["series_label", "risk_order"]).to_pandas(),
                              x="annualised_volatility", y="annualised_return", color="series_label")
            if not unconnected.is_empty():
                st.info("Non-monotonic risk controls are shown as unconnected points.")
                st.scatter_chart(unconnected.to_pandas(), x="annualised_volatility",
                                 y="annualised_return", color="series_label")
            st.dataframe(realised.to_pandas(), hide_index=True)
    with tabs[3]:
        st.subheader("Complete descriptive holdout matrix")
        st.dataframe(selected(metrics).to_pandas(), hide_index=True)
        equity = selected(tables["final_testbench_equity_curve"])
        if not equity.is_empty():
            equity = equity.with_columns(
                (pl.col("model_id") + " · " + pl.col("risk_scenario") + " · seed "
                 + pl.col("seed").cast(pl.String).fill_null("benchmark")).alias("series_label")
            ) if "seed" in equity.columns else equity.with_columns(pl.col("display_label").alias("series_label"))
            st.line_chart(equity.to_pandas(), x="session_date", y="equity", color="series_label")
            st.line_chart(equity.to_pandas(), x="session_date", y="drawdown", color="series_label")
        st.dataframe(equity.to_pandas(), hide_index=True)
    with tabs[4]:
        st.subheader("Comparable completed reports")
        protocol_hash = provenance.get("protocol_sha256")
        comparison_contract = provenance.get("comparison_contract")
        reports = [report for report in discover_expanded_reports(repository_root())
                   if (comparison_contract and
                       report["provenance"].get("comparison_contract") == comparison_contract)
                   or (not comparison_contract and protocol_hash and
                       report["provenance"].get("protocol_sha256") == protocol_hash)]
        if not reports:
            reports = [{"root": Path(path).resolve(), "provenance": provenance, "tables": tables}]
        st.caption("Each row is a generated model × seed × risk result. Reports share the same evaluation calendar, execution, cost and metric contract; no values are pooled or used for selection.")
        if len({report["provenance"].get("protocol_sha256") for report in reports}) > 1:
            st.warning("Comparable reports use different research protocols. Review budget, folds, seeds, devices and arm mappings before interpreting differences.")
        rows = []
        for report in reports:
            source = report["provenance"]
            summary = source.get("protocol_summary", {})
            arm_mappings = summary.get("arms", [])
            run_label = f"{summary.get('study_id', source['source_run_id'])} · {summary.get('budget_tier', 'unlabelled budget')}"
            implementation_scope = ", ".join(
                f"{arm.get('id')}={arm.get('component_id')}[{arm.get('requested_device')}]"
                for arm in arm_mappings
            )
            frame = report["tables"]["final_testbench_metrics"].to_pandas()
            frame.insert(0, "source_run_id", source["source_run_id"])
            frame.insert(1, "run_label", run_label)
            frame.insert(2, "implementation_scope", implementation_scope)
            frame.insert(3, "source_completed_at_utc", source.get("source_completed_at_utc"))
            frame.insert(4, "report_generated_at_utc", source.get("generated_at_utc"))
            frame.insert(5, "protocol_sha256", source.get("protocol_sha256"))
            frame.insert(6, "source_git_commit", source.get("source_git_commit"))
            frame.insert(7, "report_code_version", source.get("code_version"))
            frame.insert(8, "budget_tier", summary.get("budget_tier"))
            frame.insert(9, "outer_folds", summary.get("outer_folds"))
            frame.insert(10, "inner_folds", summary.get("inner_folds"))
            frame.insert(11, "seeds", str(summary.get("seeds")))
            frame.insert(12, "requested_device", summary.get("requested_device"))
            rows.append(frame)
        monitoring = pd.concat(rows, ignore_index=True)
        columns = [name for name in ("source_run_id", "run_label", "implementation_scope",
                   "source_completed_at_utc", "report_generated_at_utc",
                   "budget_tier", "outer_folds", "inner_folds", "seeds", "requested_device",
                   "model_id", "seed", "risk_scenario", "total_return",
                   "annualised_return", "annualised_volatility", "sharpe", "maximum_drawdown",
                   "average_turnover", "cumulative_transaction_cost", "benchmark_relative_total_return",
                   "protocol_sha256", "source_git_commit", "report_code_version")
                   if name in monitoring.columns]
        st.dataframe(monitoring[columns], hide_index=True)
        st.json({"selected_report": provenance["source_run_id"],
                 "compatible_run_count": len(reports),
                 "protocol_summary": provenance.get("protocol_summary", {}),
                 "comparison_contract": comparison_contract,
                 "protocol_sha256": protocol_hash}, expanded=False)
        with st.expander("Protocol details for compared runs"):
            st.json({report["provenance"]["source_run_id"]: {
                "protocol_sha256": report["provenance"].get("protocol_sha256"),
                "protocol_summary": report["provenance"].get("protocol_summary", {}),
                "source_git_commit": report["provenance"].get("source_git_commit"),
                "report_code_version": report["provenance"].get("code_version"),
            } for report in reports}, expanded=False)
    with tabs[5]:
        st.dataframe(tables["question_and_assumption_register"].to_pandas(), hide_index=True)


def _filtered(frame, split=None, experiments=None):
    value = frame
    if split is not None and "split" in value.columns:
        value = value.filter(value["split"] == split)
    if experiments and "experiment_id" in value.columns:
        value = value.filter(value["experiment_id"].is_in(experiments))
    return value


def _model_kind(estimator: object) -> str:
    """Return a presentation-only family label without changing model identity."""
    label = str(estimator).lower()
    if "ppo" in label or "dqn" in label or "reinforcement" in label:
        return "Reinforcement learning"
    if "lstm" in label or "transformer" in label:
        return "Deep supervised"
    if any(token in label for token in ("elastic", "gradient", "xgboost", "gbt")):
        return "Classical supervised"
    return "Fixed baseline"


def _overview_frame(comparison: pl.DataFrame, experiments: list[str]) -> pd.DataFrame:
    """Build chart-ready overview data from the immutable comparison table."""
    frame = _filtered(comparison, experiments=experiments).to_pandas()
    frame = frame[frame["split"].isin(["validation", "test"])].copy()
    frame["model_kind"] = frame["estimator_label"].map(_model_kind)
    frame["model"] = frame["experiment_id"] + " · " + frame["display_label"]
    return frame


def _metric_label(name: str) -> str:
    return {
        "total_return": "Total return",
        "annualised_return": "Annualised return",
        "sharpe": "Sharpe ratio",
        "maximum_drawdown": "Maximum drawdown",
    }.get(name, name.replace("_", " ").capitalize())


def main():
    args = _arguments()
    if not args.report and not args.expanded_report:
        reports = discover_expanded_reports(repository_root())
        if not reports:
            st.error("No completed expanded report is available. Generate one with: "
                     "python -m trading_pipeline.reporting.expanded_closeout")
            st.stop()
        args.expanded_report = str(reports[0]["root"])
    if args.expanded_report:
        _expanded_view(args.expanded_report)
        return
    provenance, tables = _report(str(Path(args.report).resolve()))
    st.set_page_config(page_title="Trading research evidence", layout="wide")
    st.title("Trading research evidence")
    st.warning("Research POC only. Final-test and RL rerun evidence are descriptive diagnostics, never selection evidence.")

    splits = sorted(tables["experiment_comparison"]["split"].unique().to_list())
    experiments = sorted(tables["experiment_comparison"]["experiment_id"].unique().to_list())
    with st.sidebar:
        st.header("Comparison controls")
        split = st.selectbox("Detail split", splits, index=splits.index("test") if "test" in splits else 0)
        selected = st.multiselect("Experiments", experiments, default=experiments)
        st.caption(f"Report schema v{provenance.get('report_schema_version')}")

    tabs = st.tabs([
        "Overview / Evidence", "Data & Splits", "Model Diagnostics",
        "Training Diagnostics", "RL Gym", "Risk / Frontiers",
        "Backtest & Benchmark", "Securities",
    ])
    with tabs[0]:
        st.header("From point-in-time data to after-cost portfolio evidence")
        st.markdown(
            "This study compares fixed baselines and supervised models using the same "
            "market/fundamental information, leakage-controlled splits, T+1 execution and "
            "transaction-cost accounting. Validation supports selection; the observed test "
            "period is descriptive only."
        )
        overview = _overview_frame(tables["experiment_comparison"], selected)
        chosen = overview[overview["split"] == split]
        ranked = chosen.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False)
        universe = tables.get("universe_summary")
        security_count = (
            int(universe["security_count"][0])
            if universe is not None and not universe.is_empty() and "security_count" in universe.columns
            else "Not recorded"
        )
        with st.container(horizontal=True):
            st.metric("Securities", security_count, border=True)
            st.metric("Experiments", int(chosen["experiment_id"].nunique()), border=True)
            st.metric("Evidence lens", split.capitalize(), border=True)
            if not ranked.empty:
                st.metric(
                    f"Highest {split} Sharpe",
                    str(ranked.iloc[0]["experiment_id"]),
                    f"{ranked.iloc[0]['sharpe']:.2f}",
                    border=True,
                )
                st.metric(
                    f"Lowest {split} Sharpe",
                    str(ranked.iloc[-1]["experiment_id"]),
                    f"{ranked.iloc[-1]['sharpe']:.2f}",
                    border=True,
                )

        st.subheader("Validation versus test")
        st.caption(
            "Large validation-to-test changes are a generalisation warning, not proof of "
            "overfitting. Test outcomes were not used to select or reject models."
        )
        metric = st.selectbox(
            "Comparison metric",
            ["sharpe", "total_return", "annualised_return", "maximum_drawdown"],
            format_func=_metric_label,
            key="overview_metric",
        )
        if not overview.empty:
            st.bar_chart(
                overview,
                x="model",
                y=metric,
                color="split",
                x_label="Experiment",
                y_label=_metric_label(metric),
                stack=False,
            )

        st.subheader("Portfolio growth against the market")
        overview_series = _filtered(tables["equity_drawdown_series"], split, selected).to_pandas()
        if not overview_series.empty:
            st.line_chart(
                overview_series,
                x="session_date",
                y="equity",
                color="display_label",
                x_label="Session date",
                y_label="Growth of one unit",
            )

        with st.expander("Models, metrics and immutable provenance"):
            if not overview.empty:
                landscape = (
                    overview[["experiment_id", "display_label", "model_kind", "estimator_label", "portfolio_label"]]
                    .drop_duplicates()
                    .sort_values(["model_kind", "experiment_id"])
                )
                st.dataframe(landscape, hide_index=True)
            if "metric_definitions" in tables:
                st.markdown("**Metric glossary**")
                st.dataframe(tables["metric_definitions"].to_pandas(), hide_index=True)
            st.markdown("**Evidence lineage**")
            st.json({
                "source_run_id": provenance.get("source_run_id"),
                "source_run_git_commit": provenance.get("source_run_git_commit"),
                "report_code_revision": provenance.get("report_code_revision"),
                "generated_at": provenance.get("generated_at"),
                "input_hashes": provenance.get("inputs"),
            }, expanded=False)
        if args.catalog:
            catalog_provenance, catalog_tables = _catalog(str(Path(args.catalog).resolve()))
            runs = catalog_tables["runs"].to_pandas()
            run_ids = runs["run_id"].tolist()
            default_run = provenance.get("source_run_id")
            default_index = run_ids.index(default_run) if default_run in run_ids else len(run_ids) - 1
            chosen_run = st.selectbox(
                "Audited run", run_ids, index=max(default_index, 0), key="catalog_run_id"
            )
            selected_runs = runs[runs["run_id"] == chosen_run]
            metrics = catalog_tables.get("metrics")
            if metrics is not None and not metrics.is_empty():
                selected_metrics = metrics.filter(pl.col("run_id") == chosen_run).to_pandas()
                if not selected_metrics.empty and "sharpe" in selected_metrics.columns:
                    st.bar_chart(
                        selected_metrics,
                        x="experiment_id",
                        y="sharpe",
                        color="split" if "split" in selected_metrics.columns else None,
                        stack=False,
                    )
            with st.expander("Selected run record"):
                st.dataframe(selected_runs, hide_index=True)
                st.caption(
                    f"Catalogue schema v{catalog_provenance.get('schema_version')}; "
                    "only complete audited runs are indexed."
                )

    with tabs[1]:
        if "pipeline_stage_summary" in tables:
            st.subheader("End-to-end pipeline funnel")
            funnel = tables["pipeline_stage_summary"].to_pandas()
            numeric = [name for name in funnel.columns if pd.api.types.is_numeric_dtype(funnel[name])]
            if numeric:
                st.bar_chart(funnel, x=funnel.columns[0], y=numeric[0])
            with st.expander("Pipeline-stage evidence table"):
                st.dataframe(funnel, hide_index=True)
        if "split_profile" not in tables:
            st.info("Data and split diagnostics were not recorded in this report version.")
        else:
            st.subheader("Coverage and target availability")
            split_profile = tables["split_profile"].to_pandas()
            if "rows" in split_profile.columns:
                st.bar_chart(split_profile, x="split", y="rows")
            with st.expander("Split evidence table"):
                st.dataframe(split_profile, hide_index=True)
            st.subheader("Feature missingness and distributions")
            features = _filtered(tables["feature_summary"], split).to_pandas()
            if {"feature", "missing_fraction"} <= set(features.columns):
                st.bar_chart(features, x="feature", y="missing_fraction")
            with st.expander("Feature evidence table"):
                st.dataframe(features, hide_index=True)
            st.subheader("Target distribution")
            with st.expander("Target evidence table", expanded=True):
                st.dataframe(_filtered(tables["target_summary"], split).to_pandas(), hide_index=True)

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
        st.subheader("Model-conditioned efficient frontiers")
        st.caption(
            "Classical frontiers apply only to supervised models that emit security-level "
            "expected returns. DQN and PPO are evaluated in the realised risk-return view."
        )
        frontier = tables.get("model_conditioned_frontier_points")
        if frontier is None or frontier.is_empty():
            st.info("No completed model-conditioned frontier evidence is available for this report.")
        else:
            frontier_pd = frontier.filter(pl.col("status") == "complete").to_pandas()
            if frontier_pd.empty:
                st.warning("All requested frontier optimisations failed; failure evidence is retained.")
                st.dataframe(frontier.to_pandas(), width="stretch")
            else:
                dates = sorted(frontier_pd["rebalance_date"].unique().tolist())
                chosen_date = st.selectbox("Rebalance date", dates, index=len(dates) - 1)
                dated = frontier_pd[frontier_pd["rebalance_date"] == chosen_date].sort_values(
                    ["model_id", "expected_volatility"]
                )
                st.line_chart(
                    dated, x="expected_volatility", y="expected_return", color="model_id"
                )
                st.dataframe(dated, width="stretch")
        st.subheader("Realised model risk-return curves")
        realised = tables.get("realised_risk_return_curve")
        if realised is None or realised.is_empty():
            st.info("No completed three-scenario realised risk-return evidence is available.")
        else:
            realised_pd = realised.sort(["model_id", "risk_order"]).to_pandas()
            monotonic = realised_pd[realised_pd["risk_control_monotonic"]]
            non_monotonic = realised_pd[~realised_pd["risk_control_monotonic"]]
            if not monotonic.empty:
                st.line_chart(
                    monotonic, x="annualised_volatility", y="annualised_return", color="model_id"
                )
            if not non_monotonic.empty:
                st.warning(
                    "Some risk controls are non-monotonic; those outcomes are shown as unconnected points."
                )
                st.scatter_chart(
                    non_monotonic, x="annualised_volatility", y="annualised_return", color="model_id"
                )
            st.dataframe(realised_pd, width="stretch")

    with tabs[6]:
        source_series = tables.get("final_testbench_equity_curve", tables["equity_drawdown_series"])
        series = _filtered(source_series, split, selected).to_pandas()
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

    with tabs[7]:
        st.caption("Generated holdings, trades and reconciled security contributions")
        security_frames = [
            tables["security_holdings_summary"], tables["security_trades_summary"],
            tables["security_contribution_summary"],
        ]
        tickers = sorted({
            ticker for frame in security_frames if "ticker" in frame.columns
            for ticker in frame["ticker"].drop_nulls().to_list()
        })
        selected_tickers = st.multiselect("Tickers", tickers, default=[], key="security_tickers")
        if "pipeline_security_summary" in tables:
            pipeline_security = tables["pipeline_security_summary"]
            if selected_tickers and "ticker" in pipeline_security.columns:
                pipeline_security = pipeline_security.filter(pl.col("ticker").is_in(selected_tickers))
            st.subheader("Pipeline contribution by security")
            st.dataframe(pipeline_security.to_pandas(), width="stretch")
        for heading, frame in zip(
            ("Holdings", "Trades", "Net contribution"), security_frames, strict=True
        ):
            filtered = _filtered(frame, split, selected)
            if selected_tickers and "ticker" in filtered.columns:
                filtered = filtered.filter(pl.col("ticker").is_in(selected_tickers))
            st.subheader(heading)
            st.dataframe(filtered.to_pandas(), width="stretch")


if __name__ == "__main__":
    main()
