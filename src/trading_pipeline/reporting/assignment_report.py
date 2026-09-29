"""Build a versioned assignment report from hash-verified generated evidence.

This module is deliberately a read-only consumer of the completed run.  It verifies
the WP7 and assignment-export inventories, then writes prose, tables and figures to a
new report directory.  It performs no fitting, tuning, ranking or selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import polars as pl


ARMS = {
    "EN_F1_STACK": ("Elastic Net", "supervised.elastic_net.v1", "SupervisedModel"),
    "HGBT_F1_STACK": ("Histogram gradient boosting", "supervised.hist_gbt.v1", "SupervisedModel"),
    "XGB_F1_STACK": ("XGBoost", "supervised.xgboost.v1", "SupervisedModel"),
    "LSTM_F1_STACK": ("LSTM", "supervised.lstm.v1", "SupervisedModel"),
    "TRANSFORMER_F1_STACK": ("Causal Transformer", "supervised.causal_transformer.v1", "SupervisedModel"),
    "DQN_SELECTOR": ("DQN selector", "rl_dqn_sb3_v1", "RLPolicy"),
    "PPO_SELECTOR": ("PPO selector", "rl_ppo_categorical_sb3_v1", "RLPolicy"),
}
PUBLIC_NOTEBOOK_PLACEHOLDER = "PUBLIC_NOTEBOOK_URL_NOT_YET_AVAILABLE"
REQUIRED_CITATIONS = {
    "chen_short-term_2024", "demiguel_transaction-cost_2020", "feng_taming_2020",
    "fieberg_machine_2023", "gu_empirical_2020", "hambly_recent_2023",
    "jegadeesh_returns_1993", "jegadeesh_profitability_2001", "jensen_machine_2026",
    "jiang_deep_2017", "lopez_de_prado_advances_2018", "mclean_does_2016",
    "novy-marx_taxonomy_2016",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _verify_inventory(directory: Path, manifest: dict, *, parquet_only: bool) -> dict[str, Path]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"Missing output inventory: {directory}")
    verified: dict[str, Path] = {}
    for item in outputs:
        name = item.get("relative_path")
        if not isinstance(name, str) or name in verified or Path(name).name != name:
            raise ValueError(f"Invalid or duplicate output declaration: {name!r}")
        path = directory / name
        if not path.is_file() or _hash(path) != item.get("sha256"):
            raise ValueError(f"Output hash mismatch: {path}")
        if path.suffix == ".parquet":
            rows = pl.scan_parquet(path).select(pl.len()).collect().item()
            if rows != item.get("rows"):
                raise ValueError(f"Output row-count mismatch: {path}")
        elif parquet_only:
            raise ValueError(f"Unexpected non-Parquet output: {path}")
        verified[name] = path
    return verified


def _bib_keys(text: str) -> set[str]:
    return set(re.findall(r"(?m)^@\w+\{([^,]+),", text))


def _word_count(markdown: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", markdown))


def _table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(cell(value) for value in row) + " |" for row in rows),
    ])


def _pct(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.2f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _figures(destination: Path, risk: pl.DataFrame, predictive: pl.DataFrame,
             equity: pl.DataFrame) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = destination / "figures"
    figures.mkdir(exist_ok=True)
    colors = dict(zip(ARMS, plt.get_cmap("tab10").colors[:7], strict=True))
    metadata = {"Software": "trading_pipeline.reporting.assignment_report"}

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {"conservative": "o", "balanced": "s", "aggressive": "^"}
    for row in risk.sort("model_id", "risk_order").to_dicts():
        ax.scatter(100 * row["annualised_volatility"], 100 * row["annualised_return"],
                   color=colors[row["model_id"]], marker=markers[row["risk_scenario"]], s=55)
        ax.annotate(row["model_id"].replace("_F1_STACK", ""),
                    (100 * row["annualised_volatility"], 100 * row["annualised_return"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="grey", linewidth=.8)
    ax.set(xlabel="Annualised volatility (%)", ylabel="Annualised return (%)",
           title="Descriptive holdout risk-return outcomes (after costs)")
    ax.grid(alpha=.25)
    fig.tight_layout()
    p1 = figures / "descriptive_risk_return.png"
    fig.savefig(p1, dpi=180, metadata=metadata)
    plt.close(fig)

    holdout = predictive.filter(pl.col("partition") == "descriptive_holdout")
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in holdout.to_dicts():
        ic = row["recomputed_finite_mean_daily_spearman_ic"]
        if ic is not None:
            ax.scatter(ic, row["rmse"], color=colors[row["arm_id"]], s=65)
            ax.annotate(row["arm_id"].replace("_F1_STACK", ""), (ic, row["rmse"]),
                        xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Mean daily Spearman IC", ylabel="RMSE",
           title="Supervised descriptive-holdout diagnostics")
    ax.grid(alpha=.25)
    fig.tight_layout()
    p2 = figures / "predictive_diagnostics.png"
    fig.savefig(p2, dpi=180, metadata=metadata)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    selected = equity.filter(
        (pl.col("risk_scenario") == "balanced") | (pl.col("model_id") == "B0_SPY")
    )
    for (model,), group in selected.partition_by("model_id", as_dict=True).items():
        label = "SPY" if model == "B0_SPY" else model.replace("_F1_STACK", "")
        ax.plot(group["session_date"], group["equity"], label=label,
                color="black" if model == "SPY" else colors.get(model), linewidth=1.5)
    ax.set(xlabel="Holdout session", ylabel="Equity (initial = 1)",
           title="Balanced-scenario descriptive equity paths and SPY")
    ax.grid(alpha=.25)
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    p3 = figures / "descriptive_equity_paths.png"
    fig.savefig(p3, dpi=180, metadata=metadata)
    plt.close(fig)
    return [p1, p2, p3]


def _report_text(*, run_id: str, wp7: dict, stages: pl.DataFrame, hpo: pl.DataFrame,
                 risk: pl.DataFrame, predictive: pl.DataFrame, availability: pl.DataFrame,
                 features: pl.DataFrame) -> str:
    protocol = wp7["protocol_summary"]
    comparison = wp7["comparison_contract"]
    stage = {r["stage"]: r for r in stages.to_dicts()}
    sample = risk.row(0, named=True)
    supervised = predictive.filter(pl.col("partition") == "descriptive_holdout")
    lowest_mae = supervised.sort("mae").row(0, named=True)
    finite_ic = supervised.filter(
        pl.col("recomputed_finite_mean_daily_spearman_ic").is_not_null()
    ).sort("recomputed_finite_mean_daily_spearman_ic", descending=True)
    highest_ic = finite_ic.row(0, named=True)
    scenario_maxima = {
        scenario: risk.filter(pl.col("risk_scenario") == scenario)
        .sort("total_return", descending=True).row(0, named=True)
        for scenario in ("conservative", "balanced", "aggressive")
    }
    arm_rows = [[arm, *ARMS[arm]] for arm in ARMS]
    hpo_rows = []
    for arm in ARMS:
        part = hpo.filter(pl.col("model_id") == arm)
        hpo_rows.append([arm, part.height, part.filter(pl.col("status") == "complete").height,
                         part.filter(pl.col("selected_global") == True).height,  # noqa: E712
                         "mean IC" if ARMS[arm][2] == "SupervisedModel" else "mean certainty equivalent"])
    pred_rows = [[r["arm_id"], _num(r["mae"], 6), _num(r["rmse"], 6),
                  _num(r["recomputed_finite_mean_daily_spearman_ic"], 5),
                  r["finite_ic_dates"], r["ic_availability_status"]]
                 for r in supervised.to_dicts()]
    result_rows = [[r["model_id"], r["risk_scenario"], _num(r["risk_control_value"], 3),
                    _pct(r["total_return"]), _pct(r["annualised_return"]),
                    _pct(r["annualised_volatility"]), _num(r["sharpe"], 3),
                    _pct(r["maximum_drawdown"]), _pct(r["average_turnover"]),
                    _pct(r["cumulative_transaction_cost"]),
                    _pct(r["benchmark_relative_total_return"])]
                   for r in risk.sort("model_id", "risk_order").to_dicts()]
    feature_groups = features.group_by("feature_group").len().sort("feature_group")
    feature_rows = [[r["feature_group"], r["len"]] for r in feature_groups.to_dicts()]
    availability_rows = [[r["item_id"], r["status"], r["reason"] or "—"]
                         for r in availability.to_dicts()]
    return f"""# Event-based machine-learning trading system: factual project-report draft

**Evidence status:** exploratory closeout; final holdout is descriptive only.  
**Canonical run:** `{run_id}`  
**Public notebook URL:** `{PUBLIC_NOTEBOOK_PLACEHOLDER}`

> This is a generated factual draft, not the final submitted document. Numeric results and tables below are rendered from hash-verified artefacts. The public-notebook field deliberately fails closed until the student publishes and verifies the required URL.

## Abstract

This project implements and evaluates an event-based, cross-sectional machine-learning pipeline for US equities. It combines point-in-time market and fundamental inputs, five supervised return-prediction arms, two reinforcement-learning portfolio selectors, risk-conditioned portfolio construction, next-session execution and transaction-cost accounting. The study uses nested chronological evaluation and a sealed final window. The completed holdout contains {sample['sessions']} weekly portfolio observations from {sample['start_date']} to {sample['end_date']}; it is reported solely to describe the locked system and must not be used to choose an arm, risk setting or parameter. The evidence supports an engineering and exploratory comparison, not a confirmatory claim of persistent profitability.

## 1. Problem definition and task interface

The implemented task maps information available at a decision date to a five-session forward-return signal or, for reinforcement learning, to a portfolio action. Supervised inputs consist of point-in-time engineered market variables, eligible filed fundamentals, causal upstream predictions and explicit availability masks. Their output is a per-security predicted forward return. RL policies instead consume the approved market/portfolio state and output allocation actions; consequently IC and RMSE are not meaningful RL diagnostics.

Predictions feed a common portfolio and execution layer. An action formed at time T is filled at the T+1 close, with {comparison['cost_bps_one_way']} basis points of one-way costs. Weekly portfolio evidence is annualised using {comparison['annualisation_periods_per_year']} periods per year and compared with {comparison['benchmark_symbol']}. This separation prevents model-specific execution assumptions from contaminating the comparison.

## 2. Research context

Cross-sectional machine learning can capture nonlinear interactions in large predictor sets, but much of the prominent evidence is monthly and therefore is not directly comparable with this project's five-session target [@gu_empirical_2020; @fieberg_machine_2023]. Factor proliferation and post-publication decay also make flexible searches vulnerable to false discovery [@feng_taming_2020; @mclean_does_2016]. Classic momentum evidence operates at multi-month horizons [@jegadeesh_returns_1993; @jegadeesh_profitability_2001], while short-horizon reversal and momentum studies remain design-specific [@chen_short-term_2024]. These sources motivate the signal families but do not validate this implementation's horizon.

Portfolio objectives must incorporate turnover and implementation costs because a statistically useful signal can be economically weak after trading [@demiguel_transaction-cost_2020; @novy-marx_taxonomy_2016; @jensen_machine_2026]. RL offers a direct sequential-decision formulation, but published results depend strongly on reward design, simulator fidelity, non-stationarity and sample efficiency [@hambly_recent_2023]. Cryptocurrency RL frameworks are methodological context rather than matched US-equity evidence [@jiang_deep_2017].

## 3. Data, point-in-time controls and feature engineering

The requested universe contained {stage['requested']['count']} securities; {stage['mapped']['count']} mapped to the admitted identifiers. The descriptive-holdout pipeline produced {stage['priced']['count']} priced security-sessions, {stage['fundamental_covered']['count']} with fundamental coverage, {stage['feature_ready']['count']} feature-ready rows, {stage['predicted']['count']} predicted rows and {stage['held']['count']} held security-sessions. Counts use the units declared by the evidence table and should not be interpreted as distinct securities unless the unit says so.

{_table(['Feature group', 'Count'], feature_rows)}

All fundamentals are joined as-of their filing availability rather than silently backfilled from later records. Upstream model predictions are causal out-of-fold inputs, with train-only imputation and availability masks. Chronological boundaries use purging and embargo where overlapping labels could leak information [@lopez_de_prado_advances_2018]. The pipeline fails rather than substituting missing identifiers, prices or estimators.

The count sequence is not a conventional nested funnel. Some stages use distinct securities and others use security-sessions; the labelled and predicted keys also reflect the target/prediction contract rather than complete cases across every exported feature. The feature-ready rule checks its specified market and fundamental fields, while the full dictionary additionally contains causal upstream predictions and availability indicators. For this reason, the stage table is evidence about processing coverage, not proof that every observation is complete in every field.

The 19-field feature contract separates observable market history, point-in-time fundamentals, causal upstream signals and missingness information. Market returns, realised volatility, liquidity and price-position features provide short-horizon state. Earnings-per-share and net-income facts add sparse accounting context. Upstream predictions allow the downstream learners to use signals created only from eligible prior folds. Availability masks prevent an imputed value from being silently treated as a genuine observation. These design choices reduce avoidable leakage, but they do not remove vendor, universe, mapping or survivorship limitations.

## 4. System and model methodology

{_table(['Arm ID', 'Model', 'Implementation component', 'Interface'], arm_rows)}

The supervised arms span a regularised linear baseline, two tree-based nonlinear learners and two sequence models. Elastic Net provides a low-variance reference; histogram gradient boosting and XGBoost represent nonlinear tree ensembles; LSTM and causal Transformer arms represent ordered sequence learners. DQN and categorical PPO represent value-based and policy-gradient allocation policies. Every arm uses the common downstream portfolio test bench, risk scenarios and cost model.

### 4.1 Theory-to-code rationale

Elastic Net combines L1 and L2 regularisation. In this system it tests whether a sparse, approximately linear mapping is adequate once the causal stacked features are supplied. Its strengths are transparency, deterministic CPU execution and shrinkage under correlated inputs. Its limitation is that nonlinear interactions and state-dependent effects must already be represented in the features.

Histogram gradient boosting and XGBoost build additive ensembles of decision trees. They can express thresholds and interactions without requiring a deep network, which is useful for heterogeneous market and fundamental variables. The histogram implementation offers a strong CPU-oriented comparison; XGBoost provides a separately implemented boosted-tree family. Neither name alone guarantees generalisation: depth, learning rate, regularisation, temporal validation and the small HPO budget determine what was actually tested.

The LSTM maintains gated recurrent state and is intended to represent ordered dependencies while reducing the vanishing-gradient problem of a simple recurrent network. The causal Transformer instead applies masked attention so an output cannot attend to future inputs. Attention can connect distant positions directly, but adds capacity and optimisation sensitivity. Both sequence arms are bounded exploratory comparisons here; the short search and single seed are insufficient for architecture-level conclusions.

DQN estimates action values and selects discrete actions from those estimates. PPO updates a stochastic policy under a clipped objective intended to limit overly large policy changes. In this project they are portfolio selectors, not return regressors. Their search objective is therefore certainty equivalent rather than IC, and their evidence must be interpreted through policy and portfolio outputs. Simulator fidelity, reward specification and market non-stationarity remain central threats [@hambly_recent_2023].

### 4.2 Shared portfolio and execution boundary

All arms enter a shared downstream contract so that a favourable result cannot be attributed to a private cost or calendar convention. Supervised signals condition risk-targeted portfolios; RL arms express scenario-dependent risk through their approved policy control. Those numeric controls are not semantically interchangeable: supervised controls are volatility targets, whereas RL controls represent inverse risk aversion. Scenario labels support within-arm comparisons, but equal labels do not imply equal ex-ante risk across interfaces.

The architecture path is: immutable data vintage → point-in-time features → chronological windows → architecture screen → inner-fold HPO → outer-fold evaluation and locked family configuration → sealed descriptive-holdout refit → shared risk-conditioned portfolio/evaluation layer. The final stage is a reporting stage, not an additional selection stage.

## 5. Validation and HPO chronology

The approved design contains {protocol['inner_folds']} inner folds, {protocol['outer_folds']} outer folds and the predeclared seed {protocol['seeds'][0]}. Each model family received {protocol['budgets']['classical_per_family']} or {protocol['budgets']['deep_supervised_per_family']} supervised proposals per evaluation context, while each RL family received {protocol['budgets']['rl_per_family_per_risk_scenario']} proposals per risk scenario. The small budget is a deadline-bounded screening design rather than an exhaustive search.

{_table(['Arm', 'HPO rows', 'Complete', 'Locked rows', 'Inner objective'], hpo_rows)}

Supervised HPO uses mean cross-sectional IC, with RMSE retained as a complementary calibration error. RL HPO uses mean certainty equivalent because its output is a policy rather than a return forecast. The architecture and parameter chronology is important: only pre-holdout evidence establishes configurations. Controlled sensitivity analysis is explicitly selection-ineligible, and the final window cannot revise any lock.

The HPO evidence contains proposal order and fold identity, but not a uniform wall-clock timeline. A defensible chronology is therefore logical rather than elapsed-time based. Controlled-sensitivity cells are local reduced-fidelity contrasts. Inner-fold candidates establish family settings. Outer folds estimate pre-holdout behaviour. The resulting family locks are then refit for the final descriptive window. A `selected_global` row means that a proposal was locked within the relevant supervised family or RL family/scenario context; it is not a flag for a final-holdout winner.

This design separates three questions. Architecture screening asks whether a bounded implementation can be trained under the approved resources. HPO asks which declared proposal should be locked using eligible inner evidence. Outer evaluation asks how locked procedures behave on later pre-holdout periods. The final test bench asks only what the already locked procedures did in the sealed window. Merging those questions after seeing the final outcomes would invalidate the study boundary.

## 6. Evaluation metrics and objective mismatch

Prediction quality is measured using MAE, RMSE and mean daily Spearman information coefficient. Economic behaviour is measured after costs using total and annualised return, annualised volatility, Sharpe ratio, maximum drawdown, turnover, transaction cost and SPY-relative return. A model may improve a statistical metric without improving realised portfolio outcomes because ranking, sizing, constraints, turnover and costs intervene between forecasts and returns. Conversely, an economically better path over a short window is not proof of a superior predictive model.

This objective mismatch is handled by reporting both levels without collapsing them into a post-hoc winner. The study's holdout cannot answer which model should be deployed; that decision would require a newly declared protocol and untouched data vintage.

The observed holdout illustrates the mismatch. `{lowest_mae['arm_id']}` has the smallest MAE ({lowest_mae['mae']:.6f}), whereas `{highest_ic['arm_id']}` has the largest finite mean daily IC ({highest_ic['recomputed_finite_mean_daily_spearman_ic']:.6f}). The largest observed total return in the conservative, balanced and aggressive scenario tables belongs respectively to `{scenario_maxima['conservative']['model_id']}`, `{scenario_maxima['balanced']['model_id']}` and `{scenario_maxima['aggressive']['model_id']}` ({_pct(scenario_maxima['conservative']['total_return'])}, {_pct(scenario_maxima['balanced']['total_return'])} and {_pct(scenario_maxima['aggressive']['total_return'])}). These are generated descriptions of different metrics, not a basis for selecting any model after exposure to the final window.

## 7. Descriptive final-holdout results

### 7.1 Supervised diagnostics

{_table(['Arm', 'MAE', 'RMSE', 'Mean daily IC', 'Finite IC dates', 'IC status'], pred_rows)}

Elastic Net has no finite daily cross-sectional IC in this output, so the recorded zero is not treated as an observed finite IC. The remaining supervised arms have finite daily IC observations as shown. These values are descriptive and are not used to change the locked configuration.

![Supervised descriptive diagnostics](figures/predictive_diagnostics.png)

### 7.2 After-cost portfolio evidence

{_table(['Arm', 'Risk', 'Control', 'Total return', 'Ann. return', 'Ann. vol.', 'Sharpe', 'Max drawdown', 'Avg turnover', 'Costs', 'SPY-relative'], result_rows)}

![Descriptive risk-return outcomes](figures/descriptive_risk_return.png)

![Balanced-scenario equity paths](figures/descriptive_equity_paths.png)

The table reports all seven arms and all three risk scenarios rather than selecting a favourable cell. Outcomes cover only {sample['sessions']} weekly observations, so annualised values and Sharpe ratios are especially uncertain. Transaction costs are included, but that does not capture every source of live slippage, borrow constraint, capacity limit or market impact.

The risk-return chart should be read as a map of realised cells rather than an efficient frontier. Its x-axis is realised annualised volatility over the short holdout and its y-axis is realised annualised return; both inherit sampling error. The equity-path figure uses the predeclared balanced scenario for visual comparability and adds SPY as a benchmark reference. The full table remains authoritative because it retains every scenario, costs, turnover, drawdown and benchmark-relative outcome.

## 8. Interpretation and limitations

The results demonstrate that the end-to-end machinery runs under the locked comparison contract and that statistical forecasts, policy actions, portfolio construction and execution accounting can be reconciled. They do not establish a persistent anomaly, causal effect or deployable edge. The evidence is an exploratory closeout with one data vintage and one seed.

Interpretation should therefore focus on mechanisms and boundaries. Variation across risk scenarios shows how the same modelling arm interacts with its approved exposure control, while variation across arms reflects the whole chain from forecasts or policy actions through portfolio construction. Neither comparison isolates a single causal ingredient. A stronger return in one cell may coexist with higher turnover, drawdown or sampling uncertainty, and a favourable benchmark-relative observation may reverse outside this short period. The appropriate use of the table is transparent diagnosis and hypothesis formation for a future pre-registered vintage.

{_table(['Evidence item', 'Status', 'Reason'], availability_rows)}

The single predeclared seed prevents seed-dispersion estimates. The short holdout makes tail risk, regime robustness and annualisation unstable. The admitted universe, data vendor coverage and point-in-time filters can induce coverage effects. Corporate actions, survivorship controls, delistings and fundamental restatements are only as strong as the pinned data contracts. Costs are a simplified fixed rule rather than a calibrated market-impact model. DQN and PPO have policy diagnostics rather than supervised predictive metrics, which limits metric symmetry. Finally, literature at monthly, international, cryptocurrency or simulated horizons is contextual rather than directly validating this five-session US-equity task.

There are also comparison limits inside the evidence. All arms use the same feature set, so the study cannot attribute differences to feature ablation. The portfolio frontier artefacts apply only to supervised arms, and solver failures must not be recoded as valid zero-weight outcomes. Telemetry fields differ by backend, so they do not form a common training-loss curve. Sensitivity deltas are matched local contrasts, not global causal effects. The SPY-relative total-return column is an arithmetic difference, while pathwise relative equity uses a ratio; those measures answer related but non-identical questions.

## 9. Reproducibility, deployment boundary and future work

The authoritative execution route remains `python -m trading_pipeline.run`. Reports and notebooks are read-only consumers of completed artefacts. Each generated input and output is hash inventoried; the report manifest records the source run, protocol, code revision, citation assets, generated timestamp and report word count. A future confirmatory study should declare a new holdout/data vintage, increase seed coverage, pre-register economic objectives, expand cost and liquidity stress tests, and test regime stability without consulting the completed final window for design choices.

Public notebook: `{PUBLIC_NOTEBOOK_PLACEHOLDER}`. This placeholder must be replaced only after the public notebook and release assets have been created and independently opened. Until then, publication is incomplete.

## 10. Conclusion

The project delivers a local-first, event-based research system with point-in-time controls, seven heterogeneous modelling arms, nested chronological evaluation, risk-conditioned portfolios, T+1 execution and after-cost descriptive evidence. Its most defensible conclusion is procedural: the locked experiment was completed and its evidence can be reproduced and audited. Any claim about financial superiority, deployment or generalisation remains outside the evidential scope of the completed holdout.

## STUDENT PROMPT — Personal reflection (student-authored)

**Student action required:** add a first-person account of what was learned, which design decisions were personally understood, what changed during the project, and what evidence supports that account. Do not convert generated implementation logs into an invented personal narrative.

## STUDENT PROMPT — Personal financial implications (student-authored)

**Student action required:** explain in the student's own words how the results affect their view of investment risk, transaction costs, model uncertainty and whether the system should influence personal financial decisions. Distinguish education/research from financial advice.

## STUDENT PROMPT — Knowledge gaps and verification (student-authored)

**Student action required:** identify remaining areas the student cannot yet explain or independently verify, and state a concrete plan for closing those gaps before submission or deployment.

## References

Citation keys are resolved by the copied, hash-verified `references.bib`. The register linking each entry to an exact Zotero item key is an input to this build. The malformed quarantined Zotero record is excluded. No PDF attachments are included.

## Provenance

Source run `{run_id}`; WP7 generated `{wp7['generated_at_utc']}`; source revision `{wp7['source_git_commit']}`; protocol SHA-256 `{wp7['protocol_sha256']}`; claim status `{wp7['claim_status']}`. Exact input and output hashes are recorded in `report_manifest.json`.
"""


def generate_assignment_report(*, repository_root: str | Path, run_id: str,
                               version: str = "v1", destination: str | Path | None = None,
                               refresh: bool = False) -> Path:
    """Generate the factual Markdown report into a new directory."""
    root = Path(repository_root).resolve()
    if Path(run_id).name != run_id or not run_id:
        raise ValueError("run_id must be a single directory name")
    assignment = root / "reports" / "assignment" / run_id / version
    wp7_dir = root / "reports" / "expanded_closeout" / run_id
    wp7_path = wp7_dir / "research_evidence_manifest.json"
    assignment_path = assignment / "assignment_evidence_manifest.json"
    citations = root / "docs" / "assignment" / "CITATION_REGISTER.md"
    bib = root / "docs" / "assignment" / "references.bib"
    for path in (wp7_path, assignment_path, citations, bib):
        if not path.is_file():
            raise FileNotFoundError(path)
    wp7, assignment_manifest = _json(wp7_path), _json(assignment_path)
    if (wp7.get("source_run_id") != run_id
            or wp7.get("claim_status") != "exploratory_closeout_not_confirmatory"):
        raise ValueError("WP7 manifest is not the required exploratory closeout")
    if (assignment_manifest.get("source_run_id") != run_id
            or assignment_manifest.get("source_wp7_manifest_sha256") != _hash(wp7_path)
            or assignment_manifest.get("claim_status") != wp7["claim_status"]
            or assignment_manifest.get("holdout_role") != "descriptive_only"):
        raise ValueError("Assignment evidence does not match WP7 or is not descriptive-only")
    wp7_files = _verify_inventory(wp7_dir, wp7, parquet_only=True)
    assignment_files = _verify_inventory(assignment, assignment_manifest, parquet_only=False)
    bib_text = bib.read_text(encoding="utf-8")
    if _bib_keys(bib_text) != REQUIRED_CITATIONS or "KJU4AWFC" in bib_text or "[cite:" in bib_text:
        raise ValueError("BibTeX is not the verified bounded citation set")
    destination_path = Path(destination).resolve() if destination else assignment / "report"
    if not destination_path.is_relative_to(assignment):
        raise ValueError(f"Report destination must be within the assignment export: {destination_path}")
    if destination_path.exists() and not refresh:
        raise FileExistsError(f"Report destination must be new and within the assignment export: {destination_path}")
    if destination_path.exists():
        prior_manifest_path = destination_path / "report_manifest.json"
        prior = _json(prior_manifest_path) if prior_manifest_path.is_file() else {}
        if prior.get("source_run_id") != run_id:
            raise ValueError("Refusing to refresh a report from a different or unverified source run")
    else:
        destination_path.mkdir(parents=True)

    stages = pl.read_parquet(wp7_files["pipeline_stage_summary.parquet"])
    hpo = pl.read_parquet(wp7_files["hpo_trial_summary.parquet"])
    risk = pl.read_parquet(wp7_files["realised_risk_return_curve.parquet"])
    equity = pl.read_parquet(wp7_files["final_testbench_equity_curve.parquet"])
    predictive = pl.read_parquet(assignment_files["predictive_diagnostics.parquet"])
    availability = pl.read_parquet(assignment_files["evidence_availability.parquet"])
    features = pl.read_parquet(assignment_files["feature_dictionary.parquet"])
    if set(risk["model_id"]) != set(ARMS) or risk.height != 3 * len(ARMS):
        raise ValueError("Risk-return evidence does not contain all seven arms and three scenarios")

    figures = _figures(destination_path, risk, predictive, equity)
    report = _report_text(run_id=run_id, wp7=wp7, stages=stages, hpo=hpo, risk=risk,
                          predictive=predictive, availability=availability, features=features)
    report_path = destination_path / "PROJECT_REPORT_DRAFT.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    shutil.copyfile(bib, destination_path / "references.bib")
    generated_at = datetime.now(timezone.utc).isoformat()
    source_paths = [wp7_path, assignment_path, citations, bib,
                    *[wp7_files[n] for n in ("pipeline_stage_summary.parquet",
                                             "hpo_trial_summary.parquet",
                                             "realised_risk_return_curve.parquet",
                                             "final_testbench_equity_curve.parquet")],
                    *[assignment_files[n] for n in ("predictive_diagnostics.parquet",
                                                    "evidence_availability.parquet",
                                                    "feature_dictionary.parquet")]]
    outputs = [report_path, destination_path / "references.bib", *figures]
    manifest = {
        "schema_version": 1, "source_run_id": run_id,
        "claim_status": wp7["claim_status"], "holdout_role": "descriptive_only",
        "generated_at_utc": generated_at, "word_count": _word_count(report),
        "public_notebook_url": PUBLIC_NOTEBOOK_PLACEHOLDER,
        "generator_module_sha256": _hash(Path(__file__)),
        "inputs": [{"relative_path": p.relative_to(root).as_posix(), "sha256": _hash(p)}
                   for p in source_paths],
        "outputs": [{"relative_path": p.relative_to(destination_path).as_posix(),
                     "sha256": _hash(p), "bytes": p.stat().st_size} for p in outputs],
        "limitations": ["single predeclared seed", "short descriptive holdout",
                        "public notebook URL not yet available", "student reflection required"],
    }
    (destination_path / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--refresh", action="store_true",
                        help="Refresh only an existing report whose manifest matches the run")
    args = parser.parse_args()
    path = generate_assignment_report(repository_root=args.repository_root,
                                      run_id=args.run_id, version=args.version,
                                      refresh=args.refresh)
    manifest = _json(path / "report_manifest.json")
    print(f"{path} ({manifest['word_count']} words)")


if __name__ == "__main__":
    main()
