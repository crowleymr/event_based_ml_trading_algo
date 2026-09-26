import polars as pl
import pytest

from trading_pipeline.config import load_config
from trading_pipeline.reporting import build_report, generate_report
from trading_pipeline.run import run


@pytest.fixture(scope="module")
def report_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("reporting")
    cfg = load_config("configs/smoke.yaml") | {
        "data_dir": str(root / "data"),
        "runs_dir": str(root / "runs"),
    }
    return run(cfg)


def test_report_derivation_is_deterministic(report_run):
    first_tables, first_markdown, first_provenance = build_report(report_run)
    second_tables, second_markdown, second_provenance = build_report(report_run)
    assert first_tables.keys() == second_tables.keys()
    assert all(first_tables[name].equals(second_tables[name]) for name in first_tables)
    assert first_markdown == second_markdown
    assert first_provenance == second_provenance
    assert {"metric", "definition", "unit"} == set(first_tables["metric_definitions"].columns)
    assert {"table", "field", "definition", "unit", "limitations", "provenance"} == set(
        first_tables["field_definitions"].columns
    )
    assert {"split_profile", "feature_summary", "target_summary", "prediction_diagnostics",
            "prediction_deciles", "feature_importance", "benchmark_relative_series",
            "benchmark_relative_metrics", "training_trace", "training_summary"} <= set(first_tables)
    assert set(first_tables["deferred_fields"]["status"]) == {"blocked/deferred"}


def test_report_schema_labels_and_source_reconciliation(report_run):
    tables, markdown, provenance = build_report(report_run)
    comparison = tables["experiment_comparison"]
    assert set(comparison["experiment_id"]) == {f"E{i}" for i in range(8)}
    assert comparison.select(
        "experiment_id", "display_label", "feature_set_label", "estimator_label",
        "portfolio_label", "selection_role", "parameters_provenance",
    ).null_count().sum_horizontal()[0] == 0
    source = pl.read_parquet(report_run / "experiment_comparison.parquet").rename(
        {"experiment": "experiment_id"}
    ).sort(["split", "experiment_id"])
    numeric = [name for name, dtype in source.schema.items() if dtype.is_numeric()]
    assert comparison.select(numeric).equals(source.select(numeric))
    assert "Final-test results are descriptive" in markdown
    assert provenance["source_run_id"] == report_run.name
    assert provenance["report_schema_version"] == 2
    assert any(item.get("role") == "external_hash_verified_feature_snapshot"
               for item in provenance["inputs"])
    assert len(provenance["inputs"]) >= 13


def test_security_contributions_reconcile_to_source_returns(report_run):
    # build_report raises before returning if daily security attribution does not
    # reconcile to every source curve return.
    tables, _, _ = build_report(report_run)
    contribution = tables["security_contribution_summary"]
    assert {"gross_contribution", "cost_contribution", "net_contribution"} <= set(
        contribution.columns
    )
    assert contribution.filter(
        pl.col("net_contribution").is_null() | ~pl.col("net_contribution").is_finite()
    ).is_empty()


def test_missing_input_fails_and_generation_is_versioned(tmp_path, report_run):
    with pytest.raises(FileNotFoundError, match="missing required inputs"):
        build_report(tmp_path / "not-a-run")
    output = tmp_path / "reports" / report_run.name / "v1"
    source_hashes = {
        name: (report_run / name).read_bytes()
        for name in ("metadata.json", "predictions.parquet", "equity_curve.parquet")
    }
    generated = generate_report(report_run, output)
    assert (generated / "report.md").is_file()
    assert (generated / "provenance.json").is_file()
    assert (generated / "experiment_comparison.csv").is_file()
    assert (generated / "experiment_comparison.parquet").is_file()
    second = generate_report(report_run, output.parent / "v2")
    for path in generated.iterdir():
        if path.name != "provenance.json":
            assert path.read_bytes() == (second / path.name).read_bytes()
    assert source_hashes == {
        name: (report_run / name).read_bytes() for name in source_hashes
    }
    with pytest.raises(FileExistsError, match="already exists"):
        generate_report(report_run, output)
