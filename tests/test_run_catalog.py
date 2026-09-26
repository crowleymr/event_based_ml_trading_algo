import json

import polars as pl

from trading_pipeline.dashboard import load_catalog
from trading_pipeline.reporting import build_metric_catalog, build_run_catalog, write_run_catalog


def _run(root, name, *, complete=True, audited=True, cost=10):
    value = root / name
    value.mkdir()
    (value / "metadata.json").write_text(json.dumps({
        "run_id": name, "status": "complete" if complete else "failed",
        "timestamp": f"2026-09-20T00:00:0{name[-1]}+00:00",
        "research_status": "software_verification_only", "mode": "synthetic",
        "label_horizon": 5, "cost_bps_one_way": cost,
        "execution": "T+1 close", "experiments": {"E1": ["F0", "elastic_net"]},
        "git_commit": "abc", "git_dirty": False,
    }), encoding="utf-8")
    (value / "audit.json").write_text(json.dumps({
        "passed": audited, "check_count": 17,
    }), encoding="utf-8")
    (value / "config.yaml").write_text("mode: synthetic\n", encoding="utf-8")
    (value / "dataset_manifest.json").write_text(json.dumps({
        "feature_sha256": "feature-hash",
    }), encoding="utf-8")
    (value / "split_manifest.json").write_text(json.dumps({
        "train": ["2020-01-01", "2020-12-31"],
    }), encoding="utf-8")
    (value / "selection.json").write_text(json.dumps({
        "criterion": "inner ranking utility",
    }), encoding="utf-8")
    pl.DataFrame({
        "experiment": ["E1"], "split": ["test"], "sharpe": [0.5],
    }).write_parquet(value / "experiment_comparison.parquet")
    return value


def test_catalog_indexes_only_completed_audited_runs_and_is_read_only(tmp_path):
    good = _run(tmp_path, "run1")
    _run(tmp_path, "run2", complete=False)
    _run(tmp_path, "run3", audited=False)
    before = {path.name: path.read_bytes() for path in good.iterdir()}
    catalog = build_run_catalog([tmp_path])
    assert catalog["run_id"].to_list() == ["run1"]
    assert catalog["compatibility_key"].null_count() == 0
    assert before == {path.name: path.read_bytes() for path in good.iterdir()}


def test_compatibility_key_changes_with_research_contract(tmp_path):
    _run(tmp_path, "run1", cost=10)
    _run(tmp_path, "run2", cost=20)
    catalog = build_run_catalog([tmp_path])
    assert catalog["compatibility_key"].n_unique() == 2


def test_catalog_output_is_versioned_and_hashed(tmp_path):
    _run(tmp_path, "run1")
    catalog = build_run_catalog([tmp_path])
    metrics = build_metric_catalog([tmp_path])
    output = write_run_catalog(catalog, tmp_path / "catalog" / "v1", metrics)
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["row_count"] == 1
    assert provenance["run_catalog_sha256"]
    assert provenance["metric_catalog_sha256"]
    loaded = load_catalog(output)
    assert set(loaded["tables"]) == {"runs", "metrics"}
    assert loaded["tables"]["metrics"]["experiment_id"].to_list() == ["E1"]
    try:
        write_run_catalog(catalog, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("catalogue overwrite should fail")


def test_catalog_loader_rejects_tampering(tmp_path):
    _run(tmp_path, "run1")
    output = write_run_catalog(
        build_run_catalog([tmp_path]), tmp_path / "catalog" / "v1",
        build_metric_catalog([tmp_path]),
    )
    (output / "run_catalog.parquet").write_bytes(b"tampered")
    try:
        load_catalog(output)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("tampered catalogue should fail")
