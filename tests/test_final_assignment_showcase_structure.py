import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "final_assignment_showcase.ipynb"


def test_final_assignment_showcase_has_required_contract():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert "LOAD_CANONICAL_EVIDENCE = True" in source
    assert "RUN_FRESH_DEMO = True" in source
    assert "RUN_FULL_EXPANDED_STUDY = False" in source
    assert "load_expanded_report" in source
    assert "trading_pipeline.run" in source
    assert "configs/smoke.yaml" in source
    assert "[STUDENT RESPONSE REQUIRED]" in source
    assert "D:\\" not in source
    assert "C:\\" not in source

    headings = [
        line.strip()
        for line in source.splitlines()
        if line.startswith("## ")
    ]
    assert any(line.startswith("## 1. ") for line in headings)
    assert any(line.startswith("## 6. ") for line in headings)
    assert any(line.startswith("## 10. ") for line in headings)


def test_final_assignment_showcase_assets_exist():
    assets = ROOT / "notebooks" / "assets"
    expected = {
        "native_etl_information_controls.png",
        "native_feature_label_execution_timeline.png",
        "native_shared_research_architecture.png",
        "native_lstm_supervised_interface.png",
        "native_transformer_supervised_interface.png",
        "native_dqn_policy_interface.png",
        "native_ppo_policy_interface.png",
        "native_nested_hpo_holdout.png",
    }
    assert expected <= {path.name for path in assets.iterdir()}