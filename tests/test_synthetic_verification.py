import hashlib
import json
from pathlib import Path

import pytest

from trading_pipeline.experiments.synthetic_verification import (
    write_synthetic_verification_receipts,
)
from trading_pipeline.experiments.schema import load_study


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs/studies/expanded_closeout_prepared_v2.yaml"


def _junit(path: Path, *, errors: int = 0, failures: int = 0) -> Path:
    path.write_text(
        f'<testsuites><testsuite name="unit" tests="3" errors="{errors}" '
        f'failures="{failures}" /></testsuites>', encoding="utf-8"
    )
    return path


def test_writes_one_provenance_receipt_per_declared_arm(tmp_path):
    junit = _junit(tmp_path / "junit.xml")
    receipts = write_synthetic_verification_receipts(
        junit_xml=junit, prepared_study=STUDY,
        output_dir=tmp_path / "receipts", mapping_path=tmp_path / "map.json",
    )
    arms = {arm["id"]: arm for arm in load_study(STUDY).config["experiment_arms"]}
    assert set(receipts) == set(arms)
    assert json.loads((tmp_path / "map.json").read_text()) == receipts
    for arm_id, receipt_file in receipts.items():
        receipt = json.loads(Path(receipt_file).read_text())
        assert receipt["study_id"] == load_study(STUDY).config["study_id"]
        assert receipt["evidence_kind"] == "synthetic_verification"
        assert receipt["status"] == "passed"
        assert receipt["arm_id"] == arm_id
        assert receipt["component_id"] == arms[arm_id]["component_id"]
        assert receipt["interface"] == arms[arm_id]["interface"]
        assert receipt["junit_xml_sha256"] == hashlib.sha256(junit.read_bytes()).hexdigest()
        assert receipt["junit_totals"] == {"tests": 3, "errors": 0, "failures": 0}
        assert receipt["generated_at_utc"]


@pytest.mark.parametrize(("errors", "failures"), [(1, 0), (0, 1)])
def test_rejects_junit_errors_or_failures(tmp_path, errors, failures):
    with pytest.raises(ValueError, match="zero errors/failures"):
        write_synthetic_verification_receipts(
            junit_xml=_junit(tmp_path / "junit.xml", errors=errors, failures=failures),
            prepared_study=STUDY, output_dir=tmp_path / "receipts",
            mapping_path=tmp_path / "map.json",
        )


def test_receipts_are_immutable(tmp_path):
    kwargs = dict(junit_xml=_junit(tmp_path / "junit.xml"), prepared_study=STUDY,
                  output_dir=tmp_path / "receipts", mapping_path=tmp_path / "map.json")
    write_synthetic_verification_receipts(**kwargs)
    with pytest.raises(FileExistsError):
        write_synthetic_verification_receipts(**kwargs)
