"""Trusted discovery and the exact declared study roster."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from trading_pipeline.experiments import ComponentSpec, SupervisedModel
from trading_pipeline.experiments.default_registry import default_registry
from trading_pipeline.experiments.discovery import register_package
from trading_pipeline.experiments.registry import ComponentRegistry
from trading_pipeline.experiments.schema import load_study
from trading_pipeline.experiments import study_runner


def test_trusted_new_module_can_self_register(monkeypatch):
    package_name = "trading_pipeline.modelling.supervised_models"
    module_name = f"{package_name}.synthetic_test_adapter"
    package = ModuleType(package_name)
    package.__path__ = ["trusted"]
    module = ModuleType(module_name)

    class SyntheticModel(SupervisedModel):
        pass

    SyntheticModel.__module__ = module_name
    module.COMPONENT_REGISTRATIONS = ((ComponentSpec(
        "supervised.synthetic_test.v1", "SupervisedModel", f"{module_name}.SyntheticModel"
    ), SyntheticModel),)
    monkeypatch.setattr("trading_pipeline.experiments.discovery.iter_modules",
                        lambda path: [SimpleNamespace(name="synthetic_test_adapter", ispkg=False)])
    monkeypatch.setattr("trading_pipeline.experiments.discovery.import_module",
                        lambda name: module if name == module_name else None)
    registry = ComponentRegistry()
    register_package(registry, package)
    assert registry.ids("SupervisedModel") == ("supervised.synthetic_test.v1",)
    with pytest.raises(ValueError, match="trusted model package"):
        register_package(ComponentRegistry(), ModuleType("user.supplied.module"))


def test_yaml_arm_roster_drives_validation_and_unknown_import_is_rejected(tmp_path):
    source = yaml.safe_load(open("configs/studies/expanded_closeout_prepared_v3.yaml", encoding="utf-8"))
    first, second = source["experiment_arms"][:2]
    root = tmp_path

    def validated(arms, *, gate_arms=None):
        config = {"study_id": "synthetic", "authority": {}, "experiment_arms": arms}
        gate_arms = arms if gate_arms is None else gate_arms
        gate = {"study_id": "synthetic", "score_blind_calibration": True,
                "outcome_rankings_opened": False,
                "arm_capabilities": {arm["id"]: {
                    "component_id": arm["component_id"], "interface": arm["interface"],
                    "synthetic_verification_passed": True,
                    "real_data_bridge_passed": True,
                } for arm in gate_arms}}
        encoded = json.dumps(gate).encode()
        (root / "gate.json").write_bytes(encoded)
        config["authority"] = {"capability_gate": "gate.json",
                               "capability_gate_sha256": hashlib.sha256(encoded).hexdigest()}
        return study_runner._validate_declared_arms(config, default_registry(), root)

    assert [arm["id"] for arm in validated([first])] == [first["id"]]
    assert [arm["id"] for arm in validated([first, second])] == [first["id"], second["id"]]
    with pytest.raises(PermissionError, match="exact declared arms"):
        validated([first, second], gate_arms=[first])
    unknown = deepcopy(second)
    unknown["component_id"] = "os.system"
    with pytest.raises(ValueError, match="Unknown component id"):
        validated([first, unknown])
    duplicate = deepcopy(second)
    duplicate["component_id"] = first["component_id"]
    with pytest.raises(ValueError, match="must each be unique"):
        validated([first, duplicate])
    config_path = root / "study.yaml"
    source["experiment_arms"] = [first]
    config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    assert len(load_study(config_path).config["experiment_arms"]) == 1
