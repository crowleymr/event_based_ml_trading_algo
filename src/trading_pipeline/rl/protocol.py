"""Frozen Phase 2 protocol and stable policy identifiers."""

from __future__ import annotations

from pathlib import Path
import yaml

ACTION_IDS = {0: "CASH", 1: "E0", 2: "E1", 3: "E2", 4: "E3", 5: "E4", 6: "E5"}
POLICY_IDS = {
    "RL0_RANDOM_SELECTOR": "Seeded uniform random strategy selector",
    "RL1_DQN_SELECTOR": "Fixed-budget DQN strategy selector",
    **{f"RL_B{i}_ALWAYS_E{i}": f"Always select frozen Slice 1 sleeve E{i}" for i in range(6)},
}


def load_protocol(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    actions = {int(key): value for key, value in data["actions"].items()}
    if actions != ACTION_IDS:
        raise ValueError(f"Action mapping differs from frozen mapping: {actions}")
    if data["research_status"] not in {
        "exploratory_not_confirmatory", "diagnostic_reproduction_not_model_selection"
    }:
        raise ValueError("Phase 2 pilot must remain explicitly non-confirmatory")
    if data.get("protocol_version") not in {1, 2}:
        raise ValueError("Unsupported RL protocol version")
    if data["dqn"].get("device", "cpu") not in {"cpu", "cuda", "auto"}:
        raise ValueError("DQN device must be cpu, cuda or auto")
    return data
