"""Frozen Phase 2 protocol and stable policy identifiers."""

from __future__ import annotations

from pathlib import Path
import yaml

ACTION_IDS = {0: "CASH", 1: "E0", 2: "E1", 3: "E2", 4: "E3", 5: "E4", 6: "E5"}
POLICY_IDS = {
    "RL0_RANDOM_SELECTOR": "Seeded uniform random strategy selector",
    "RL1_DQN_SELECTOR": "Fixed-budget CPU DQN strategy selector",
    **{f"RL_B{i}_ALWAYS_E{i}": f"Always select frozen Slice 1 sleeve E{i}" for i in range(6)},
}


def load_protocol(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    actions = {int(key): value for key, value in data["actions"].items()}
    if actions != ACTION_IDS:
        raise ValueError(f"Action mapping differs from frozen mapping: {actions}")
    if data["research_status"] != "exploratory_not_confirmatory":
        raise ValueError("Phase 2 pilot must remain explicitly exploratory")
    if data["dqn"]["device"] != "cpu":
        raise ValueError("The frozen pilot is CPU-first")
    return data
