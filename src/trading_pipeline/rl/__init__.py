"""Exploratory strategy-selection RL with optional dependency lazy loading."""

__all__ = ["PilotDataset", "StrategySelectorEnv", "prepare_dataset"]


def __getattr__(name):
    # Keep registry and policy-adapter imports usable in the base installation. The
    # Gymnasium dependency is required only when the legacy environment is requested.
    if name in {"PilotDataset", "prepare_dataset"}:
        from .dataset import PilotDataset, prepare_dataset
        return {"PilotDataset": PilotDataset, "prepare_dataset": prepare_dataset}[name]
    if name == "StrategySelectorEnv":
        from .environment import StrategySelectorEnv
        return StrategySelectorEnv
    raise AttributeError(name)
