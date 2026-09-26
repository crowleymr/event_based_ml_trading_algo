from pathlib import Path
import yaml


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("horizon", 5) != 5:
        raise ValueError("Slice 1 requires a five-session target")
    if cfg.get("cost_bps", 10) != 10:
        raise ValueError("Headline cost changes require approval")
    if cfg.get("mode") not in {"live", "synthetic"}:
        raise ValueError("mode must be live or synthetic")
    xgboost = cfg.get("xgboost", {})
    if xgboost.get("device", "cpu") not in {"cpu", "cuda", "auto"}:
        raise ValueError("xgboost.device must be cpu, cuda or auto")
    if xgboost.get("enabled") and xgboost.get("early_stopping_rounds", 20) < 1:
        raise ValueError("xgboost.early_stopping_rounds must be positive")
    return cfg
