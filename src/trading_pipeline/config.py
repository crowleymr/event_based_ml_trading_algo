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
    return cfg
