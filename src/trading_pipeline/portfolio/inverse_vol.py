"""Long-only inverse-volatility weights for a selected cross-section."""

import numpy as np


def inverse_vol_weights(security_ids: list[str], volatilities: list[float]) -> dict[str, float]:
    if not security_ids:
        return {}
    raw = 1 / np.asarray(volatilities, dtype=float)
    normalized = raw / raw.sum()
    return dict(zip(security_ids, normalized))
