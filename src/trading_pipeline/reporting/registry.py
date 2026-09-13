"""Stable semantic labels layered over immutable machine identifiers."""

BASE = {
    "E0": {
        "display_label": "Momentum · Equal Weight",
        "feature_set_label": "Momentum (20-session return)",
        "estimator_label": "Deterministic momentum rule",
        "portfolio_label": "Equal-weight top 10",
        "selection_role": "Non-ML strategy baseline",
    },
    "E1": {
        "display_label": "Market Elastic Net · Equal Weight",
        "feature_set_label": "F0 Market",
        "estimator_label": "Elastic Net",
        "portfolio_label": "Equal-weight top 10",
        "selection_role": "Validation-tuned candidate",
    },
    "E2": {
        "display_label": "Market Histogram GBT · Equal Weight",
        "feature_set_label": "F0 Market",
        "estimator_label": "Histogram GBT",
        "portfolio_label": "Equal-weight top 10",
        "selection_role": "Validation-tuned candidate",
    },
    "E3": {
        "display_label": "Market + SEC Elastic Net · Equal Weight",
        "feature_set_label": "F1 Market + PIT SEC",
        "estimator_label": "Elastic Net",
        "portfolio_label": "Equal-weight top 10",
        "selection_role": "Validation-tuned candidate",
    },
    "E4": {
        "display_label": "Market + SEC Histogram GBT · Equal Weight",
        "feature_set_label": "F1 Market + PIT SEC",
        "estimator_label": "Histogram GBT",
        "portfolio_label": "Equal-weight top 10",
        "selection_role": "Validation-tuned candidate",
    },
    "E5": {
        "display_label": "Validation Winner · Inverse Volatility",
        "feature_set_label": "Winner feature set",
        "estimator_label": "Frozen validation winner",
        "portfolio_label": "Inverse-volatility top 10",
        "selection_role": "Risk-engine comparison",
    },
    "B0": {
        "display_label": "SPY Buy and Hold",
        "feature_set_label": "None",
        "estimator_label": "Broad-market price baseline",
        "portfolio_label": "Buy once and hold",
        "selection_role": "Descriptive benchmark",
    },
}


def registry(selection: dict) -> dict:
    """Return run-aware labels without changing immutable IDs."""
    result = {key: value.copy() for key, value in BASE.items()}
    source = selection["e5_source"]
    if source not in result or source not in selection["models"]:
        raise ValueError(f"Unsupported E5 source: {source}")
    result["E5"]["feature_set_label"] = result[source]["feature_set_label"]
    result["E5"]["estimator_label"] = f"Frozen {source} predictions"
    return result
