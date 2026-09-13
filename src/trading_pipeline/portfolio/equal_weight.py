"""Long-only equal weights for a selected cross-section."""


def equal_weights(security_ids: list[str]) -> dict[str, float]:
    if not security_ids:
        return {}
    weight = 1.0 / len(security_ids)
    return {security_id: weight for security_id in security_ids}
