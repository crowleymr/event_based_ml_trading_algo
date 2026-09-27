"""Declared complete, score-blind expanded-study proposal budgets.

Each tuple is (classical, deep supervised, RL per risk scenario) per family.
The ordering is deliberate: calibration chooses the first complete tier that
fits its deadline, without consulting any model outcome.
"""

TIERS = {
    "full": (24, 18, 12),
    "reduced": (16, 12, 8),
    "minimum_defensible": (8, 8, 6),
    "deadline_complete": (2, 2, 2),
}
