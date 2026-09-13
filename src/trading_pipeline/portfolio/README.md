# Portfolio package

Owns the conversion of frozen scores into holdings and the daily self-financing backtest. Public interfaces are `select_weights`, equal/inverse-volatility weighting helpers, `backtest` and `financial_metrics`.

Input predictions contain session, security, score and trailing 20-session volatility. Market bars provide adjusted-close valuations. P1 chooses the deterministic top K and assigns equal weights. P2 chooses the identical names and normalizes inverse volatility. B0 buys SPY once.

Signals occur on the first observed session of each ISO week. Existing holdings earn the return into the next-session close; target holdings are then traded and begin earning afterward. Costs apply to absolute traded dollars and initial entry. Positions drift between rebalances; terminal positions are marked, not liquidated.

Outputs are daily curves, security-level positions and trades. Weights are long-only and at most fully invested; cash cannot be materially negative; held prices cannot be missing; execution must be exactly T+1. This layer does not fit or select models and must compare portfolio engines using identical frozen predictions when that is the research question.
