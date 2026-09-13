# trading_pipeline package

This package is the layered local Slice 1 research application. `run.py` is the only end-to-end application service; `audit.py` is a read-only completed-run check.

| Area | Owns | Does not own |
|---|---|---|
| `data` | Source access, identifiers, schemas, caches and canonical tables | Features or research selection |
| `features` | PIT-safe predictors | Labels, fitting or portfolio policy |
| `modelling` | Target, split, fit, selection and prediction metrics | Trade execution |
| `portfolio` | Ranking-to-weight rules and accounting | Model fitting |
| `tracking` | Persistence helpers and research plots | Choosing winners |
| `validation` | Read-only integrity checks | Repairing artefacts |

Public operator interfaces are `python -m trading_pipeline.run --config <path>` and `python -m trading_pipeline.audit --run <run-dir>`. Modules exchange explicit Polars tables and JSON-compatible manifests. Configuration validation fixes the five-session horizon, live/synthetic mode and positive cost/top-K values.

Core invariants are strict point-in-time availability, train-only preprocessing, validation-only selection, T+1 execution, deterministic CPU-first operation and immutable run contracts. Presentation code must remain a read-only downstream consumer.
