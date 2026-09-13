# Reporting package

This package is a read-only presentation adapter for one specified immutable run. Run:

```powershell
python -m trading_pipeline.reporting --run runs/<run_id> --output reports/<run_id>/v1
```

It validates the required contract, joins stable semantic labels, derives comparison,
universe, security holdings/trades/contribution summaries and equity/drawdown,
turnover/cost and IC series, metric definitions/units and an explicit deferred-field
register, then writes CSV, Parquet, Markdown and provenance JSON.

The output directory must not already exist. Corrections are made in source code/data
and regenerated into a new version, never patched in place. The module does not fit,
select, audit-with-writes or modify its source run. Industry, historical market cap,
P/E and other unavailable PIT fields are explicitly deferred.
