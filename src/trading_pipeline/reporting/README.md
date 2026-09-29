# Reporting package

This package is a read-only presentation adapter for one specified immutable run. Run:

```powershell
python -m trading_pipeline.reporting --run runs/<run_id> --rl-run runs/rl/<run_id> --output reports/<run_id>/v2
```

It validates the required contract, joins stable semantic labels, derives comparison,
universe, security holdings/trades/contribution summaries and equity/drawdown,
turnover/cost and IC series, split/feature/target profiles, prediction and decile
diagnostics, semantic feature importance, SPY-relative evidence, training telemetry or
explicit not-recorded states, field definitions/units/limitations/provenance and an
explicit deferred-field register, then writes CSV, Parquet, Markdown and provenance JSON.

The feature snapshot is external to the immutable Slice 1 run directory, so v2 requires
the path and SHA-256 frozen in `dataset_manifest.json`. An optional RL input must be
complete and audited; its input files are hashed into report provenance.

The output directory must not already exist. Corrections are made in source code/data
and regenerated into a new version, never patched in place. The module does not fit,
select, audit-with-writes or modify its source run. Industry, historical market cap,
P/E and other unavailable PIT fields are explicitly deferred.

## Multi-run catalogue

The catalogue command indexes only completed, audited runs and writes a new versioned
output with source hashes and a compatibility key:

```powershell
.venv/Scripts/python -m trading_pipeline.reporting.catalog_cli `
  --runs runs `
  --runs runs/rl `
  --output reports/catalog/v1
```

`run_catalog.parquet` contains run/config/protocol identity; `metric_catalog.parquet`
contains persisted experiment-comparison rows where available. Different compatibility
keys may be inspected together but must not be pooled or ranked as a fair comparison.
The catalogue never trains, audits with writes, or changes source runs.

## Assignment evidence export

After the expanded closeout report has been generated, create a new assignment-specific
evidence version with:

```powershell
.venv/Scripts/python -m trading_pipeline.reporting.assignment_export `
  --run-id <expanded-run-id> --version v1
```

The command verifies the completed run audit and every declared WP7 report output, then
writes paired CSV/Parquet tables beneath `reports/assignment/<run-id>/<version>/`.
These tables cover controlled sensitivity, outer-versus-holdout predictive diagnostics,
recorded fit telemetry, the model-input feature dictionary, split-level descriptive
statistics, explicit unavailable evidence and provenance. The destination must be new.
The command does not train, select, or write under `runs/`; the sealed holdout remains
descriptive only.
