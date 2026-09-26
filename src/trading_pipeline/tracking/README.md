# Tracking package

Owns generic JSON persistence, consumed-input hashing and static research-plot generation. `input_vintage` lists only raw files consumed by a run; `json_write` rejects non-JSON NaN values; `plots` writes equity, drawdown and comparison images.

Inputs are completed in-memory tables, configuration and explicit paths. Outputs belong to a new run directory. The application service decides what to persist and in what sequence.

Tracking must not select models, recompute research policy, scrape prose or mutate a completed run. Paths/hashes, run ID, code revision and timestamps should accompany generated evidence. Reporting belongs in its own read-only downstream package because reports are versioned derivatives, not source-run artefacts.

`telemetry.py` defines the stable nullable Parquet schemas shared by supervised and RL
training. Long-form traces contain only metrics actually exposed or deterministically
derived from the fitted estimator. Summaries record requested/actual device, fallback,
duration, data shape, iterations/epochs, stopping reason, peak GPU memory when available,
package/CUDA versions, seed and determinism details. Older immutable runs remain valid
and presentation layers show explicit not-recorded states.

Supervised XGBoost runs also emit `device_benchmark.parquet`: paired CPU/CUDA status,
actual device, fallback, elapsed time, validation RMSE, selected iteration and maximum
prediction delta. This diagnostic is explicitly non-selection evidence.
