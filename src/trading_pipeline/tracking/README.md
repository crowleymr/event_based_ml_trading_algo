# Tracking package

Owns generic JSON persistence, consumed-input hashing and static research-plot generation. `input_vintage` lists only raw files consumed by a run; `json_write` rejects non-JSON NaN values; `plots` writes equity, drawdown and comparison images.

Inputs are completed in-memory tables, configuration and explicit paths. Outputs belong to a new run directory. The application service decides what to persist and in what sequence.

Tracking must not select models, recompute research policy, scrape prose or mutate a completed run. Paths/hashes, run ID, code revision and timestamps should accompany generated evidence. Reporting belongs in its own read-only downstream package because reports are versioned derivatives, not source-run artefacts.
