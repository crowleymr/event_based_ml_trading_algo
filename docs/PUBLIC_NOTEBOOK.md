# Public-notebook preparation

The notebook is prepared for a self-contained hosted-notebook workflow but has not
been published and is not claimed to be publicly accessible.

1. Generate a versioned report from the intended immutable run.
2. Copy or mount that report directory in the notebook environment, subject to data
   and licensing review.
3. Set `TRADING_REPORT_DIR` to the directory containing `provenance.json`.
4. Install the declared Python dependencies and run all cells.
5. Confirm the displayed source run ID and input hashes before sharing any output.

The notebook consumes generated Parquet only. It does not need raw market/SEC data,
model files or training code, and it does not recompute published metrics. Before
external publication, review Yahoo/SEC terms, repository privacy and assignment rules.
