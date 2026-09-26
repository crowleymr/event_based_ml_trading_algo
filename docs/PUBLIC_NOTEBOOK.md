# Public-notebook preparation

The [final evidence notebook](../notebooks/final_evidence.ipynb) is prepared for a
hosted-notebook workflow, but has not been published and is not claimed to be publicly
accessible. No public URL or licensing decision has been recorded; both remain pending
human review.

1. Generate a versioned report from the intended immutable run.
2. Copy or mount that report directory in the notebook environment, subject to data
   and licensing review.
3. Set `TRADING_REPORT_DIR` to the generated report directory containing
   `provenance.json` (for example, a versioned directory under `reports/`). The loader
   checks report schema and declared output hashes before showing tables. Optional
   expanded-study tables are shown only when present; they are not reconstructed by
   the notebook.
4. Install the declared Python dependencies and run all cells.
5. Confirm the displayed source run ID and input hashes before sharing any output.

The notebook consumes generated Parquet only. It does not need raw market/SEC data,
model files or training code, and it does not recompute published metrics. From the
repository root, the optional local read-only dashboard can be started with
`.venv/Scripts/python -m trading_pipeline.dashboard --report "$env:TRADING_REPORT_DIR"`;
the notebook also shows inline evidence when Streamlit is unavailable. Before external
publication, review Yahoo/SEC terms, repository privacy and assignment rules. Student
reflection prompts in the notebook require the student's own evidence and wording.
