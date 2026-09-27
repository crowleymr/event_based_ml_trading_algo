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

For an audited expanded close-out run, first generate the separate eleven-table
report with `.venv/Scripts/python -m trading_pipeline.reporting.expanded_closeout`.
The command chooses the newest completed audited run and prints the report path.
Set `TRADING_REPORT_DIR` to that path if you want to pin a particular report; with
the variable unset, the notebook discovers the newest generated report. It detects
the expanded report's
`research_evidence_manifest.json`, verifies all eleven output hashes and schemas,
and displays the descriptive test bench. The optional dashboard command is simply
`.venv/Scripts/python -m trading_pipeline.dashboard`; `--expanded-report` remains
available to pin a run, while `--report` selects a Slice 1 report.

The notebook consumes generated Parquet only. It does not need raw market/SEC data,
model files or training code, and it does not recompute published metrics. From the
repository root, the optional local read-only dashboard can be started with
`.venv/Scripts/python -m trading_pipeline.dashboard`; use `--expanded-report` only
when deliberately pinning a historical expanded report. The notebook also shows inline
evidence when Streamlit is unavailable. Before external publication, review Yahoo/SEC
terms, repository privacy and assignment rules. Student
reflection prompts in the notebook require the student's own evidence and wording.
