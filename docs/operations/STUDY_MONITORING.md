# Expanded study operational monitoring

Launch an approved study through the monitoring wrapper from the repository root:

```powershell
.venv\Scripts\python.exe -m trading_pipeline.operations.study_supervisor --study configs/studies/expanded_closeout_approved_v4.yaml --log-dir reports/operations/expanded-closeout-v4-attempt-1
```

Use a new `--log-dir` for each attempt; the wrapper refuses an existing directory.
The child command is always `python -m trading_pipeline.run --study <protocol>`.
The monitor does not inspect model outputs, scores, manifests or the holdout.

Approved-v4 first runs the selection-ineligible controlled-sensitivity matrix, then
the full-fidelity HPO/outer-selection matrix, and opens the descriptive holdout only
after all family locks are sealed. Score-blind calibration estimates approximately
3.0 hours for sensitivity plus HPO; allow the documented 6-hour window for final fits,
holdout evaluation, reporting reserve and retries. The terminal must remain open, but
the GPU is used only by the declared XGBoost, LSTM and Transformer fits.

To continue a terminally failed run that contains verified cell checkpoints, create a
new operational log directory and pass the immutable parent run:

```powershell
.venv\Scripts\python.exe -m trading_pipeline.operations.study_supervisor --study configs/studies/<approved-protocol>.yaml --resume-from runs/expanded_closeout/<failed-run-id> --log-dir reports/operations/<new-attempt-label>
```

Continuation always creates a new derived research run. It never edits the parent.
The authoritative runner rejects missing or altered checkpoints, protocol/input/code
drift, ledger disagreement, and attempts to resume a completed or pre-checkpoint run.
The failed `20260927T075513Z-3a5052c9` attempt predates this facility and cannot be
resumed; its evidence remains immutable.

The launching terminal displays only concise lifecycle and phase changes plus
warnings, errors and final status. It does not stream the complete child log.
Detailed output and resource samples remain available in the files below.

`events.jsonl` records UTC-timestamped startup, every child output line as an info,
warning or error event, interruption, failure and final exit status. `stdout.log`
and `stderr.log` preserve child output with one operator-local timestamp at
whole-second precision. The supervisor removes child timestamp milliseconds and does
not prepend a second GMT/UTC timestamp to these human-readable logs or concise terminal
messages. `metrics.jsonl` records a UTC-timestamped heartbeat every ten seconds by default,
including elapsed time, process-tree RSS, cumulative CPU seconds and interval CPU
utilization, host RAM, CPU utilization and one/five/fifteen-minute load averages, and
NVIDIA GPU utilization and used/total VRAM. It also sums study process-tree VRAM
when NVIDIA exposes per-process memory. Device totals include other applications.
Windows WDDM may hide process VRAM; `null` means unavailable, not zero. If
`nvidia-smi` is unavailable, the GPU record reports `available: false`.

The wrapper exits with the child status (130 after Ctrl+C). It leaves all logs
in place when the child fails or is interrupted. Operational logs are separate
from immutable research artefacts under `runs/` and cannot establish research
completion; inspect the authoritative run audit and manifests for that.
