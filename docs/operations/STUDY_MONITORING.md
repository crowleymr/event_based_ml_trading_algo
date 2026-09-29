# Expanded study operational monitoring

Launch an approved study through the monitoring wrapper from the repository root:

```powershell
.venv\Scripts\python.exe -m trading_pipeline.operations.study_supervisor --study configs/studies/expanded_closeout_approved_v5.yaml --log-dir reports/operations/expanded-closeout-v5-attempt-1
```

Use a new `--log-dir` for each attempt; the wrapper refuses an existing directory.
The child command is always `python -m trading_pipeline.run --study <protocol>`.
The monitor does not inspect model outputs, scores, manifests or the holdout.

Approved-v5 first runs the selection-ineligible controlled-sensitivity matrix, then
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
Controlled-sensitivity candidate start/completion messages include the candidate and
cell counters, arm, fold, risk scenario, trial identifier and elapsed seconds, but no
score or ranking. This makes long-running progress visible without opening outcomes.

For a full-screen local PowerShell dashboard, open a second terminal at the repository
root and run:

```powershell
.\scripts\watch-study.ps1
```

The script selects the most recently updated operational monitoring directory. Pin a
specific attempt when required:

```powershell
.\scripts\watch-study.ps1 -LogDir reports/operations/<attempt-label>
```

`Ctrl+C` closes only this read-only dashboard; it does not signal the supervised study.
Use `-Once` to print one snapshot without refreshing.

The dashboard reports the broad execution phase (`HPO`, `FINAL TEST`, `COMPLETED` or
`FAILED`) separately from the current activity. The `Run status` section incorporates
the latest event as vertical fields, so its timestamp, trial and candidate are shown
once rather than repeated in a second event panel. `System status` contains process
and system CPU, process RSS alongside available host RAM, and GPU/VRAM; the internal
supervisor heartbeat timestamp is intentionally omitted from the operator display.

The progression section reads the current run's immutable protocol and trial ledgers.
Progress is grouped into three compact tables rather than one excessively wide row.
The completed controlled-sensitivity phase remains visible in its own compact table,
including the overall completed/planned total and per-arm counts for the first inner
fold under each outer history. It is explicitly labelled selection-ineligible so its
220 candidate evaluations cannot be confused with the subsequent selection HPO.
`Outer fold 1` and `Outer fold 2` each contain `Inner 1`, `Inner 2`, and `Outer eval`
columns. Every cell leads with its own completed/planned count and adds cumulative
runtime in parentheses after work completes, for example `1/2 (00:08:10)`. This makes
all four nested HPO folds and both unbiased outer evaluations explicit. Stopping,
purge, and embargo windows are safeguards within these cells, not additional model-fit
stages. During controlled sensitivity only the first declared inner fold under each
outer history is shown because that is the approved reduced-fidelity design.

The separate `Final evaluation` table contains `Final training` and `Holdout`. Both
show completed/expected cells from the outset (`0/1` for each supervised arm and `0/3`
for each RL arm's three risk scenarios), then advance as the immutable holdout-fit
ledger is appended. A started-but-unsealed final cell is shown as training `running`
and holdout `waiting`; the runner performs these within one authoritative fit/score
call, so both become complete only when its ledger row is durable.

Every progress-table heading is followed by an educational `Objective:` subtitle.
Controlled sensitivity explains held-constant marginal-effect measurement and its
selection exclusion; each outer-fold subtitle explains inner tuning followed by unseen
outer assessment; final evaluation explains locked refitting and descriptive holdout
reporting without further selection.

`Overall estimated time remaining` is displayed before the arm tables. It counts every
remaining HPO inner cell, outer evaluation, and combined final-fit/holdout cell. Each
arm uses the larger of its live observed median and its hash-verified score-blind
production calibration, so an unobserved stage cannot silently become a zero-duration
omission. The estimate is rounded to five minutes and labelled as a planning estimate;
it is not a deadline or research result. If the capability-gate or calibration hash
does not verify, the dashboard reports ETA unavailable instead of substituting a
partial lower bound.

The overall estimate is reconciled immediately below into remaining inner-HPO cells,
outer evaluations, final fit/holdout, and controlled sensitivity when still active.
Each component is rounded to five minutes and the displayed total is the sum of those
rounded components. Times in the progress tables remain backward-looking cumulative
durations of completed cells; blank future-cell times are represented in the ETA
breakdown through live per-arm timing or calibration rather than written into the
progress table as if they had already been observed.

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

Status precedence follows that same evidence boundary. When `completion.json` says
`complete`, `audit.json` says `passed`, the run identifiers agree, and the audit file's
SHA-256 matches the hash sealed into `completion.json`, the dashboard reports the
research run as `COMPLETED`. A later non-zero supervisor/child-process exit remains
visible as a yellow post-completion operational warning; it cannot overwrite the
already sealed and hash-verified research outcome. Before a verified completion seal
exists, a supervisor failure continues to report the run as `FAILED`.
