# Decisions and limitations

Implementation choices (not human research decisions): sklearn HistGradientBoostingRegressor;
filesystem tracking; deterministic fixed universe with accepted survivorship bias;
offline synthetic fixtures are software checks only, never research evidence.
SEC contact must be supplied through SEC_USER_AGENT; no fabricated contact.

2026-09-12: Corrected MMC to MRSH for the same issuer (CIK 0000062709).
Official SEC mapping and the issuer announcement confirm the 14 January 2026 ticker rename:
https://www.marsh.com/en/corp/about/news/marsh-mclennan-to-change-nyse-symbol-to-mrsh.html
This preserves the intended company universe; it is not an issuer substitution.

Execution detail: weekly signal on first observed session of each ISO week, then T+1 close.
The FSD specifies next-session execution but does not specify open/close. Close fills
avoid assuming access to adjusted open data. Old positions earn the return into the
execution close, new positions earn returns only after that close. Positions drift
between rebalances. Costs apply to actual absolute traded dollars, including initial
entry; terminal positions are marked to market without forced liquidation.

Split implementation: common eligible calendar after 20-session feature warmup,
60/20/20 boundaries, remove labels reaching the next boundary and embargo the first
five sessions after each boundary. Test tail is retained for valuation and excluded
from label metrics where its five-session label is unavailable. Latest fundamental
values are selected by filed date, latest fiscal end, shortest duration, accession,
then value; only USD and USD/shares enter models. Fiscal growth features deferred.

2026-09-13 documentation decision: retain a layered local research application with
`run.py` as the application-service orchestrator. Package boundaries improve review
and testing but are not microservices. Reporting and notebooks are read-only derivatives
of one specified immutable run, with outputs stored separately.

Experiment identifiers E0-E5 and B0 are immutable machine IDs. Human-readable semantic
labels are joined at presentation time from the registry. Future IDs are append-only;
completed run data is never retrospectively renamed.

Assignment-support material remains Markdown until generated evidence and the student's
own interpretation are ready. AI-action records remain factual. First-person reflection,
understanding, critique and knowledge gaps are student-authored only.

2026-09-19 Phase 2 pilot decision: use a discrete portfolio/risk selector over cash
and the six frozen E0-E5 strategy sleeves. This tests dynamic allocation while keeping
the existing alpha models, security universe, execution timing and transaction-cost
accounting fixed. The selector acts after close T and its chosen underlying sleeve
weights fill at T+1 close. Reward is the next weekly net log return after the canonical
self-financing one-way costs.

The validation episode is used for fixed-budget DQN training. The already-observed
Slice 1 test episode is descriptive evaluation only and cannot tune features, reward,
architecture, budget or parameters. Seeds 41, 42 and 43 and the complete configuration
are frozen before evaluation; summaries report all seeds and the median, never the best.

Continuous control of 100 security weights with PPO/SAC is deferred. It would expand
the action space, constraint handling and reward-hacking surface before the discrete
harness has established basic parity, timing and leakage correctness. Recurrent models,
GPU tuning and reward/hyperparameter search are also deferred.

2026-09-20 diagnostic reproduction decision: protocol v2 retains the frozen MDP,
features, actions, reward, budget, seeds and descriptive test boundary, while permitting
`cpu`, `cuda` or `auto` execution. Device availability is an engineering concern, not a
selection criterion. `auto` records requested/actual device and falls back to CPU with a
reason. A short CPU/GPU DQN timing comparison is diagnostic only; it cannot choose a
device, architecture, parameter or seed. All observed-test reruns are labelled
`diagnostic_reproduction_not_model_selection`.

Common training telemetry is schema-versioned and records only values exposed reliably
by the fitted library. Missing histories are reported as not recorded. In particular,
Elastic Net has solver iterations and convergence diagnostics rather than epochs, and no
synthetic epoch loss curve may be created. Report schema v2 remains a read-only,
hash-backed derivative and may embed only a completed audited RL run.

2026-09-20 supervised diagnostic decision: preserve sklearn Histogram GBT as M2/E2/E4
and add XGBoost separately as M3/E6/E7. E5 remains eligible only for the original E1-E4
set, so the new family cannot rewrite the frozen selection decision. XGBoost uses
`tree_method="hist"`, validation-only early stopping and `cpu`/`cuda`/`auto` with an
executed CUDA preflight and explicit CPU fallback. Paired CPU/GPU timing and prediction
parity are engineering diagnostics, not model/device selection. All use of the already
observed final test is labelled `diagnostic_reproduction_not_model_selection`. SHAP is
optional; native gain and deterministic validation permutation importance are the
guaranteed importance contract.

2026-09-27 expanded-study upstream-layer decision: every one of the seven downstream
arms receives the same F1 plus causal-stack information contract. The fixed helper
layer is not a compared finalist: it generates expanding, purged out-of-fold E1
Elastic Net, E2 HistGBT and E3 XGBoost predictions from F1, plus an equal-weight E4
ensemble in fixed 60-session prediction blocks, using seed 41 and predeclared
parameters. The longer score block bounds repeated helper-layer fitting while every
prediction still comes from information frozen before the block begins. Missing outputs remain null with
an availability mask and any median imputation is fit on the consumer's training
partition only. The helper layer never reads the new holdout. For the frozen discrete
RL action space, E0 is momentum, E1-E4 are those fresh-vintage helper sleeves and E5
uses the score-blind predeclared E4 ensemble source with inverse-volatility weights. The
volatility estimator uses 20 sessions; top K is 10. This supplies shared causal
information without recursively using a candidate model's own outer/holdout result
as its training feature.

The same expanded protocol freezes 20 maximum deep-training epochs with patience 4
and 5,000 environment steps per RL candidate. The three classical search spaces each
contain 24 predeclared combinations spanning regularisation/capacity; score-blind
deadline calibration selects a complete 8, 16 or 24-candidate subset uniformly within
the classical families. These are exploratory-study ranges and do not alter Slice 1.

2026-09-27 deadline-closeout protocol revision: the first approved full-tier run
was stopped by the operator after four inner-fit cells on the 496-security vintage,
before outer selection or holdout access. Its roughly five-minute elapsed time
showed that the earlier 8-security bridge-smoke rate was not representative of
full-vintage fitting. This is resource evidence only; no score, ranking or other
model outcome is used for this revision. A new complete `deadline_complete` tier
declares two proposals per classical, deep and RL family, including each of the
three RL risk scenarios. The existing architecture and HPO search ranges remain
available; the deterministic proposal sampler must produce two valid distinct
candidates per family. The revised protocol freezes two outer and two inner folds
and seed 41 only. All seven arms, conservative/balanced/aggressive risk scenarios,
purge, embargo, stopping-window separation and sealed descriptive holdout remain
mandatory. The one-seed design cannot estimate seed dispersion, and two-by-two
fold evidence is deadline-constrained exploratory evidence, not confirmatory.
The original prepared/approved protocol and stopped run remain historical records;
the revised protocol requires fresh manifests, calibration, capability gate and
approval artefacts with their own hashes before execution. Calibration estimates
use per-arm timed cells multiplied by the complete proposal/fold/seed/scenario
matrix. Full-vintage timing records are required for a credible deadline estimate;
small-universe bridge timings cannot be extrapolated by this formula. This
estimate excludes data preparation, final fits, holdout, reporting and retries,
so a documented reserve is needed when the deadline is chosen.

2026-09-27 production-device and deep-resource revision: executed engineering
benchmarks, not model outcomes, determine the deadline protocol's devices and
resource ceiling. PyTorch CUDA executed on the RTX 4060 Ti, but the initial deep
adapter attempted full-dataset VRAM transfer and failed at roughly 24 GB. Training,
loss evaluation and prediction now stream deterministic hyperparameter-sized
minibatches; the 400-security full-history rerun stayed near 1.1--1.4 GB VRAM.
Small MLP DQN/PPO CUDA smokes were materially slower than CPU and Stable-Baselines
warned that MLP PPO is CPU-oriented, so both RL arms remain CPU. At 1,112,000 fit
rows XGBoost CUDA completed its score-blind fit/predict bridge in 11.42 seconds
versus 13.29 seconds on CPU, so XGBoost uses CUDA. LSTM and Transformer use CUDA;
Elastic Net and HistGBT use CPU. The original 20-epoch/patience-4 ceiling exceeded
34 minutes for one production-shaped LSTM-plus-Transformer smoke and would threaten
the six-hour execution window. The deadline protocol therefore freezes eight epochs
with patience two while retaining two distinct architecture/HPO candidates per deep
family. Device and resource choices are engineering constraints and never use model
scores, rankings or holdout outcomes.

2026-09-27 RL observed-price eligibility correction: the approved-v2 production
attempt failed closed before any complete RL candidate and before holdout access when
one selected security lacked an adjusted-close row on an intervening daily session.
The discrete RL policy acts weekly and its reward requires exact prices only at the
T+1 execution and weekly endpoint. Requiring every intervening daily row was therefore
stricter than the return definition and was not an information or valuation need.
The revised protocol uses exact observed endpoint valuation with no carry-forward,
imputation or identifier substitution; a missing required endpoint remains fatal.
Selection at each signal also requires a contiguous observed trailing 20-session
price history. That screen uses only then-available history, records per-signal
exclusions and reasons, and permits re-entry only after the history window recovers.
A whole-future-panel completeness screen is prohibited as lookahead. The change is
a pre-RL-result method correction, not outcome-based selection, and requires new
prepared, capability and approved artefacts while preserving approved v2 and its
failed attempt unchanged.
