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
