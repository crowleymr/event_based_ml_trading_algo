# Modelling package

Owns the five-session label, chronological split, estimator factories, fixed candidate grids, training, validation-only selection, prediction schemas and forecast metrics.

Public interfaces include `add_target`, `temporal_split`, `build_elastic_net`, `build_gbt`, the optional `xgboost_model` harness, `choose_models`, `predict_test` and `metrics`. Inputs are one row per security/session with F0/F1 features. Outputs are fitted pipelines, selection provenance, predictions and daily IC.

The label is T-to-T+5 adjusted-close return. Split assignment uses a common daily calendar: 60/20/20, labels crossing boundaries purged, five post-boundary sessions embargoed. Preprocessing fits only training data. Four fixed candidates per family are ranked on validation mean daily IC, then RMSE, then stable grid order. The selection lock must precede final-test prediction.

Prediction keys are model ID, split, session date and security ID. Values and ranks must be finite and deterministic. The package does not construct or backtest positions. Final-test results cannot inform any future model, feature or parameter choice.

Training telemetry distinguishes estimator mechanics from evaluation evidence. Elastic
Net exposes solver iterations, convergence warnings and final diagnostics, not an epoch
loss curve; no curve may be invented. Histogram GBT staged prediction metrics, where a
fitted API supports them, are prediction diagnostics rather than optimizer loss. MAE/RMSE
measure forecast error, daily IC measures cross-sectional ranking, RL reward measures a
weekly control objective, and the after-cost portfolio objective is a separate realised
financial outcome.

New runs emit the common `training_trace.parquet` and `training_summary.parquet` contract.
Elastic Net records `n_iter`, convergence/warning state, dual gap, selected parameters,
the explicitly defined final objective and train/validation RMSE. Histogram GBT records
train/validation RMSE for every supported boosting stage, selected parameters/iteration,
duration, seed and CPU determinism metadata. These diagnostics never alter validation
selection and the completed immutable reference run is not retrofitted.

XGBoost is registered separately as E6/E7 with `tree_method="hist"` and configurable
`cpu`/`cuda`/`auto`. Train-only median imputation is shared with the tree baseline;
validation is used for both fixed-grid ranking and early stopping, never the test.
The harness records every exposed train/validation RMSE round, one-based selected
iteration, requested/actual device, explicit fallback, package/CUDA build information
and elapsed time. A paired CPU/CUDA timing/parity diagnostic is persisted separately and
cannot select a device or model. E5 remains locked to the original E1-E4 candidate set.

Because sklearn histogram GBT has no native impurity importance, new runs also persist a
fixed-seed, three-repeat validation permutation importance using change in negative RMSE.
It is a diagnostic of the frozen validation-selected model, not a new selection input.
XGBoost additionally records native gain importance. SHAP remains optional and was not
installed for the approved rerun; no missing SHAP values are fabricated.

## Phase 3 component contracts

`supervised_models/` and `unsupervised_models/` implement the research-disabled
`SupervisedModel` and `UnsupervisedModel` contracts defined in `experiments/`.
The supervised adapters preserve the existing Elastic Net, histogram GBT and XGBoost
builders behind registry metadata. The unsupervised lane provides a no-regime baseline
and deterministic train-only GMM scaffold. These components are engineering capability
only: registry creation with `require_research_enabled=True` rejects them until a new
protocol and data vintage are approved.

Model components declare fit/predict-or-transform behavior and serialisable search
spaces. They do not own temporal splits, objectives, candidate selection or final-test
access; those belong to the shared optimisation evaluator.
