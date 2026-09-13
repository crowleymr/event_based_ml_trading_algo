# Modelling package

Owns the five-session label, chronological split, estimator factories, fixed candidate grids, training, validation-only selection, prediction schemas and forecast metrics.

Public interfaces include `add_target`, `temporal_split`, `build_elastic_net`, `build_gbt`, `choose_models`, `predict_test` and `metrics`. Inputs are one row per security/session with F0/F1 features. Outputs are fitted sklearn pipelines, selection provenance, predictions and daily IC.

The label is T-to-T+5 adjusted-close return. Split assignment uses a common daily calendar: 60/20/20, labels crossing boundaries purged, five post-boundary sessions embargoed. Preprocessing fits only training data. Four fixed candidates per family are ranked on validation mean daily IC, then RMSE, then stable grid order. The selection lock must precede final-test prediction.

Prediction keys are model ID, split, session date and security ID. Values and ranks must be finite and deterministic. The package does not construct or backtest positions. Final-test results cannot inform any future model, feature or parameter choice.
