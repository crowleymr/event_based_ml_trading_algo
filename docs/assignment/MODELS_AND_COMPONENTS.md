# Models, algorithms and code map

## Component map

| Concern | Actual code | Role |
|---|---|---|
| Source preservation and schemas | `trading_pipeline.data` | Cache Yahoo/SEC inputs, map identifiers, validate and snapshot canonical tables |
| Feature engineering | `trading_pipeline.features.market`, `fundamentals` | Backward-looking rolling transforms and strict SEC filed-date as-of joins |
| Target and temporal split | `trading_pipeline.modelling.targets`, `splits` | Five-session label, boundary purge and embargo |
| Estimators | `elastic_net`, `gbt`, `xgboost_model` | Construct deterministic CPU baselines and the separate CPU/CUDA XGBoost diagnostic |
| Fit/select/predict | `training` | Fixed candidate grids, train-only fitting and validation-only selection |
| Model evaluation | `evaluate` | MAE, RMSE and daily Spearman IC |
| Portfolio decision | `trading_pipeline.portfolio` | Shared top-K selection, equal/inverse-vol weights and T+1 accounting |
| Orchestration | `trading_pipeline.run` | Single authoritative execution sequence and selection lock |
| Audit | `trading_pipeline.validation.leakage` | Read-only persisted-run integrity and reconciliation checks |

## Elastic Net

For training rows ((x_i,y_i)), Elastic Net minimizes a squared-error objective with combined L1 and L2 coefficient penalties. In sklearn's parameterization:

```text
(1 / (2n)) * ||y - Xw||²
+ alpha * l1_ratio * ||w||₁
+ 0.5 * alpha * (1 - l1_ratio) * ||w||²₂
```

The L1 term can set weak coefficients to zero; the L2 term stabilizes correlated features. This is a useful transparent linear baseline when finance features have different scales and noisy correlation.

Fitting procedure:

1. fit median imputation on training rows only;
2. fit standardization on the imputed training matrix only;
3. fit Elastic Net with a fixed seed, `max_iter=10000` and `tol=1e-5`;
4. evaluate four predeclared combinations: `alpha` in {0.0001, 0.001} and `l1_ratio` in {0.1, 0.5};
5. select by validation mean daily IC, then RMSE, then grid order.

The exact winner for any run must be read from that run's `selection.json`, not copied from prose.

## Histogram gradient-boosted trees

The GBT model builds an additive predictor:

```text
F_m(x) = F_(m-1)(x) + learning_rate * tree_m(x)
```

Each new shallow histogram-based regression tree is fitted to reduce squared-error loss relative to the current ensemble. Histogram binning makes tree search efficient; splits capture nonlinearities and interactions without scaling. Median imputation remains in the pipeline so all inputs have an explicit training-fitted missing-value policy.

Fitting procedure:

1. fit median imputation on training rows only;
2. fit `HistGradientBoostingRegressor` for 100 boosting iterations with learning rate 0.05, early stopping disabled and a fixed seed;
3. evaluate four predeclared combinations: `max_leaf_nodes` in {7, 15} and `l2_regularization` in {1.0, 10.0};
4. apply the same validation-only selection rule.

`max_leaf_nodes` controls tree capacity; `l2_regularization` shrinks leaf values; learning rate controls each tree's contribution; iteration count controls ensemble length. The small grid limits search degrees of freedom.

## XGBoost diagnostic family

E6/E7 add XGBoost as a separate family rather than replacing Histogram GBT. The harness
uses histogram tree construction, a fixed seed and one numerical thread, with requested
device `cpu`, `cuda` or `auto`. A real CUDA fit preflight determines availability; any
fallback to CPU is explicit in persisted metadata.

The training matrix is imputed using train-only medians. XGBoost receives training and
validation evaluation sets, records RMSE at each boosting round and may early-stop using
validation only. The final test never controls iteration count or parameters. Native gain
and fixed-seed validation permutation importance are descriptive diagnostics. E6/E7 and
their CPU/GPU timing comparison carry `diagnostic_reproduction_not_model_selection`;
E5 still reuses the original E1-E4 validation winner.

## Portfolio algorithms

P1 assigns (1/K) to each of the K highest predictions. P2 chooses the identical securities and normalizes inverse trailing volatility. Both are long-only and sum to at most one. Positions drift between weekly rebalances. Costs equal 10 bps times absolute traded dollars, including initial entry, and the accounting solves post-cost net asset value self-consistently.

E5 is not a fifth predictive fit. It reuses the validation-selected candidate's predictions and holdings selection, changing only the weighting engine. This isolates a portfolio-risk question from the model question.

## Four different objective families

- **Training loss** is the estimator's optimization criterion, such as squared error plus Elastic Net penalties. It determines fitted parameters.
- **Prediction metrics** MAE and RMSE measure numerical forecast error in return units. RMSE penalizes large errors more strongly.
- **Ranking IC** measures same-date Spearman ordering between predicted and realised returns. It matches the cross-sectional selection use more closely than raw error.
- **After-cost portfolio objectives** measure the consequences of ranking, weighting, execution and turnover. Return, Sharpe and drawdown are downstream system properties, not differentiable training losses here.

Good training loss does not guarantee good IC; good IC does not guarantee a good portfolio after concentration and costs. Slice 1 selects hyperparameters and E5's source by validation IC, with RMSE only as a tie-break. Final-test portfolio metrics are descriptive.
