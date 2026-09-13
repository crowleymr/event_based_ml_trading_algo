# Validation package

Owns read-only integrity checks over a completed run contract. `audit(run_dir)` verifies raw/snapshot/feature hashes, strict filing availability, unique and finite predictions, long-only weights, cash, exact T+1 execution, boundary purge, E5 prediction and top-K identity, transaction costs and equity compounding.

Inputs are a completed run directory and the paths recorded in its manifests. Output is an audit result; the original application writes `audit.json` during run completion.

An audit does not fit, tune, select or repair. Failure raises an error and the underlying source/code must be corrected and regenerated. An integrity pass demonstrates internal contract consistency, not profitable performance, unbiased data or stage-gate approval.
