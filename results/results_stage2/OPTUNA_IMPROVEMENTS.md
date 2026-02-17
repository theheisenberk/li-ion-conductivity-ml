# Optuna Pipeline Improvements (vs. Stage 2)

**Date:** 2026-02-03

## Baseline Experiments

- **stage2_baseline_default**: sklearn default HistGradientBoostingRegressor (no Optuna). Documents the default model; often best generalizer on test set.
- **stage2_baseline_simple**: Optuna with SIMPLE_SEARCH_SPACE — searches for even simpler models (shallower trees, more regularization).
- **stage2_physics**: Optuna with standard search space.

Double-model strategies use `stage2_baseline_default` as Model 1.

## Problem Identified from Stage 2

Stage 2 Optuna optimization showed signs of overfitting:

- **Baseline**: CV R² improved ~1% (0.737 → 0.742) but **test R² got worse** (CV 0.74 vs test 0.54)
- Hyperparameters were tuned to minimize CV RMSE on the same splits used for final evaluation
- This led to models that fit the CV structure rather than generalize to unseen data

## Improvements Implemented in `stage2_physics_optuna.py`

### 1. 5-Fold CV with Generalization Penalty

- **Before**: Optuna optimized on full 5-fold CV (same data used for evaluation), no penalty
- **After**: 5-fold CV with penalty applied **in every fold**. Per-fold objective:
  - `score = val_metric + 0.2 * max(0, val_metric - train_metric)`
- Favors hyperparameters that perform similarly on train and validation across all folds
- More robust than a single 80/20 split; uses all data for both training and validation

### 2. Generalization Penalty (Applied Per Fold)

- Objective function penalizes large train–validation RMSE gap in each fold
- If validation RMSE >> train RMSE in a fold, the fold score is penalized (weight = 0.2)
- Final objective: mean of penalized scores across folds

### 3. Conservative Hyperparameter Search Space

| Parameter        | Stage 2 (old) | Optuna (new) | Rationale                    |
|-----------------|---------------|--------------|------------------------------|
| max_depth       | 3–15          | 3–8          | Shallower trees generalize   |
| max_leaf_nodes  | 15–255        | 15–100       | Fewer leaves = simpler model |
| min_samples_leaf| 1–50          | 5–35         | More regularization          |
| learning_rate   | 0.01–0.3      | 0.01–0.15    | Lower LR often more stable   |
| max_iter        | 50–300        | 50–200       | Avoid overfitting iterations  |

## Output Location

All results are saved to **`results/results_optuna/`** (not `results_deep`):

- `optuna.log` — Execution log
- `interim_report_optuna.md` — Full report with methodology
- `stage2_*_optuna_cv_parity.png` — CV parity plots
- `stage2_*_optuna_test_parity.png` — Test parity plots
- `stage2_*_optuna_predictions.csv` — Test predictions
- `*_quantile_uncertainty_*.png` — Uncertainty plots
- `*_quantile_predictions.csv` — Quantile predictions

## How to Run

```bash
python stage2_physics_optuna.py
```

Stage 2 (`li_mobility_advanced_deep.py` and `results_deep`) is unchanged.
