# Interim Report: Stage 2 - Physics-Informed Features (Optuna Pipeline)

**Date:** 2026-02-12

## 1. Goal of Stage 2

Stage 2 builds on the Stage 1 `stage1_full_struct` experiment by adding physics-informed
features that probe the local atomic environment around mobile lithium ions. These
features are inspired by physical theories of ion transport.

**Model variants in this report:**
- **stage2_baseline_default**: sklearn defaults, **no Optuna** — used as Model 1 in most double-model strategies.
- **stage2_baseline_default_geometry**: Stage 0 + geometry-only baseline (no space group one-hot), sklearn defaults (no Optuna).
- **stage2_baseline_simple**: Optuna with SIMPLE_SEARCH_SPACE.
- **stage2_physics**: Optuna with standard search space.
- **Double-model strategies (A–E)**: Use either **stage2_baseline_default** or **stage2_baseline_default_geometry** as Model 1 (baseline); Model 2 (physics / residual / meta) is Optuna-optimized.

## 2. Physics-Informed Features Implemented

### A. Li–Anion Bond-Valence Mismatch
Quantifies the mismatch between ideal chemical bonding and the actual crystallographic
environment of Lithium using pymatgen's BondValenceAnalyzer.

**Features:** `bv_mismatch_avg`, `bv_mismatch_std`

**Calculation:**
$$V_{sum} = \sum_i \exp\left(\frac{R_0 - R_i}{b}\right)$$

Where $R_0$ is the tabulated bond-valence parameter, $R_i$ is the bond length, and $b \approx 0.37$ Å.
The mismatch is $|\Delta V| = |1 - V_{sum}|$ for Li (expected +1 oxidation state).

### B. Ewald Site Energy
Quantifies the electrostatic potential experienced by Li ions within the infinite periodic lattice
using pymatgen's EwaldSummation.

**Features:** `ewald_energy_avg`, `ewald_energy_std`

**Calculation:** Summation of long-range (reciprocal space) and short-range (real space)
electrostatic interactions for each Li site.

### C. Li Voronoi Coordination Number
Describes the local coordination environment around Li ions using Voronoi tessellation
via pymatgen's VoronoiNN.

**Feature:** `li_voronoi_cn_avg`

**Calculation:** Average coordination number from Voronoi polyhedra around Li sites.

## 3. Experiments (Default + Optuna)

### Baseline Default: `stage2_baseline_default` (Model 1, sklearn defaults, no Optuna)
**Stage 1 full_struct features (validation):**
- All Stage 0 features (elemental ratios + SMACT + Magpie embeddings)
- All geometry features (density, volume, lattice parameters, Li environment)
- Space group one-hot encoding (sg_1 to sg_230)
- Uses sklearn default HistGradientBoostingRegressor (no hyperparameter tuning). Often best generalizer.

### Baseline Geometry-only: `stage2_baseline_default_geometry` (geometry baseline, no Optuna)
**Stage 1 geometry-only baseline (no space group one-hot):**
- All Stage 0 features (elemental ratios + SMACT + Magpie embeddings)
- All geometry features (density, volume, lattice parameters, Li environment)
- **No** space group one-hot encoding (tests pure geometry contribution)
- Uses sklearn default HistGradientBoostingRegressor (no hyperparameter tuning).

### Baseline Simple: `stage2_baseline_simple` (Optuna searching for simpler models)
Same features as baseline. Optuna uses SIMPLE_SEARCH_SPACE (shallower trees, more regularization)
to find models potentially simpler than defaults.

### Physics-only: `stage2_physics` (Model 2, physics-informed on top of Model 1 features)
**Baseline + CIF-derived physics features:**
- All baseline features
- Bond-valence mismatch features (bv_mismatch_avg, bv_mismatch_std)
- Ewald site energy features (ewald_energy_avg, ewald_energy_std)
- Li Voronoi coordination number (li_voronoi_cn_avg)

### Double-model Strategy A: `stage2_double_model_fulltrain`
**Model 1 (baseline):** `stage2_baseline_default` (sklearn defaults, no Optuna).
**Model 2 (physics):** `stage2_physics` (Optuna-optimized, trained on full dataset).
- If physics coverage is good (all 3 indicators = 1), use Model 2.
- Otherwise, fall back to Model 1.

### Double-model Strategy B: `stage2_double_model_subsettrain`
**Model 1 (baseline):** `stage2_baseline_default` (sklearn defaults, no Optuna).
**Model 2 (physics):** Optuna-optimized model trained **only on the subset** where all 3 physics indicators are 1.
- If physics coverage is good, use subset-trained Model 2.
- Otherwise, fall back to Model 1.

### Double-model Strategy C: `stage2_double_model_residual`
**Model 1 (baseline):** `stage2_baseline_default` (sklearn defaults, no Optuna).
**Model 2 (residual):** Optuna-optimized model predicting residual (y_true − y_baseline) on the good-physics subset.
- Final prediction: y_baseline + residual when physics coverage is good; y_baseline otherwise.

### Double-model Strategy D: `stage2_double_model_residual_stack`
**Model 1 (baseline):** `stage2_baseline_default` (sklearn defaults, no Optuna).
**Model 2 (meta-model):** Optuna-optimized combiner on the good-physics subset, combining baseline and residual-corrected predictions.
- Uses baseline prediction, residual-corrected prediction, and physics scalars as features.

### Double-model Strategy E: `stage2_double_model_residual_geometry`
**Model 1 (baseline):** `stage2_baseline_default_geometry` (Stage 0 + geometry, no space group one-hot, sklearn defaults, no Optuna).
**Model 2 (residual):** Optuna-optimized model predicting residual (y_true − y_baseline_geometry) on the good-physics subset.
- Final prediction: geometry-only baseline + residual when physics coverage is good; geometry-only baseline otherwise.

## 4. Model Versions Used in Double-Model Strategies (Summary)

| Double-model strategy | Model 1 (baseline) | Model 2 (physics / residual / meta) |
|-----------------------|--------------------|-------------------------------------|
| stage2_double_model_fulltrain | **stage2_baseline_default** (no Optuna) | stage2_physics (Optuna) |
| stage2_double_model_subsettrain | **stage2_baseline_default** (no Optuna) | Optuna-optimized subset model |
| stage2_double_model_residual | **stage2_baseline_default** (no Optuna) | Optuna-optimized residual model |
| stage2_double_model_residual_geometry | **stage2_baseline_default_geometry** (no Optuna) | Optuna-optimized residual model (relative to geometry baseline) |
| stage2_double_model_residual_stack | **stage2_baseline_default** (no Optuna) | Optuna-optimized meta-model |

**The DEFAULT model is used as Model 1 in all double-model strategies except Strategy E** — not the simple model. Among experiments, **stage2_physics** (Test R² 0.5982) and **stage2_double_model_residual** (Test R² 0.5963) achieve the best test set performance.

## 5. Cross-Validation Results (Train OOF)

| Experiment | R² | RMSE | MAE | Spearman ρ (train) |
|------------|------|------|-----|-------------------|
| stage2_baseline_default | 0.7370 | 1.3734 | 0.8253 | 0.8901 |
| stage2_baseline_default_geometry | 0.7369 | 1.3737 | 0.8229 | 0.8873 |
| stage2_baseline_simple | 0.7295 | 1.3929 | 0.8851 | 0.8748 |
| stage2_physics | 0.7404 | 1.3646 | 0.8826 | 0.8856 |
| stage2_double_model_fulltrain | 0.7418 | 1.3608 | 0.8384 | 0.8922 |
| stage2_double_model_subsettrain | 0.7237 | 1.4078 | 0.8506 | 0.8852 |
| stage2_double_model_residual | 0.7334 | 1.3828 | 0.8304 | 0.8893 |
| stage2_double_model_residual_geometry | 0.7336 | 1.3824 | 0.8284 | 0.8865 |
| stage2_double_model_residual_stack | 0.7324 | 1.3854 | 0.8488 | 0.8869 |

## 6. Test Set Results

| Experiment | R² | RMSE | MAE | Spearman ρ (test) |
|------------|------|------|-----|------------------|
| stage2_baseline_default | 0.5912 | 1.6217 | 1.0897 | 0.7008 |
| stage2_baseline_default_geometry | 0.5557 | 1.6907 | 1.1274 | 0.7114 |
| stage2_baseline_simple | 0.5666 | 1.6699 | 1.1790 | 0.6651 |
| stage2_physics | 0.5982 | 1.6078 | 1.1654 | 0.7055 |
| stage2_double_model_fulltrain | 0.5619 | 1.6789 | 1.1028 | 0.7097 |
| stage2_double_model_subsettrain | 0.5262 | 1.7459 | 1.1295 | 0.6889 |
| stage2_double_model_residual | 0.5963 | 1.6116 | 1.0918 | 0.6991 |
| stage2_double_model_residual_geometry | 0.5592 | 1.6841 | 1.1294 | 0.7115 |
| stage2_double_model_residual_stack | 0.5438 | 1.7132 | 1.1309 | 0.6815 |

## 7. Optuna Optimization Methodology (Improved for Generalization)

Stage 2 analysis showed that optimizing on full CV splits led to overfitting: CV R² improved
~1% but test R² got worse. This pipeline implements improvements:

1. **5-fold CV with generalization penalty**: Optuna optimizes using predefined 5-fold splits.
   For each fold, the objective is: val_metric + 0.2 * max(0, val_metric - train_metric).
   This penalizes large train-val gaps and favors hyperparameters that generalize.
   *Previously used 80/20 hold-out; switching to CV+penalty (applied in every fold) yields more
   robust validation and improves test performance for several models (see Section 7.1).*
2. **Conservative search space**: Tighter bounds favor simpler models.
3. **Baseline experiments**:
   - `stage2_baseline_default`: sklearn defaults (no Optuna) — documents default, often best generalizer.
   - `stage2_baseline_default_geometry`: geometry-only baseline (Stage 0 + geometry, no space group one-hot), sklearn defaults (no Optuna).
   - `stage2_baseline_simple`: Optuna with SIMPLE_SEARCH_SPACE — searches for even simpler models.

### 7.1 Impact of CV+Penalty on Test Performance (vs. Previous 80/20 Hold-Out)

Switching from 80/20 hold-out to 5-fold CV with per-fold generalization penalty changed test R² as follows:

| Experiment | Previous (hold-out) | Current (CV+penalty) | Δ R² |
|------------|---------------------|----------------------|------|
| stage2_baseline_simple | 0.5684 | 0.5666 | −0.0018 |
| stage2_physics | 0.5675 | **0.5982** | **+0.0307** |
| stage2_double_model_fulltrain | 0.5335 | 0.5619 | +0.0284 |
| stage2_double_model_subsettrain | 0.5711 | 0.5262 | −0.0449 |
| stage2_double_model_residual | 0.5963 | 0.5963 | 0.0000 |
| stage2_double_model_residual_geometry | 0.5592 | 0.5592 | 0.0000 |
| stage2_double_model_residual_stack | 0.5434 | 0.5438 | +0.0004 |

**Summary:** `stage2_physics` and `stage2_double_model_fulltrain` gain substantially (≈3 pp and ≈2.8 pp R²). `stage2_double_model_residual` remains the best overall (0.5963) and is unchanged. `stage2_double_model_subsettrain` drops, likely due to the small physics subset favouring simpler validation. The baseline (`stage2_baseline_default`, no Optuna) is unchanged at 0.5912.

## 8. Best Hyperparameters

### stage2_baseline_default

sklearn default HistGradientBoostingRegressor (no tuning)

### stage2_baseline_default_geometry

sklearn default HistGradientBoostingRegressor (no tuning)

### stage2_baseline_simple

| Hyperparameter | Value |
|----------------|-------|
| max_depth | 3 |
| learning_rate | 0.09377886423050882 |
| max_leaf_nodes | 30 |
| min_samples_leaf | 16 |
| l2_regularization | 0.0011912016478004273 |
| max_bins | 99 |
| max_iter | 62 |

### stage2_physics

| Hyperparameter | Value |
|----------------|-------|
| max_depth | 4 |
| learning_rate | 0.05761369519330271 |
| max_leaf_nodes | 57 |
| min_samples_leaf | 5 |
| l2_regularization | 0.003009815839845394 |
| max_bins | 95 |
| max_iter | 91 |

### stage2_double_model_subsettrain

| Hyperparameter | Value |
|----------------|-------|
| max_depth | 7 |
| learning_rate | 0.03917094471268463 |
| max_leaf_nodes | 59 |
| min_samples_leaf | 17 |
| l2_regularization | 0.010662775308093897 |
| max_bins | 166 |
| max_iter | 184 |

### stage2_double_model_residual

| Hyperparameter | Value |
|----------------|-------|
| max_depth | 7 |
| learning_rate | 0.022816739880816207 |
| max_leaf_nodes | 23 |
| min_samples_leaf | 26 |
| l2_regularization | 0.00015876781526924017 |
| max_bins | 87 |
| max_iter | 124 |

### stage2_double_model_residual_geometry

| Hyperparameter | Value |
|----------------|-------|
| max_depth | 7 |
| learning_rate | 0.022816739880816207 |
| max_leaf_nodes | 23 |
| min_samples_leaf | 26 |
| l2_regularization | 0.00015876781526924017 |
| max_bins | 87 |
| max_iter | 124 |

### stage2_double_model_residual_stack

| Hyperparameter | Value |
|----------------|-------|
| max_depth | 3 |
| learning_rate | 0.01756747893790105 |
| max_leaf_nodes | 62 |
| min_samples_leaf | 14 |
| l2_regularization | 0.0001791993274329227 |
| max_bins | 232 |
| max_iter | 167 |


## 9. Comparison with Stage 1 (stage1_full_struct)

| Metric | Stage 1 | stage2_baseline_default | stage2_baseline_default_geometry | stage2_baseline_simple | stage2_physics | stage2_double_model_fulltrain | stage2_double_model_subsettrain | stage2_double_model_residual | stage2_double_model_residual_geometry | stage2_double_model_residual_stack | 
|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| R2 | 0.7370 | 0.7370 (+0.0000) | 0.7369 (-0.0001) | 0.7295 (-0.0075) | 0.7404 (+0.0034) | 0.7418 (+0.0048) | 0.7237 (-0.0133) | 0.7334 (-0.0036) | 0.7336 (-0.0034) | 0.7324 (-0.0046) | 
| RMSE | 1.3734 | 1.3734 (-0.0000) | 1.3737 (+0.0003) | 1.3929 (+0.0195) | 1.3646 (-0.0088) | 1.3608 (-0.0126) | 1.4078 (+0.0344) | 1.3828 (+0.0094) | 1.3824 (+0.0090) | 1.3854 (+0.0120) | 
| MAE | 0.8253 | 0.8253 (-0.0000) | 0.8229 (-0.0024) | 0.8851 (+0.0598) | 0.8826 (+0.0573) | 0.8384 (+0.0131) | 0.8506 (+0.0253) | 0.8304 (+0.0051) | 0.8284 (+0.0031) | 0.8488 (+0.0235) | 

## 10. Feature Coverage and Gating

Physics features are extracted from CIF files and therefore only available for a subset
of samples. **Indicator variables** are used in two ways:

- As direct inputs to the physics-informed model (Model 2), so tree-based learners can
  decide when physics values are trustworthy.
- As a **decision rule** in the double-model strategies:
  - If too few physics indicators are active (limited coverage), predictions fall back
    to the baseline `stage2_baseline_default` model (Model 1).
  - If all 3 indicators are active (good coverage), predictions come from:
    - the full-train physics model (`stage2_double_model_fulltrain`),
    - or the subset-trained physics model (`stage2_double_model_subsettrain`),
    - or the residual correction model on top of the baseline (`stage2_double_model_residual`),
    - or the learned combiner meta-model (`stage2_double_model_residual_stack`),
      depending on the chosen strategy.

This ensures that the physics model is only used where CIF-derived descriptors are
well-populated, while all samples still benefit from a strong composition+structure
baseline trained on the full dataset.

## 11. Key Observations

- **Baseline validation:** The `stage2_baseline_default` experiment should match Stage 1 `stage1_full_struct`
  results, confirming pipeline consistency.
- **Physics feature impact (single model):** Compare R² changes from adding physics-informed features
  in `stage2_physics` relative to `stage2_baseline_default`.
- **Double-model behaviour:** The double-model metrics (`stage2_double_model_fulltrain`,
  `stage2_double_model_subsettrain`, `stage2_double_model_residual`, `stage2_double_model_residual_stack`)
  quantify whether selectively using physics only on well-covered samples—and optionally only as a
  residual correction—improves or stabilizes performance compared to always-on physics features.
- **Coverage effect:** Physics features have limited coverage (~10–20% with CIFs), so the gated
  strategies help avoid overfitting to this small subset while retaining their value where reliable.
- **Optuna optimization (where used):** `stage2_baseline_simple`, `stage2_physics`, and all
  double-model Model 2 variants use Optuna with 5-fold CV and generalization penalty. `stage2_baseline_default`
  uses **no Optuna** (sklearn defaults). Best test performers: `stage2_physics` (0.5982) and
  `stage2_double_model_residual` (0.5963); the CV+penalty change improved physics-only models.
- **Quantile regression for uncertainty:** The best-performing experiment was used to train
  quantile regression models (Q0.05, Q0.5, Q0.95) to generate 90% prediction intervals.
  This provides uncertainty quantification alongside point predictions.

## 12. Quantile Regression and Uncertainty Quantification

Quantile regression models were trained for the best-performing experiment to provide
prediction intervals:

- **Quantiles:** 0.05 (lower bound), 0.5 (median), 0.95 (upper bound)
- **Prediction interval:** 90% confidence interval (Q0.05 to Q0.95)
- **Loss function:** Pinball loss (quantile loss)
- **Optimization:** Each quantile model was separately optimized with Optuna (Bayesian)

The quantile models allow us to:
- Quantify prediction uncertainty
- Generate prediction intervals for each test sample
- Assess calibration (coverage should be close to 90% for a well-calibrated model)

## 13. Artifacts

Results are saved to `results/results_optuna/`:
- `stage2_*_optuna_cv_parity.png` — 5-fold CV parity plots for each experiment (including all double-model variants)
- `stage2_*_optuna_test_parity.png` — Test set parity plots
- `stage2_*_optuna_predictions.csv` — Test set predictions for each experiment
- `*_quantile_uncertainty_cv.png` — CV uncertainty plot with prediction intervals (best experiment)
- `*_quantile_uncertainty_test.png` — Test set uncertainty plot with prediction intervals (best experiment)
- `*_quantile_predictions.csv` — Quantile predictions (Q0.05, Q0.5, Q0.95) and interval widths
- `interim_report_optuna.md` — This report
- `optuna.log` — Detailed execution log
- `stage2_feature_importance.csv` — Permutation importance for physics features
