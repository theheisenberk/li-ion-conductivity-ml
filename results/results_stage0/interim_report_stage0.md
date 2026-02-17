# Interim Report: Stage 0 - Composition-Only Baseline

**Date:** 2026-02-16

## 1. Goal of Stage 0

Stage 0 establishes a composition-only baseline for predicting ionic conductivity.
The intent is to build a reproducible, chemistry-informed starting point before
introducing structure-based descriptors.

## 2. Data Leakage Fix (v2)

An earlier version of this pipeline suffered from **data leakage**: the raw
`Ionic conductivity (S cm-1)` column was inadvertently left in the feature set.
After `add_target_log10_sigma()` converts it from string to numeric, it became a
numeric column that passed the `select_dtypes(include=np.number)` filter. Since
`log10_sigma = log10(Ionic conductivity)`, the model was effectively given the
answer as a feature, producing artificially high R-square values on both CV and test
sets.

The `sigma_is_coerced` column (a binary flag derived from the target column,
indicating whether the conductivity value was a detection-limit string like
"<1E-10") was also removed from the feature set to prevent any indirect leakage.

**Fix applied:** Both `Ionic conductivity (S cm-1)` and `sigma_is_coerced` are now
explicitly excluded from the feature set via the `structural_metadata_cols`
exclusion list.

All prior results (predictions, metrics, plots) generated before this fix have been
deleted. The results in this directory were regenerated after the fix and reflect
the true composition-only predictive performance.

## 3. Exact Pipeline (Stage 0)

1. Load `train.csv` and `test.csv`.
2. Plot **raw target distributions** (sigma and log10(sigma)) before any clipping.
3. Clean data:
   - Strip whitespace from column names.
   - Replace +/-inf with NaN.
   - Drop duplicate rows.
4. Create `log10_sigma` target from the ionic conductivity column.
5. Drop rows with missing targets **from training only**.
6. Build baseline features on train and test, **excluding the raw conductivity
   column and any target-derived columns from the feature set**.
7. Generate **one fixed 5-fold CV split** on the training set and reuse it for all experiments.
8. For each embedding experiment:
   - Train 5-fold CV models -> **out-of-fold (OOF) predictions**.
   - Save **5-fold CV parity plot** and metrics.
   - Train on full training set -> predict test set.
   - Save **test parity plot** and predictions.
9. Run the Magpie-only linear-sigma experiment and repeat the same plot/prediction steps.
10. Export a metrics summary table.

## 4. Data Cleaning and Target Construction

The raw training data is cleaned using a consistent routine:
- Strip whitespace from column names.
- Replace infinite values with NaN.
- Drop exact duplicate rows.

The target is the ionic conductivity column, transformed as:
- Convert "Ionic conductivity (S cm-1)" to numeric, coercing non-numeric strings to NaN.
- Clip non-positive values to a small positive constant (1e-30) to avoid log10 issues.
- Compute log10(sigma) and store as `log10_sigma`.

The cleaned data is saved to `data/processed/`, and rows missing `log10_sigma` are
removed from training. Two histograms are saved before any clipping:
- `stage0_log10_sigma_hist_raw.png` (log10 scale)
- `stage0_sigma_hist_raw.png` (linear sigma scale)

## 5. Baseline Features (Composition-Only)

Two baseline feature blocks are generated from the reduced composition:

1. Elemental ratios:
   - `li_fraction`: atomic fraction of Li in the formula.
   - `anion_fraction`: total fraction of common anions (O, S, F, Cl, Br, I, N, P).
   - `total_elements`: count of distinct elements in the formula.

2. SMACT stoichiometry vector:
   - A 103-length vector where each position represents an element in the periodic
     table and stores its atomic fraction in the formula.

These form the "baseline features" used in all Stage 0 experiments.

**Excluded from features:** The raw conductivity column (`Ionic conductivity (S cm-1)`),
the derived target (`log10_sigma`), and the target-derived flag (`sigma_is_coerced`)
are all explicitly removed before training to prevent data leakage.

## 6. Element Embeddings Compared

Stage 0 evaluates three embedding schemes on top of the baseline features:
- Magpie: hand-engineered elemental properties.
- Mat2Vec: embeddings learned from element co-occurrence in text.
- MegNet16: embeddings learned from a graph neural network pre-trained on materials data.

A final experiment combines all three embeddings. Each embedding is represented as
a composition-weighted average vector across the elements in the formula.

Stage 0 also includes a **Magpie-only, no-baseline** experiment on the **linear sigma**
target (embeddings only), to compare against the log-scale models without changing
the feature set.

## 7. Cross-Validation Protocol

Stage 0 uses **5-fold cross-validation** to estimate performance:
- The training set is split into 5 folds.
- For each fold, the model is trained on 4 folds and evaluated on the held-out fold.
- This yields out-of-fold predictions for every training sample.

To ensure a fair comparison across embeddings, **the exact same fold splits** are
generated once and reused for all experiments.

**Important:** the reported R-square/RMSE/MAE values are **cross-validation metrics**
computed on the training data (out-of-fold). They are **not** test-set metrics.
After CV, a final model is trained on the full training set and used to produce
predictions for the test set (saved as CSV), but no test-set targets are available
for evaluation.

## 8. Plots and Metrics (How to Read Them)

- **5-fold CV parity plots**: use OOF predictions on training data.
  - Plots report **R-square (log)** and **R-square_linear** from back-transformed predictions.
- **Test parity plots**: use predictions on the test set (if targets are present).
  - Same R-square annotations as above for log-target experiments.
- The **metrics summary table** consolidates both log-space and linear-space metrics
  across all experiments in `results/results_stage0/stage0_metrics_summary.csv`.

## 9. Artifacts

- Parity plots for each log-scale embedding experiment are saved in `results/results_stage0/`.
  These plots include both R-square (log) and R-square_linear computed after back-transforming
  predictions with sigma_hat = 10^pred.
- A parity plot and predictions are also saved for the **Magpie-only linear sigma** run.
- Prediction CSVs for the test set are saved alongside the plots.
- Both log-scale and linear-scale histograms are saved in the same directory.
