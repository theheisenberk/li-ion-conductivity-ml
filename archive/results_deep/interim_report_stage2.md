# Interim Report: Stage 2 - Physics-Informed Features

**Date:** 2026-02-03

## 1. Goal of Stage 2

Stage 2 builds on the Stage 1 `stage1_full_struct` experiment by adding physics-informed
features that probe the local atomic environment around mobile lithium ions. These
features are inspired by physical theories of ion transport.

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

## 3. Experiments

### Baseline: `stage2_baseline` (Model 1, stage1_full_struct replica)
**Stage 1 full_struct features (validation):**
- All Stage 0 features (elemental ratios + SMACT + Magpie embeddings)
- All geometry features (density, volume, lattice parameters, Li environment)
- Space group one-hot encoding (sg_1 to sg_230)

### Physics-only: `stage2_physics` (Model 2, physics-informed on top of Model 1 features)
**Baseline + CIF-derived physics features:**
- All baseline features
- Bond-valence mismatch features (bv_mismatch_avg, bv_mismatch_std)
- Ewald site energy features (ewald_energy_avg, ewald_energy_std)
- Li Voronoi coordination number (li_voronoi_cn_avg)

### Double-model Strategy A: `stage2_double_model_fulltrain`
Physics model (Model 2) trained on the **full training set**, including rows without physics
coverage (zero-filled, with indicators), then combined with the baseline via a coverage-based
decision tree:
- If physics coverage is good (all 3 indicators = 1), use Model 2.
- Otherwise, fall back to Model 1 (baseline).

### Double-model Strategy B: `stage2_double_model_subsettrain`
Physics model (Model 2) trained **only on the subset of samples where all 3 physics indicators
are 1** (no zero-filled physics rows in its training set). At prediction time:
- If physics coverage is good (all 3 indicators = 1), use this subset-trained Model 2.
- Otherwise, fall back to Model 1 (baseline).

### Double-model Strategy C: `stage2_double_model_residual`
Residual correction model trained on the good-physics subset:
- First, Model 1 (baseline) predicts log10(σ) for all samples.
- On the subset with full physics coverage, Model 2 is trained to predict the residual
  r = y_true − y_baseline from physics features.
- Final prediction uses:
  - y_final = y_baseline + r_physics when all 3 indicators = 1.
  - y_final = y_baseline otherwise.

## 4. Cross-Validation Results

| Experiment | R² Score | RMSE | MAE |
|------------|----------|------|-----|
| stage2_baseline | 0.7370 | 1.3734 | 0.8253 |
| stage2_physics | 0.6999 | 1.4671 | 0.8798 |
| stage2_double_model_fulltrain | 0.7155 | 1.4286 | 0.8569 |
| stage2_double_model_subsettrain | 0.7128 | 1.4352 | 0.8550 |
| stage2_double_model_residual | 0.6808 | 1.5130 | 0.8752 |
| stage2_double_model_residual_stack | 0.6956 | 1.4775 | 0.8797 |

### Comparison with Stage 1 (stage1_full_struct)

| Metric | Stage 1 | stage2_baseline | stage2_physics | stage2_double_model_fulltrain | stage2_double_model_subsettrain | stage2_double_model_residual | stage2_double_model_residual_stack | 
|--------|---------|---------|---------|---------|---------|---------|---------|
| R2 | 0.7370 | 0.7370 (+0.0000) | 0.6999 (-0.0371) | 0.7155 (-0.0215) | 0.7128 (-0.0242) | 0.6808 (-0.0562) | 0.6956 (-0.0414) | 
| RMSE | 1.3734 | 1.3734 (-0.0000) | 1.4671 (+0.0937) | 1.4286 (+0.0552) | 1.4352 (+0.0618) | 1.5130 (+0.1396) | 1.4775 (+0.1041) | 
| MAE | 0.8253 | 0.8253 (-0.0000) | 0.8798 (+0.0545) | 0.8569 (+0.0316) | 0.8550 (+0.0297) | 0.8752 (+0.0499) | 0.8797 (+0.0544) | 

## 5. Feature Coverage and Gating

Physics features are extracted from CIF files and therefore only available for a subset
of samples. **Indicator variables** are used in two ways:

- As direct inputs to the physics-informed model (Model 2), so tree-based learners can
  decide when physics values are trustworthy.
- As a **decision rule** in the double-model strategies:
  - If too few physics indicators are active (limited coverage), predictions fall back
    to the baseline `stage2_baseline` model (Model 1).
  - If all 3 indicators are active (good coverage), predictions come from:
    - the full-train physics model (`stage2_double_model_fulltrain`),
    - or the subset-trained physics model (`stage2_double_model_subsettrain`),
    - or the residual correction model on top of the baseline (`stage2_double_model_residual`),
      depending on the chosen strategy.

This ensures that the physics model is only used where CIF-derived descriptors are
well-populated, while all samples still benefit from a strong composition+structure
baseline trained on the full dataset.

## 6. Key Observations

- **Baseline validation:** The `stage2_baseline` experiment should match Stage 1 `stage1_full_struct`
  results, confirming pipeline consistency.
- **Physics feature impact (single model):** Compare R² changes from adding physics-informed features
  in `stage2_physics` relative to `stage2_baseline`.
- **Double-model behaviour:** The double-model metrics (`stage2_double_model_fulltrain`,
  `stage2_double_model_subsettrain`, `stage2_double_model_residual`) quantify whether selectively
  using physics only on well-covered samples—and optionally only as a residual correction—improves
  or stabilizes performance compared to always-on physics features.
- **Coverage effect:** Physics features have limited coverage (~10–20% with CIFs), so the gated
  strategies help avoid overfitting to this small subset while retaining their value where reliable.

## 7. Artifacts

Results are saved to `results/results_deep/`:
- `stage2_*_cv_parity.png` — 5-fold CV parity plots for each experiment (including all double-model variants)
- `stage2_*_test_parity.png` — Test set parity plots
- `stage2_*_predictions.csv` — Test set predictions for each experiment
- `interim_report_stage2.md` — This report
- `stage2.log` — Detailed execution log
- `stage2_feature_importance.csv` — Permutation importance for physics features
