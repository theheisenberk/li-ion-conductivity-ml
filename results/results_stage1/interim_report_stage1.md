# Interim Report: Stage 1 - Structural Features

**Date:** 2026-02-16

## 1. Goal of Stage 1

Stage 1 builds on the Stage 0 composition-only baseline by incorporating structural
features extracted from CIF files and lattice metadata in the CSV. The objective is
to capture geometry-dependent effects that are not present in composition alone.

## 2. Exact Pipeline (Stage 1)

1. Load `train.csv` and `test.csv` and list available CIFs.
2. Plot the raw target distributions (sigma/log10(sigma)) before clipping.
3. Clean data and create `log10_sigma` targets (map values like "<E-10" to 1e-10 first).
4. Generate Stage 0 baseline features (ratios + SMACT + Magpie embeddings only).
5. Extract structural features:
   - CSV‑derived lattice/space‑group metadata for all rows.
   - CIF‑derived geometry features where CIFs exist.
   - Zero‑fill CIF‑only columns and add `has_cif_struct` indicator.
6. One‑hot encode space groups.
7. Build fixed 5‑fold CV splits from the training set and reuse them across experiments.
8. Run each experiment:
   - 5‑fold CV → OOF predictions → CV parity plots.
   - Train on full training set → predict test set → test parity plots.
9. Export predictions and feature‑importance artifacts.

## 3. Data Preparation

The same cleaning and target construction as Stage 0 is applied. The log10(sigma)
target is created from the raw conductivity column after numeric conversion,
mapping threshold strings like "<E-10" to 1e-10 and applying stability clipping.
Rows without a valid target are removed from training.

**Coerced sigma indicator:** A binary feature `sigma_is_coerced` is added to flag
rows where the conductivity value was coerced from a threshold string (e.g., "<E-10"
→ 1e-10). This allows the model to learn to weight these imprecise measurements
differently, similar to how `has_cif_struct` gates CIF-derived features.

## 4. Stage 0 Baseline Features

Stage 0 features are generated first (elemental ratios, SMACT stoichiometry, and
Magpie embeddings only). These are carried forward into all Stage 1 experiments.

## 5. Structural Feature Sets

Stage 1 introduces structural descriptors from two sources:

1. CSV-derived structural metadata (available for all samples):
   - lattice parameters (a, b, c) and angles (alpha, beta, gamma)
   - space group number and Z
   - derived quantities such as volume, density, and volume per atom

2. CIF-derived geometry (available for a subset):
   - lattice anisotropy and orthogonality measures
   - Li site counts, Li-Li and Li-anion distances
   - Li coordination and site multiplicity metrics

Rows without CIF data are retained by zero-filling CIF-only columns and adding a
`has_cif_struct` indicator to let the model gate structural features.

## 6. Experiments and Feature Sets

Stage 1 compares progressively richer feature sets. Below are the exact features
used in each experiment:

### Experiment 1: `stage0_magpie` (Baseline)
**Stage 0 composition-only features:**
- `li_fraction`, `anion_fraction`, `total_elements` (elemental ratios)
- `smact_stoich_0` ... `smact_stoich_102` (103 SMACT stoichiometry features)
- `magpie_emb_0` ... `magpie_emb_21` (22 Magpie embedding features)
- `sigma_is_coerced` — 1 if sigma was coerced from threshold string (e.g., "<E-10"), else 0

### Experiment 2: `stage1_basic_struct`
**Stage 0 features + basic structural scalars:**
- All Stage 0 features (above)
- `density` — mass density (g/cm³)
- `volume_per_atom` — unit cell volume / total atoms (Å³/atom)
- `n_li_sites` — number of Li atoms in unit cell
- `n_total_atoms` — total atoms in unit cell

### Experiment 3: `stage1_geometry`
**Stage 0 features + full geometry descriptors:**
- All Stage 0 features
- All basic structural features (above)
- **Lattice parameters:**
  - `lattice_a`, `lattice_b`, `lattice_c` — cell edge lengths (Å)
  - `lattice_alpha`, `lattice_beta`, `lattice_gamma` — inter-axial angles (°)
- **Anisotropy measures:**
  - `lattice_anisotropy_bc_a` — (b+c)/(2a), measures elongation vs a-axis
  - `lattice_anisotropy_max_min` — max(a,b,c)/min(a,b,c), distortion ratio
- **Orthogonality:**
  - `angle_deviation_from_ortho` — mean |angle − 90°|
  - `is_cubic_like` — 1 if nearly cubic, else 0
- **Li environment:**
  - `li_fraction` — n_Li / n_total_atoms
  - `li_concentration` — n_Li / cell_volume (Li per Å³)
  - `framework_density` — (n_total − n_Li) / cell_volume
  - `li_li_min_dist` — minimum Li-Li distance (Å)
  - `li_li_avg_dist` — mean Li-Li nearest-neighbor distance < 5 Å
  - `li_anion_min_dist` — minimum Li-anion distance (Å)
  - `li_coordination_avg` — average Li coordination number (VoronoiNN)
  - `li_site_avg_multiplicity` — average Wyckoff multiplicity of Li sites
- **CIF indicator:**
  - `has_cif_struct` — 1 if CIF available, else 0 (gates CIF-only features)

### Experiment 4: `stage1_full_struct`
**Stage 0 features + geometry + space group one-hot:**
- All Stage 0 features
- All geometry features (above)
- `sg_1` ... `sg_230` — 230 binary space group indicators

For each experiment, parity plots and prediction CSVs are saved to
`results/results_stage1/`. Feature-importance plots (permutation importance) are
also saved in that directory.

## 7. Cross-Validation Results

| Experiment | R² Score | RMSE | MAE |
|------------|----------|------|-----|
| stage0_magpie | 0.7393 | 1.3675 | 0.8613 |
| stage1_basic_struct | 0.7369 | 1.3737 | 0.8681 |
| stage1_geometry | 0.7369 | 1.3737 | 0.8229 |
| stage1_full_struct | 0.7370 | 1.3734 | 0.8253 |

**Note:** The stage0_magpie experiment should produce results consistent with
Stage 0 results from the baseline pipeline (`results/results_stage0/`). If there are
discrepancies, verify that no data leakage is occurring (e.g., the raw conductivity
column must not be included in features).

## 8. Key Observations

- **Baseline consistency:** Stage 0 Magpie results should match between this pipeline
  and the original Stage 0 pipeline to confirm no leakage.
- **Structural feature value:** Compare R² gains from adding structural features.
- **Feature importance:** Check the permutation importance plot to identify which
  structural features contribute most to prediction accuracy.
