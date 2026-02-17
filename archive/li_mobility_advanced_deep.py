# =================================================================================================
# Li-ion Mobility Prediction - Stage 2: Physics-Informed Features
# =================================================================================================
# This script implements Stage 2 of the machine learning pipeline for predicting lithium-ion
# conductivity. It builds upon Stage 1's best-performing experiment (stage1_full_struct) and
# adds physics-informed features that probe the local atomic environment around Li ions.
#
# Stage 2 Physics Features (from CIF parsing via pymatgen):
# - Li–anion bond-valence mismatch (using BondValenceAnalyzer)
# - Average Ewald site energy (using EwaldSummation)
# - Li Voronoi coordination number (using VoronoiNN)
#
# The baseline experiment replicates stage1_full_struct EXACTLY to validate pipeline consistency.
# =================================================================================================

import os
import sys
import shutil
import warnings
from pathlib import Path

# Prevent Python from writing bytecode (.pyc files) to avoid caching issues
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def clear_pycache(root_path: Path):
    """
    Recursively removes all __pycache__ directories to ensure fresh code loading.
    """
    for pycache in root_path.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            print(f"Cleared cache: {pycache}")
        except Exception as e:
            print(f"Failed to clear cache {pycache}: {e}")


# =================================================================================================
# Project Structure Setup
# =================================================================================================

def find_project_root(start_path: str = None, marker_file: str = "requirements.txt") -> Path:
    """Find the project root directory by searching upwards for a marker file."""
    if start_path is None:
        current = Path.cwd()
    else:
        path = Path(start_path).resolve()
        current = path.parent if path.is_file() else path
    
    for path in [current] + list(current.parents):
        marker = path / marker_file
        if marker.is_file():
            return path
    
    raise FileNotFoundError(f"Could not find '{marker_file}' in current directory or any parent directory.")


PROJECT_ROOT = find_project_root(__file__)
clear_pycache(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =================================================================================================
# Custom Module Imports
# =================================================================================================

from src.utils import (
    setup_logger,
    seed_everything,
    save_dataframe,
    save_parity_plot,
    save_histogram,
    compute_regression_metrics,
)
from src.data_processing import load_data, clean_data, add_target_log10_sigma, coerce_sigma_series, TARGET_COL
from src.features import (
    stage0_elemental_ratios, 
    stage0_smact_features, 
    stage0_element_embeddings,
    stage1_csv_structural_features,
    stage1_structural_features,
    stage1_spacegroup_onehot,
)
from src.model_training import ExperimentConfig, run_cv_with_predefined_splits, fit_full_and_predict


# =================================================================================================
# Stage 2: Physics-Informed Feature Extraction Functions
# =================================================================================================

def extract_physics_features_from_structure(structure, verbose: bool = False):
    """
    Extracts physics-informed features from a pymatgen Structure object.
    
    Features extracted:
    1. Li–anion bond-valence mismatch (avg, std) + has_bv_mismatch indicator
    2. Average Ewald site energy for Li (avg, std) + has_ewald_energy indicator
    3. Li Voronoi coordination number (avg) + has_voronoi_cn indicator
    
    The indicator variables (has_*) allow tree-based models to learn when to
    trust the feature values vs. when they were zero-filled due to failed extraction.
    
    Args:
        structure: pymatgen Structure object
        verbose: Whether to print debug info
    
    Returns:
        dict: Dictionary of physics features (NaN if extraction fails) and indicators
    """
    result = {
        # Bond-valence mismatch features
        "bv_mismatch_avg": np.nan,
        "bv_mismatch_std": np.nan,
        "has_bv_mismatch": 0,  # Indicator: 1 if calculated, 0 if will be zero-filled
        # Ewald site energy features
        "ewald_energy_avg": np.nan,
        "ewald_energy_std": np.nan,
        "has_ewald_energy": 0,  # Indicator: 1 if calculated, 0 if will be zero-filled
        # Voronoi coordination features
        "li_voronoi_cn_avg": np.nan,
        "has_voronoi_cn": 0,  # Indicator: 1 if calculated, 0 if will be zero-filled
    }
    
    if structure is None:
        return result
    
    # Find Li site indices
    # Use species_string for compatibility with mixed occupancy sites
    li_indices = [i for i, site in enumerate(structure) if "Li" in str(site.species)]
    if len(li_indices) == 0:
        return result
    
    # -----------------------------------------------------------------------------------------
    # 1. Bond-Valence Mismatch for Li sites
    # Uses bond valence sum calculated from actual bond lengths
    # -----------------------------------------------------------------------------------------
    try:
        from pymatgen.analysis.local_env import CrystalNN
        
        # Use CrystalNN for robust neighbor finding
        cnn = CrystalNN()
        
        # Bond valence parameters for Li-X bonds (R0 in Å, B = 0.37)
        # From International Tables for Crystallography
        bv_params = {
            "O": 1.466, "S": 2.052, "F": 1.36, "Cl": 1.91,
            "Br": 2.07, "I": 2.34, "N": 1.61, "Se": 2.22,
        }
        B = 0.37  # Universal constant
        
        li_bvs_mismatches = []
        
        for li_idx in li_indices[:min(10, len(li_indices))]:
            try:
                # Get neighbors of Li site
                neighbors = cnn.get_nn_info(structure, li_idx)
                
                if neighbors:
                    bv_sum = 0.0
                    for neighbor in neighbors:
                        neighbor_site = neighbor["site"]
                        neighbor_element = str(neighbor_site.species.elements[0]) if hasattr(neighbor_site.species, 'elements') else str(neighbor_site.species).split(":")[-1].strip()
                        
                        # Get distance to neighbor
                        distance = neighbor["site"].distance(structure[li_idx])
                        
                        # Find R0 for this Li-X pair
                        r0 = None
                        for anion, r0_val in bv_params.items():
                            if anion in neighbor_element:
                                r0 = r0_val
                                break
                        
                        if r0 is not None and distance > 0:
                            # Bond valence = exp((R0 - R) / B)
                            bv = np.exp((r0 - distance) / B)
                            bv_sum += bv
                    
                    if bv_sum > 0:
                        # BVS mismatch = |1 - BVS| (Li should have BVS ≈ 1)
                        mismatch = abs(1.0 - bv_sum)
                        li_bvs_mismatches.append(mismatch)
            except Exception:
                pass
        
        if li_bvs_mismatches:
            result["bv_mismatch_avg"] = float(np.mean(li_bvs_mismatches))
            result["bv_mismatch_std"] = float(np.std(li_bvs_mismatches)) if len(li_bvs_mismatches) > 1 else 0.0
            result["has_bv_mismatch"] = 1  # Successfully calculated
    except Exception as e:
        if verbose:
            print(f"  BV analysis failed: {e}")
    
    # -----------------------------------------------------------------------------------------
    # 2. Ewald Site Energy for Li sites
    # -----------------------------------------------------------------------------------------
    try:
        from pymatgen.analysis.ewald import EwaldSummation
        from pymatgen.core.structure import Structure
        
        # Try to assign oxidation states if not already present
        try:
            from pymatgen.analysis.bond_valence import BVAnalyzer
            bva = BVAnalyzer()
            oxi_struct = bva.get_oxi_state_decorated_structure(structure)
        except Exception:
            # If BVA fails, try a simple oxidation state assignment
            oxi_struct = structure.copy()
            try:
                oxi_struct.add_oxidation_state_by_guess()
            except Exception:
                oxi_struct = None
        
        if oxi_struct is not None:
            # Calculate Ewald summation
            ewald = EwaldSummation(oxi_struct)
            
            # Get site energies for Li sites
            li_ewald_energies = []
            for li_idx in li_indices:
                if li_idx < len(oxi_struct):
                    try:
                        # Get the site energy contribution
                        site_energy = ewald.get_site_energy(li_idx)
                        li_ewald_energies.append(site_energy)
                    except Exception:
                        pass
            
            if li_ewald_energies:
                result["ewald_energy_avg"] = float(np.mean(li_ewald_energies))
                result["ewald_energy_std"] = float(np.std(li_ewald_energies)) if len(li_ewald_energies) > 1 else 0.0
                result["has_ewald_energy"] = 1  # Successfully calculated
    except Exception as e:
        if verbose:
            print(f"  Ewald analysis failed: {e}")
    
    # -----------------------------------------------------------------------------------------
    # 3. Li Voronoi Coordination Number
    # -----------------------------------------------------------------------------------------
    try:
        from pymatgen.analysis.local_env import VoronoiNN
        
        vnn = VoronoiNN(cutoff=5.0, allow_pathological=True)
        li_cns = []
        
        # Sample up to 10 Li sites for computational efficiency
        for li_idx in li_indices[:min(10, len(li_indices))]:
            try:
                cn = vnn.get_cn(structure, li_idx)
                li_cns.append(cn)
            except Exception:
                pass
        
        if li_cns:
            result["li_voronoi_cn_avg"] = float(np.mean(li_cns))
            result["has_voronoi_cn"] = 1  # Successfully calculated
    except Exception as e:
        if verbose:
            print(f"  Voronoi analysis failed: {e}")
    
    return result


def stage2_physics_features(
    df: pd.DataFrame,
    id_col: str = "ID",
    cif_dir: str = None,
    verbose: bool = True
) -> tuple:
    """
    Extracts Stage 2 physics-informed features from CIF files.
    
    Args:
        df: Input DataFrame with ID column matching CIF filenames
        id_col: Name of ID column
        cif_dir: Directory containing CIF files
        verbose: Whether to print progress
    
    Returns:
        Tuple of (DataFrame with physics features, extraction statistics dict)
    """
    df = df.copy()
    
    # Initialize physics feature columns (values + indicator variables)
    physics_cols = [
        # Bond-valence mismatch
        "bv_mismatch_avg", "bv_mismatch_std", "has_bv_mismatch",
        # Ewald site energy
        "ewald_energy_avg", "ewald_energy_std", "has_ewald_energy",
        # Voronoi coordination
        "li_voronoi_cn_avg", "has_voronoi_cn",
    ]
    
    for col in physics_cols:
        df[col] = np.nan
    
    stats = {
        "total": len(df),
        "processed": 0,
        "with_bv": 0,
        "with_ewald": 0,
        "with_voronoi": 0,
    }
    
    if cif_dir is None or not os.path.exists(cif_dir):
        return df, stats
    
    # Import pymatgen for structure parsing
    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifParser
    except ImportError:
        warnings.warn("pymatgen not available for physics feature extraction")
        return df, stats
    
    for idx, row in df.iterrows():
        material_id = row[id_col]
        cif_path = os.path.join(cif_dir, f"{material_id}.cif")
        
        if not os.path.exists(cif_path):
            continue
        
        # Parse structure
        structure = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structure = Structure.from_file(cif_path)
        except Exception:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parser = CifParser(cif_path, occupancy_tolerance=1.0)
                    structures = parser.parse_structures(primitive=False)
                    if structures:
                        structure = structures[0]
            except Exception:
                pass
        
        if structure is None:
            continue
        
        stats["processed"] += 1
        
        # Extract physics features
        features = extract_physics_features_from_structure(structure, verbose=False)
        
        # Update DataFrame
        for col, value in features.items():
            if col in df.columns:
                df.at[idx, col] = value
        
        # Update stats
        if not np.isnan(features.get("bv_mismatch_avg", np.nan)):
            stats["with_bv"] += 1
        if not np.isnan(features.get("ewald_energy_avg", np.nan)):
            stats["with_ewald"] += 1
        if not np.isnan(features.get("li_voronoi_cn_avg", np.nan)):
            stats["with_voronoi"] += 1
    
    return df, stats


def generate_stage2_report(
    path: str,
    metrics: dict,
    stage1_ref: dict = None,
) -> None:
    """
    Generate the Stage 2 interim report.
    
    Args:
        path: Destination path for the Markdown report
        metrics: Dict mapping experiment name -> metrics dict
        stage1_ref: Reference metrics from Stage 1 for comparison
    """
    report_content = f"""# Interim Report: Stage 2 - Physics-Informed Features

**Date:** {pd.to_datetime('today').strftime('%Y-%m-%d')}

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
$$V_{{sum}} = \\sum_i \\exp\\left(\\frac{{R_0 - R_i}}{{b}}\\right)$$

Where $R_0$ is the tabulated bond-valence parameter, $R_i$ is the bond length, and $b \\approx 0.37$ Å.
The mismatch is $|\\Delta V| = |1 - V_{{sum}}|$ for Li (expected +1 oxidation state).

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
"""
    # Add metrics rows
    for exp_name, m in metrics.items():
        r2 = m.get('r2', float('nan'))
        rmse = m.get('rmse', float('nan'))
        mae = m.get('mae', float('nan'))
        report_content += f"| {exp_name} | {r2:.4f} | {rmse:.4f} | {mae:.4f} |\n"
    
    # Add comparison with Stage 1
    if stage1_ref:
        report_content += f"""
### Comparison with Stage 1 (stage1_full_struct)

| Metric | Stage 1 | """
        for exp_name in metrics.keys():
            report_content += f"{exp_name} | "
        report_content += "\n|--------|---------|"
        for _ in metrics:
            report_content += "---------|"
        report_content += "\n"
        
        for metric_name in ['r2', 'rmse', 'mae']:
            stage1_val = stage1_ref.get(metric_name, float('nan'))
            report_content += f"| {metric_name.upper()} | {stage1_val:.4f} | "
            for exp_name, m in metrics.items():
                exp_val = m.get(metric_name, float('nan'))
                diff = exp_val - stage1_val
                report_content += f"{exp_val:.4f} ({diff:+.4f}) | "
            report_content += "\n"
    
    report_content += """
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
"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except Exception as e:
        print(f"Could not write report: {e}")


# =================================================================================================
# Main Execution Block
# =================================================================================================

def main():
    """
    Stage 2: Physics-Informed Features Pipeline.
    
    This pipeline:
    1. Replicates stage1_full_struct EXACTLY as baseline (for validation)
    2. Adds physics-informed features: bond-valence mismatch, Ewald energy, Voronoi CN
    3. Compares results to ensure no data leakage and validate physics feature contribution
    """
    # -----------------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------------
    seed_everything(42)
    
    results_dir = os.path.join(PROJECT_ROOT, "results", "results_deep")
    os.makedirs(results_dir, exist_ok=True)
    
    log_path = os.path.join(results_dir, "stage2.log")
    logger = setup_logger("stage2", log_file=log_path)
    logger.info("=" * 80)
    logger.info("Stage 2: Physics-Informed Features Pipeline")
    logger.info("=" * 80)
    
    # Reference metrics from Stage 1 (stage1_full_struct)
    stage1_ref = {
        "r2": 0.7370,
        "rmse": 1.3734,
        "mae": 0.8253,
    }
    
    # -----------------------------------------------------------------------------------------
    # Data Loading and Preprocessing
    # -----------------------------------------------------------------------------------------
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    
    logger.info(f"Loading data from: {data_dir}")
    train_df_full, test_df_full = load_data(data_dir)
    logger.info(f"Loaded {len(train_df_full)} training samples and {len(test_df_full)} test samples.")
    
    # Histogram of log10(sigma) before threshold clipping
    sigma_col = "Ionic conductivity (S cm-1)"
    if sigma_col in train_df_full.columns:
        sigma_numeric = coerce_sigma_series(train_df_full[sigma_col])
        sigma_positive = sigma_numeric[sigma_numeric > 0]
        log10_sigma_raw = np.log10(sigma_positive)
        hist_path = os.path.join(results_dir, "stage2_log10_sigma_hist_raw.png")
        save_histogram(
            log10_sigma_raw.values,
            hist_path,
            title="Stage 2: log10(sigma) before threshold clipping",
            x_label="log10(sigma)",
            vlines={np.log10(1e-30): "clip threshold (1e-30)"},
        )
        logger.info(f"log10(sigma) histogram saved to: {hist_path}")
    
    # Identify available CIF files
    train_cif_dir = os.path.join(data_dir, "train_cifs")
    test_cif_dir = os.path.join(data_dir, "test_cifs")
    train_cif_files = sorted([f.replace(".cif", "") for f in os.listdir(train_cif_dir) if f.endswith(".cif")])
    test_cif_files = sorted([f.replace(".cif", "") for f in os.listdir(test_cif_dir) if f.endswith(".cif")])
    logger.info(f"Available CIF files: {len(train_cif_files)} training, {len(test_cif_files)} test")
    
    # Clean and transform data (also adds sigma_is_coerced indicator)
    train_df = add_target_log10_sigma(clean_data(train_df_full), target_sigma_col="Ionic conductivity (S cm-1)")
    test_df = add_target_log10_sigma(clean_data(test_df_full), target_sigma_col="Ionic conductivity (S cm-1)")
    
    # Log coerced sigma statistics
    if "sigma_is_coerced" in train_df.columns:
        n_coerced_train = train_df["sigma_is_coerced"].sum()
        logger.info(f"Coerced sigma values (train): {n_coerced_train}/{len(train_df)} ({100*n_coerced_train/len(train_df):.1f}%)")
    if "sigma_is_coerced" in test_df.columns:
        n_coerced_test = test_df["sigma_is_coerced"].sum()
        logger.info(f"Coerced sigma values (test): {n_coerced_test}/{len(test_df)} ({100*n_coerced_test/len(test_df):.1f}%)")
    
    # Drop rows with missing target
    if TARGET_COL in train_df.columns:
        train_df.dropna(subset=[TARGET_COL], inplace=True)
    
    logger.info(f"Training samples (total): {len(train_df)} (with CIFs: {sum(train_df['ID'].isin(train_cif_files))})")
    logger.info(f"Test samples (total): {len(test_df)} (with CIFs: {sum(test_df['ID'].isin(test_cif_files))})")
    
    # -----------------------------------------------------------------------------------------
    # Feature Engineering - Stage 0 (Composition-Only Baseline)
    # -----------------------------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("Generating Stage 0 baseline features (Magpie embeddings)...")
    
    base_train_df = stage0_elemental_ratios(train_df.copy(), "Reduced Composition")
    base_train_df = stage0_smact_features(base_train_df, "Reduced Composition")
    base_train_df = stage0_element_embeddings(base_train_df, "Reduced Composition", embedding_names=["magpie"])
    
    base_test_df = stage0_elemental_ratios(test_df.copy(), "Reduced Composition")
    base_test_df = stage0_smact_features(base_test_df, "Reduced Composition")
    base_test_df = stage0_element_embeddings(base_test_df, "Reduced Composition", embedding_names=["magpie"])
    
    # Define metadata columns to exclude from features
    # CRITICAL: Include the raw conductivity column to prevent data leakage!
    metadata_cols = [
        "Space group", "Space group #", "a", "b", "c", "alpha", "beta", "gamma", "Z",
        "IC (Total)", "IC (Bulk)", "ID", "Family", "DOI", "Checked", "Ref", "Cif ID",
        "Cif ref_1", "Cif ref_2", "note", "close match", "close match DOI", "ICSD ID",
        "Laskowski ID", "Liion ID", "True Composition", "Reduced Composition",
        "Ionic conductivity (S cm-1)",
    ]
    
    # Get Stage 0 feature columns
    numeric_cols = base_train_df.select_dtypes(include=np.number).columns.tolist()
    stage0_feature_cols = [c for c in numeric_cols if c != TARGET_COL and c not in metadata_cols]
    logger.info(f"Stage 0 features: {len(stage0_feature_cols)} columns")
    
    # -----------------------------------------------------------------------------------------
    # Feature Engineering - Stage 1 (Structural Features from CSV + CIF)
    # -----------------------------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("Extracting Stage 1 structural features (CSV + CIF hybrid)...")
    
    # CSV-derived structural metadata (available for all rows)
    csv_train_struct = stage1_csv_structural_features(base_train_df.copy())
    csv_test_struct = stage1_csv_structural_features(base_test_df.copy())
    
    # CIF-derived advanced structural features (available subset)
    struct_train_df, train_stats = stage1_structural_features(
        base_train_df.copy(), id_col="ID", cif_dir=train_cif_dir, extended=True, verbose=False
    )
    struct_test_df, test_stats = stage1_structural_features(
        base_test_df.copy(), id_col="ID", cif_dir=test_cif_dir, extended=True, verbose=False
    )
    
    logger.info(f"Stage 1 CIF parsing statistics:")
    logger.info(f"  Training -> full: {train_stats['parsed_full']}, partial: {train_stats['parsed_partial']}, missing: {train_stats['missing_cif']}")
    logger.info(f"  Test     -> full: {test_stats['parsed_full']}, partial: {test_stats['parsed_partial']}, missing: {test_stats['missing_cif']}")
    
    # Merge CSV metadata into CIF DataFrame (fill gaps)
    csv_struct_cols = [
        "spacegroup_number", "formula_units_z",
        "lattice_a", "lattice_b", "lattice_c",
        "lattice_alpha", "lattice_beta", "lattice_gamma",
        "lattice_volume", "lattice_anisotropy", "angle_deviation_from_ortho", "is_cubic_like",
        "density", "volume_per_atom", "n_li_sites"
    ]
    
    for col in csv_struct_cols:
        if col in csv_train_struct.columns:
            if col in struct_train_df.columns:
                struct_train_df[col] = struct_train_df[col].fillna(csv_train_struct[col])
            else:
                struct_train_df[col] = csv_train_struct[col]
        if col in csv_test_struct.columns:
            if col in struct_test_df.columns:
                struct_test_df[col] = struct_test_df[col].fillna(csv_test_struct[col])
            else:
                struct_test_df[col] = csv_test_struct[col]
    
    # Indicator for CIF availability
    struct_train_df["has_cif_struct"] = struct_train_df["ID"].isin(train_cif_files).astype(int)
    struct_test_df["has_cif_struct"] = struct_test_df["ID"].isin(test_cif_files).astype(int)
    
    # CIF-only features: fill missing with 0 so they contribute only when available
    cif_only_cols = [
        "li_fraction", "li_concentration", "framework_density",
        "li_li_min_dist", "li_li_avg_dist", "li_anion_min_dist",
        "li_coordination_avg", "li_site_avg_multiplicity",
        "lattice_anisotropy_bc_a", "lattice_anisotropy_max_min"
    ]
    for col in cif_only_cols:
        if col in struct_train_df.columns:
            struct_train_df[col] = struct_train_df[col].fillna(0)
        if col in struct_test_df.columns:
            struct_test_df[col] = struct_test_df[col].fillna(0)
    
    # Add space group one-hot encoding
    full_train_df = stage1_spacegroup_onehot(struct_train_df.copy(), spacegroup_col="spacegroup_number")
    full_test_df = stage1_spacegroup_onehot(struct_test_df.copy(), spacegroup_col="spacegroup_number")
    
    # -----------------------------------------------------------------------------------------
    # Feature Engineering - Stage 2 (Physics-Informed Features)
    # -----------------------------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("Extracting Stage 2 physics-informed features...")
    
    physics_train_df, train_physics_stats = stage2_physics_features(
        full_train_df.copy(), id_col="ID", cif_dir=train_cif_dir, verbose=False
    )
    physics_test_df, test_physics_stats = stage2_physics_features(
        full_test_df.copy(), id_col="ID", cif_dir=test_cif_dir, verbose=False
    )
    
    logger.info(f"Stage 2 physics feature extraction:")
    logger.info(f"  Training -> processed: {train_physics_stats['processed']}, with BV: {train_physics_stats['with_bv']}, with Ewald: {train_physics_stats['with_ewald']}, with Voronoi: {train_physics_stats['with_voronoi']}")
    logger.info(f"  Test     -> processed: {test_physics_stats['processed']}, with BV: {test_physics_stats['with_bv']}, with Ewald: {test_physics_stats['with_ewald']}, with Voronoi: {test_physics_stats['with_voronoi']}")
    
    # Physics feature columns: values (zero-fill NaN) and indicators (already 0/1)
    physics_value_cols = [
        "bv_mismatch_avg", "bv_mismatch_std",
        "ewald_energy_avg", "ewald_energy_std",
        "li_voronoi_cn_avg",
    ]
    physics_indicator_cols = [
        "has_bv_mismatch", "has_ewald_energy", "has_voronoi_cn",
    ]
    physics_feature_cols = physics_value_cols + physics_indicator_cols
    
    # Zero-fill physics VALUE features for samples without valid calculations
    # Indicators are already 0/1 (0 = zero-filled, 1 = calculated)
    for col in physics_value_cols:
        if col in physics_train_df.columns:
            physics_train_df[col] = physics_train_df[col].fillna(0)
        if col in physics_test_df.columns:
            physics_test_df[col] = physics_test_df[col].fillna(0)
    
    # Fill indicator columns with 0 where NaN (shouldn't happen, but be safe)
    for col in physics_indicator_cols:
        if col in physics_train_df.columns:
            physics_train_df[col] = physics_train_df[col].fillna(0).astype(int)
        if col in physics_test_df.columns:
            physics_test_df[col] = physics_test_df[col].fillna(0).astype(int)
    
    # Log physics feature coverage (using indicator columns for accurate reporting)
    logger.info("Physics feature coverage (training) - based on indicator variables:")
    for indicator_col in physics_indicator_cols:
        if indicator_col in physics_train_df.columns:
            calculated = physics_train_df[indicator_col].sum()
            logger.info(f"  {indicator_col}: {calculated}/{len(physics_train_df)} ({100*calculated/len(physics_train_df):.1f}%)")
    
    # Also log the value columns for reference
    logger.info("Physics value features (non-zero after zero-fill):")
    for col in physics_value_cols:
        if col in physics_train_df.columns:
            non_zero = (physics_train_df[col] != 0).sum()
            logger.info(f"  {col}: {non_zero}/{len(physics_train_df)} ({100*non_zero/len(physics_train_df):.1f}%)")
    
    # -----------------------------------------------------------------------------------------
    # Define Feature Sets for Experiments
    # -----------------------------------------------------------------------------------------
    
    # Basic structural columns (no space group one-hot)
    basic_struct_cols = [
        "density", "volume_per_atom", "n_li_sites", "n_total_atoms"
    ]
    
    # Full geometry columns (same as stage1_geometry)
    geometry_cols = basic_struct_cols + [
        "lattice_a", "lattice_b", "lattice_c",
        "lattice_alpha", "lattice_beta", "lattice_gamma",
        "lattice_anisotropy_bc_a", "lattice_anisotropy_max_min",
        "angle_deviation_from_ortho", "is_cubic_like",
        "li_fraction", "li_concentration", "framework_density",
        "li_li_min_dist", "li_li_avg_dist", "li_anion_min_dist",
        "li_coordination_avg", "li_site_avg_multiplicity",
        "has_cif_struct"
    ]
    
    # Space group one-hot columns
    spacegroup_cols = [f"sg_{i}" for i in range(1, 231)]
    
    # Stage 1 full_struct = Stage 0 + geometry + space group one-hot
    stage1_full_struct_cols = stage0_feature_cols + geometry_cols + spacegroup_cols
    
    # Stage 2 physics = Stage 1 full_struct + physics features
    stage2_physics_cols = stage1_full_struct_cols + physics_feature_cols
    
    logger.info(f"Feature set sizes:")
    logger.info(f"  Stage 0 features: {len(stage0_feature_cols)}")
    logger.info(f"  Geometry features: {len(geometry_cols)}")
    logger.info(f"  Space group one-hot: {len(spacegroup_cols)}")
    logger.info(f"  Baseline (stage1_full_struct): {len(stage1_full_struct_cols)}")
    logger.info(f"  Physics features: {len(physics_feature_cols)}")
    logger.info(f"  Full (baseline + physics): {len(stage2_physics_cols)}")
    
    # -----------------------------------------------------------------------------------------
    # Cross-Validation Setup
    # -----------------------------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("Setting up cross-validation...")
    
    group_col = "group" if "group" in physics_train_df.columns else None
    y = physics_train_df[TARGET_COL]
    
    if group_col:
        groups = physics_train_df[group_col].values
        logger.info(f"Using column '{group_col}' for GroupKFold cross-validation.")
    else:
        groups = np.arange(len(physics_train_df))
        logger.info("No group column found. Using standard KFold behavior.")
    
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X=physics_train_df, y=y, groups=groups))
    logger.info(f"Created {len(splits)} CV splits.")
    
    # -----------------------------------------------------------------------------------------
    # Define Experiments
    # -----------------------------------------------------------------------------------------
    experiments = {
        "stage2_baseline": {
            "description": "Stage 1 full_struct replica (validation) — this is Model 1 in the double-model scheme",
            "train_df": full_train_df,
            "test_df": full_test_df,
            "feature_cols": stage1_full_struct_cols,
        },
        "stage2_physics": {
            "description": "Baseline + physics features (BV, Ewald, Voronoi) — this is Model 2 in the double-model scheme",
            "train_df": physics_train_df,
            "test_df": physics_test_df,
            "feature_cols": stage2_physics_cols,
        },
    }
    
    # -----------------------------------------------------------------------------------------
    # Run Experiments
    # -----------------------------------------------------------------------------------------
    all_experiment_metrics = {}
    all_experiment_oof = {}
    all_experiment_test_preds = {}
    
    for exp_name, exp_config in experiments.items():
        logger.info("=" * 80)
        logger.info(f"Running Experiment: {exp_name}")
        logger.info(f"Description: {exp_config['description']}")
        
        train_exp = exp_config["train_df"]
        test_exp = exp_config["test_df"]
        feature_cols = exp_config["feature_cols"]
        
        # Filter to available columns
        available_cols = [c for c in feature_cols if c in train_exp.columns]
        logger.info(f"Using {len(available_cols)} features.")
        
        # Verify key indicator features
        if "sigma_is_coerced" in available_cols:
            logger.info(f"  [OK] sigma_is_coerced included in training features")
        if "has_cif_struct" in available_cols:
            logger.info(f"  [OK] has_cif_struct included in training features")
        
        # Prepare feature matrix
        X = train_exp[available_cols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.fillna(0)
        
        # Model configuration
        config = ExperimentConfig(
            model_name="hgbt",
            n_splits=5,
            random_state=42,
            params={},
            group_col=group_col
        )
        
        # Run cross-validation
        fold_scores, overall_metrics, oof = run_cv_with_predefined_splits(X, y, splits, config)
        
        # Log results
        logger.info(f"[{exp_name}] Fold RMSEs: {[f'{s:.4f}' for s in fold_scores]}")
        logger.info(f"[{exp_name}] Overall CV Metrics:")
        logger.info(f"    R2:   {overall_metrics['r2']:.4f}")
        logger.info(f"    RMSE: {overall_metrics['rmse']:.4f}")
        logger.info(f"    MAE:  {overall_metrics['mae']:.4f}")
        
        all_experiment_metrics[exp_name] = overall_metrics
        all_experiment_oof[exp_name] = oof
        
        # Save CV parity plot
        parity_plot_path = os.path.join(results_dir, f"{exp_name}_cv_parity.png")
        save_parity_plot(
            y.values,
            oof,
            parity_plot_path,
            title=f"Stage 2 5-Fold CV: {exp_name}",
            r2_linear_from_log=False,
        )
        
        # Train final model and predict on test set
        logger.info(f"[{exp_name}] Training final model on full training set...")
        
        X_test = test_exp[[c for c in available_cols if c in test_exp.columns]].copy()
        X_test = X_test.reindex(columns=X.columns, fill_value=0)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test = X_test.fillna(0)
        
        preds = fit_full_and_predict(X, y, X_test, config)
        
        # Save predictions
        output_path = os.path.join(results_dir, f"{exp_name}_predictions.csv")
        out_df = pd.DataFrame({"ID": test_df["ID"], "prediction": preds})
        save_dataframe(out_df, output_path)
        logger.info(f"[{exp_name}] Predictions saved to: {output_path}")
        all_experiment_test_preds[exp_name] = preds
        
        # Test-set parity plot
        if TARGET_COL in test_df.columns:
            test_y = test_df[TARGET_COL].values
            test_parity_path = os.path.join(results_dir, f"{exp_name}_test_parity.png")
            save_parity_plot(
                test_y,
                preds,
                test_parity_path,
                title=f"Stage 2 Test: {exp_name}",
                r2_linear_from_log=False,
            )
            
            # Log test-set metrics analogous to CV metrics
            test_metrics = compute_regression_metrics(test_y, preds)
            logger.info(f"[{exp_name}] Test Metrics:")
            logger.info(f"    R2:   {test_metrics['r2']:.4f}")
            logger.info(f"    RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"    MAE:  {test_metrics['mae']:.4f}")
    
    # -----------------------------------------------------------------------------------------
    # Double-Model Strategies: Model 1 (baseline) + Model 2 (physics-informed)
    # -----------------------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Double-model strategies: combining baseline (Model 1) with physics-informed models (Model 2)")
    
    try:
        # Define a simple coverage score: count of successfully computed physics blocks
        train_coverage_score = physics_train_df[physics_indicator_cols].sum(axis=1)
        test_coverage_score = physics_test_df[physics_indicator_cols].sum(axis=1)
        
        # Threshold: require ALL 3 physics feature groups to be present
        coverage_threshold = 3
        train_has_good_physics = train_coverage_score >= coverage_threshold
        test_has_good_physics = test_coverage_score >= coverage_threshold
        
        frac_good_train = float(train_has_good_physics.mean())
        frac_good_test = float(test_has_good_physics.mean())
        frac_baseline_train = 1.0 - frac_good_train
        frac_baseline_test = 1.0 - frac_good_test
        
        logger.info(
            "Physics coverage gating:"
            f" threshold = {coverage_threshold} active indicators"
            f" -> Model 2 (physics-informed) will be used for {frac_good_train*100:.1f}% of training samples"
            f" and {frac_good_test*100:.1f}% of test samples; "
            f"Model 1 (baseline) will handle the remaining {frac_baseline_train*100:.1f}% (train)"
            f" and {frac_baseline_test*100:.1f}% (test)."
        )
        
        if "stage2_baseline" in all_experiment_oof and "stage2_physics" in all_experiment_oof:
            # Common data for all double-model variants
            oof_baseline = all_experiment_oof["stage2_baseline"]
            oof_physics_full = all_experiment_oof["stage2_physics"]
            preds_baseline = all_experiment_test_preds.get("stage2_baseline")
            preds_physics_full = all_experiment_test_preds.get("stage2_physics")
            
            # Sanity check on predictions
            if preds_baseline is None or preds_physics_full is None:
                logger.warning("Missing baseline or physics test predictions; skipping double-model strategies.")
            else:
                # ---------------------------------------------------------------------------------
                # Strategy A: Full-train double-model gating (existing approach, renamed)
                # ---------------------------------------------------------------------------------
                logger.info("-" * 80)
                logger.info("Strategy A: stage2_double_model_fulltrain (physics model trained on full dataset)")
                
                oof_fulltrain = np.where(train_has_good_physics.values, oof_physics_full, oof_baseline)
                fulltrain_metrics = compute_regression_metrics(y.values, oof_fulltrain)
                
                logger.info("stage2_double_model_fulltrain CV metrics:")
                logger.info(f"  R2:   {fulltrain_metrics['r2']:.4f}")
                logger.info(f"  RMSE: {fulltrain_metrics['rmse']:.4f}")
                logger.info(f"  MAE:  {fulltrain_metrics['mae']:.4f}")
                
                all_experiment_metrics["stage2_double_model_fulltrain"] = fulltrain_metrics
                
                fulltrain_parity_path = os.path.join(results_dir, "stage2_double_model_fulltrain_cv_parity.png")
                save_parity_plot(
                    y.values,
                    oof_fulltrain,
                    fulltrain_parity_path,
                    title="Stage 2 5-Fold CV: double-model fulltrain (baseline + physics)",
                    r2_linear_from_log=False,
                )
                
                test_preds_fulltrain = np.where(test_has_good_physics.values, preds_physics_full, preds_baseline)
                fulltrain_output_path = os.path.join(results_dir, "stage2_double_model_fulltrain_predictions.csv")
                fulltrain_out_df = pd.DataFrame({"ID": test_df["ID"], "prediction": test_preds_fulltrain})
                save_dataframe(fulltrain_out_df, fulltrain_output_path)
                logger.info(f"stage2_double_model_fulltrain predictions saved to: {fulltrain_output_path}")
                
                if TARGET_COL in test_df.columns:
                    test_y = test_df[TARGET_COL].values
                    fulltrain_test_parity_path = os.path.join(results_dir, "stage2_double_model_fulltrain_test_parity.png")
                    save_parity_plot(
                        test_y,
                        test_preds_fulltrain,
                        fulltrain_test_parity_path,
                        title="Stage 2 Test: double-model fulltrain (baseline + physics)",
                        r2_linear_from_log=False,
                    )
                    
                    fulltrain_test_metrics = compute_regression_metrics(test_y, test_preds_fulltrain)
                    logger.info("stage2_double_model_fulltrain Test metrics:")
                    logger.info(f"  R2:   {fulltrain_test_metrics['r2']:.4f}")
                    logger.info(f"  RMSE: {fulltrain_test_metrics['rmse']:.4f}")
                    logger.info(f"  MAE:  {fulltrain_test_metrics['mae']:.4f}")
                
                # ---------------------------------------------------------------------------------
                # Strategy B: Physics model trained ONLY on good-physics subset (direct target y)
                #            + gating between baseline and this subset-trained model
                # ---------------------------------------------------------------------------------
                logger.info("-" * 80)
                logger.info("Strategy B: stage2_double_model_subsettrain (Model 2 trained only where all 3 physics indicators are 1)")
                
                physics_exp = experiments["stage2_physics"]
                train_physics = physics_exp["train_df"]
                test_physics = physics_exp["test_df"]
                available_physics_cols = [c for c in physics_exp["feature_cols"] if c in train_physics.columns]
                
                X_physics_train = train_physics[available_physics_cols].copy()
                X_physics_train.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_physics_train = X_physics_train.fillna(0)
                
                X_physics_test = test_physics[available_physics_cols].copy()
                X_physics_test.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_physics_test = X_physics_test.fillna(0)
                
                # Subset indices where all physics indicators are present
                subset_train_indices = np.where(train_has_good_physics.values)[0]
                subset_test_indices = np.where(test_has_good_physics.values)[0]
                
                if len(subset_train_indices) > 0:
                    X_subset_train = X_physics_train.iloc[subset_train_indices]
                    y_subset = y.iloc[subset_train_indices]
                    
                    # Build GroupKFold splits on the subset only
                    if group_col and group_col in train_physics.columns:
                        subset_groups = train_physics.iloc[subset_train_indices][group_col].values
                    else:
                        subset_groups = np.arange(len(X_subset_train))
                    
                    gkf_subset = GroupKFold(n_splits=5)
                    subset_splits = list(gkf_subset.split(X_subset_train, y_subset, groups=subset_groups))
                    
                    subset_config = ExperimentConfig(
                        model_name="hgbt",
                        n_splits=5,
                        random_state=42,
                        params={},
                        group_col=group_col,
                    )
                    
                    subset_fold_scores, subset_model_metrics, subset_oof = run_cv_with_predefined_splits(
                        X_subset_train, y_subset, subset_splits, subset_config
                    )
                    
                    logger.info(f"  Physics-only subset model CV RMSEs: {[f'{s:.4f}' for s in subset_fold_scores]}")
                    logger.info(
                        f"  Physics-only subset model CV metrics: "
                        f"R2={subset_model_metrics['r2']:.4f}, "
                        f"RMSE={subset_model_metrics['rmse']:.4f}, "
                        f"MAE={subset_model_metrics['mae']:.4f}"
                    )
                    
                    # Double-model gating for Strategy B: baseline everywhere, but replace
                    # baseline predictions with subset-model predictions where physics is good
                    oof_subsettrain = oof_baseline.copy()
                    oof_subsettrain[subset_train_indices] = subset_oof
                    subsettrain_metrics = compute_regression_metrics(y.values, oof_subsettrain)
                    
                    logger.info("stage2_double_model_subsettrain CV metrics:")
                    logger.info(f"  R2:   {subsettrain_metrics['r2']:.4f}")
                    logger.info(f"  RMSE: {subsettrain_metrics['rmse']:.4f}")
                    logger.info(f"  MAE:  {subsettrain_metrics['mae']:.4f}")
                    
                    all_experiment_metrics["stage2_double_model_subsettrain"] = subsettrain_metrics
                    
                    subsettrain_parity_path = os.path.join(results_dir, "stage2_double_model_subsettrain_cv_parity.png")
                    save_parity_plot(
                        y.values,
                        oof_subsettrain,
                        subsettrain_parity_path,
                        title="Stage 2 5-Fold CV: double-model subsettrain (baseline + physics subset)",
                        r2_linear_from_log=False,
                    )
                    
                    # Final subset-trained physics model and test-set predictions
                    X_subset_train_full = X_subset_train  # all subset rows
                    y_subset_full = y_subset
                    X_subset_test = X_physics_test.iloc[subset_test_indices] if len(subset_test_indices) > 0 else None
                    
                    if X_subset_test is not None and len(X_subset_test) > 0:
                        subset_preds_test = fit_full_and_predict(
                            X_subset_train_full,
                            y_subset_full,
                            X_subset_test,
                            subset_config,
                        )
                        
                        test_preds_subsettrain = preds_baseline.copy()
                        test_preds_subsettrain[subset_test_indices] = subset_preds_test
                        
                        subsettrain_output_path = os.path.join(results_dir, "stage2_double_model_subsettrain_predictions.csv")
                        subsettrain_out_df = pd.DataFrame({"ID": test_df["ID"], "prediction": test_preds_subsettrain})
                        save_dataframe(subsettrain_out_df, subsettrain_output_path)
                        logger.info(f"stage2_double_model_subsettrain predictions saved to: {subsettrain_output_path}")
                        
                        if TARGET_COL in test_df.columns:
                            test_y = test_df[TARGET_COL].values
                            subsettrain_test_parity_path = os.path.join(results_dir, "stage2_double_model_subsettrain_test_parity.png")
                            save_parity_plot(
                                test_y,
                                test_preds_subsettrain,
                                subsettrain_test_parity_path,
                                title="Stage 2 Test: double-model subsettrain (baseline + physics subset)",
                                r2_linear_from_log=False,
                            )
                            
                            subsettrain_test_metrics = compute_regression_metrics(test_y, test_preds_subsettrain)
                            logger.info("stage2_double_model_subsettrain Test metrics:")
                            logger.info(f"  R2:   {subsettrain_test_metrics['r2']:.4f}")
                            logger.info(f"  RMSE: {subsettrain_test_metrics['rmse']:.4f}")
                            logger.info(f"  MAE:  {subsettrain_test_metrics['mae']:.4f}")
                    else:
                        logger.warning("No test samples with full physics coverage; skipping subsettrain test predictions.")
                else:
                    logger.warning("No training samples with full physics coverage; skipping subsettrain strategy.")
                
                # ---------------------------------------------------------------------------------
                # Strategy C: Residual correction model on good-physics subset
                #             (Model 2 predicts residual y - y_baseline, then added back)
                # ---------------------------------------------------------------------------------
                logger.info("-" * 80)
                logger.info("Strategy C: stage2_double_model_residual (physics residual model on good-physics subset)")
                
                if len(subset_train_indices) > 0:
                    X_subset_train = X_physics_train.iloc[subset_train_indices]
                    y_subset = y.iloc[subset_train_indices]
                    baseline_subset_oof = oof_baseline[subset_train_indices]
                    
                    # Residual target: what physics should correct on top of baseline
                    residual_target = y_subset.values - baseline_subset_oof
                    
                    # Build GroupKFold splits on the subset only (same as Strategy B)
                    if group_col and group_col in train_physics.columns:
                        subset_groups = train_physics.iloc[subset_train_indices][group_col].values
                    else:
                        subset_groups = np.arange(len(X_subset_train))
                    
                    gkf_subset_resid = GroupKFold(n_splits=5)
                    subset_resid_splits = list(gkf_subset_resid.split(X_subset_train, residual_target, groups=subset_groups))
                    
                    resid_config = ExperimentConfig(
                        model_name="hgbt",
                        n_splits=5,
                        random_state=42,
                        params={},
                        group_col=group_col,
                    )
                    
                    resid_fold_scores, resid_model_metrics, resid_oof = run_cv_with_predefined_splits(
                        X_subset_train,
                        pd.Series(residual_target),
                        subset_resid_splits,
                        resid_config,
                    )
                    
                    logger.info(f"  Residual model CV RMSEs (on residual target): {[f'{s:.4f}' for s in resid_fold_scores]}")
                    logger.info(
                        f"  Residual model CV metrics (on residual target): "
                        f"R2={resid_model_metrics['r2']:.4f}, "
                        f"RMSE={resid_model_metrics['rmse']:.4f}, "
                        f"MAE={resid_model_metrics['mae']:.4f}"
                    )
                    
                    # Double-model residual gating: baseline everywhere, plus residual on good-physics subset
                    oof_residual = oof_baseline.copy()
                    oof_residual[subset_train_indices] = baseline_subset_oof + resid_oof
                    residual_metrics = compute_regression_metrics(y.values, oof_residual)
                    
                    logger.info("stage2_double_model_residual CV metrics:")
                    logger.info(f"  R2:   {residual_metrics['r2']:.4f}")
                    logger.info(f"  RMSE: {residual_metrics['rmse']:.4f}")
                    logger.info(f"  MAE:  {residual_metrics['mae']:.4f}")
                    
                    all_experiment_metrics["stage2_double_model_residual"] = residual_metrics
                    
                    residual_parity_path = os.path.join(results_dir, "stage2_double_model_residual_cv_parity.png")
                    save_parity_plot(
                        y.values,
                        oof_residual,
                        residual_parity_path,
                        title="Stage 2 5-Fold CV: double-model residual (baseline + physics residual)",
                        r2_linear_from_log=False,
                    )
                    
                    # Final residual model and test-set predictions
                    X_subset_train_full = X_subset_train
                    residual_target_full = residual_target
                    X_subset_test = X_physics_test.iloc[subset_test_indices] if len(subset_test_indices) > 0 else None
                    
                    if X_subset_test is not None and len(X_subset_test) > 0:
                        resid_preds_test = fit_full_and_predict(
                            X_subset_train_full,
                            pd.Series(residual_target_full),
                            X_subset_test,
                            resid_config,
                        )
                        
                        test_preds_residual = preds_baseline.copy()
                        # baseline + residual correction where physics is good
                        test_preds_residual[subset_test_indices] = (
                            preds_baseline[subset_test_indices] + resid_preds_test
                        )
                        
                        residual_output_path = os.path.join(results_dir, "stage2_double_model_residual_predictions.csv")
                        residual_out_df = pd.DataFrame({"ID": test_df["ID"], "prediction": test_preds_residual})
                        save_dataframe(residual_out_df, residual_output_path)
                        logger.info(f"stage2_double_model_residual predictions saved to: {residual_output_path}")
                        
                        if TARGET_COL in test_df.columns:
                            test_y = test_df[TARGET_COL].values
                            residual_test_parity_path = os.path.join(results_dir, "stage2_double_model_residual_test_parity.png")
                            save_parity_plot(
                                test_y,
                                test_preds_residual,
                                residual_test_parity_path,
                                title="Stage 2 Test: double-model residual (baseline + physics residual)",
                                r2_linear_from_log=False,
                            )
                            
                            residual_test_metrics = compute_regression_metrics(test_y, test_preds_residual)
                            logger.info("stage2_double_model_residual Test metrics:")
                            logger.info(f"  R2:   {residual_test_metrics['r2']:.4f}")
                            logger.info(f"  RMSE: {residual_test_metrics['rmse']:.4f}")
                            logger.info(f"  MAE:  {residual_test_metrics['mae']:.4f}")
                    else:
                        logger.warning("No test samples with full physics coverage; skipping residual strategy predictions.")
                else:
                    logger.warning("No training samples with full physics coverage; skipping residual strategy.")
                
                # ---------------------------------------------------------------------------------
                # Strategy D: Learned combiner on top of residual model (stacking on good-physics subset)
                # ---------------------------------------------------------------------------------
                logger.info("-" * 80)
                logger.info("Strategy D: stage2_double_model_residual_stack (meta-model combining baseline and residual-corrected predictions on good-physics subset)")
                
                if len(subset_train_indices) > 0 and 'test_preds_residual' in locals():
                    # Build meta-model features on good-physics subset (train)
                    # Features: baseline OOF prediction, residual-corrected OOF prediction
                    meta_train_baseline = oof_baseline[subset_train_indices]
                    meta_train_residual = oof_residual[subset_train_indices]
                    
                    meta_train_df = pd.DataFrame({
                        "baseline_pred": meta_train_baseline,
                        "residual_pred": meta_train_residual,
                    })
                    
                    # Optionally add a few physics scalars for extra flexibility
                    for extra_col in ["ewald_energy_avg", "li_voronoi_cn_avg", "bv_mismatch_avg"]:
                        if extra_col in physics_train_df.columns:
                            meta_train_df[extra_col] = physics_train_df.iloc[subset_train_indices][extra_col].values
                    
                    y_meta = y.iloc[subset_train_indices]
                    
                    # Build GroupKFold splits on the subset (same grouping as before)
                    if group_col and group_col in physics_train_df.columns:
                        subset_groups = physics_train_df.iloc[subset_train_indices][group_col].values
                    else:
                        subset_groups = np.arange(len(meta_train_df))
                    
                    gkf_meta = GroupKFold(n_splits=5)
                    meta_splits = list(gkf_meta.split(meta_train_df, y_meta, groups=subset_groups))
                    
                    meta_config = ExperimentConfig(
                        model_name="hgbt",
                        n_splits=5,
                        random_state=42,
                        params={},
                        group_col=None,  # groups are handled via predefined splits
                    )
                    
                    meta_fold_scores, meta_model_metrics, meta_oof = run_cv_with_predefined_splits(
                        meta_train_df,
                        y_meta,
                        meta_splits,
                        meta_config,
                    )
                    
                    logger.info(f"  Residual-stack meta-model CV RMSEs: {[f'{s:.4f}' for s in meta_fold_scores]}")
                    logger.info(
                        f"  Residual-stack meta-model CV metrics (on y): "
                        f"R2={meta_model_metrics['r2']:.4f}, "
                        f"RMSE={meta_model_metrics['rmse']:.4f}, "
                        f"MAE={meta_model_metrics['mae']:.4f}"
                    )
                    
                    # Global CV prediction: baseline everywhere, meta-model where physics is good
                    oof_residual_stack = oof_baseline.copy()
                    oof_residual_stack[subset_train_indices] = meta_oof
                    residual_stack_metrics = compute_regression_metrics(y.values, oof_residual_stack)
                    
                    logger.info("stage2_double_model_residual_stack CV metrics:")
                    logger.info(f"  R2:   {residual_stack_metrics['r2']:.4f}")
                    logger.info(f"  RMSE: {residual_stack_metrics['rmse']:.4f}")
                    logger.info(f"  MAE:  {residual_stack_metrics['mae']:.4f}")
                    
                    all_experiment_metrics["stage2_double_model_residual_stack"] = residual_stack_metrics
                    
                    residual_stack_parity_path = os.path.join(results_dir, "stage2_double_model_residual_stack_cv_parity.png")
                    save_parity_plot(
                        y.values,
                        oof_residual_stack,
                        residual_stack_parity_path,
                        title="Stage 2 5-Fold CV: double-model residual_stack (baseline + learned combiner)",
                        r2_linear_from_log=False,
                    )
                    
                    # Final meta-model and test-set predictions (on good-physics subset)
                    subset_test_indices = np.where(test_has_good_physics.values)[0]
                    if len(subset_test_indices) > 0:
                        meta_test_baseline = preds_baseline[subset_test_indices]
                        meta_test_residual = test_preds_residual[subset_test_indices]
                        
                        meta_test_df = pd.DataFrame({
                            "baseline_pred": meta_test_baseline,
                            "residual_pred": meta_test_residual,
                        })
                        
                        for extra_col in ["ewald_energy_avg", "li_voronoi_cn_avg", "bv_mismatch_avg"]:
                            if extra_col in physics_test_df.columns:
                                meta_test_df[extra_col] = physics_test_df.iloc[subset_test_indices][extra_col].values
                        
                        meta_preds_test = fit_full_and_predict(
                            meta_train_df,
                            y_meta,
                            meta_test_df,
                            meta_config,
                        )
                        
                        test_preds_residual_stack = preds_baseline.copy()
                        test_preds_residual_stack[subset_test_indices] = meta_preds_test
                        
                        residual_stack_output_path = os.path.join(results_dir, "stage2_double_model_residual_stack_predictions.csv")
                        residual_stack_out_df = pd.DataFrame({"ID": test_df["ID"], "prediction": test_preds_residual_stack})
                        save_dataframe(residual_stack_out_df, residual_stack_output_path)
                        logger.info(f"stage2_double_model_residual_stack predictions saved to: {residual_stack_output_path}")
                        
                        if TARGET_COL in test_df.columns:
                            test_y = test_df[TARGET_COL].values
                            residual_stack_test_parity_path = os.path.join(results_dir, "stage2_double_model_residual_stack_test_parity.png")
                            save_parity_plot(
                                test_y,
                                test_preds_residual_stack,
                                residual_stack_test_parity_path,
                                title="Stage 2 Test: double-model residual_stack (baseline + learned combiner)",
                                r2_linear_from_log=False,
                            )
                            
                            residual_stack_test_metrics = compute_regression_metrics(test_y, test_preds_residual_stack)
                            logger.info("stage2_double_model_residual_stack Test metrics:")
                            logger.info(f"  R2:   {residual_stack_test_metrics['r2']:.4f}")
                            logger.info(f"  RMSE: {residual_stack_test_metrics['rmse']:.4f}")
                            logger.info(f"  MAE:  {residual_stack_test_metrics['mae']:.4f}")
                    else:
                        logger.warning("No test samples with full physics coverage; skipping residual_stack test predictions.")
                else:
                    logger.warning("Not enough data to fit residual_stack meta-model; skipping Strategy D.")
        else:
            logger.warning("Cannot construct double-model strategies: missing baseline or physics OOF predictions.")
    except Exception as e:
        logger.warning(f"Double-model strategies step failed: {e}")
    
    # -----------------------------------------------------------------------------------------
    # Comparison with Stage 1
    # -----------------------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Comparison with Stage 1 (stage1_full_struct):")
    logger.info(f"  Reference: R2={stage1_ref['r2']:.4f}, RMSE={stage1_ref['rmse']:.4f}, MAE={stage1_ref['mae']:.4f}")
    
    for exp_name, m in all_experiment_metrics.items():
        r2_diff = m['r2'] - stage1_ref['r2']
        rmse_diff = m['rmse'] - stage1_ref['rmse']
        mae_diff = m['mae'] - stage1_ref['mae']
        
        logger.info(f"  {exp_name}:")
        logger.info(f"    R2:   {m['r2']:.4f} (diff: {r2_diff:+.4f})")
        logger.info(f"    RMSE: {m['rmse']:.4f} (diff: {rmse_diff:+.4f})")
        logger.info(f"    MAE:  {m['mae']:.4f} (diff: {mae_diff:+.4f})")
        
        # Check if baseline matches Stage 1
        if exp_name == "stage2_baseline":
            if abs(r2_diff) < 0.001 and abs(rmse_diff) < 0.001 and abs(mae_diff) < 0.001:
                logger.info(f"    [PASS] Results match Stage 1 within tolerance!")
            else:
                logger.warning(f"    [WARN] Results differ from Stage 1 reference")
    
    # -----------------------------------------------------------------------------------------
    # Feature Importance Analysis (Physics Features)
    # -----------------------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("FEATURE IMPORTANCE ANALYSIS (Physics Features)")
    logger.info("-" * 80)
    
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        
        # Use physics experiment's full feature set
        physics_exp = experiments["stage2_physics"]
        train_physics = physics_exp["train_df"]
        available_physics_cols = [c for c in physics_exp["feature_cols"] if c in train_physics.columns]
        
        X_physics = train_physics[available_physics_cols].copy()
        X_physics.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_physics = X_physics.fillna(0)
        
        # Fit model and get permutation importance
        model = HistGradientBoostingRegressor(random_state=42)
        model.fit(X_physics, y)
        
        perm_importance = permutation_importance(model, X_physics, y, n_repeats=10, random_state=42, n_jobs=-1)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X_physics.columns.tolist(),
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        
        # Log physics feature importances
        logger.info("Physics Feature Importances:")
        physics_importance = importance_df[importance_df['feature'].isin(physics_feature_cols)]
        for _, row in physics_importance.iterrows():
            logger.info(f"  {row['feature']:35s} | {row['importance']:.4f} +/- {row['std']:.4f}")
        
        # Save feature importance plot
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Top 20 overall
            top20 = importance_df.head(20)
            ax1 = axes[0]
            colors = ['#e74c3c' if f in physics_feature_cols else '#3498db' for f in top20['feature']]
            ax1.barh(range(len(top20)), top20['importance'], xerr=top20['std'], color=colors, alpha=0.8)
            ax1.set_yticks(range(len(top20)))
            ax1.set_yticklabels(top20['feature'])
            ax1.invert_yaxis()
            ax1.set_xlabel('Permutation Importance')
            ax1.set_title('Top 20 Features (Red=Physics, Blue=Other)')
            ax1.grid(axis='x', alpha=0.3)
            
            # Physics only
            ax2 = axes[1]
            physics_sorted = physics_importance.sort_values('importance', ascending=False)
            ax2.barh(range(len(physics_sorted)), physics_sorted['importance'], xerr=physics_sorted['std'], color='#e74c3c', alpha=0.8)
            ax2.set_yticks(range(len(physics_sorted)))
            ax2.set_yticklabels(physics_sorted['feature'])
            ax2.invert_yaxis()
            ax2.set_xlabel('Permutation Importance')
            ax2.set_title('Physics Feature Importances')
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "stage2_feature_importance.png"), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Feature importance plot saved.")
        except Exception as e:
            logger.warning(f"Could not generate plot: {e}")
        
        importance_df.to_csv(os.path.join(results_dir, "stage2_feature_importance.csv"), index=False)
        
    except Exception as e:
        logger.warning(f"Feature importance analysis failed: {e}")
    
    # -----------------------------------------------------------------------------------------
    # Generate Final Report
    # -----------------------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Generating final report...")
    
    report_path = os.path.join(results_dir, "interim_report_stage2.md")
    generate_stage2_report(report_path, all_experiment_metrics, stage1_ref)
    logger.info(f"Report saved to: {report_path}")
    
    # Print summary
    logger.info("-" * 80)
    logger.info("SUMMARY: Stage 2 Experiment Results")
    logger.info("-" * 80)
    for exp_name, metrics in all_experiment_metrics.items():
        logger.info(f"{exp_name:25s} | R2: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")
    
    logger.info("=" * 80)
    logger.info("Stage 2 pipeline completed successfully!")


if __name__ == "__main__":
    main()
