# =================================================================================================
# Li-ion Mobility Prediction - Stage 2: Physics-Informed Features WITH OPTUNA OPTIMIZATION
# =================================================================================================
# This script implements Stage 2 of the machine learning pipeline for predicting lithium-ion
# conductivity WITH Optuna hyperparameter optimization for all models.
# It builds upon Stage 1's best-performing experiment (stage1_full_struct) and
# adds physics-informed features that probe the local atomic environment around Li ions.
#
# Stage 2 Physics Features (from CIF parsing via pymatgen):
# - Li–anion bond-valence mismatch (using BondValenceAnalyzer)
# - Average Ewald site energy (using EwaldSummation)
# - Li Voronoi coordination number (using VoronoiNN)
#
# ALL MODELS ARE OPTIMIZED WITH OPTUNA BEFORE TRAINING
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
from sklearn.model_selection import GroupKFold, train_test_split
import optuna
from optuna.samplers import TPESampler
import logging


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
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import load_data, clean_data, add_target_log10_sigma, coerce_sigma_series, TARGET_COL
from src.features import (
    stage0_elemental_ratios, 
    stage0_smact_features, 
    stage0_element_embeddings,
    stage1_csv_structural_features,
    stage1_structural_features,
    stage1_spacegroup_onehot,
)
from src.model_training import (
    ExperimentConfig,
    run_cv_with_predefined_splits,
    run_cv_with_predefined_splits_and_gaps,
    fit_full_and_predict,
)


# =================================================================================================
# Helper metrics
# =================================================================================================

def spearman_rho(y_true, y_pred) -> float:
    """
    Compute Spearman rank correlation without requiring scipy.
    """
    s_true = pd.Series(y_true)
    s_pred = pd.Series(y_pred)
    mask = s_true.notna() & s_pred.notna()
    if mask.sum() == 0:
        return float("nan")
    r_true = s_true[mask].rank(method="average")
    r_pred = s_pred[mask].rank(method="average")
    return float(np.corrcoef(r_true, r_pred)[0, 1])


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


# =================================================================================================
# Optuna Hyperparameter Optimization Functions (IMPROVED FOR GENERALIZATION)
# =================================================================================================
#
# Stage 2 analysis showed: Optuna optimized on full CV splits led to overfitting:
# - CV R² improved ~1% but test R² got worse (CV 0.74 vs test 0.54)
# - Hyperparameters were tuned to fit the CV structure, not to generalize
#
# Improvements implemented:
# 1. 5-FOLD CV WITH GENERALIZATION PENALTY: Optuna optimizes on each fold; per-fold objective
#    penalizes large train-val gap (weight=0.2) to favor hyperparameters that generalize.
# 2. CONSERVATIVE SEARCH SPACE: Tighter bounds favor simpler models (shallower trees, more
#    regularization) to reduce overfitting.
# =================================================================================================

# Conservative hyperparameter bounds (reduces overfitting vs. original wide search)
OPTUNA_SEARCH_SPACE = {
    "max_depth": (3, 8),           # was 3-15: shallower trees generalize better
    "learning_rate": (0.01, 0.15),  # was 0.01-0.3: lower LR often more stable
    "max_leaf_nodes": (15, 100),    # was 15-255: fewer leaves = simpler model
    "min_samples_leaf": (5, 35),     # was 1-50: higher min = more regularization
    "l2_regularization": (1e-6, 0.1),  # keep wide for flexibility
    "max_bins": (64, 255),
    "max_iter": (50, 200),          # was 50-300: avoid overfitting with too many iterations
}

# Even simpler search space: aims to find models simpler than sklearn defaults
# (defaults: max_leaf_nodes=31, min_samples_leaf=20, max_depth=None)
SIMPLE_SEARCH_SPACE = {
    "max_depth": (2, 5),            # very shallow
    "learning_rate": (0.02, 0.1),
    "max_leaf_nodes": (10, 31),     # fewer than or equal to default
    "min_samples_leaf": (15, 50),   # higher = more regularization
    "l2_regularization": (1e-4, 0.05),  # stronger regularization
    "max_bins": (64, 128),
    "max_iter": (50, 120),
}

# Hold-out fraction for Optuna validation (never used for final training)
OPTUNA_VAL_FRACTION = 0.2
OPTUNA_VAL_RANDOM_STATE = 42


def create_optuna_objective(
    X_train, y_train, X_val, y_val, exp_name, logger, quantile=None,
    use_generalization_penalty=True, penalty_weight=0.2,
    search_space=None,
):
    """
    Creates an Optuna objective using HOLD-OUT VALIDATION (not CV).
    
    Optuna optimizes on validation set only. This prevents overfitting to the CV structure
    and selects hyperparameters that generalize better to unseen data.
    
    Args:
        X_train: Training feature matrix (80% of data)
        y_train: Training target
        X_val: Validation feature matrix (20% of data) - used for Optuna objective
        y_val: Validation target
        exp_name: Experiment name for logging
        logger: Logger instance
        quantile: If provided, use quantile loss
        use_generalization_penalty: If True, penalize large train-val RMSE gap
        penalty_weight: Weight for generalization penalty (default 0.2)
    
    Returns:
        Objective function for Optuna study
    """
    space = search_space if search_space is not None else OPTUNA_SEARCH_SPACE
    
    def objective(trial: optuna.Trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", space["max_depth"][0], space["max_depth"][1]),
            "learning_rate": trial.suggest_float("learning_rate", space["learning_rate"][0], space["learning_rate"][1], log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", space["max_leaf_nodes"][0], space["max_leaf_nodes"][1]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", space["min_samples_leaf"][0], space["min_samples_leaf"][1]),
            "l2_regularization": trial.suggest_float("l2_regularization", space["l2_regularization"][0], space["l2_regularization"][1], log=True),
            "max_bins": trial.suggest_int("max_bins", space["max_bins"][0], space["max_bins"][1]),
            "max_iter": trial.suggest_int("max_iter", space["max_iter"][0], space["max_iter"][1]),
        }
        
        if quantile is not None:
            params["loss"] = "quantile"
            params["quantile"] = quantile
        
        try:
            # Train on train split, evaluate on validation split (hold-out)
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.pipeline import Pipeline
            
            model = HistGradientBoostingRegressor(random_state=42, **params)
            pipeline = Pipeline(steps=[("model", model)])
            
            features = list(X_train.columns)
            pipeline.fit(X_train[features], y_train)
            
            val_preds = pipeline.predict(X_val[features])
            val_metrics = compute_regression_metrics(y_val.values, val_preds)
            
            if quantile is not None:
                score = val_metrics["mae"]
            else:
                score = val_metrics["rmse"]
            
            # Generalization penalty: penalize if train RMSE is much better than val RMSE
            if use_generalization_penalty and not quantile:
                train_preds = pipeline.predict(X_train[features])
                train_metrics = compute_regression_metrics(y_train.values, train_preds)
                gap = val_metrics["rmse"] - train_metrics["rmse"]
                if gap > 0:  # val worse than train = potential overfitting
                    score = score + penalty_weight * gap
            
            return score
        except Exception as e:
            logger.warning(f"Trial failed with params {params}: {e}")
            return float('inf')
    
    return objective


def optimize_hyperparameters(
    X, y, splits, group_col, exp_name, logger, n_trials=50, quantile=None,
    use_holdout_validation=True, search_space=None
):
    """
    Optimize hyperparameters using Optuna with K-FOLD CV and generalization penalty.
    
    Uses predefined splits. For each fold, computes train and val metrics and applies
    a penalty for large train-val gaps. Optuna minimizes mean (across folds) of:
        val_metric + penalty_weight * max(0, val_metric - train_metric)
    
    Final model is trained on FULL data. No penalty for quantile regression.
    
    Args:
        X: Feature matrix (full training data)
        y: Target vector
        splits: Predefined CV splits (train_idx, val_idx) per fold
        group_col: Group column name (or None)
        exp_name: Experiment name
        logger: Logger instance
        n_trials: Number of Optuna trials
        quantile: If provided, optimize for quantile regression at this quantile (0-1)
        use_holdout_validation: Ignored (kept for API compatibility)
        search_space: Optuna search space dict (or None for default)
    
    Returns:
        Best hyperparameters dict
    """
    quantile_str = f" (quantile={quantile})" if quantile is not None else ""
    penalty_weight = 0.2
    logger.info(
        f"[{exp_name}] Starting Optuna Bayesian optimization ({n_trials} trials, "
        f"5-fold CV with generalization penalty weight={penalty_weight}){quantile_str}..."
    )

    metric = "mae" if quantile is not None else "rmse"
    apply_penalty = quantile is None  # No penalty for quantile regression

    def _cv_objective_with_penalty(trial):
        space = search_space if search_space is not None else OPTUNA_SEARCH_SPACE
        params = {
            "max_depth": trial.suggest_int("max_depth", space["max_depth"][0], space["max_depth"][1]),
            "learning_rate": trial.suggest_float("learning_rate", space["learning_rate"][0], space["learning_rate"][1], log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", space["max_leaf_nodes"][0], space["max_leaf_nodes"][1]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", space["min_samples_leaf"][0], space["min_samples_leaf"][1]),
            "l2_regularization": trial.suggest_float("l2_regularization", space["l2_regularization"][0], space["l2_regularization"][1], log=True),
            "max_bins": trial.suggest_int("max_bins", space["max_bins"][0], space["max_bins"][1]),
            "max_iter": trial.suggest_int("max_iter", space["max_iter"][0], space["max_iter"][1]),
        }
        if quantile is not None:
            params["loss"] = "quantile"
            params["quantile"] = quantile
        try:
            config = ExperimentConfig(
                model_name="hgbt",
                n_splits=5,
                random_state=42,
                params=params,
                group_col=group_col,
            )
            fold_val_scores, overall_metrics, _, train_metric_per_fold = run_cv_with_predefined_splits_and_gaps(
                X, y, splits, config, metric=metric
            )
            if apply_penalty:
                penalized_per_fold = [
                    val_s + penalty_weight * max(0.0, val_s - train_s)
                    for val_s, train_s in zip(fold_val_scores, train_metric_per_fold)
                ]
                return float(np.mean(penalized_per_fold))
            return overall_metrics[metric]
        except Exception as e:
            logger.warning(f"Trial failed with params {params}: {e}")
            return float("inf")

    objective_func = _cv_objective_with_penalty

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name=f"{exp_name}_optuna" + (f"_q{quantile}" if quantile else ""),
    )
    
    study.optimize(objective_func, n_trials=n_trials, show_progress_bar=False)
    
    metric_name = "MAE" if quantile is not None else "RMSE"
    logger.info(f"[{exp_name}] Optuna optimization completed!")
    logger.info(f"  Best {metric_name}: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")
    
    return study.best_params


def save_uncertainty_plot(
    y_true: np.ndarray,
    y_pred_median: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    path: str,
    title: str = "Prediction Intervals",
    confidence_level: float = 0.90,
) -> None:
    """
    Generates and saves an uncertainty plot with prediction intervals.
    
    Args:
        y_true: True target values
        y_pred_median: Median predictions (quantile 0.5)
        y_pred_lower: Lower bound predictions (e.g., quantile 0.05)
        y_pred_upper: Upper bound predictions (e.g., quantile 0.95)
        path: Output file path
        title: Plot title
        confidence_level: Confidence level for the interval (e.g., 0.90 for 90%)
    """
    try:
        plt.figure(figsize=(10, 8))
        sns.set_theme(style="whitegrid")
        
        # Filter out NaN/Inf
        mask = (
            np.isfinite(y_true) & np.isfinite(y_pred_median) &
            np.isfinite(y_pred_lower) & np.isfinite(y_pred_upper)
        )
        y_true_clean = y_true[mask]
        y_pred_median_clean = y_pred_median[mask]
        y_pred_lower_clean = y_pred_lower[mask]
        y_pred_upper_clean = y_pred_upper[mask]
        
        # Sort by true values for cleaner visualization
        sort_idx = np.argsort(y_true_clean)
        y_true_sorted = y_true_clean[sort_idx]
        y_pred_median_sorted = y_pred_median_clean[sort_idx]
        y_pred_lower_sorted = y_pred_lower_clean[sort_idx]
        y_pred_upper_sorted = y_pred_upper_clean[sort_idx]
        
        # Plot prediction intervals (shaded area)
        plt.fill_between(
            range(len(y_true_sorted)),
            y_pred_lower_sorted,
            y_pred_upper_sorted,
            alpha=0.3,
            color='lightblue',
            label=f'{confidence_level*100:.0f}% Prediction Interval'
        )
        
        # Plot median predictions
        plt.plot(
            range(len(y_true_sorted)),
            y_pred_median_sorted,
            'b-',
            linewidth=2,
            label='Median Prediction (Q50)'
        )
        
        # Plot true values
        plt.plot(
            range(len(y_true_sorted)),
            y_true_sorted,
            'ro',
            markersize=4,
            alpha=0.6,
            label='True Values'
        )
        
        # Calculate coverage (what fraction of true values fall within the interval)
        coverage = np.mean(
            (y_true_clean >= y_pred_lower_clean) & (y_true_clean <= y_pred_upper_clean)
        )
        
        plt.xlabel('Sample Index (sorted by true value)', fontsize=12)
        plt.ylabel(r'log10($\sigma$)', fontsize=12)
        plt.title(f'{title}\nCoverage: {coverage*100:.1f}% (target: {confidence_level*100:.0f}%)', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Uncertainty plot saved to {path}")
        
    except ImportError:
        logging.warning("Matplotlib or Seaborn not installed. Skipping uncertainty plot generation.")
    except Exception as e:
        logging.error(f"Could not generate uncertainty plot: {e}")


def generate_stage2_report(
    path: str,
    metrics: dict,
    stage1_ref: dict = None,
    best_params: dict = None,
    test_metrics: dict = None,
) -> None:
    """
    Generate the Stage 2 interim report with Optuna optimization results.
    
    Args:
        path: Destination path for the Markdown report
        metrics: Dict mapping experiment name -> metrics dict
        stage1_ref: Reference metrics from Stage 1 for comparison
        best_params: Dict mapping experiment name -> best hyperparameters
    """
    report_content = f"""# Interim Report: Stage 2 - Physics-Informed Features (Optuna Pipeline)

**Date:** {pd.to_datetime('today').strftime('%Y-%m-%d')}

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

**The DEFAULT model is used as Model 1 in all double-model strategies except Strategy E** — not the simple model. Among experiments, stage2_double_model_residual often achieves the best test set performance.

## 5. Cross-Validation Results (Train OOF)

| Experiment | R² | RMSE | MAE | Spearman ρ (train) |
|------------|------|------|-----|-------------------|
"""
    # Add metrics rows (CV / train-side)
    for exp_name, m in metrics.items():
        r2 = m.get('r2', float('nan'))
        rmse = m.get('rmse', float('nan'))
        mae = m.get('mae', float('nan'))
        rho_train = m.get('rho_train', float('nan'))
        report_content += f"| {exp_name} | {r2:.4f} | {rmse:.4f} | {mae:.4f} | {rho_train:.4f} |\n"
    
    # Add test set results if available
    if test_metrics:
        report_content += """
## 6. Test Set Results

| Experiment | R² | RMSE | MAE | Spearman ρ (test) |
|------------|------|------|-----|------------------|
"""
        for exp_name, m in test_metrics.items():
            r2 = m.get('r2', float('nan'))
            rmse = m.get('rmse', float('nan'))
            mae = m.get('mae', float('nan'))
            rho_test = m.get('rho_test', float('nan'))
            report_content += f"| {exp_name} | {r2:.4f} | {rmse:.4f} | {mae:.4f} | {rho_test:.4f} |\n"
    
    # Add Optuna methodology section (improvements over Stage 2)
    report_content += """
## 7. Optuna Optimization Methodology (Improved for Generalization)

Stage 2 analysis showed that optimizing on full CV splits led to overfitting: CV R² improved
~1% but test R² got worse. This pipeline implements improvements:

1. **5-fold CV with generalization penalty**: Optuna optimizes using predefined 5-fold splits.
   For each fold, the objective is: val_metric + 0.2 * max(0, val_metric - train_metric).
   This penalizes large train-val gaps and favors hyperparameters that generalize.
2. **Conservative search space**: Tighter bounds favor simpler models.
3. **Baseline experiments**:
   - `stage2_baseline_default`: sklearn defaults (no Optuna) — documents default, often best generalizer.
   - `stage2_baseline_default_geometry`: geometry-only baseline (Stage 0 + geometry, no space group one-hot), sklearn defaults (no Optuna).
   - `stage2_baseline_simple`: Optuna with SIMPLE_SEARCH_SPACE — searches for even simpler models.

"""
    
    # Add best hyperparameters section
    if best_params:
        report_content += "\n## 8. Best Hyperparameters\n\n"
        for exp_name, params in best_params.items():
            report_content += f"### {exp_name}\n\n"
            if not params:
                report_content += "sklearn default HistGradientBoostingRegressor (no tuning)\n\n"
            else:
                report_content += "| Hyperparameter | Value |\n"
                report_content += "|----------------|-------|\n"
                for param_name, param_value in params.items():
                    report_content += f"| {param_name} | {param_value} |\n"
                report_content += "\n"
    
    # Add comparison with Stage 1
    if stage1_ref:
        report_content += f"""
## 9. Comparison with Stage 1 (stage1_full_struct)

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
  uses **no Optuna** (sklearn defaults). Best test performer is often `stage2_double_model_residual`.
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

Results are saved to `results/results_stage2/`:
- `stage2_*_optuna_cv_parity.png` — 5-fold CV parity plots for each experiment (including all double-model variants)
- `stage2_*_optuna_test_parity.png` — Test set parity plots
- `stage2_*_optuna_predictions.csv` — Test set predictions for each experiment
- `*_quantile_uncertainty_cv.png` — CV uncertainty plot with prediction intervals (best experiment)
- `*_quantile_uncertainty_test.png` — Test set uncertainty plot with prediction intervals (best experiment)
- `*_quantile_predictions.csv` — Quantile predictions (Q0.05, Q0.5, Q0.95) and interval widths
- `interim_report_optuna.md` — This report
- `optuna.log` — Detailed execution log
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
    Stage 2: Physics-Informed Features Pipeline WITH OPTUNA OPTIMIZATION.
    
    This pipeline:
    1. Replicates stage1_full_struct EXACTLY as baseline (for validation)
    2. Adds physics-informed features: bond-valence mismatch, Ewald energy, Voronoi CN
    3. Optimizes ALL models with Optuna before training
    4. Compares results to ensure no data leakage and validate physics feature contribution
    """
    # -----------------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------------
    seed_everything(42)
    
    results_dir = os.path.join(PROJECT_ROOT, "results", "results_stage2")
    os.makedirs(results_dir, exist_ok=True)
    
    log_path = os.path.join(results_dir, "optuna.log")
    # Use a unique logger name so we don't collide with Optuna's own "optuna" logger,
    # which already adds handlers and prevents our FileHandler from being attached.
    logger = setup_logger("stage2_optuna_pipeline", log_file=log_path)
    logger.info("=" * 80)
    logger.info("Stage 2: Physics-Informed Features Pipeline WITH OPTUNA OPTIMIZATION")
    logger.info("=" * 80)
    
    
    # Reference metrics from Stage 1 (stage1_full_struct)
    stage1_ref = {
        "r2": 0.7370,
        "rmse": 1.3734,
        "mae": 0.8253,
    }
    
    # Optuna configuration
    N_TRIALS = 50  # Number of Optuna trials per model
    
    # -----------------------------------------------------------------------------------------
    # Data Loading and Preprocessing (SAME AS ORIGINAL)
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
    
    # Stage 1 geometry-only baseline = Stage 0 + geometry (no space group one-hot)
    stage1_geometry_cols = stage0_feature_cols + geometry_cols
    
    # Stage 1 full_struct = Stage 0 + geometry + space group one-hot
    stage1_full_struct_cols = stage0_feature_cols + geometry_cols + spacegroup_cols
    
    # Stage 2 physics = Stage 1 full_struct + physics features
    stage2_physics_cols = stage1_full_struct_cols + physics_feature_cols
    
    # Stage 2 physics (geometry-only) = Stage 1 geometry + physics features (NO space group one-hot)
    stage2_physics_geometry_cols = stage1_geometry_cols + physics_feature_cols
    
    logger.info(f"Feature set sizes:")
    logger.info(f"  Stage 0 features: {len(stage0_feature_cols)}")
    logger.info(f"  Geometry features: {len(geometry_cols)}")
    logger.info(f"  Space group one-hot: {len(spacegroup_cols)}")
    logger.info(f"  Baseline geometry-only (stage1_geometry): {len(stage1_geometry_cols)}")
    logger.info(f"  Baseline (stage1_full_struct): {len(stage1_full_struct_cols)}")
    logger.info(f"  Physics features: {len(physics_feature_cols)}")
    logger.info(f"  Full (baseline + physics): {len(stage2_physics_cols)}")
    logger.info(f"  Physics geometry-only (no SG one-hot): {len(stage2_physics_geometry_cols)}")
    
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
    # optuna_mode: "default" = sklearn defaults (no Optuna), "simple" = Optuna with SIMPLE_SEARCH_SPACE,
    #              "standard" = Optuna with OPTUNA_SEARCH_SPACE
    experiments = {
        "stage2_baseline_default": {
            "description": "Stage 1 full_struct with sklearn default HistGradientBoostingRegressor (no Optuna) — documents default, best generalizer",
            "train_df": full_train_df,
            "test_df": full_test_df,
            "feature_cols": stage1_full_struct_cols,
            "optuna_mode": "default",
        },
        "stage2_baseline_default_geometry": {
            "description": "Stage 1 geometry-only baseline: Stage 0 + geometry (no space group one-hot), sklearn defaults (no Optuna)",
            "train_df": full_train_df,
            "test_df": full_test_df,
            "feature_cols": stage1_geometry_cols,
            "optuna_mode": "default",
        },
        "stage2_baseline_simple": {
            "description": "Stage 1 full_struct with Optuna searching for simpler models (SIMPLE_SEARCH_SPACE)",
            "train_df": full_train_df,
            "test_df": full_test_df,
            "feature_cols": stage1_full_struct_cols,
            "optuna_mode": "simple",
        },
        "stage2_physics": {
            "description": "Baseline + physics features (BV, Ewald, Voronoi) — this is Model 2 in the double-model scheme",
            "train_df": physics_train_df,
            "test_df": physics_test_df,
            "feature_cols": stage2_physics_cols,
            "optuna_mode": "standard",
        },
        "stage2_physics_geometry": {
            "description": "Geometry-only baseline + physics features (no space-group one-hot) — cleaner Stage 2 model",
            "train_df": physics_train_df,
            "test_df": physics_test_df,
            "feature_cols": stage2_physics_geometry_cols,
            "optuna_mode": "standard",
        },
        "stage1_geometry_optuna": {
            "description": "Stage 1 geometry only (no SG one-hot, no physics) with Optuna",
            "train_df": struct_train_df,
            "test_df": struct_test_df,
            "feature_cols": stage1_geometry_cols,
            "optuna_mode": "standard",
        },
    }
    
    # -----------------------------------------------------------------------------------------
    # Run Experiments WITH OPTUNA OPTIMIZATION
    # -----------------------------------------------------------------------------------------
    all_experiment_metrics = {}
    all_experiment_oof = {}
    all_experiment_test_preds = {}
    all_experiment_best_params = {}  # Store best hyperparameters
    all_experiment_test_metrics = {}  # Store test metrics for summary
    
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
        
        optuna_mode = exp_config.get("optuna_mode", "standard")
        
        if optuna_mode == "default":
            # sklearn defaults (no Optuna) — documents default model, often best generalizer
            best_params = None  # HistGradientBoostingRegressor defaults
            logger.info(f"[{exp_name}] Using sklearn default hyperparameters (no Optuna)")
            all_experiment_best_params[exp_name] = {}
        elif optuna_mode == "simple":
            # Optuna with SIMPLE_SEARCH_SPACE — search for even simpler models
            best_params = optimize_hyperparameters(
                X, y, splits, group_col, exp_name, logger, n_trials=N_TRIALS,
                search_space=SIMPLE_SEARCH_SPACE
            )
            all_experiment_best_params[exp_name] = best_params
        else:
            # standard: Optuna with OPTUNA_SEARCH_SPACE
            best_params = optimize_hyperparameters(
                X, y, splits, group_col, exp_name, logger, n_trials=N_TRIALS
            )
            all_experiment_best_params[exp_name] = best_params
        
        # Model configuration
        config = ExperimentConfig(
            model_name="hgbt",
            n_splits=5,
            random_state=42,
            params=best_params,
            group_col=group_col
        )
        
        # Run cross-validation with optimized hyperparameters
        fold_scores, overall_metrics, oof = run_cv_with_predefined_splits(X, y, splits, config)
        
        # Compute Spearman rho on OOF predictions (train-side metric)
        rho_train = spearman_rho(y.values, oof)
        
        # Log results
        logger.info(f"[{exp_name}] Fold RMSEs: {[f'{s:.4f}' for s in fold_scores]}")
        logger.info(f"[{exp_name}] Overall CV Metrics:")
        logger.info(f"    R2:   {overall_metrics['r2']:.4f}")
        logger.info(f"    RMSE: {overall_metrics['rmse']:.4f}")
        logger.info(f"    MAE:  {overall_metrics['mae']:.4f}")
        logger.info(f"    Spearman rho (train OOF): {rho_train:.4f}")
        
        overall_metrics["rho_train"] = float(rho_train)
        all_experiment_metrics[exp_name] = overall_metrics
        all_experiment_oof[exp_name] = oof
        
        # Save CV parity plot
        parity_plot_path = os.path.join(results_dir, f"{exp_name}_optuna_cv_parity.png")
        save_parity_plot(
            y.values,
            oof,
            parity_plot_path,
            title=f"Stage 2 5-Fold CV (Optuna): {exp_name}",
            r2_linear_from_log=False,
        )
        logger.info(f"[{exp_name}] CV parity plot saved to: {parity_plot_path}")
        
        # Train final model and predict on test set
        logger.info(f"[{exp_name}] Training final model on full training set with optimized hyperparameters...")
        
        X_test = test_exp[[c for c in available_cols if c in test_exp.columns]].copy()
        X_test = X_test.reindex(columns=X.columns, fill_value=0)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test = X_test.fillna(0)
        
        preds = fit_full_and_predict(X, y, X_test, config)
        
        # Save predictions
        output_path = os.path.join(results_dir, f"{exp_name}_optuna_predictions.csv")
        out_df = pd.DataFrame({"ID": test_df["ID"], "prediction": preds})
        save_dataframe(out_df, output_path)
        logger.info(f"[{exp_name}] Predictions saved to: {output_path}")
        all_experiment_test_preds[exp_name] = preds
        
        # Test-set parity plot
        if TARGET_COL in test_df.columns:
            test_y = test_df[TARGET_COL].values
            test_parity_path = os.path.join(results_dir, f"{exp_name}_optuna_test_parity.png")
            save_parity_plot(
                test_y,
                preds,
                test_parity_path,
                title=f"Stage 2 Test (Optuna): {exp_name}",
                r2_linear_from_log=False,
            )
            
            # Log test-set metrics analogous to CV metrics
            test_metrics = compute_regression_metrics(test_y, preds)
            rho_test = spearman_rho(test_y, preds)
            test_metrics["rho_test"] = float(rho_test)
            all_experiment_test_metrics[exp_name] = test_metrics
            logger.info(f"[{exp_name}] Test Metrics:")
            logger.info(f"    R2:   {test_metrics['r2']:.4f}")
            logger.info(f"    RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"    MAE:  {test_metrics['mae']:.4f}")
            logger.info(f"    Spearman rho (test): {rho_test:.4f}")
            logger.info(f"[{exp_name}] Test parity plot saved to: {test_parity_path}")
    
    # -----------------------------------------------------------------------------------------
    # Double-Model Strategies: Model 1 (baseline) + Model 2 (physics-informed) WITH OPTUNA
    # -----------------------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Double-model strategies: combining baseline (Model 1) with physics-informed models (Model 2)")
    logger.info("ALL DOUBLE-MODEL STRATEGIES USE OPTUNA-OPTIMIZED BASELINE AND PHYSICS MODELS")
    
    residual_quantile_data = None  # Populated when stage2_double_model_residual is built (for quantile block)
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
        
        if "stage2_baseline_default" in all_experiment_oof and "stage2_physics" in all_experiment_oof:
            # Common data for all double-model variants (use default baseline — best generalizer)
            oof_baseline = all_experiment_oof["stage2_baseline_default"]
            oof_physics_full = all_experiment_oof["stage2_physics"]
            preds_baseline = all_experiment_test_preds.get("stage2_baseline_default")
            preds_physics_full = all_experiment_test_preds.get("stage2_physics")
            
            # Optional geometry-only baseline (Stage 0 + geometry, no space group)
            oof_baseline_geom = all_experiment_oof.get("stage2_baseline_default_geometry")
            preds_baseline_geom = all_experiment_test_preds.get("stage2_baseline_default_geometry")
            
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
                fulltrain_rho = spearman_rho(y.values, oof_fulltrain)
                
                logger.info("stage2_double_model_fulltrain CV metrics:")
                logger.info(f"  R2:   {fulltrain_metrics['r2']:.4f}")
                logger.info(f"  RMSE: {fulltrain_metrics['rmse']:.4f}")
                logger.info(f"  MAE:  {fulltrain_metrics['mae']:.4f}")
                logger.info(f"  Spearman rho (train OOF): {fulltrain_rho:.4f}")
                
                fulltrain_metrics["rho_train"] = float(fulltrain_rho)
                all_experiment_metrics["stage2_double_model_fulltrain"] = fulltrain_metrics
                
                fulltrain_parity_path = os.path.join(results_dir, "stage2_double_model_fulltrain_optuna_cv_parity.png")
                save_parity_plot(
                    y.values,
                    oof_fulltrain,
                    fulltrain_parity_path,
                    title="Stage 2 5-Fold CV (Optuna): double-model fulltrain (baseline + physics)",
                    r2_linear_from_log=False,
                )
                logger.info(f"stage2_double_model_fulltrain CV parity plot saved to: {fulltrain_parity_path}")
                
                test_preds_fulltrain = np.where(test_has_good_physics.values, preds_physics_full, preds_baseline)
                fulltrain_output_path = os.path.join(results_dir, "stage2_double_model_fulltrain_optuna_predictions.csv")
                fulltrain_out_df = pd.DataFrame({"ID": test_df["ID"], "prediction": test_preds_fulltrain})
                save_dataframe(fulltrain_out_df, fulltrain_output_path)
                logger.info(f"stage2_double_model_fulltrain predictions saved to: {fulltrain_output_path}")
                
                if TARGET_COL in test_df.columns:
                    test_y = test_df[TARGET_COL].values
                    fulltrain_test_parity_path = os.path.join(results_dir, "stage2_double_model_fulltrain_optuna_test_parity.png")
                    save_parity_plot(
                        test_y,
                        test_preds_fulltrain,
                        fulltrain_test_parity_path,
                        title="Stage 2 Test (Optuna): double-model fulltrain (baseline + physics)",
                        r2_linear_from_log=False,
                    )
                    
                    fulltrain_test_metrics = compute_regression_metrics(test_y, test_preds_fulltrain)
                    fulltrain_rho_test = spearman_rho(test_y, test_preds_fulltrain)
                    fulltrain_test_metrics["rho_test"] = float(fulltrain_rho_test)
                    all_experiment_test_metrics["stage2_double_model_fulltrain"] = fulltrain_test_metrics
                    logger.info("stage2_double_model_fulltrain Test metrics:")
                    logger.info(f"  R2:   {fulltrain_test_metrics['r2']:.4f}")
                    logger.info(f"  RMSE: {fulltrain_test_metrics['rmse']:.4f}")
                    logger.info(f"  MAE:  {fulltrain_test_metrics['mae']:.4f}")
                    logger.info(f"  Spearman rho (test): {fulltrain_rho_test:.4f}")
                    logger.info(f"stage2_double_model_fulltrain Test parity plot saved to: {fulltrain_test_parity_path}")
                
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
                    
                    # OPTUNA OPTIMIZATION FOR SUBSET MODEL
                    subset_best_params = optimize_hyperparameters(
                        X_subset_train, y_subset, subset_splits, group_col,
                        f"{exp_name}_subsettrain", logger, n_trials=N_TRIALS
                    )
                    all_experiment_best_params["stage2_double_model_subsettrain"] = subset_best_params
                    
                    subset_config = ExperimentConfig(
                        model_name="hgbt",
                        n_splits=5,
                        random_state=42,
                        params=subset_best_params,
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
                    subsettrain_rho = spearman_rho(y.values, oof_subsettrain)
                    
                    logger.info("stage2_double_model_subsettrain CV metrics:")
                    logger.info(f"  R2:   {subsettrain_metrics['r2']:.4f}")
                    logger.info(f"  RMSE: {subsettrain_metrics['rmse']:.4f}")
                    logger.info(f"  MAE:  {subsettrain_metrics['mae']:.4f}")
                    logger.info(f"  Spearman rho (train OOF): {subsettrain_rho:.4f}")
                    
                    subsettrain_metrics["rho_train"] = float(subsettrain_rho)
                    all_experiment_metrics["stage2_double_model_subsettrain"] = subsettrain_metrics
                    
                    subsettrain_parity_path = os.path.join(results_dir, "stage2_double_model_subsettrain_optuna_cv_parity.png")
                    save_parity_plot(
                        y.values,
                        oof_subsettrain,
                        subsettrain_parity_path,
                        title="Stage 2 5-Fold CV (Optuna): double-model subsettrain (baseline + physics subset)",
                        r2_linear_from_log=False,
                    )
                    logger.info(f"stage2_double_model_subsettrain CV parity plot saved to: {subsettrain_parity_path}")
                    
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
                        
                        subsettrain_output_path = os.path.join(results_dir, "stage2_double_model_subsettrain_optuna_predictions.csv")
                        subsettrain_out_df = pd.DataFrame({"ID": test_df["ID"], "prediction": test_preds_subsettrain})
                        save_dataframe(subsettrain_out_df, subsettrain_output_path)
                        logger.info(f"stage2_double_model_subsettrain predictions saved to: {subsettrain_output_path}")
                        
                        if TARGET_COL in test_df.columns:
                            test_y = test_df[TARGET_COL].values
                            subsettrain_test_parity_path = os.path.join(results_dir, "stage2_double_model_subsettrain_optuna_test_parity.png")
                            save_parity_plot(
                                test_y,
                                test_preds_subsettrain,
                                subsettrain_test_parity_path,
                                title="Stage 2 Test (Optuna): double-model subsettrain (baseline + physics subset)",
                                r2_linear_from_log=False,
                            )
                            
                            subsettrain_test_metrics = compute_regression_metrics(test_y, test_preds_subsettrain)
                            subsettrain_rho_test = spearman_rho(test_y, test_preds_subsettrain)
                            subsettrain_test_metrics["rho_test"] = float(subsettrain_rho_test)
                            all_experiment_test_metrics["stage2_double_model_subsettrain"] = subsettrain_test_metrics
                            logger.info("stage2_double_model_subsettrain Test metrics:")
                            logger.info(f"  R2:   {subsettrain_test_metrics['r2']:.4f}")
                            logger.info(f"  RMSE: {subsettrain_test_metrics['rmse']:.4f}")
                            logger.info(f"  MAE:  {subsettrain_test_metrics['mae']:.4f}")
                            logger.info(f"  Spearman rho (test): {subsettrain_rho_test:.4f}")
                            logger.info(f"stage2_double_model_subsettrain Test parity plot saved to: {subsettrain_test_parity_path}")
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
                    
                    # OPTUNA OPTIMIZATION FOR RESIDUAL MODEL
                    resid_best_params = optimize_hyperparameters(
                        X_subset_train, pd.Series(residual_target), subset_resid_splits, group_col,
                        f"{exp_name}_residual", logger, n_trials=N_TRIALS
                    )
                    all_experiment_best_params["stage2_double_model_residual"] = resid_best_params
                    
                    resid_config = ExperimentConfig(
                        model_name="hgbt",
                        n_splits=5,
                        random_state=42,
                        params=resid_best_params,
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
                    residual_rho = spearman_rho(y.values, oof_residual)
                    
                    logger.info("stage2_double_model_residual CV metrics:")
                    logger.info(f"  R2:   {residual_metrics['r2']:.4f}")
                    logger.info(f"  RMSE: {residual_metrics['rmse']:.4f}")
                    logger.info(f"  MAE:  {residual_metrics['mae']:.4f}")
                    logger.info(f"  Spearman rho (train OOF): {residual_rho:.4f}")
                    
                    residual_metrics["rho_train"] = float(residual_rho)
                    all_experiment_metrics["stage2_double_model_residual"] = residual_metrics
                    
                    residual_parity_path = os.path.join(results_dir, "stage2_double_model_residual_optuna_cv_parity.png")
                    save_parity_plot(
                        y.values,
                        oof_residual,
                        residual_parity_path,
                        title="Stage 2 5-Fold CV (Optuna): double-model residual (baseline + physics residual)",
                        r2_linear_from_log=False,
                    )
                    logger.info(f"stage2_double_model_residual CV parity plot saved to: {residual_parity_path}")
                    
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
                        
                        residual_output_path = os.path.join(results_dir, "stage2_double_model_residual_optuna_predictions.csv")
                        residual_out_df = pd.DataFrame({"ID": test_df["ID"], "prediction": test_preds_residual})
                        save_dataframe(residual_out_df, residual_output_path)
                        logger.info(f"stage2_double_model_residual predictions saved to: {residual_output_path}")
                        
                        if TARGET_COL in test_df.columns:
                            test_y = test_df[TARGET_COL].values
                            residual_test_parity_path = os.path.join(results_dir, "stage2_double_model_residual_optuna_test_parity.png")
                            save_parity_plot(
                                test_y,
                                test_preds_residual,
                                residual_test_parity_path,
                                title="Stage 2 Test (Optuna): double-model residual (baseline + physics residual)",
                                r2_linear_from_log=False,
                            )
                            
                            residual_test_metrics = compute_regression_metrics(test_y, test_preds_residual)
                            residual_rho_test = spearman_rho(test_y, test_preds_residual)
                            residual_test_metrics["rho_test"] = float(residual_rho_test)
                            all_experiment_test_metrics["stage2_double_model_residual"] = residual_test_metrics
                            logger.info("stage2_double_model_residual Test metrics:")
                            logger.info(f"  R2:   {residual_test_metrics['r2']:.4f}")
                            logger.info(f"  RMSE: {residual_test_metrics['rmse']:.4f}")
                            logger.info(f"  MAE:  {residual_test_metrics['mae']:.4f}")
                            logger.info(f"  Spearman rho (test): {residual_rho_test:.4f}")
                            logger.info(f"stage2_double_model_residual Test parity plot saved to: {residual_test_parity_path}")
                        
                        # Store data for quantile regression on this composite model
                        residual_quantile_data = {
                            "subset_train_indices": subset_train_indices,
                            "subset_test_indices": subset_test_indices,
                            "X_subset_train": X_subset_train,
                            "X_subset_test": X_subset_test,
                            "residual_target": residual_target,
                            "preds_baseline": preds_baseline.copy(),
                            "oof_baseline": oof_baseline.copy(),
                            "train_has_good_physics": train_has_good_physics,
                            "test_has_good_physics": test_has_good_physics,
                            "baseline_exp": experiments["stage2_baseline_default"],
                            "subset_resid_splits": subset_resid_splits,
                            "group_col": group_col,
                        }
                    else:
                        logger.warning("No test samples with full physics coverage; skipping residual strategy predictions.")
                else:
                    logger.warning("No training samples with full physics coverage; skipping residual strategy.")
                
                # ---------------------------------------------------------------------------------
                # Strategy E: Residual correction model with geometry-only baseline (Stage 0 + geometry, no space group)
                #             (Model 2 still predicts residual y - y_baseline_geometry on good-physics subset)
                # ---------------------------------------------------------------------------------
                if oof_baseline_geom is not None and preds_baseline_geom is not None:
                    logger.info("-" * 80)
                    logger.info("Strategy E: stage2_double_model_residual_geometry (geometry baseline + physics residual on good-physics subset)")
                    
                    if len(subset_train_indices) > 0:
                        X_subset_train_geom = X_physics_train.iloc[subset_train_indices]
                        y_subset_geom = y.iloc[subset_train_indices]
                        baseline_geom_subset_oof = oof_baseline_geom[subset_train_indices]
                        
                        # Residual target relative to geometry-only baseline
                        residual_geom_target = y_subset_geom.values - baseline_geom_subset_oof
                        
                        # GroupKFold splits on the subset (same grouping as before)
                        if group_col and group_col in train_physics.columns:
                            subset_groups_geom = train_physics.iloc[subset_train_indices][group_col].values
                        else:
                            subset_groups_geom = np.arange(len(X_subset_train_geom))
                        
                        gkf_subset_resid_geom = GroupKFold(n_splits=5)
                        subset_resid_splits_geom = list(
                            gkf_subset_resid_geom.split(X_subset_train_geom, residual_geom_target, groups=subset_groups_geom)
                        )
                        
                        # OPTUNA OPTIMIZATION FOR GEOMETRY-BASELINE RESIDUAL MODEL
                        resid_geom_best_params = optimize_hyperparameters(
                            X_subset_train_geom,
                            pd.Series(residual_geom_target),
                            subset_resid_splits_geom,
                            group_col,
                            f"{exp_name}_residual_geometry",
                            logger,
                            n_trials=N_TRIALS,
                        )
                        all_experiment_best_params["stage2_double_model_residual_geometry"] = resid_geom_best_params
                        
                        resid_geom_config = ExperimentConfig(
                            model_name="hgbt",
                            n_splits=5,
                            random_state=42,
                            params=resid_geom_best_params,
                            group_col=group_col,
                        )
                        
                        resid_geom_fold_scores, resid_geom_model_metrics, resid_geom_oof = run_cv_with_predefined_splits(
                            X_subset_train_geom,
                            pd.Series(residual_geom_target),
                            subset_resid_splits_geom,
                            resid_geom_config,
                        )
                        
                        logger.info(
                            f"  Residual-geometry model CV RMSEs (on residual target): "
                            f"{[f'{s:.4f}' for s in resid_geom_fold_scores]}"
                        )
                        logger.info(
                            f"  Residual-geometry model CV metrics (on residual target): "
                            f"R2={resid_geom_model_metrics['r2']:.4f}, "
                            f"RMSE={resid_geom_model_metrics['rmse']:.4f}, "
                            f"MAE={resid_geom_model_metrics['mae']:.4f}"
                        )
                        
                        # Double-model residual gating: geometry baseline everywhere, plus residual on good-physics subset
                        oof_residual_geom = oof_baseline_geom.copy()
                        oof_residual_geom[subset_train_indices] = baseline_geom_subset_oof + resid_geom_oof
                        residual_geom_metrics = compute_regression_metrics(y.values, oof_residual_geom)
                        residual_geom_rho = spearman_rho(y.values, oof_residual_geom)
                        
                        logger.info("stage2_double_model_residual_geometry CV metrics:")
                        logger.info(f"  R2:   {residual_geom_metrics['r2']:.4f}")
                        logger.info(f"  RMSE: {residual_geom_metrics['rmse']:.4f}")
                        logger.info(f"  MAE:  {residual_geom_metrics['mae']:.4f}")
                        logger.info(f"  Spearman rho (train OOF): {residual_geom_rho:.4f}")
                        
                        residual_geom_metrics["rho_train"] = float(residual_geom_rho)
                        all_experiment_metrics["stage2_double_model_residual_geometry"] = residual_geom_metrics
                        
                        residual_geom_parity_path = os.path.join(
                            results_dir, "stage2_double_model_residual_geometry_optuna_cv_parity.png"
                        )
                        save_parity_plot(
                            y.values,
                            oof_residual_geom,
                            residual_geom_parity_path,
                            title=(
                                "Stage 2 5-Fold CV (Optuna): "
                                "double-model residual_geometry (geometry baseline + physics residual)"
                            ),
                            r2_linear_from_log=False,
                        )
                        logger.info(
                            f"stage2_double_model_residual_geometry CV parity plot saved to: {residual_geom_parity_path}"
                        )
                        
                        # Final residual-geometry model and test-set predictions
                        X_subset_train_geom_full = X_subset_train_geom
                        residual_geom_target_full = residual_geom_target
                        X_subset_test_geom = (
                            X_physics_test.iloc[subset_test_indices] if len(subset_test_indices) > 0 else None
                        )
                        
                        if X_subset_test_geom is not None and len(X_subset_test_geom) > 0:
                            resid_geom_preds_test = fit_full_and_predict(
                                X_subset_train_geom_full,
                                pd.Series(residual_geom_target_full),
                                X_subset_test_geom,
                                resid_geom_config,
                            )
                            
                            test_preds_residual_geom = preds_baseline_geom.copy()
                            # geometry baseline + residual correction where physics is good
                            test_preds_residual_geom[subset_test_indices] = (
                                preds_baseline_geom[subset_test_indices] + resid_geom_preds_test
                            )
                            
                            residual_geom_output_path = os.path.join(
                                results_dir, "stage2_double_model_residual_geometry_optuna_predictions.csv"
                            )
                            residual_geom_out_df = pd.DataFrame(
                                {"ID": test_df["ID"], "prediction": test_preds_residual_geom}
                            )
                            save_dataframe(residual_geom_out_df, residual_geom_output_path)
                            logger.info(
                                f"stage2_double_model_residual_geometry predictions saved to: {residual_geom_output_path}"
                            )
                            
                            if TARGET_COL in test_df.columns:
                                test_y = test_df[TARGET_COL].values
                                residual_geom_test_parity_path = os.path.join(
                                    results_dir, "stage2_double_model_residual_geometry_optuna_test_parity.png"
                                )
                                save_parity_plot(
                                    test_y,
                                    test_preds_residual_geom,
                                    residual_geom_test_parity_path,
                                    title=(
                                        "Stage 2 Test (Optuna): "
                                        "double-model residual_geometry (geometry baseline + physics residual)"
                                    ),
                                    r2_linear_from_log=False,
                                )
                                
                                residual_geom_test_metrics = compute_regression_metrics(
                                    test_y, test_preds_residual_geom
                                )
                                residual_geom_rho_test = spearman_rho(test_y, test_preds_residual_geom)
                                residual_geom_test_metrics["rho_test"] = float(residual_geom_rho_test)
                                all_experiment_test_metrics[
                                    "stage2_double_model_residual_geometry"
                                ] = residual_geom_test_metrics
                                logger.info("stage2_double_model_residual_geometry Test metrics:")
                                logger.info(f"  R2:   {residual_geom_test_metrics['r2']:.4f}")
                                logger.info(f"  RMSE: {residual_geom_test_metrics['rmse']:.4f}")
                                logger.info(f"  MAE:  {residual_geom_test_metrics['mae']:.4f}")
                                logger.info(
                                    f"stage2_double_model_residual_geometry Test parity plot saved to: "
                                    f"{residual_geom_test_parity_path}"
                                )
                        else:
                            logger.warning(
                                "Missing geometry-only baseline predictions; skipping residual_geometry strategy."
                            )
                
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
                    
                    # OPTUNA OPTIMIZATION FOR META-MODEL
                    meta_best_params = optimize_hyperparameters(
                        meta_train_df, y_meta, meta_splits, None,
                        f"{exp_name}_residual_stack", logger, n_trials=N_TRIALS
                    )
                    all_experiment_best_params["stage2_double_model_residual_stack"] = meta_best_params
                    
                    meta_config = ExperimentConfig(
                        model_name="hgbt",
                        n_splits=5,
                        random_state=42,
                        params=meta_best_params,
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
                    residual_stack_rho = spearman_rho(y.values, oof_residual_stack)
                    
                    logger.info("stage2_double_model_residual_stack CV metrics:")
                    logger.info(f"  R2:   {residual_stack_metrics['r2']:.4f}")
                    logger.info(f"  RMSE: {residual_stack_metrics['rmse']:.4f}")
                    logger.info(f"  MAE:  {residual_stack_metrics['mae']:.4f}")
                    logger.info(f"  Spearman rho (train OOF): {residual_stack_rho:.4f}")
                    
                    residual_stack_metrics["rho_train"] = float(residual_stack_rho)
                    all_experiment_metrics["stage2_double_model_residual_stack"] = residual_stack_metrics
                    
                    residual_stack_parity_path = os.path.join(results_dir, "stage2_double_model_residual_stack_optuna_cv_parity.png")
                    save_parity_plot(
                        y.values,
                        oof_residual_stack,
                        residual_stack_parity_path,
                        title="Stage 2 5-Fold CV (Optuna): double-model residual_stack (baseline + learned combiner)",
                        r2_linear_from_log=False,
                    )
                    logger.info(f"stage2_double_model_residual_stack CV parity plot saved to: {residual_stack_parity_path}")
                    
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
                        
                        residual_stack_output_path = os.path.join(results_dir, "stage2_double_model_residual_stack_optuna_predictions.csv")
                        residual_stack_out_df = pd.DataFrame({"ID": test_df["ID"], "prediction": test_preds_residual_stack})
                        save_dataframe(residual_stack_out_df, residual_stack_output_path)
                        logger.info(f"stage2_double_model_residual_stack predictions saved to: {residual_stack_output_path}")
                        
                        if TARGET_COL in test_df.columns:
                            test_y = test_df[TARGET_COL].values
                            residual_stack_test_parity_path = os.path.join(results_dir, "stage2_double_model_residual_stack_optuna_test_parity.png")
                            save_parity_plot(
                                test_y,
                                test_preds_residual_stack,
                                residual_stack_test_parity_path,
                                title="Stage 2 Test (Optuna): double-model residual_stack (baseline + learned combiner)",
                                r2_linear_from_log=False,
                            )
                            
                            residual_stack_test_metrics = compute_regression_metrics(test_y, test_preds_residual_stack)
                            residual_stack_rho_test = spearman_rho(test_y, test_preds_residual_stack)
                            residual_stack_test_metrics["rho_test"] = float(residual_stack_rho_test)
                            all_experiment_test_metrics["stage2_double_model_residual_stack"] = residual_stack_test_metrics
                            logger.info("stage2_double_model_residual_stack Test metrics:")
                            logger.info(f"  R2:   {residual_stack_test_metrics['r2']:.4f}")
                            logger.info(f"  RMSE: {residual_stack_test_metrics['rmse']:.4f}")
                            logger.info(f"  MAE:  {residual_stack_test_metrics['mae']:.4f}")
                            logger.info(f"  Spearman rho (test): {residual_stack_rho_test:.4f}")
                            logger.info(f"stage2_double_model_residual_stack Test parity plot saved to: {residual_stack_test_parity_path}")
                    else:
                        logger.warning("No test samples with full physics coverage; skipping residual_stack test predictions.")
                else:
                    logger.warning("Not enough data to fit residual_stack meta-model; skipping Strategy D.")
        else:
            logger.warning("Cannot construct double-model strategies: missing baseline or physics OOF predictions.")
    except Exception as e:
        logger.warning(f"Double-model strategies step failed: {e}")
    
    # Quantile regression for uncertainty has been removed in this stage.
    
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
        
        # Check if baseline default matches Stage 1
        if exp_name == "stage2_baseline_default":
            if abs(r2_diff) < 0.001 and abs(rmse_diff) < 0.001 and abs(mae_diff) < 0.001:
                logger.info(f"    [PASS] Results match Stage 1 within tolerance!")
            else:
                logger.warning(f"    [WARN] Results differ from Stage 1 reference")
    
    # -----------------------------------------------------------------------------------------
    # Generate Final Report
    # -----------------------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Generating final report...")
    
    report_path = os.path.join(results_dir, "interim_report_optuna.md")
    generate_stage2_report(
        report_path,
        all_experiment_metrics,
        stage1_ref,
        all_experiment_best_params,
        test_metrics=all_experiment_test_metrics,
    )
    logger.info(f"Report saved to: {report_path}")
    
    # Print summary
    logger.info("-" * 80)
    logger.info("SUMMARY: Stage 2 Experiment Results (Default + Optuna pipeline)")
    logger.info("-" * 80)
    logger.info("CROSS-VALIDATION METRICS:")
    logger.info(f"{'Experiment':<40s} | {'R2':>8s} | {'RMSE':>8s} | {'MAE':>8s} | {'rho_train':>8s}")
    logger.info("-" * 80)
    for exp_name, metrics in all_experiment_metrics.items():
        rho_train = metrics.get("rho_train", float("nan"))
        logger.info(f"{exp_name:<40s} | {metrics['r2']:8.4f} | {metrics['rmse']:8.4f} | {metrics['mae']:8.4f} | {rho_train:8.4f}")
    
    if all_experiment_test_metrics:
        logger.info("")
        logger.info("TEST SET METRICS:")
        logger.info(f"{'Experiment':<40s} | {'R2':>8s} | {'RMSE':>8s} | {'MAE':>8s} | {'rho_test':>8s}")
        logger.info("-" * 80)
        for exp_name, test_metrics in all_experiment_test_metrics.items():
            rho_test = test_metrics.get("rho_test", float("nan"))
            logger.info(f"{exp_name:<40s} | {test_metrics['r2']:8.4f} | {test_metrics['rmse']:8.4f} | {test_metrics['mae']:8.4f} | {rho_test:8.4f}")
    
    logger.info("")
    logger.info("All parity plots and predictions saved to:")
    logger.info(f"  Results directory: {results_dir}")
    logger.info("  CV plots: *_optuna_cv_parity.png")
    logger.info("  Test plots: *_optuna_test_parity.png")
    logger.info("  Predictions: *_optuna_predictions.csv")
    
    logger.info("")
    logger.info("OPTUNA OPTIMIZATION METHOD:")
    logger.info("  - Bayesian hyperparameter optimization using TPESampler (Tree-structured Parzen Estimator)")
    logger.info("  - Each model optimized separately with 50 trials")
    
    logger.info("=" * 80)
    logger.info("Stage 2 pipeline with Optuna optimization completed successfully!")


# =================================================================================================
# Stage 0 Magpie Optuna: Standalone Optimization of Composition-Only Baseline
# =================================================================================================
#
# This function optimizes the stage0_magpie model independently, without requiring
# Stage 1/2 feature engineering (CIF parsing, physics features). It uses only
# compositional features: elemental ratios + SMACT + Magpie embeddings.
#
# Hyperparameter optimization follows the exact same methodology as the later-stage
# models in main(): 5-fold GroupKFold CV with a generalization penalty that penalizes
# large train-val RMSE gaps to discourage overfitting.
#
# Metrics: R², RMSE, MAE, Spearman rho (no linearized R²).
# =================================================================================================

def run_stage0_magpie_optuna():
    """
    Optuna-optimized Stage 0 Magpie baseline model.

    Uses only Stage 0 compositional features (elemental ratios + SMACT + Magpie embeddings).
    Optimizes hyperparameters using 5-fold GroupKFold CV with generalization penalty
    (penalty_weight=0.2), identical to the approach used for later-stage models.

    Outputs:
        - Optuna optimization log appended to results/results_stage2/optuna.log
        - CV parity plot: stage0_magpie_optuna_cv_parity.png
        - Test parity plot: stage0_magpie_optuna_test_parity.png
        - Test predictions CSV: stage0_magpie_optuna_predictions.csv
    """
    # -----------------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------------
    seed_everything(42)

    results_dir = os.path.join(PROJECT_ROOT, "results", "results_stage2")
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "optuna.log")
    logger = setup_logger("stage0_magpie_optuna_pipeline", log_file=log_path)
    logger.info("=" * 80)
    logger.info("Stage 0 Magpie Optuna: Optimizing hyperparameters for composition-only baseline")
    logger.info("=" * 80)

    N_TRIALS = 50

    # -----------------------------------------------------------------------------------------
    # Data Loading and Preprocessing (identical to main())
    # -----------------------------------------------------------------------------------------
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")

    logger.info(f"Loading data from: {data_dir}")
    train_df_full, test_df_full = load_data(data_dir)
    logger.info(f"Loaded {len(train_df_full)} training samples and {len(test_df_full)} test samples.")

    train_df = add_target_log10_sigma(
        clean_data(train_df_full), target_sigma_col="Ionic conductivity (S cm-1)"
    )
    test_df = add_target_log10_sigma(
        clean_data(test_df_full), target_sigma_col="Ionic conductivity (S cm-1)"
    )

    if "sigma_is_coerced" in train_df.columns:
        n_coerced_train = train_df["sigma_is_coerced"].sum()
        logger.info(
            f"Coerced sigma values (train): {n_coerced_train}/{len(train_df)} "
            f"({100 * n_coerced_train / len(train_df):.1f}%)"
        )
    if "sigma_is_coerced" in test_df.columns:
        n_coerced_test = test_df["sigma_is_coerced"].sum()
        logger.info(
            f"Coerced sigma values (test): {n_coerced_test}/{len(test_df)} "
            f"({100 * n_coerced_test / len(test_df):.1f}%)"
        )

    if TARGET_COL in train_df.columns:
        train_df.dropna(subset=[TARGET_COL], inplace=True)

    logger.info(f"Training samples after cleaning: {len(train_df)}")
    logger.info(f"Test samples after cleaning: {len(test_df)}")

    # -----------------------------------------------------------------------------------------
    # Feature Engineering - Stage 0 Only (elemental ratios + SMACT + Magpie)
    # -----------------------------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("Generating Stage 0 baseline features (elemental ratios + SMACT + Magpie)...")

    base_train_df = stage0_elemental_ratios(train_df.copy(), "Reduced Composition")
    base_train_df = stage0_smact_features(base_train_df, "Reduced Composition")
    base_train_df = stage0_element_embeddings(
        base_train_df, "Reduced Composition", embedding_names=["magpie"]
    )

    base_test_df = stage0_elemental_ratios(test_df.copy(), "Reduced Composition")
    base_test_df = stage0_smact_features(base_test_df, "Reduced Composition")
    base_test_df = stage0_element_embeddings(
        base_test_df, "Reduced Composition", embedding_names=["magpie"]
    )

    # Metadata columns to exclude from features (same list as main())
    metadata_cols = [
        "Space group", "Space group #", "a", "b", "c", "alpha", "beta", "gamma", "Z",
        "IC (Total)", "IC (Bulk)", "ID", "Family", "DOI", "Checked", "Ref", "Cif ID",
        "Cif ref_1", "Cif ref_2", "note", "close match", "close match DOI", "ICSD ID",
        "Laskowski ID", "Liion ID", "True Composition", "Reduced Composition",
        "Ionic conductivity (S cm-1)",
    ]

    numeric_cols = base_train_df.select_dtypes(include=np.number).columns.tolist()
    stage0_feature_cols = [c for c in numeric_cols if c != TARGET_COL and c not in metadata_cols]
    logger.info(f"Stage 0 features: {len(stage0_feature_cols)} columns")

    # -----------------------------------------------------------------------------------------
    # Cross-Validation Setup (identical GroupKFold strategy)
    # -----------------------------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("Setting up cross-validation...")

    group_col = "group" if "group" in base_train_df.columns else None
    y = base_train_df[TARGET_COL]

    if group_col:
        groups = base_train_df[group_col].values
        logger.info(f"Using column '{group_col}' for GroupKFold cross-validation.")
    else:
        groups = np.arange(len(base_train_df))
        logger.info("No group column found. Using standard KFold behavior.")

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X=base_train_df, y=y, groups=groups))
    logger.info(f"Created {len(splits)} CV splits.")

    # -----------------------------------------------------------------------------------------
    # Prepare Feature Matrix
    # -----------------------------------------------------------------------------------------
    exp_name = "stage0_magpie_optuna"
    available_cols = [c for c in stage0_feature_cols if c in base_train_df.columns]
    logger.info(f"[{exp_name}] Using {len(available_cols)} features.")

    X = base_train_df[available_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(0)

    # -----------------------------------------------------------------------------------------
    # Optuna Hyperparameter Optimization (5-fold CV with generalization penalty)
    # -----------------------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info(f"[{exp_name}] Starting Optuna hyperparameter optimization...")

    best_params = optimize_hyperparameters(
        X, y, splits, group_col, exp_name, logger, n_trials=N_TRIALS
    )
    logger.info(f"[{exp_name}] Best hyperparameters: {best_params}")

    # -----------------------------------------------------------------------------------------
    # Cross-Validation with Optimized Hyperparameters
    # -----------------------------------------------------------------------------------------
    config = ExperimentConfig(
        model_name="hgbt",
        n_splits=5,
        random_state=42,
        params=best_params,
        group_col=group_col,
    )

    fold_scores, overall_metrics, oof = run_cv_with_predefined_splits(X, y, splits, config)

    rho_train = spearman_rho(y.values, oof)

    logger.info(f"[{exp_name}] Fold RMSEs: {[f'{s:.4f}' for s in fold_scores]}")
    logger.info(f"[{exp_name}] Overall CV Metrics:")
    logger.info(f"    R2:   {overall_metrics['r2']:.4f}")
    logger.info(f"    RMSE: {overall_metrics['rmse']:.4f}")
    logger.info(f"    MAE:  {overall_metrics['mae']:.4f}")
    logger.info(f"    Spearman rho (train OOF): {rho_train:.4f}")

    overall_metrics["rho_train"] = float(rho_train)

    # -----------------------------------------------------------------------------------------
    # CV Parity Plot
    # -----------------------------------------------------------------------------------------
    parity_plot_path = os.path.join(results_dir, f"{exp_name}_cv_parity.png")
    save_parity_plot(
        y.values,
        oof,
        parity_plot_path,
        title=f"Stage 0 5-Fold CV (Optuna): {exp_name}",
        r2_linear_from_log=False,
    )
    logger.info(f"[{exp_name}] CV parity plot saved to: {parity_plot_path}")

    # -----------------------------------------------------------------------------------------
    # Train Final Model on Full Training Set and Predict Test
    # -----------------------------------------------------------------------------------------
    logger.info(f"[{exp_name}] Training final model on full training set with optimized hyperparameters...")

    X_test = base_test_df[[c for c in available_cols if c in base_test_df.columns]].copy()
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test = X_test.fillna(0)

    preds = fit_full_and_predict(X, y, X_test, config)

    output_path = os.path.join(results_dir, f"{exp_name}_predictions.csv")
    out_df = pd.DataFrame({"ID": test_df["ID"], "prediction": preds})
    save_dataframe(out_df, output_path)
    logger.info(f"[{exp_name}] Predictions saved to: {output_path}")

    # -----------------------------------------------------------------------------------------
    # Test-Set Parity Plot and Metrics
    # -----------------------------------------------------------------------------------------
    if TARGET_COL in test_df.columns:
        test_y = test_df[TARGET_COL].values
        test_parity_path = os.path.join(results_dir, f"{exp_name}_test_parity.png")
        save_parity_plot(
            test_y,
            preds,
            test_parity_path,
            title=f"Stage 0 Test (Optuna): {exp_name}",
            r2_linear_from_log=False,
        )

        test_metrics = compute_regression_metrics(test_y, preds)
        rho_test = spearman_rho(test_y, preds)
        test_metrics["rho_test"] = float(rho_test)
        logger.info(f"[{exp_name}] Test Metrics:")
        logger.info(f"    R2:   {test_metrics['r2']:.4f}")
        logger.info(f"    RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"    MAE:  {test_metrics['mae']:.4f}")
        logger.info(f"    Spearman rho (test): {rho_test:.4f}")
        logger.info(f"[{exp_name}] Test parity plot saved to: {test_parity_path}")

    # -----------------------------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info(f"SUMMARY: {exp_name}")
    logger.info("-" * 80)
    logger.info(f"  Best hyperparameters: {best_params}")
    logger.info(
        f"  CV  -> R2: {overall_metrics['r2']:.4f}, "
        f"RMSE: {overall_metrics['rmse']:.4f}, "
        f"MAE: {overall_metrics['mae']:.4f}, "
        f"rho: {rho_train:.4f}"
    )
    if TARGET_COL in test_df.columns:
        logger.info(
            f"  Test -> R2: {test_metrics['r2']:.4f}, "
            f"RMSE: {test_metrics['rmse']:.4f}, "
            f"MAE: {test_metrics['mae']:.4f}, "
            f"rho: {rho_test:.4f}"
        )
    logger.info("=" * 80)
    logger.info("Stage 0 Magpie Optuna pipeline completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stage0_magpie":
        run_stage0_magpie_optuna()
    else:
        main()
