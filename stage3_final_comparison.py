# =================================================================================================
# Li-ion Mobility Prediction - Final Model Comparison
# =================================================================================================
# Collects the three representative models (one per project stage) and produces
# combined parity plots that show the progression of predictive performance.
#
# Models compared (all use HistGradientBoostingRegressor):
#   1. stage0_magpie_optuna  - Stage 0: composition-only (elemental ratios + SMACT + Magpie)
#                              Optuna-optimized hyperparameters.
#   2. stage2_baseline_default_geometry - Stage 1: composition + structural geometry features
#                              (no space-group one-hot). sklearn defaults (best generalizer).
#   3. stage2_physics_geometry - Stage 2: composition + geometry + physics (no space-group one-hot)
#                              (BV mismatch, Ewald energy, Voronoi CN). Optuna-optimized.
#
# Outputs (in results/results_stage3_final/):
#   - final_combined_cv_parity.png   - combined 5-fold CV parity plot
#   - final_combined_test_parity.png - combined held-out test parity plot
#   - final_models.log               - full run log
# =================================================================================================

import os
import sys
import shutil
import warnings
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold


def clear_pycache(root_path: Path):
    for pycache in root_path.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
        except Exception:
            pass


def find_project_root(start_path: str = None, marker_file: str = "requirements.txt") -> Path:
    if start_path is None:
        current = Path.cwd()
    else:
        path = Path(start_path).resolve()
        current = path.parent if path.is_file() else path
    for path in [current] + list(current.parents):
        if (path / marker_file).is_file():
            return path
    raise FileNotFoundError(f"Could not find '{marker_file}' in any parent directory.")


PROJECT_ROOT = find_project_root(__file__)
clear_pycache(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    setup_logger,
    seed_everything,
    compute_regression_metrics,
)
from src.data_processing import (
    load_data,
    clean_data,
    add_target_log10_sigma,
    TARGET_COL,
)
from src.features import (
    stage0_elemental_ratios,
    stage0_smact_features,
    stage0_element_embeddings,
    stage1_csv_structural_features,
    stage1_structural_features,
)
from src.model_training import (
    ExperimentConfig,
    run_cv_with_predefined_splits,
    fit_full_and_predict,
)

# Stage 2 physics feature extraction (defined in the optuna script)
from stage2_physics_optuna import stage2_physics_features


# =================================================================================================
# Helper: Spearman rho
# =================================================================================================

def spearman_rho(y_true, y_pred) -> float:
    s_true = pd.Series(y_true)
    s_pred = pd.Series(y_pred)
    mask = s_true.notna() & s_pred.notna()
    if mask.sum() == 0:
        return float("nan")
    r_true = s_true[mask].rank(method="average")
    r_pred = s_pred[mask].rank(method="average")
    return float(np.corrcoef(r_true, r_pred)[0, 1])


# =================================================================================================
# Combined Parity Plot
# =================================================================================================

def save_combined_parity_plot(
    models: dict,
    path: str,
    title: str = "Combined Parity Plot",
):
    """
    Generates and saves a combined parity plot with multiple models overlaid.

    Args:
        models: dict mapping display_name -> {
            "y_true": np.ndarray,
            "y_pred": np.ndarray,
            "color": str,
            "marker": str,
        }
        path: output file path
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.set_theme(style="whitegrid")

    # Compute global axis limits across all models
    all_vals = []
    for info in models.values():
        yt = np.asarray(info["y_true"], dtype=float)
        yp = np.asarray(info["y_pred"], dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        all_vals.extend(yt[mask].tolist())
        all_vals.extend(yp[mask].tolist())

    min_raw = min(all_vals)
    max_raw = max(all_vals)
    pad = 0.05 * (max_raw - min_raw) if max_raw > min_raw else 1.0
    min_val = min_raw - pad
    max_val = max_raw + pad

    # y=x reference line
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        "k--", lw=1.5, alpha=0.5, label="Perfect prediction",
    )

    # Plot each model
    for name, info in models.items():
        yt = np.asarray(info["y_true"], dtype=float)
        yp = np.asarray(info["y_pred"], dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt_clean = yt[mask]
        yp_clean = yp[mask]

        metrics = compute_regression_metrics(yt_clean, yp_clean)
        rho = spearman_rho(yt_clean, yp_clean)

        label = (
            f"{name}\n"
            f"  R\u00b2 = {metrics['r2']:.4f},  \u03c1 = {rho:.4f}"
        )

        ax.scatter(
            yt_clean,
            yp_clean,
            c=info["color"],
            marker=info["marker"],
            alpha=0.55,
            s=40,
            edgecolors="none",
            label=label,
        )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Actual log$_{10}$($\sigma$ / S cm$^{-1}$)", fontsize=13)
    ax.set_ylabel(r"Predicted log$_{10}$($\sigma$ / S cm$^{-1}$)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")

    ax.legend(
        loc="upper left",
        fontsize=9.5,
        framealpha=0.9,
        handletextpad=0.4,
        borderpad=0.6,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =================================================================================================
# Visualization 1: Side-by-Side Triptych (one panel per model, shared axes)
# =================================================================================================

def save_triptych_parity(models: dict, path: str, title: str = "Model Comparison"):
    """Three side-by-side parity subplots with identical axis limits."""
    names = list(models.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)
    sns.set_theme(style="whitegrid")

    # Global axis limits
    all_vals = []
    for info in models.values():
        yt = np.asarray(info["y_true"], dtype=float)
        yp = np.asarray(info["y_pred"], dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        all_vals.extend(yt[mask].tolist())
        all_vals.extend(yp[mask].tolist())
    mn = min(all_vals)
    mx = max(all_vals)
    pad = 0.05 * (mx - mn) if mx > mn else 1.0
    lo, hi = mn - pad, mx + pad

    for ax, name in zip(axes, names):
        info = models[name]
        yt = np.asarray(info["y_true"], dtype=float)
        yp = np.asarray(info["y_pred"], dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt_c, yp_c = yt[mask], yp[mask]

        m = compute_regression_metrics(yt_c, yp_c)
        rho = spearman_rho(yt_c, yp_c)

        ax.scatter(yt_c, yp_c, c=info["color"], alpha=0.50, s=30, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(
            f"{name}\nR\u00b2 = {m['r2']:.4f}   \u03c1 = {rho:.4f}",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel(r"Actual log$_{10}$($\sigma$)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel(r"Predicted log$_{10}$($\sigma$)", fontsize=11)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =================================================================================================
# Visualization 2: Pairwise Overlay (2 models at a time)
# =================================================================================================

def save_pairwise_parity(
    name_a: str, data_a: dict,
    name_b: str, data_b: dict,
    path: str,
    title: str = "Pairwise Comparison",
):
    """Overlay exactly two models with different colors on one parity plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.set_theme(style="whitegrid")

    all_vals = []
    for d in [data_a, data_b]:
        yt = np.asarray(d["y_true"], dtype=float)
        yp = np.asarray(d["y_pred"], dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        all_vals.extend(yt[mask].tolist())
        all_vals.extend(yp[mask].tolist())
    mn = min(all_vals)
    mx = max(all_vals)
    pad = 0.05 * (mx - mn)
    lo, hi = mn - pad, mx + pad

    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.4, label="Perfect prediction")

    for name, d in [(name_a, data_a), (name_b, data_b)]:
        yt = np.asarray(d["y_true"], dtype=float)
        yp = np.asarray(d["y_pred"], dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt_c, yp_c = yt[mask], yp[mask]
        m = compute_regression_metrics(yt_c, yp_c)
        rho = spearman_rho(yt_c, yp_c)
        label = f"{name}  (R\u00b2={m['r2']:.4f}, \u03c1={rho:.4f})"
        ax.scatter(
            yt_c, yp_c, c=d["color"], marker=d["marker"],
            alpha=0.45, s=35, edgecolors="none", label=label,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Actual log$_{10}$($\sigma$ / S cm$^{-1}$)", fontsize=12)
    ax.set_ylabel(r"Predicted log$_{10}$($\sigma$ / S cm$^{-1}$)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =================================================================================================
# Visualization 3: Residual Improvement Scatter (where does the newer model help?)
# =================================================================================================

def save_residual_improvement(
    name_old: str, data_old: dict,
    name_new: str, data_new: dict,
    path: str,
    title: str = "Residual Improvement",
):
    """
    Scatter: x = actual value, y = |error_old| - |error_new|.
    Points above zero = the newer model improved that sample.
    """
    yt = np.asarray(data_old["y_true"], dtype=float)
    yp_old = np.asarray(data_old["y_pred"], dtype=float)
    yp_new = np.asarray(data_new["y_pred"], dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp_old) & np.isfinite(yp_new)
    yt_c = yt[mask]
    err_old = np.abs(yt_c - yp_old[mask])
    err_new = np.abs(yt_c - yp_new[mask])
    improvement = err_old - err_new  # positive = new is better

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_theme(style="whitegrid")

    colors = np.where(improvement >= 0, "#2ca02c", "#d62728")
    ax.scatter(yt_c, improvement, c=colors, alpha=0.6, s=35, edgecolors="none")
    ax.axhline(0, color="k", lw=1.2, ls="--", alpha=0.5)

    n_improved = int((improvement > 0).sum())
    n_worsened = int((improvement < 0).sum())
    n_same = int((improvement == 0).sum())
    mean_imp = float(np.mean(improvement))

    ax.set_xlabel(r"Actual log$_{10}$($\sigma$ / S cm$^{-1}$)", fontsize=12)
    ax.set_ylabel(r"|error$_{\rm old}$| $-$ |error$_{\rm new}$|", fontsize=12)
    ax.set_title(
        f"{title}\n"
        f"Improved: {n_improved}  |  Worsened: {n_worsened}  |  Same: {n_same}  |  "
        f"Mean \u0394 = {mean_imp:+.4f}",
        fontsize=12, fontweight="bold",
    )

    # Add legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label=f"{name_new} better ({n_improved})"),
        Patch(facecolor="#d62728", label=f"{name_old} better ({n_worsened})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =================================================================================================
# Visualization 4: Cumulative Absolute Error Distribution (CDF)
# =================================================================================================

def save_error_cdf(models: dict, path: str, title: str = "Cumulative Error Distribution"):
    """
    CDF of |prediction error| for each model. Lines further left = more accurate.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    for name, info in models.items():
        yt = np.asarray(info["y_true"], dtype=float)
        yp = np.asarray(info["y_pred"], dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        abs_err = np.abs(yt[mask] - yp[mask])
        sorted_err = np.sort(abs_err)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)

        m = compute_regression_metrics(yt[mask], yp[mask])
        rho = spearman_rho(yt[mask], yp[mask])
        label = f"{name}  (MAE={m['mae']:.3f}, R\u00b2={m['r2']:.3f})"
        ax.plot(sorted_err, cdf, lw=2.2, color=info["color"], label=label)

    ax.set_xlabel(r"|Prediction Error|  (log$_{10}$ units)", fontsize=12)
    ax.set_ylabel("Cumulative Fraction of Samples", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =================================================================================================
# Best hyperparameters (from Optuna logs / prior runs)
# =================================================================================================

# stage0_magpie_optuna: Optuna-optimized (50 trials, 5-fold CV + generalization penalty)
STAGE0_MAGPIE_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.06987117776912892,
    "max_leaf_nodes": 54,
    "min_samples_leaf": 16,
    "l2_regularization": 0.004784957200496693,
    "max_bins": 135,
    "max_iter": 53,
}

# stage2_baseline_default_geometry: sklearn defaults (no Optuna) — best generalizer
STAGE1_GEOMETRY_PARAMS = None  # None = sklearn defaults

# stage2_physics_geometry: Optuna-optimized (50 trials, 5-fold CV + generalization penalty)
# Geometry + physics features, NO space-group one-hot encoding
STAGE2_PHYSICS_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.02604297933348663,
    "max_leaf_nodes": 81,
    "min_samples_leaf": 5,
    "l2_regularization": 0.07942148861284407,
    "max_bins": 86,
    "max_iter": 114,
}


# =================================================================================================
# Main
# =================================================================================================

def main():
    seed_everything(42)

    results_dir = os.path.join(PROJECT_ROOT, "results", "results_stage3_final")
    os.makedirs(results_dir, exist_ok=True)

    logger = setup_logger("final_models", log_file=os.path.join(results_dir, "final_models.log"))
    logger.info("=" * 80)
    logger.info("Final Model Comparison: Stage 0 -> Stage 1 -> Stage 2")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------------------
    # Data Loading (identical to the Optuna pipeline)
    # -------------------------------------------------------------------------------------
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    logger.info(f"Loading data from: {data_dir}")

    train_df_full, test_df_full = load_data(data_dir)
    train_df = add_target_log10_sigma(
        clean_data(train_df_full), target_sigma_col="Ionic conductivity (S cm-1)"
    )
    test_df = add_target_log10_sigma(
        clean_data(test_df_full), target_sigma_col="Ionic conductivity (S cm-1)"
    )

    if TARGET_COL in train_df.columns:
        train_df.dropna(subset=[TARGET_COL], inplace=True)

    logger.info(f"Training samples: {len(train_df)},  Test samples: {len(test_df)}")

    # CIF directories (needed for Stage 1 + 2)
    train_cif_dir = os.path.join(data_dir, "train_cifs")
    test_cif_dir = os.path.join(data_dir, "test_cifs")
    train_cif_files = sorted(
        [f.replace(".cif", "") for f in os.listdir(train_cif_dir) if f.endswith(".cif")]
    )
    test_cif_files = sorted(
        [f.replace(".cif", "") for f in os.listdir(test_cif_dir) if f.endswith(".cif")]
    )

    # -------------------------------------------------------------------------------------
    # Feature Engineering - Stage 0 (composition only)
    # -------------------------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("Feature Engineering: Stage 0 (elemental ratios + SMACT + Magpie)")

    base_train = stage0_elemental_ratios(train_df.copy(), "Reduced Composition")
    base_train = stage0_smact_features(base_train, "Reduced Composition")
    base_train = stage0_element_embeddings(base_train, "Reduced Composition", embedding_names=["magpie"])

    base_test = stage0_elemental_ratios(test_df.copy(), "Reduced Composition")
    base_test = stage0_smact_features(base_test, "Reduced Composition")
    base_test = stage0_element_embeddings(base_test, "Reduced Composition", embedding_names=["magpie"])

    metadata_cols = [
        "Space group", "Space group #", "a", "b", "c", "alpha", "beta", "gamma", "Z",
        "IC (Total)", "IC (Bulk)", "ID", "Family", "DOI", "Checked", "Ref", "Cif ID",
        "Cif ref_1", "Cif ref_2", "note", "close match", "close match DOI", "ICSD ID",
        "Laskowski ID", "Liion ID", "True Composition", "Reduced Composition",
        "Ionic conductivity (S cm-1)",
    ]

    numeric_cols = base_train.select_dtypes(include=np.number).columns.tolist()
    stage0_feature_cols = [c for c in numeric_cols if c != TARGET_COL and c not in metadata_cols]
    logger.info(f"  Stage 0 features: {len(stage0_feature_cols)}")

    # -------------------------------------------------------------------------------------
    # Feature Engineering - Stage 1 (structural geometry from CSV + CIF)
    # -------------------------------------------------------------------------------------
    logger.info("Feature Engineering: Stage 1 (CSV + CIF structural geometry)")

    csv_train_struct = stage1_csv_structural_features(base_train.copy())
    csv_test_struct = stage1_csv_structural_features(base_test.copy())

    struct_train, _ = stage1_structural_features(
        base_train.copy(), id_col="ID", cif_dir=train_cif_dir, extended=True, verbose=False,
    )
    struct_test, _ = stage1_structural_features(
        base_test.copy(), id_col="ID", cif_dir=test_cif_dir, extended=True, verbose=False,
    )

    csv_struct_cols = [
        "spacegroup_number", "formula_units_z",
        "lattice_a", "lattice_b", "lattice_c",
        "lattice_alpha", "lattice_beta", "lattice_gamma",
        "lattice_volume", "lattice_anisotropy", "angle_deviation_from_ortho", "is_cubic_like",
        "density", "volume_per_atom", "n_li_sites",
    ]
    for col in csv_struct_cols:
        if col in csv_train_struct.columns:
            if col in struct_train.columns:
                struct_train[col] = struct_train[col].fillna(csv_train_struct[col])
            else:
                struct_train[col] = csv_train_struct[col]
        if col in csv_test_struct.columns:
            if col in struct_test.columns:
                struct_test[col] = struct_test[col].fillna(csv_test_struct[col])
            else:
                struct_test[col] = csv_test_struct[col]

    struct_train["has_cif_struct"] = struct_train["ID"].isin(train_cif_files).astype(int)
    struct_test["has_cif_struct"] = struct_test["ID"].isin(test_cif_files).astype(int)

    cif_only_cols = [
        "li_fraction", "li_concentration", "framework_density",
        "li_li_min_dist", "li_li_avg_dist", "li_anion_min_dist",
        "li_coordination_avg", "li_site_avg_multiplicity",
        "lattice_anisotropy_bc_a", "lattice_anisotropy_max_min",
    ]
    for col in cif_only_cols:
        if col in struct_train.columns:
            struct_train[col] = struct_train[col].fillna(0)
        if col in struct_test.columns:
            struct_test[col] = struct_test[col].fillna(0)

    # -------------------------------------------------------------------------------------
    # Feature Engineering - Stage 2 (physics-informed features on geometry, NO space-group one-hot)
    # -------------------------------------------------------------------------------------
    logger.info("Feature Engineering: Stage 2 (BV mismatch, Ewald energy, Voronoi CN)")

    physics_train, train_phys_stats = stage2_physics_features(
        struct_train.copy(), id_col="ID", cif_dir=train_cif_dir, verbose=False,
    )
    physics_test, test_phys_stats = stage2_physics_features(
        struct_test.copy(), id_col="ID", cif_dir=test_cif_dir, verbose=False,
    )

    logger.info(
        f"  Physics extraction (train): processed={train_phys_stats['processed']}, "
        f"BV={train_phys_stats['with_bv']}, Ewald={train_phys_stats['with_ewald']}, "
        f"Voronoi={train_phys_stats['with_voronoi']}"
    )

    physics_value_cols = [
        "bv_mismatch_avg", "bv_mismatch_std",
        "ewald_energy_avg", "ewald_energy_std",
        "li_voronoi_cn_avg",
    ]
    physics_indicator_cols = ["has_bv_mismatch", "has_ewald_energy", "has_voronoi_cn"]
    physics_feature_cols = physics_value_cols + physics_indicator_cols

    for col in physics_value_cols:
        if col in physics_train.columns:
            physics_train[col] = physics_train[col].fillna(0)
        if col in physics_test.columns:
            physics_test[col] = physics_test[col].fillna(0)
    for col in physics_indicator_cols:
        if col in physics_train.columns:
            physics_train[col] = physics_train[col].fillna(0).astype(int)
        if col in physics_test.columns:
            physics_test[col] = physics_test[col].fillna(0).astype(int)

    # -------------------------------------------------------------------------------------
    # Define feature sets
    # -------------------------------------------------------------------------------------
    basic_struct_cols = ["density", "volume_per_atom", "n_li_sites", "n_total_atoms"]

    geometry_cols = basic_struct_cols + [
        "lattice_a", "lattice_b", "lattice_c",
        "lattice_alpha", "lattice_beta", "lattice_gamma",
        "lattice_anisotropy_bc_a", "lattice_anisotropy_max_min",
        "angle_deviation_from_ortho", "is_cubic_like",
        "li_fraction", "li_concentration", "framework_density",
        "li_li_min_dist", "li_li_avg_dist", "li_anion_min_dist",
        "li_coordination_avg", "li_site_avg_multiplicity",
        "has_cif_struct",
    ]
    stage1_geometry_cols = stage0_feature_cols + geometry_cols
    # Stage 2 physics uses geometry + physics (NO space-group one-hot)
    stage2_physics_cols = stage1_geometry_cols + physics_feature_cols

    logger.info(f"  Stage 0 features: {len(stage0_feature_cols)}")
    logger.info(f"  Stage 1 geometry features: {len(stage1_geometry_cols)}")
    logger.info(f"  Stage 2 physics features: {len(stage2_physics_cols)}")

    # -------------------------------------------------------------------------------------
    # CV splits (same GroupKFold as the Optuna pipeline)
    # -------------------------------------------------------------------------------------
    group_col = "group" if "group" in physics_train.columns else None
    y_train = physics_train[TARGET_COL]

    if group_col:
        groups = physics_train[group_col].values
    else:
        groups = np.arange(len(physics_train))

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X=physics_train, y=y_train, groups=groups))
    logger.info(f"CV splits: {len(splits)} folds")

    # -------------------------------------------------------------------------------------
    # Define the three models
    # -------------------------------------------------------------------------------------
    models_spec = {
        "Stage 0: Composition (Magpie)": {
            "train_df": base_train,
            "test_df": base_test,
            "feature_cols": stage0_feature_cols,
            "params": STAGE0_MAGPIE_PARAMS,
            "color": "#1f77b4",   # blue
            "marker": "o",
        },
        "Stage 1: + Geometry": {
            "train_df": struct_train,
            "test_df": struct_test,
            "feature_cols": stage1_geometry_cols,
            "params": STAGE1_GEOMETRY_PARAMS,
            "color": "#ff7f0e",   # orange
            "marker": "s",
        },
        "Stage 2: + Physics": {
            "train_df": physics_train,
            "test_df": physics_test,
            "feature_cols": stage2_physics_cols,
            "params": STAGE2_PHYSICS_PARAMS,
            "color": "#2ca02c",   # green
            "marker": "D",
        },
    }

    # -------------------------------------------------------------------------------------
    # Run each model: CV (for OOF) + final fit (for test predictions)
    # -------------------------------------------------------------------------------------
    cv_plot_data = {}
    test_plot_data = {}

    for name, spec in models_spec.items():
        logger.info("=" * 80)
        logger.info(f"Model: {name}")

        available = [c for c in spec["feature_cols"] if c in spec["train_df"].columns]
        logger.info(f"  Features: {len(available)}")

        X = spec["train_df"][available].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.fillna(0)

        config = ExperimentConfig(
            model_name="hgbt",
            n_splits=5,
            random_state=42,
            params=spec["params"],
            group_col=group_col,
        )

        # 5-fold CV
        fold_scores, cv_metrics, oof = run_cv_with_predefined_splits(X, y_train, splits, config)
        rho_cv = spearman_rho(y_train.values, oof)

        logger.info(f"  CV  Fold RMSEs: {[f'{s:.4f}' for s in fold_scores]}")
        logger.info(f"  CV  R2={cv_metrics['r2']:.4f}  RMSE={cv_metrics['rmse']:.4f}  "
                     f"MAE={cv_metrics['mae']:.4f}  rho={rho_cv:.4f}")

        cv_plot_data[name] = {
            "y_true": y_train.values,
            "y_pred": oof,
            "color": spec["color"],
            "marker": spec["marker"],
        }

        # Final model + test predictions
        X_test = spec["test_df"][[c for c in available if c in spec["test_df"].columns]].copy()
        X_test = X_test.reindex(columns=X.columns, fill_value=0)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test = X_test.fillna(0)

        preds = fit_full_and_predict(X, y_train, X_test, config)

        if TARGET_COL in test_df.columns:
            test_y = test_df[TARGET_COL].values
            test_metrics = compute_regression_metrics(test_y, preds)
            rho_test = spearman_rho(test_y, preds)
            logger.info(f"  Test R2={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.4f}  "
                         f"MAE={test_metrics['mae']:.4f}  rho={rho_test:.4f}")

            test_plot_data[name] = {
                "y_true": test_y,
                "y_pred": preds,
                "color": spec["color"],
                "marker": spec["marker"],
            }

    # -------------------------------------------------------------------------------------
    # Visualization: Combined Overlay (original, kept for reference)
    # -------------------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Generating visualizations...")

    cv_parity_path = os.path.join(results_dir, "final_combined_cv_parity.png")
    save_combined_parity_plot(
        cv_plot_data, cv_parity_path,
        title="5-Fold Cross-Validation: Model Progression",
    )
    logger.info(f"  Combined CV overlay -> {cv_parity_path}")

    test_parity_path = os.path.join(results_dir, "final_combined_test_parity.png")
    save_combined_parity_plot(
        test_plot_data, test_parity_path,
        title="Held-Out Test Set: Model Progression",
    )
    logger.info(f"  Combined test overlay -> {test_parity_path}")

    # -------------------------------------------------------------------------------------
    # Visualization 1: Side-by-side triptych (one panel per model)
    # -------------------------------------------------------------------------------------
    save_triptych_parity(
        cv_plot_data,
        os.path.join(results_dir, "final_triptych_cv.png"),
        title="5-Fold Cross-Validation",
    )
    logger.info("  Triptych CV -> final_triptych_cv.png")

    save_triptych_parity(
        test_plot_data,
        os.path.join(results_dir, "final_triptych_test.png"),
        title="Held-Out Test Set",
    )
    logger.info("  Triptych test -> final_triptych_test.png")

    # -------------------------------------------------------------------------------------
    # Visualization 2: Pairwise overlays (Stage 0 vs 1, Stage 1 vs 2)
    # -------------------------------------------------------------------------------------
    names = list(models_spec.keys())

    for dataset_label, plot_data in [("CV", cv_plot_data), ("Test", test_plot_data)]:
        save_pairwise_parity(
            names[0], plot_data[names[0]],
            names[1], plot_data[names[1]],
            os.path.join(results_dir, f"final_pairwise_s0_vs_s1_{dataset_label.lower()}.png"),
            title=f"{dataset_label}: Stage 0 vs Stage 1",
        )
        save_pairwise_parity(
            names[1], plot_data[names[1]],
            names[2], plot_data[names[2]],
            os.path.join(results_dir, f"final_pairwise_s1_vs_s2_{dataset_label.lower()}.png"),
            title=f"{dataset_label}: Stage 1 vs Stage 2",
        )
    logger.info("  Pairwise overlays -> final_pairwise_*.png")

    # -------------------------------------------------------------------------------------
    # Visualization 3: Residual improvement scatter (where does each stage help?)
    # -------------------------------------------------------------------------------------
    for dataset_label, plot_data in [("CV", cv_plot_data), ("Test", test_plot_data)]:
        save_residual_improvement(
            names[0], plot_data[names[0]],
            names[1], plot_data[names[1]],
            os.path.join(results_dir, f"final_residual_improvement_s0_to_s1_{dataset_label.lower()}.png"),
            title=f"{dataset_label}: Improvement from Stage 0 \u2192 Stage 1",
        )
        save_residual_improvement(
            names[1], plot_data[names[1]],
            names[2], plot_data[names[2]],
            os.path.join(results_dir, f"final_residual_improvement_s1_to_s2_{dataset_label.lower()}.png"),
            title=f"{dataset_label}: Improvement from Stage 1 \u2192 Stage 2",
        )
    logger.info("  Residual improvement -> final_residual_improvement_*.png")

    # -------------------------------------------------------------------------------------
    # Visualization 4: Cumulative error CDF (all 3 models, clean comparison)
    # -------------------------------------------------------------------------------------
    save_error_cdf(
        cv_plot_data,
        os.path.join(results_dir, "final_error_cdf_cv.png"),
        title="Cumulative |Error| Distribution (5-Fold CV)",
    )
    save_error_cdf(
        test_plot_data,
        os.path.join(results_dir, "final_error_cdf_test.png"),
        title="Cumulative |Error| Distribution (Test Set)",
    )
    logger.info("  Error CDF -> final_error_cdf_*.png")

    # -------------------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("-" * 80)
    header = f"{'Model':<35s} | {'CV R2':>7s} | {'CV rho':>7s} | {'Test R2':>7s} | {'Test rho':>7s}"
    logger.info(header)
    logger.info("-" * 80)
    for name in models_spec:
        cv_m = compute_regression_metrics(
            cv_plot_data[name]["y_true"], cv_plot_data[name]["y_pred"]
        )
        cv_r = spearman_rho(cv_plot_data[name]["y_true"], cv_plot_data[name]["y_pred"])
        if name in test_plot_data:
            te_m = compute_regression_metrics(
                test_plot_data[name]["y_true"], test_plot_data[name]["y_pred"]
            )
            te_r = spearman_rho(test_plot_data[name]["y_true"], test_plot_data[name]["y_pred"])
            logger.info(
                f"{name:<35s} | {cv_m['r2']:7.4f} | {cv_r:7.4f} | {te_m['r2']:7.4f} | {te_r:7.4f}"
            )
        else:
            logger.info(f"{name:<35s} | {cv_m['r2']:7.4f} | {cv_r:7.4f} |     N/A |     N/A")

    logger.info("=" * 80)
    logger.info("Final model comparison completed successfully!")


if __name__ == "__main__":
    main()
