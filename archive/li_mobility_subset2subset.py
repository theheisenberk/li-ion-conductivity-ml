import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold

import optuna

from src.utils import (
    find_project_root,
    setup_logger,
    seed_everything,
    save_histogram,
    save_dataframe,
    compute_regression_metrics,
)
from src.data_processing import (
    clean_data,
    add_target_log10_sigma,
    coerce_sigma_series,
    TARGET_COL,
)
from src.features import (
    stage0_elemental_ratios,
    stage0_smact_features,
    stage0_element_embeddings,
    stage1_csv_structural_features,
    stage1_structural_features,
    stage1_spacegroup_onehot,
)
from src.model_training import ExperimentConfig, run_cv_with_predefined_splits, fit_full_and_predict

# Reuse Stage 2 physics + Optuna helpers
from li_mobility_advanced_v2_optuna import (
    stage2_physics_features,
    OPTUNA_SEARCH_SPACE,
    SIMPLE_SEARCH_SPACE,
    OPTUNA_VAL_FRACTION,
    OPTUNA_VAL_RANDOM_STATE,
    create_optuna_objective,
    optimize_hyperparameters,
)


def _create_results_dir(project_root: Path) -> Path:
    results_dir = project_root / "results" / "results_subset2subset"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _setup_logger(results_dir: Path):
    log_path = results_dir / "subset2subset.log"
    logger = setup_logger("subset2subset_pipeline", log_file=str(log_path))
    return logger


def _load_and_preprocess_data(project_root: Path, results_dir: Path, logger):
    """
    Data loading + preprocessing, identical to Stage 2 pipeline.
    """
    data_dir = os.path.join(project_root, "data", "raw")

    # We reuse the same logic as in Stage 2: use src.data_processing.load_data
    from src.data_processing import load_data

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

    # Clean and transform data (adds sigma_is_coerced and TARGET_COL)
    train_df = add_target_log10_sigma(clean_data(train_df_full), target_sigma_col=sigma_col)
    test_df = add_target_log10_sigma(clean_data(test_df_full), target_sigma_col=sigma_col)

    # Log coerced sigma statistics
    if "sigma_is_coerced" in train_df.columns:
        n_coerced_train = train_df["sigma_is_coerced"].sum()
        logger.info(f"Coerced sigma values (train): {n_coerced_train}/{len(train_df)} ({100*n_coerced_train/len(train_df):.1f}%)")
    if "sigma_is_coerced" in test_df.columns:
        n_coerced_test = test_df["sigma_is_coerced"].sum()
        logger.info(f"Coerced sigma values (test): {n_coerced_test}/{len(test_df)} ({100*n_coerced_test/len(test_df):.1f}%)")

    # Drop rows with missing target
    if TARGET_COL in train_df.columns:
        train_df = train_df.dropna(subset=[TARGET_COL])

    logger.info(
        f"Training samples (total): {len(train_df)} "
        f"(with CIFs: {sum(train_df['ID'].isin(train_cif_files))})"
    )
    logger.info(
        f"Test samples (total): {len(test_df)} "
        f"(with CIFs: {sum(test_df['ID'].isin(test_cif_files))})"
    )

    return (
        train_df,
        test_df,
        train_cif_dir,
        test_cif_dir,
        train_cif_files,
        test_cif_files,
    )


def _build_stage0_features(train_df: pd.DataFrame, test_df: pd.DataFrame, logger):
    logger.info("-" * 80)
    logger.info("Generating Stage 0 baseline features (Magpie embeddings)...")

    base_train_df = stage0_elemental_ratios(train_df.copy(), "Reduced Composition")
    base_train_df = stage0_smact_features(base_train_df, "Reduced Composition")
    base_train_df = stage0_element_embeddings(base_train_df, "Reduced Composition", embedding_names=["magpie"])

    base_test_df = stage0_elemental_ratios(test_df.copy(), "Reduced Composition")
    base_test_df = stage0_smact_features(base_test_df, "Reduced Composition")
    base_test_df = stage0_element_embeddings(base_test_df, "Reduced Composition", embedding_names=["magpie"])

    # Define metadata columns to exclude from features (same as Stage 2)
    metadata_cols = [
        "Space group",
        "Space group #",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "Z",
        "IC (Total)",
        "IC (Bulk)",
        "ID",
        "Family",
        "DOI",
        "Checked",
        "Ref",
        "Cif ID",
        "Cif ref_1",
        "Cif ref_2",
        "note",
        "close match",
        "close match DOI",
        "ICSD ID",
        "Laskowski ID",
        "Liion ID",
        "True Composition",
        "Reduced Composition",
        "Ionic conductivity (S cm-1)",
    ]

    numeric_cols = base_train_df.select_dtypes(include=np.number).columns.tolist()
    stage0_feature_cols = [c for c in numeric_cols if c != TARGET_COL and c not in metadata_cols]
    logger.info(f"Stage 0 features: {len(stage0_feature_cols)} columns")

    return base_train_df, base_test_df, stage0_feature_cols


def _build_stage1_features(
    base_train_df: pd.DataFrame,
    base_test_df: pd.DataFrame,
    train_cif_dir: str,
    test_cif_dir: str,
    train_cif_files: List[str],
    test_cif_files: List[str],
    logger,
):
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

    logger.info("Stage 1 CIF parsing statistics:")
    logger.info(
        f"  Training -> full: {train_stats['parsed_full']}, "
        f"partial: {train_stats['parsed_partial']}, missing: {train_stats['missing_cif']}"
    )
    logger.info(
        f"  Test     -> full: {test_stats['parsed_full']}, "
        f"partial: {test_stats['parsed_partial']}, missing: {test_stats['missing_cif']}"
    )

    # Merge CSV metadata into CIF DataFrame (fill gaps)
    csv_struct_cols = [
        "spacegroup_number",
        "formula_units_z",
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "lattice_alpha",
        "lattice_beta",
        "lattice_gamma",
        "lattice_volume",
        "lattice_anisotropy",
        "angle_deviation_from_ortho",
        "is_cubic_like",
        "density",
        "volume_per_atom",
        "n_li_sites",
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
        "li_fraction",
        "li_concentration",
        "framework_density",
        "li_li_min_dist",
        "li_li_avg_dist",
        "li_anion_min_dist",
        "li_coordination_avg",
        "li_site_avg_multiplicity",
        "lattice_anisotropy_bc_a",
        "lattice_anisotropy_max_min",
    ]
    for col in cif_only_cols:
        if col in struct_train_df.columns:
            struct_train_df[col] = struct_train_df[col].fillna(0)
        if col in struct_test_df.columns:
            struct_test_df[col] = struct_test_df[col].fillna(0)

    # Add space group one-hot encoding
    full_train_df = stage1_spacegroup_onehot(struct_train_df.copy(), spacegroup_col="spacegroup_number")
    full_test_df = stage1_spacegroup_onehot(struct_test_df.copy(), spacegroup_col="spacegroup_number")

    return full_train_df, full_test_df


def _add_stage2_physics_features(
    full_train_df: pd.DataFrame,
    full_test_df: pd.DataFrame,
    train_cif_dir: str,
    test_cif_dir: str,
    logger,
):
    logger.info("-" * 80)
    logger.info("Extracting Stage 2 physics-informed features...")

    physics_train_df, train_physics_stats = stage2_physics_features(
        full_train_df.copy(), id_col="ID", cif_dir=train_cif_dir, verbose=False
    )
    physics_test_df, test_physics_stats = stage2_physics_features(
        full_test_df.copy(), id_col="ID", cif_dir=test_cif_dir, verbose=False
    )

    logger.info(
        "Stage 2 physics feature extraction: "
        f"Training -> processed: {train_physics_stats['processed']}, "
        f"with BV: {train_physics_stats['with_bv']}, "
        f"with Ewald: {train_physics_stats['with_ewald']}, "
        f"with Voronoi: {train_physics_stats['with_voronoi']}"
    )
    logger.info(
        "Stage 2 physics feature extraction: "
        f"Test -> processed: {test_physics_stats['processed']}, "
        f"with BV: {test_physics_stats['with_bv']}, "
        f"with Ewald: {test_physics_stats['with_ewald']}, "
        f"with Voronoi: {test_physics_stats['with_voronoi']}"
    )

    physics_value_cols = [
        "bv_mismatch_avg",
        "bv_mismatch_std",
        "ewald_energy_avg",
        "ewald_energy_std",
        "li_voronoi_cn_avg",
    ]
    physics_indicator_cols = [
        "has_bv_mismatch",
        "has_ewald_energy",
        "has_voronoi_cn",
    ]
    physics_feature_cols = physics_value_cols + physics_indicator_cols

    # Zero-fill physics VALUE features for samples without valid calculations
    for col in physics_value_cols:
        if col in physics_train_df.columns:
            physics_train_df[col] = physics_train_df[col].fillna(0)
        if col in physics_test_df.columns:
            physics_test_df[col] = physics_test_df[col].fillna(0)

    # Fill indicator columns with 0 where NaN (defensive)
    for col in physics_indicator_cols:
        if col in physics_train_df.columns:
            physics_train_df[col] = physics_train_df[col].fillna(0).astype(int)
        if col in physics_test_df.columns:
            physics_test_df[col] = physics_test_df[col].fillna(0).astype(int)

    logger.info("Physics feature coverage (training) - based on indicator variables:")
    for indicator_col in physics_indicator_cols:
        if indicator_col in physics_train_df.columns:
            calculated = physics_train_df[indicator_col].sum()
            logger.info(
                f"  {indicator_col}: {calculated}/{len(physics_train_df)} "
                f"({100*calculated/len(physics_train_df):.1f}%)"
            )

    logger.info("Physics value features (non-zero after zero-fill):")
    for col in physics_value_cols:
        if col in physics_train_df.columns:
            non_zero = (physics_train_df[col] != 0).sum()
            logger.info(
                f"  {col}: {non_zero}/{len(physics_train_df)} "
                f"({100*non_zero/len(physics_train_df):.1f}%)"
            )

    return physics_train_df, physics_test_df, physics_feature_cols


def _build_feature_sets(stage0_feature_cols: List[str]):
    # Basic structural columns (no space group one-hot)
    basic_struct_cols = [
        "density",
        "volume_per_atom",
        "n_li_sites",
        "n_total_atoms",
    ]

    # Full geometry columns (same as stage1_geometry)
    geometry_cols = basic_struct_cols + [
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "lattice_alpha",
        "lattice_beta",
        "lattice_gamma",
        "lattice_anisotropy_bc_a",
        "lattice_anisotropy_max_min",
        "angle_deviation_from_ortho",
        "is_cubic_like",
        "li_fraction",
        "li_concentration",
        "framework_density",
        "li_li_min_dist",
        "li_li_avg_dist",
        "li_anion_min_dist",
        "li_coordination_avg",
        "li_site_avg_multiplicity",
        "has_cif_struct",
    ]

    # Space group one-hot columns
    spacegroup_cols = [f"sg_{i}" for i in range(1, 231)]

    # Stage 1 full_struct = Stage 0 + geometry + space group one-hot
    stage1_full_struct_cols = stage0_feature_cols + geometry_cols + spacegroup_cols

    return stage1_full_struct_cols


def _spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation without requiring scipy."""
    s_true = pd.Series(y_true)
    s_pred = pd.Series(y_pred)
    mask = s_true.notna() & s_pred.notna()
    if mask.sum() == 0:
        return float("nan")
    r_true = s_true[mask].rank(method="average")
    r_pred = s_pred[mask].rank(method="average")
    return float(np.corrcoef(r_true, r_pred)[0, 1])


def _create_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    model_name: str,
    logger,
) -> None:
    """
    Create a parity plot with mean predictions only (no quantile regression).
    Shows R² and Spearman ρ for mean predictions.
    """
    y_true_plot = np.asarray(y_true, dtype=float)
    y_pred_plot = np.asarray(y_pred, dtype=float)
    mask_plot = np.isfinite(y_true_plot) & np.isfinite(y_pred_plot)
    y_true_plot = y_true_plot[mask_plot]
    y_pred_plot = y_pred_plot[mask_plot]

    if len(y_true_plot) == 0:
        logger.warning(f"[{model_name}] No valid data points for parity plot.")
        return

    # Compute metrics
    metrics = compute_regression_metrics(y_true_plot, y_pred_plot)
    rho = _spearman_rho(y_true_plot, y_pred_plot)

    # Create plot
    plt.figure(figsize=(8, 8))
    sns.set_theme(style="whitegrid")

    ax = sns.scatterplot(
        x=y_true_plot,
        y=y_pred_plot,
        alpha=0.6,
        s=50,
        edgecolor="k",
        label="Predictions",
    )

    # y = x reference line
    min_raw = float(min(np.nanmin(y_true_plot), np.nanmin(y_pred_plot)))
    max_raw = float(max(np.nanmax(y_true_plot), np.nanmax(y_pred_plot)))
    pad = 0.05 * (max_raw - min_raw) if max_raw > min_raw else 1.0
    min_val = min_raw - pad
    max_val = max_raw + pad
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

    ax.set_xlabel(r"Actual log10($\sigma$)", fontsize=14)
    ax.set_ylabel(r"Predicted log10($\sigma$)", fontsize=14)
    ax.set_title(f"Test Set: {model_name}", fontsize=16)

    # Text box with metrics
    text_lines = [
        rf"$R^2 = {metrics['r2']:.3f}$",
        rf"$\rho = {rho:.3f}$" if np.isfinite(rho) else r"$\rho = \text{nan}$",
        rf"RMSE = {metrics['rmse']:.3f}",
        rf"MAE = {metrics['mae']:.3f}",
    ]
    ax.text(
        0.05,
        0.95,
        "\n".join(text_lines),
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.legend()
    plt.tight_layout()
    os.makedirs(path.parent, exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()

    logger.info(
        f"[{model_name}] Parity plot saved to {path} "
        f"(R²={metrics['r2']:.4f}, ρ={rho:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f})"
    )


def _compute_subset2subset_splits(
    df: pd.DataFrame,
    target_col: str,
    logger,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], int, str]:
    """
    Build GroupKFold splits for the subset2subset dataset.
    """
    y = df[target_col]
    group_col = "group" if "group" in df.columns else None

    if group_col:
        groups = df[group_col].values
        logger.info(f"Using column '{group_col}' for GroupKFold in subset2subset.")
    else:
        groups = np.arange(len(df))
        logger.info("No group column found for subset2subset; using standard KFold behavior.")

    unique_groups = np.unique(groups)
    n_unique_groups = len(unique_groups)
    n_splits = min(5, n_unique_groups)
    if n_splits < 2:
        raise ValueError(
            f"Not enough groups/samples for cross-validation in subset2subset: n_unique_groups={n_unique_groups}"
        )

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X=df, y=y, groups=groups))
    logger.info(f"Created {len(splits)} CV splits for subset2subset (n_splits={n_splits}).")

    return y, splits, n_splits, group_col


def main():
    # Project / seed setup
    project_root = find_project_root(__file__)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    seed_everything(42)

    # Results and logging
    results_dir = _create_results_dir(project_root)
    logger = _setup_logger(results_dir)

    logger.info("=" * 80)
    logger.info("Subset2Subset Stage 2 Physics-Informed Model with Optuna")
    logger.info("Train subset: full coverage of BV, Ewald, Voronoi;")
    logger.info("Test subset: same full-coverage filter applied.")
    logger.info("=" * 80)

    # Data + preprocessing (same as Stage 2)
    (
        train_df,
        test_df,
        train_cif_dir,
        test_cif_dir,
        train_cif_files,
        test_cif_files,
    ) = _load_and_preprocess_data(project_root, results_dir, logger)

    # Stage 0
    base_train_df, base_test_df, stage0_feature_cols = _build_stage0_features(train_df, test_df, logger)

    # Stage 1
    full_train_df, full_test_df = _build_stage1_features(
        base_train_df,
        base_test_df,
        train_cif_dir,
        test_cif_dir,
        train_cif_files,
        test_cif_files,
        logger,
    )

    # Stage 2 physics features
    physics_train_df, physics_test_df, physics_feature_cols = _add_stage2_physics_features(
        full_train_df,
        full_test_df,
        train_cif_dir,
        test_cif_dir,
        logger,
    )

    # Build full Stage 2 physics feature set (Stage 1 full_struct + physics features)
    stage1_full_struct_cols = _build_feature_sets(stage0_feature_cols)
    stage2_physics_cols = stage1_full_struct_cols + physics_feature_cols

    logger.info(
        f"Feature set sizes (subset2subset): "
        f"Stage 0 = {len(stage0_feature_cols)}, "
        f"Stage 2 physics extras = {len(physics_feature_cols)}, "
        f"Total Stage 2 physics = {len(stage2_physics_cols)}"
    )

    # -----------------------------------------------------------------------------------------
    # Subset2Subset selection: full coverage of BV, Ewald, Voronoi
    # -----------------------------------------------------------------------------------------
    full_cov_mask_train = (
        (physics_train_df["has_bv_mismatch"] == 1)
        & (physics_train_df["has_ewald_energy"] == 1)
        & (physics_train_df["has_voronoi_cn"] == 1)
    )
    full_cov_mask_test = (
        (physics_test_df["has_bv_mismatch"] == 1)
        & (physics_test_df["has_ewald_energy"] == 1)
        & (physics_test_df["has_voronoi_cn"] == 1)
    )

    subset_train_df = physics_train_df.loc[full_cov_mask_train].copy()
    subset_test_df = physics_test_df.loc[full_cov_mask_test].copy()

    logger.info(
        f"Subset2Subset train size: {len(subset_train_df)} "
        f"(out of {len(physics_train_df)})"
    )
    logger.info(
        f"Subset2Subset test size: {len(subset_test_df)} "
        f"(out of {len(physics_test_df)})"
    )

    # Prepare feature matrix for subset2subset model
    feature_cols = [c for c in stage2_physics_cols if c in subset_train_df.columns]
    logger.info(f"Subset2Subset: using {len(feature_cols)} features.")

    X_subset_full = subset_train_df[feature_cols].copy()
    X_subset_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_subset_full = X_subset_full.fillna(0)

    # Build CV splits and identify group column
    y_subset, subset_splits, n_splits, group_col = _compute_subset2subset_splits(
        subset_train_df, TARGET_COL, logger
    )

    # -----------------------------------------------------------------------------------------
    # Optuna hyperparameter optimization (reuse Stage 2 helper)
    # -----------------------------------------------------------------------------------------
    N_TRIALS = 50
    exp_name = "stage2_physics_subset2subset"
    logger.info(
        f"[{exp_name}] Starting Optuna optimization on subset2subset "
        f"(n_samples={len(X_subset_full)}, n_features={len(feature_cols)}, n_splits={n_splits})"
    )

    best_params = optimize_hyperparameters(
        X_subset_full,
        y_subset,
        subset_splits,
        group_col,
        exp_name,
        logger,
        n_trials=N_TRIALS,
    )

    # -----------------------------------------------------------------------------------------
    # Cross-validation with best params + final training on full subset
    # -----------------------------------------------------------------------------------------
    config = ExperimentConfig(
        model_name="hgbt",
        n_splits=n_splits,
        random_state=42,
        params=best_params,
        group_col=group_col,
    )

    # For CV, we need a DataFrame that includes group_col (if any)
    if group_col:
        X_cv = subset_train_df[feature_cols + [group_col]].copy()
    else:
        X_cv = subset_train_df[feature_cols].copy()

    fold_scores, overall_metrics, oof_preds = run_cv_with_predefined_splits(
        X_cv,
        y_subset,
        subset_splits,
        config,
    )

    logger.info(f"[{exp_name}] Fold RMSEs (subset2subset): {[f'{s:.4f}' for s in fold_scores]}")
    logger.info(
        f"[{exp_name}] OOF metrics (subset2subset): "
        f"R2={overall_metrics['r2']:.4f}, "
        f"RMSE={overall_metrics['rmse']:.4f}, "
        f"MAE={overall_metrics['mae']:.4f}"
    )

    # Save OOF predictions for reference
    oof_df = pd.DataFrame(
        {
            "ID": subset_train_df["ID"].values,
            TARGET_COL: y_subset.values,
            "oof_prediction": oof_preds,
        }
    )
    oof_path = results_dir / f"{exp_name}_oof_predictions.csv"
    save_dataframe(oof_df, str(oof_path))
    logger.info(f"[{exp_name}] Saved OOF predictions to {oof_path}")

    # Final training on full subset and prediction on subset of test set
    X_subset_train_full = X_cv  # already includes group_col if present
    if group_col:
        X_subset_test = subset_test_df[feature_cols + [group_col]].copy()
    else:
        X_subset_test = subset_test_df[feature_cols].copy()

    X_subset_train_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_subset_train_full = X_subset_train_full.fillna(0)
    X_subset_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_subset_test = X_subset_test.fillna(0)

    test_preds = fit_full_and_predict(
        X_subset_train_full,
        y_subset,
        X_subset_test,
        config,
    )

    preds_df = pd.DataFrame(
        {
            "ID": subset_test_df["ID"].values,
            "prediction": test_preds,
        }
    )
    preds_filename = f"{exp_name}_optuna_predictions.csv"
    preds_path = results_dir / preds_filename
    save_dataframe(preds_df, str(preds_path))
    logger.info(f"[{exp_name}] Saved subset2subset test predictions to {preds_path}")

    # -----------------------------------------------------------------------------------------
    # Create parity plot with mean predictions only (no quantile regression)
    # -----------------------------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("Creating parity plot for subset2subset model...")

    # Merge predictions with ground truth
    merged = (
        subset_test_df[["ID", TARGET_COL]]
        .merge(preds_df[["ID", "prediction"]], on="ID", how="inner")
        .dropna(subset=[TARGET_COL, "prediction"])
    )

    if merged.empty:
        logger.warning(f"[{exp_name}] No overlapping IDs between test subset and predictions.")
    else:
        y_true = merged[TARGET_COL].values
        y_pred = merged["prediction"].values

        # Create parity plot
        parity_path = results_dir / f"{exp_name}_parity.png"
        _create_parity_plot(y_true, y_pred, parity_path, exp_name, logger)

    logger.info("Subset2Subset pipeline with Optuna completed.")


if __name__ == "__main__":
    main()

