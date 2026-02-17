# =================================================================================================
# Li-ion Mobility Prediction - Stage 1: Structural Features from CIFs
# =================================================================================================
# This script implements Stage 1 of the machine learning pipeline for predicting lithium-ion
# conductivity. It builds upon the best-performing Stage 0 features and adds structural 
# information extracted from CIF files (density, volume per atom, Li coordination, etc.).
#
# Stage 1 Features Added (from CIF parsing):
# - Bulk structural properties: density, volume per atom, Li site count
# - Lattice parameters and angles
# - Li coordination metrics and distances
# - Space group one-hot encoding
#
# =================================================================================================

import os
import sys
import shutil
from pathlib import Path

# Prevent Python from writing bytecode (.pyc files) to avoid caching issues
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def clear_pycache(root_path: Path):
    """
    recursively removes all __pycache__ directories to ensure fresh code loading.
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
# Clear cache immediately upon finding project root
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
	generate_stage1_report,
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
# Main Execution Block
# =================================================================================================

def main():
	"""
	Stage 1: Structural features extracted from CIF files.
	
	This pipeline combines:
	- Stage 0 baseline: elemental ratios + SMACT stoichiometry + Magpie embeddings
	- Stage 1 structural: density, volume per atom, Li coordination, lattice metrics (from CIFs)
	
	We compare experiments progressively adding structural features.
	"""
	# -----------------------------------------------------------------------------------------
	# Initialization
	# -----------------------------------------------------------------------------------------
	seed_everything(42)
	
	results_dir = os.path.join(PROJECT_ROOT, "results", "results_stage1")
	os.makedirs(results_dir, exist_ok=True)
	
	log_path = os.path.join(results_dir, "stage1.log")
	logger = setup_logger("stage1", log_file=log_path)
	logger.info("=" * 80)
	logger.info("Stage 1: Structural Features Pipeline (CIF-based)")
	logger.info("=" * 80)
	
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
		hist_path = os.path.join(results_dir, "stage1_log10_sigma_hist_raw.png")
		save_histogram(
			log10_sigma_raw.values,
			hist_path,
			title="Stage 1: log10(sigma) before threshold clipping",
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
	
	# Warn about CIF files without CSV rows
	train_ids_in_csv = set(train_df["ID"])
	test_ids_in_csv = set(test_df["ID"])
	train_cifs_missing = [cid for cid in train_cif_files if cid not in train_ids_in_csv]
	test_cifs_missing = [cid for cid in test_cif_files if cid not in test_ids_in_csv]
	if train_cifs_missing:
		logger.warning(f"CIF files without CSV entries (train): {train_cifs_missing}")
	if test_cifs_missing:
		logger.warning(f"CIF files without CSV entries (test): {test_cifs_missing}")
	
	logger.info(f"Training samples (total): {len(train_df)} (with CIFs: {sum(train_df['ID'].isin(train_cif_files))})")
	logger.info(f"Test samples (total): {len(test_df)} (with CIFs: {sum(test_df['ID'].isin(test_cif_files))})")
	
	# -----------------------------------------------------------------------------------------
	# Feature Engineering - Stage 0 (Composition-Only Baseline)
	# -----------------------------------------------------------------------------------------
	logger.info("-" * 80)
	logger.info("Generating Stage 0 baseline features (Magpie embeddings)...")
	
	# Generate Stage 0 features
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
		# Prevent leakage: exclude raw target column
		"Ionic conductivity (S cm-1)",
	]
	
	# Get Stage 0 feature columns
	numeric_cols = base_train_df.select_dtypes(include=np.number).columns.tolist()
	stage0_feature_cols = [c for c in numeric_cols if c != TARGET_COL and c not in metadata_cols]
	logger.info(f"Stage 0 features: {len(stage0_feature_cols)} columns")
	
	# Verify sigma_is_coerced is included in training features
	if "sigma_is_coerced" in stage0_feature_cols:
		logger.info("  [OK] sigma_is_coerced is included in Stage 0 features (will be used in training)")
	else:
		logger.warning("  [MISSING] sigma_is_coerced is NOT in Stage 0 features!")
	
	# -----------------------------------------------------------------------------------------
	# Feature Engineering - Stage 1 (Structural Features from CSV)
	# -----------------------------------------------------------------------------------------
	logger.info("-" * 80)
	logger.info("Extracting Stage 1 structural features (CSV + CIF hybrid)...")
	
	# CSV-derived structural metadata (available for all rows)
	csv_train_struct = stage1_csv_structural_features(base_train_df.copy())
	csv_test_struct = stage1_csv_structural_features(base_test_df.copy())
	
	# CIF-derived advanced structural features (available subset)
	struct_train_df, train_stats = stage1_structural_features(
		base_train_df.copy(), id_col="ID", cif_dir=train_cif_dir, extended=True, verbose=True
	)
	struct_test_df, test_stats = stage1_structural_features(
		base_test_df.copy(), id_col="ID", cif_dir=test_cif_dir, extended=True, verbose=True
	)
	
	logger.info("-" * 80)
	logger.info("CIF parsing statistics:")
	logger.info(f"  Training -> full: {train_stats['parsed_full']}, partial: {train_stats['parsed_partial']}, missing: {train_stats['missing_cif']}, failed: {train_stats['failed']}")
	logger.info(f"  Test     -> full: {test_stats['parsed_full']}, partial: {test_stats['parsed_partial']}, missing: {test_stats['missing_cif']}, failed: {test_stats['failed']}")
	if train_stats["failed_ids"]:
		logger.warning(f"Failed training CIFs: {train_stats['failed_ids']}")
	if train_stats["missing_cif_ids"]:
		logger.warning(f"Missing training CIF files for IDs: {train_stats['missing_cif_ids']}")
	if test_stats["failed_ids"]:
		logger.warning(f"Failed test CIFs: {test_stats['failed_ids']}")
	if test_stats["missing_cif_ids"]:
		logger.warning(f"Missing test CIF files for IDs: {test_stats['missing_cif_ids']}")
	
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
	
	# Log structural feature coverage
	structural_cols = [
		"density", "volume_per_atom", "n_li_sites", "n_total_atoms",
		"lattice_a", "lattice_b", "lattice_c",
		"lattice_alpha", "lattice_beta", "lattice_gamma",
		"lattice_anisotropy", "lattice_anisotropy_bc_a", "lattice_anisotropy_max_min",
		"angle_deviation_from_ortho", "is_cubic_like",
		"li_fraction", "li_concentration", "framework_density",
		"li_li_min_dist", "li_li_avg_dist", "li_anion_min_dist",
		"li_coordination_avg", "li_site_avg_multiplicity"
	]
	
	logger.info("Structural feature coverage (training):")
	for col in structural_cols:
		if col in struct_train_df.columns:
			if col in cif_only_cols:
				non_nan = (struct_train_df[col] != 0).sum()
			else:
				non_nan = struct_train_df[col].notna().sum()
			logger.info(f"  {col}: {non_nan}/{len(struct_train_df)} ({100*non_nan/len(struct_train_df):.1f}%)")
	
	# Add space group one-hot encoding
	full_train_df = stage1_spacegroup_onehot(struct_train_df.copy(), spacegroup_col="spacegroup_number")
	full_test_df = stage1_spacegroup_onehot(struct_test_df.copy(), spacegroup_col="spacegroup_number")
	
	# Define feature sets for experiments
	basic_struct_cols = [
		"density", "volume_per_atom", "n_li_sites", "n_total_atoms"
	]
	geometry_cols = basic_struct_cols + [
		"lattice_a", "lattice_b", "lattice_c",
		"lattice_alpha", "lattice_beta", "lattice_gamma",
		"lattice_anisotropy_bc_a", "lattice_anisotropy_max_min",
		"angle_deviation_from_ortho", "is_cubic_like",
		"li_fraction", "li_concentration", "framework_density",
		"li_li_min_dist", "li_li_avg_dist", "li_anion_min_dist",
		"li_coordination_avg", "li_site_avg_multiplicity"
	]
	geometry_cols += ["has_cif_struct"]
	spacegroup_cols = [f"sg_{i}" for i in range(1, 231)]
	
	# -----------------------------------------------------------------------------------------
	# Cross-Validation Setup
	# -----------------------------------------------------------------------------------------
	logger.info("-" * 80)
	logger.info("Setting up cross-validation...")
	
	group_col = "group" if "group" in train_df.columns else None
	y = full_train_df[TARGET_COL]
	
	if group_col:
		groups = full_train_df[group_col].values
		logger.info(f"Using column '{group_col}' for GroupKFold cross-validation.")
	else:
		groups = np.arange(len(full_train_df))
		logger.info("No group column found. Using standard KFold behavior.")
	
	gkf = GroupKFold(n_splits=5)
	splits = list(gkf.split(X=full_train_df, y=y, groups=groups))
	logger.info(f"Created {len(splits)} CV splits.")
	
	# -----------------------------------------------------------------------------------------
	# Define Experiments
	# -----------------------------------------------------------------------------------------
	experiments = {
		"stage0_magpie": {
			"description": "Stage 0 baseline (elemental ratios + SMACT + Magpie)",
			"train_df": base_train_df,
			"test_df": base_test_df,
			"feature_cols": stage0_feature_cols,
		},
		"stage1_basic_struct": {
			"description": "Stage 0 + basic structural (lattice, density, Li sites)",
			"train_df": struct_train_df,
			"test_df": struct_test_df,
			"feature_cols": stage0_feature_cols + basic_struct_cols,
		},
		"stage1_geometry": {
			"description": "Stage 0 + full geometry (lattice, angles, derived features)",
			"train_df": struct_train_df,
			"test_df": struct_test_df,
			"feature_cols": stage0_feature_cols + geometry_cols,
		},
		"stage1_full_struct": {
			"description": "Stage 0 + geometry + spacegroup one-hot",
			"train_df": full_train_df,
			"test_df": full_test_df,
			"feature_cols": stage0_feature_cols + geometry_cols + spacegroup_cols,
		},
	}
	
	# -----------------------------------------------------------------------------------------
	# Run Experiments
	# -----------------------------------------------------------------------------------------
	all_experiment_metrics = {}
	
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
		
		# Verify sigma_is_coerced is in training features for this experiment
		if "sigma_is_coerced" in available_cols:
			logger.info(f"  [OK] sigma_is_coerced included in {exp_name} training features")
		else:
			logger.warning(f"  [MISSING] sigma_is_coerced NOT in {exp_name} training features!")
		
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
		
		# Save parity plot
		parity_plot_path = os.path.join(results_dir, f"stage1_cv_parity_{exp_name}.png")
		save_parity_plot(
			y.values,
			oof,
			parity_plot_path,
			title=f"Stage 1 5-Fold CV: {exp_name}",
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
		output_path = os.path.join(results_dir, f"stage1_predictions_{exp_name}.csv")
		out_df = pd.DataFrame({"ID": test_df["ID"], "prediction": preds})
		save_dataframe(out_df, output_path)
		logger.info(f"[{exp_name}] Predictions saved to: {output_path}")

		# Test-set parity plot (log target, back-transformed R² shown)
		if TARGET_COL in test_df.columns:
			test_y = test_df[TARGET_COL].values
			test_parity_path = os.path.join(results_dir, f"stage1_test_parity_{exp_name}.png")
			save_parity_plot(
				test_y,
				preds,
				test_parity_path,
				title=f"Stage 1 Test: {exp_name}",
				r2_linear_from_log=False,
			)
	
	# -----------------------------------------------------------------------------------------
	# Generate Final Report
	# -----------------------------------------------------------------------------------------
	logger.info("=" * 80)
	logger.info("Generating final report...")
	
	report_path = os.path.join(results_dir, "interim_report_stage1.md")
	generate_stage1_report(report_path, metrics=all_experiment_metrics)
	logger.info(f"Report saved to: {report_path}")
	
	# Print summary
	logger.info("-" * 80)
	logger.info("SUMMARY: Stage 1 Experiment Results")
	logger.info("-" * 80)
	for exp_name, metrics in all_experiment_metrics.items():
		logger.info(f"{exp_name:25s} | R2: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")
	
	# -----------------------------------------------------------------------------------------
	# Feature Importance Analysis
	# -----------------------------------------------------------------------------------------
	logger.info("=" * 80)
	logger.info("FEATURE IMPORTANCE ANALYSIS")
	logger.info("-" * 80)
	
	from sklearn.ensemble import HistGradientBoostingRegressor
	from sklearn.inspection import permutation_importance
	
	# Use geometry experiment's feature set
	geometry_feature_cols = [c for c in geometry_cols if c in struct_train_df.columns]
	X_geometry = struct_train_df[stage0_feature_cols + geometry_feature_cols].copy()
	X_geometry.replace([np.inf, -np.inf], np.nan, inplace=True)
	X_geometry = X_geometry.fillna(0)
	
	# Fit model and get permutation importances
	model = HistGradientBoostingRegressor(random_state=42)
	model.fit(X_geometry, y)
	
	perm_importance = permutation_importance(model, X_geometry, y, n_repeats=10, random_state=42, n_jobs=-1)
	
	# Create importance DataFrame
	importance_df = pd.DataFrame({
		'feature': X_geometry.columns.tolist(),
		'importance': perm_importance.importances_mean,
		'std': perm_importance.importances_std
	}).sort_values('importance', ascending=False)
	
	# Log top 20 features
	logger.info("Top 20 Most Important Features:")
	logger.info("-" * 60)
	for _, row in importance_df.head(20).iterrows():
		logger.info(f"  {row['feature']:35s} | {row['importance']:.4f} +/- {row['std']:.4f}")
	
	# Log structural features only
	logger.info("-" * 80)
	logger.info("STRUCTURAL FEATURE IMPORTANCE:")
	logger.info("-" * 60)
	structural_importance = importance_df[importance_df['feature'].isin(geometry_feature_cols)]
	for _, row in structural_importance.iterrows():
		logger.info(f"  {row['feature']:35s} | {row['importance']:.4f} +/- {row['std']:.4f}")
	
	# Save feature importance
	try:
		import matplotlib.pyplot as plt
		
		fig, axes = plt.subplots(1, 2, figsize=(16, 8))
		
		# Top 20 overall
		top20 = importance_df.head(20)
		ax1 = axes[0]
		colors = ['#e74c3c' if f in geometry_feature_cols else '#3498db' for f in top20['feature']]
		ax1.barh(range(len(top20)), top20['importance'], xerr=top20['std'], color=colors, alpha=0.8)
		ax1.set_yticks(range(len(top20)))
		ax1.set_yticklabels(top20['feature'])
		ax1.invert_yaxis()
		ax1.set_xlabel('Permutation Importance')
		ax1.set_title('Top 20 Features (Red=Structural, Blue=Composition)')
		ax1.grid(axis='x', alpha=0.3)
		
		# Structural only
		ax2 = axes[1]
		struct_sorted = structural_importance.sort_values('importance', ascending=False)
		ax2.barh(range(len(struct_sorted)), struct_sorted['importance'], xerr=struct_sorted['std'], color='#e74c3c', alpha=0.8)
		ax2.set_yticks(range(len(struct_sorted)))
		ax2.set_yticklabels(struct_sorted['feature'])
		ax2.invert_yaxis()
		ax2.set_xlabel('Permutation Importance')
		ax2.set_title('Structural Feature Importances')
		ax2.grid(axis='x', alpha=0.3)
		
		plt.tight_layout()
		plt.savefig(os.path.join(results_dir, "stage1_feature_importance.png"), dpi=300, bbox_inches='tight')
		plt.close()
		logger.info(f"Feature importance plot saved.")
	except Exception as e:
		logger.warning(f"Could not generate plot: {e}")
	
	importance_df.to_csv(os.path.join(results_dir, "stage1_feature_importance.csv"), index=False)
	logger.info("=" * 80)
	logger.info("Stage 1 pipeline completed successfully!")


if __name__ == "__main__":
	main()
