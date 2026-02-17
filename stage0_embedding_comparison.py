# =================================================================================================
# Module Imports
# =================================================================================================
# The 'os' module is used for interacting with the operating system, such as constructing file paths.
# The 'sys' module allows us to manipulate Python's runtime environment, like modifying the system path.
# 'pathlib.Path' provides an object-oriented way to handle filesystem paths, which is more modern
# and intuitive than using strings.
import os
import sys
from pathlib import Path

# =================================================================================================
# Scientific Computing Libraries
# =================================================================================================
# 'numpy' is the fundamental package for scientific computing in Python. It provides a powerful
# N-dimensional array object and a host of functions for mathematical operations.
# 'pandas' is an essential library for data analysis and manipulation. It offers data structures
# like the DataFrame, which is perfect for handling tabular data like our CSV files.
import numpy as np
import pandas as pd
# Import GroupKFold for creating the cross-validation splits.
from sklearn.model_selection import GroupKFold
# r2_score import removed - using explicit R² = 1 - SSE/SST calculation

# =================================================================================================
# Project Structure Setup
# =================================================================================================
# To maintain a clean and modular project structure, all reusable code (like data processing functions)
# is placed in the 'src' directory. To allow Python to find and import these modules, we need to
# add the project's root directory to the system path. This block of code is designed to
# automatically find the project root and add it to the path.

def find_project_root(start_path: str = None, marker_file: str = "requirements.txt") -> Path:
	"""
	Find the project root directory by searching upwards from a starting point for a marker file.

	This is a robust way to locate the project's base directory, regardless of where the script
	is being run from. It works by looking for a file that is known to exist at the root,
	such as 'requirements.txt' or '.git'.

	Args:
		start_path (str, optional): The path to start the search from. If it's a file,
									its parent directory is used. If None, the current
									working directory is used. Defaults to None.
		marker_file (str, optional): The name of the file to search for.
									 Defaults to "requirements.txt".

	Raises:
		FileNotFoundError: If the marker file cannot be found in the directory hierarchy.

	Returns:
		Path: An object representing the path to the project root directory.
	"""
	if start_path is None:
		current = Path.cwd()
	else:
		path = Path(start_path).resolve()
		current = path.parent if path.is_file() else path
	
	for path in [current] + list(current.parents):
		marker = path / marker_file
		if marker.is_file():
			return path
	
	raise FileNotFoundError(
		f"Could not find '{marker_file}' in current directory or any parent directory. "
		f"Started from: {current}"
	)

# Find the project root using the location of the current script (__file__) and add it to sys.path.
# This ensures that subsequent imports from the 'src' directory will work correctly.
PROJECT_ROOT = str(find_project_root(__file__))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)


# =================================================================================================
# Custom Module Imports from 'src'
# =================================================================================================
# Now that the project root is in the system path, we can import our custom modules.
# These modules encapsulate the logic for different parts of the ML pipeline, promoting
# code reusability and maintainability.

# Import utility functions like logger setup, random seed management, and saving data.
from src.utils import (
	setup_logger,
	seed_everything,
	save_dataframe,
	save_parity_plot,
	save_histogram,
	generate_stage0_report,
	compute_regression_metrics,
)

# Import functions related to data handling: loading, cleaning, target transformation, and splitting.
from src.data_processing import load_data, clean_data, add_target_log10_sigma, coerce_sigma_series, split_features_target, TARGET_COL

# Import the specific feature engineering functions needed for Stage 0.
from src.features import (
	stage0_elemental_ratios, 
	stage0_smact_features, 
	stage0_element_embeddings
)

# Import the configuration class and the new, refactored training functions.
from src.model_training import ExperimentConfig, run_cv_with_predefined_splits, fit_full_and_predict


# =================================================================================================
# Main Execution Block
# =================================================================================================
# This function orchestrates the entire Stage 0 pipeline, which now includes the comparison
# of different element embedding schemes as outlined for Weeks 3-5.
def main():
	# ---------------------------------------------------------------------------------------------
	# Initialization (Weeks 1-2)
	# ---------------------------------------------------------------------------------------------
	# Set a universal random seed for reproducibility.
	seed_everything(42)
	
	# Set up a logger to record the script's progress.
	log_path = os.path.join(PROJECT_ROOT, "results", "results_stage0", "stage0.log")
	logger = setup_logger("stage0", log_file=log_path)
	logger.info("--- Starting Stage 0: Composition-Only Baseline & Embedding Comparison ---")

	# ---------------------------------------------------------------------------------------------
	# Data Loading and Preprocessing (Weeks 1-2)
	# ---------------------------------------------------------------------------------------------
	# Load the raw training and testing datasets.
	data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
	logger.info(f"Loading data from: {data_dir}")
	train_df, test_df = load_data(data_dir)
	logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples.")

	# Histogram of log10(sigma) before threshold clipping
	sigma_col = "Ionic conductivity (S cm-1)"
	if sigma_col in train_df.columns:
		sigma_numeric = coerce_sigma_series(train_df[sigma_col])
		# Linear-scale histogram (raw numeric sigma)
		sigma_linear = sigma_numeric.dropna()
		hist_path_linear = os.path.join(PROJECT_ROOT, "results", "results_stage0", "stage0_sigma_hist_raw.png")
		save_histogram(
			sigma_linear.values,
			hist_path_linear,
			title="Stage 0: sigma (S cm-1) before threshold clipping",
			x_label=r"$\sigma$ (S cm$^{-1}$)",
		)
		logger.info(f"sigma histogram saved to: {hist_path_linear}")

		sigma_positive = sigma_numeric[sigma_numeric > 0]
		log10_sigma_raw = np.log10(sigma_positive)
		hist_path = os.path.join(PROJECT_ROOT, "results", "results_stage0", "stage0_log10_sigma_hist_raw.png")
		save_histogram(
			log10_sigma_raw.values,
			hist_path,
			title="Stage 0: log10(sigma) before threshold clipping",
			x_label="log10(sigma)",
			vlines={np.log10(1e-30): "clip threshold (1e-30)"},
		)
		logger.info(f"log10(sigma) histogram saved to: {hist_path}")

	# Clean the data, create the log10(sigma) target variable, and save the cleaned data.
	train_df = add_target_log10_sigma(clean_data(train_df), target_sigma_col="Ionic conductivity (S cm-1)")
	test_df = add_target_log10_sigma(clean_data(test_df), target_sigma_col="Ionic conductivity (S cm-1)")
	logger.info("Applied initial data cleaning and target transformation (log10_sigma).")

	# Save the cleaned data to the 'processed' directory.
	processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
	save_dataframe(train_df, os.path.join(processed_dir, "train_cleaned.csv"))
	save_dataframe(test_df, os.path.join(processed_dir, "test_cleaned.csv"))
	logger.info(f"Saved cleaned data to: {processed_dir}")

	# Drop rows with a missing target. This is done *after* saving the cleaned data so that the
	# saved file contains all original rows, but the training process only uses valid ones.
	if TARGET_COL in train_df.columns:
		train_df.dropna(subset=[TARGET_COL], inplace=True)
		logger.info(f"Training samples after dropping NaN targets: {len(train_df)}")

	# ---------------------------------------------------------------------------------------------
	# Feature Engineering (Weeks 3-5)
	# ---------------------------------------------------------------------------------------------
	# First, generate all the features that are common to every experiment. This "base" feature set
	# includes elemental ratios and SMACT vectors.
	logger.info("Generating baseline (non-embedding) features...")
	base_train_df = stage0_elemental_ratios(train_df, "Reduced Composition")
	base_train_df = stage0_smact_features(base_train_df, "Reduced Composition")
	
	base_test_df = stage0_elemental_ratios(test_df, "Reduced Composition")
	base_test_df = stage0_smact_features(base_test_df, "Reduced Composition")

	# Define the list of columns that are structural or metadata, which we need to exclude from the feature set.
	# CRITICAL: "Ionic conductivity (S cm-1)" must be excluded -- it is the raw target variable
	# (log10_sigma = log10 of this column). Including it would be direct data leakage.
	# "sigma_is_coerced" is also excluded because it is derived from the target column
	# (flags whether the conductivity was a detection-limit value like "<1E-10").
	structural_metadata_cols = [
		"Ionic conductivity (S cm-1)", "sigma_is_coerced",
		"Space group", "Space group #", "a", "b", "c", "alpha", "beta", "gamma", "Z",
		"IC (Total)", "IC (Bulk)", "ID", "Family", "DOI", "Checked", "Ref", "Cif ID",
		"Cif ref_1", "Cif ref_2", "note", "close match", "close match DOI", "ICSD ID",
		"Laskowski ID", "Liion ID", "True Composition", "Reduced Composition",
	]
	# Create the list of base feature names by taking all numeric columns and removing the target and metadata.
	numeric_cols = base_train_df.select_dtypes(include=np.number).columns.tolist()
	base_feature_cols = [c for c in numeric_cols if c != TARGET_COL and c not in structural_metadata_cols]
	
	logger.info(f"Generated {len(base_feature_cols)} baseline features.")

	# ---------------------------------------------------------------------------------------------
	# Embedding Comparison Experiment (Weeks 3-5)
	# ---------------------------------------------------------------------------------------------
	# To ensure a fair comparison between embedding schemes, we must test each model on the
	# exact same subsets of the training data. This block generates the cross-validation
	# splits (e.g., the train/validation indices for each of the 5 folds) *once* before the loop.
	group_col = "group" if "group" in train_df.columns else None
	y = base_train_df[TARGET_COL] # The target variable is the same for all experiments.
	
	if group_col:
		groups = base_train_df[group_col].values
		logger.info(f"Using column '{group_col}' for GroupKFold cross-validation.")
	else:
		groups = np.arange(len(base_train_df)) # Fallback for standard KFold

	gkf = GroupKFold(n_splits=5)
	splits = list(gkf.split(X=base_train_df, y=y, groups=groups))
	logger.info(f"Created {len(splits)} identical CV splits for the comparison.")

	# Define the different experiments we want to run. We will test each embedding scheme
	# individually, and then test a combination of all of them.
	embedding_schemes = ["mat2vec", "megnet16", "magpie"]
	experiments = {scheme: [scheme] for scheme in embedding_schemes}
	experiments["all_embeddings"] = embedding_schemes

	# Create a dictionary to store the metrics from each experiment for the final report.
	all_experiment_metrics = {}
	metrics_rows = []

	# Now, loop through each defined experiment.
	for name, embeddings_to_use in experiments.items():
		logger.info(f"--- Running Experiment: {name} ---")
		
		# Generate the specific embedding features for the current experiment.
		train_df_exp = stage0_element_embeddings(base_train_df.copy(), "Reduced Composition", embeddings_to_use)
		test_df_exp = stage0_element_embeddings(base_test_df.copy(), "Reduced Composition", embeddings_to_use)

		# Identify the names of the new embedding columns that were just added.
		emb_cols = [c for c in train_df_exp.columns if "_emb_" in c and any(e in c for e in embeddings_to_use)]
		
		# The final feature set for this experiment is the combination of the base features and the new embedding features.
		final_feature_cols = base_feature_cols + emb_cols
		X = train_df_exp[final_feature_cols]

		logger.info(f"Training with {len(final_feature_cols)} features ({len(base_feature_cols)} base + {len(emb_cols)} embedding).")

		# Define the model configuration. This is the same for all experiments.
		config = ExperimentConfig(
			model_name="hgbt", 
			n_splits=5, 
			random_state=42, 
			params={},
			group_col=group_col
		)
		
		# Run cross-validation using the pre-defined, identical splits. This is the key to a fair comparison.
		fold_scores, overall_metrics, oof = run_cv_with_predefined_splits(X, y, splits, config)

		# Log the final metrics for this specific experiment.
		logger.info(f"[{name}] Cross-validation complete. Fold RMSEs: {[f'{s:.4f}' for s in fold_scores]}")
		logger.info(f"[{name}] Overall CV Metrics -> R²: {overall_metrics['r2']:.4f} | RMSE: {overall_metrics['rmse']:.4f} | MAE: {overall_metrics['mae']:.4f}")

		# Store the metrics for the report.
		all_experiment_metrics[name] = overall_metrics

		# Compute linear-scale metrics from log predictions
		# Convert log predictions to linear: y_linear = 10^(log_prediction)
		y_true_linear = np.power(10.0, y.values)
		y_pred_linear = np.power(10.0, oof)
		
		# Explicit R² calculation: R² = 1 - SSE/SST
		# SSE = Σ(y_pred_linear_i - y_true_linear_i)²
		sse_linear = np.sum((y_pred_linear - y_true_linear) ** 2)
		# SST = Σ(y_true_linear_i - mean(y_true_linear))² = total variance of true linear values
		sst_linear = np.sum((y_true_linear - np.mean(y_true_linear)) ** 2)
		r2_linear = 1.0 - (sse_linear / sst_linear) if sst_linear > 0 else np.nan
		
		rmse_linear = float(np.sqrt(np.mean((y_true_linear - y_pred_linear) ** 2)))
		mae_linear = float(np.mean(np.abs(y_true_linear - y_pred_linear)))
		metrics_rows.append({
			"experiment": name,
			"target_scale": "log10_sigma",
			"r2": overall_metrics["r2"],
			"rmse": overall_metrics["rmse"],
			"mae": overall_metrics["mae"],
			"r2_linear": r2_linear,
			"rmse_linear": rmse_linear,
			"mae_linear": mae_linear,
		})

		# Save a uniquely named parity plot for this experiment's results.
		parity_plot_path = os.path.join(PROJECT_ROOT, "results", "results_stage0", f"stage0_cv_parity_{name}.png")
		save_parity_plot(
			y.values,
			oof,
			parity_plot_path,
			title=f"Stage 0 5-Fold CV Performance ({name})",
			r2_linear_from_log=True,
		)
		
		# ---------------------------------------------------------------------------------------------
		# Final Model Training & Prediction for this Experiment
		# ---------------------------------------------------------------------------------------------
		# As a final step, train a model on the full dataset with these features and predict on the test set.
		logger.info(f"[{name}] Training final model on full dataset and predicting on the test set...")
		
		X_test = test_df_exp[[c for c in final_feature_cols if c in test_df_exp.columns]]
		# Ensure the test set has the exact same columns in the same order as the training set.
		X_test = X_test.reindex(columns=X.columns, fill_value=0)

		preds = fit_full_and_predict(X, y, X_test, config)
		
		# Save the test predictions to a uniquely named CSV file for this experiment.
		output_path = os.path.join(PROJECT_ROOT, "results", "results_stage0", f"stage0_predictions_{name}.csv")
		out = pd.DataFrame({"ID": test_df["ID"], "prediction": preds})
		save_dataframe(out, output_path)
		logger.info(f"[{name}] Test predictions saved to: {output_path}")

		# Test-set parity plot (log target, back-transformed R² shown)
		if TARGET_COL in test_df.columns:
			test_y = test_df[TARGET_COL].values
			test_parity_path = os.path.join(
				PROJECT_ROOT, "results", "results_stage0", f"stage0_test_parity_{name}.png"
			)
			save_parity_plot(
				test_y,
				preds,
				test_parity_path,
				title=f"Stage 0 Test Performance ({name})",
				# Keep linearized R² consistent with CV: compute R² on 10**(log10_sigma) vs 10**(log10_sigma_pred).
				r2_linear_from_log=True,
			)

	# ---------------------------------------------------------------------------------------------
	# Magpie-only embeddings on linear sigma (no baseline features)
	# ---------------------------------------------------------------------------------------------
	logger.info("--- Running Experiment: magpie_only_linear_sigma (embeddings only, linear target) ---")
	y_linear = coerce_sigma_series(train_df[sigma_col])
	valid_idx = y_linear.dropna().index
	train_df_linear = train_df.loc[valid_idx].copy()
	test_df_linear = test_df.copy()

	train_df_linear = stage0_element_embeddings(train_df_linear, "Reduced Composition", ["magpie"])
	test_df_linear = stage0_element_embeddings(test_df_linear, "Reduced Composition", ["magpie"])

	emb_cols_linear = [c for c in train_df_linear.columns if c.startswith("magpie_emb_")]
	X_linear = train_df_linear[emb_cols_linear]
	y_linear = y_linear.loc[valid_idx]

	# Filter splits to valid indices
	indexer = {idx: i for i, idx in enumerate(train_df_linear.index)}
	splits_linear = []
	for trn_idx, val_idx in splits:
		trn_ids = [base_train_df.index[i] for i in trn_idx if base_train_df.index[i] in indexer]
		val_ids = [base_train_df.index[i] for i in val_idx if base_train_df.index[i] in indexer]
		splits_linear.append((
			np.array([indexer[i] for i in trn_ids], dtype=int),
			np.array([indexer[i] for i in val_ids], dtype=int),
		))

	config_linear = ExperimentConfig(
		model_name="hgbt",
		n_splits=5,
		random_state=42,
		params={},
		group_col=group_col,
		non_negative=True,
	)
	fold_scores_linear, overall_metrics_linear, oof_linear = run_cv_with_predefined_splits(
		X_linear, y_linear, splits_linear, config_linear
	)
	# Predictions are already clamped to be non-negative by the model wrapper
	overall_metrics_linear = compute_regression_metrics(y_linear.values, oof_linear)
	logger.info(f"[magpie_only_linear_sigma] Cross-validation complete. Fold RMSEs: {[f'{s:.4f}' for s in fold_scores_linear]}")
	logger.info(
		f"[magpie_only_linear_sigma] Overall CV Metrics -> R²: {overall_metrics_linear['r2']:.4f} | "
		f"RMSE: {overall_metrics_linear['rmse']:.4f} | MAE: {overall_metrics_linear['mae']:.4f}"
	)
	metrics_rows.append({
		"experiment": "magpie_only_linear_sigma",
		"target_scale": "linear_sigma",
		"r2": np.nan,
		"rmse": np.nan,
		"mae": np.nan,
		"r2_linear": overall_metrics_linear["r2"],
		"rmse_linear": overall_metrics_linear["rmse"],
		"mae_linear": overall_metrics_linear["mae"],
	})

	parity_plot_path_linear = os.path.join(
		PROJECT_ROOT, "results", "results_stage0", "stage0_cv_parity_magpie_only_linear_sigma.png"
	)
	save_parity_plot(
		y_linear.values,
		oof_linear,
		parity_plot_path_linear,
		title="Stage 0 5-Fold CV Performance (magpie only, linear σ)",
		x_label=r"Actual $\sigma$ (S cm$^{-1}$)",
		y_label=r"Predicted $\sigma$ (S cm$^{-1}$)",
	)

	X_test_linear = test_df_linear[[c for c in emb_cols_linear if c in test_df_linear.columns]]
	X_test_linear = X_test_linear.reindex(columns=X_linear.columns, fill_value=0)
	preds_linear = fit_full_and_predict(X_linear, y_linear, X_test_linear, config_linear)
	output_path_linear = os.path.join(
		PROJECT_ROOT, "results", "results_stage0", "stage0_predictions_magpie_only_linear_sigma.csv"
	)
	out_linear = pd.DataFrame({"ID": test_df["ID"], "prediction": preds_linear})
	save_dataframe(out_linear, output_path_linear)
	logger.info(f"[magpie_only_linear_sigma] Test predictions saved to: {output_path_linear}")

	# Test-set parity plot for linear sigma
	if sigma_col in test_df.columns:
		test_sigma = coerce_sigma_series(test_df[sigma_col]).values
		test_parity_linear_path = os.path.join(
			PROJECT_ROOT, "results", "results_stage0", "stage0_test_parity_magpie_only_linear_sigma.png"
		)
		save_parity_plot(
			test_sigma,
			preds_linear,
			test_parity_linear_path,
			title="Stage 0 Test Performance (magpie only, linear σ)",
			x_label=r"Actual $\sigma$ (S cm$^{-1}$)",
			y_label=r"Predicted $\sigma$ (S cm$^{-1}$)",
		)

	# ---------------------------------------------------------------------------------------------
	# Save summary metrics table
	# ---------------------------------------------------------------------------------------------
	metrics_df = pd.DataFrame(metrics_rows)
	metrics_path = os.path.join(PROJECT_ROOT, "results", "results_stage0", "stage0_metrics_summary.csv")
	save_dataframe(metrics_df, metrics_path)
	logger.info(f"Stage 0 metrics summary saved to: {metrics_path}")

	logger.info("--- All Stage 0 experiments finished successfully! ---")

	# ---------------------------------------------------------------------------------------------
	# Generate Final Report
	# ---------------------------------------------------------------------------------------------
	# After all experiments are complete, generate the Markdown report summarizing the findings.
	report_path = os.path.join(PROJECT_ROOT, "results", "results_stage0", "interim_report_stage0.md")
	generate_stage0_report(report_path)


# =================================================================================================
# Script Entry Point
# =================================================================================================
if __name__ == "__main__":
	main()


