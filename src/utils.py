# =================================================================================================
# Module Imports
# =================================================================================================
# Import 'logging' for creating logs, 'os' for interacting with the operating system,
# 'pathlib' for object-oriented filesystem paths, and 'typing' for type hints.
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Import 'numpy' and 'pandas' for numerical operations and data manipulation.
import numpy as np
import pandas as pd

# Import specific metrics from scikit-learn for model evaluation.
# mean_absolute_error (MAE): Average of the absolute differences between predictions and actual values.
# mean_squared_error (MSE): Average of the squared differences. RMSE (root MSE) is often used.
# r2_score (R²): Coefficient of determination, indicating the proportion of variance in the
#                dependent variable that is predictable from the independent variables.
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import plotting libraries. Matplotlib is the foundational library, and Seaborn provides
# an attractive high-level interface. We import them inside the function to make them
# optional dependencies for the project.
import matplotlib.pyplot as plt
import seaborn as sns


# =================================================================================================
# Utility Functions
# =================================================================================================
# This section contains general-purpose helper functions that can be used across the project.
# Keeping them here avoids code duplication and makes the main scripts cleaner.

def find_project_root(start_path: Optional[str] = None, marker_file: str = "requirements.txt") -> Path:
	"""
	Finds the project root directory by searching upwards for a specific marker file.

	This is a robust utility to ensure that file paths are always relative to the project
	root, regardless of where a script is executed from. It's essential for a modular
	project structure where scripts in different subdirectories need to access common
	resources like the 'data' or 'src' folders.

	Args:
		start_path (Optional[str], optional): The path to start searching from. If a file is given,
											  its parent directory is used. Defaults to the current
											  working directory.
		marker_file (str, optional): The name of the file that marks the root directory.
									 Defaults to "requirements.txt".

	Raises:
		FileNotFoundError: If the marker file cannot be found by traversing up the directory tree.

	Returns:
		Path: A Path object representing the absolute path to the project root.
	"""
	if start_path is None:
		current = Path.cwd()
	else:
		path = Path(start_path).resolve()
		# If it's a file, use its parent directory; if it's a directory, use it directly
		current = path.parent if path.is_file() else path
	
	# Traverse up the directory tree looking for the marker file.
	for path in [current] + list(current.parents):
		marker = path / marker_file
		if marker.is_file():
			return path
	
	# If the loop finishes without finding the file, raise an error.
	raise FileNotFoundError(
		f"Could not find '{marker_file}' in current directory or any parent directory. "
		f"Started from: {current}"
	)


def setup_logger(name: str = "ml_pipeline", level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
	"""
	Creates and configures a logger for printing and saving experiment progress.

	A well-configured logger is crucial for tracking experiments, debugging code, and
	recording results. This function sets up a logger that can print messages to the
	console and optionally save them to a file. It avoids duplicate handlers if called
	multiple times.

	Args:
		name (str, optional): The name of the logger. Defaults to "ml_pipeline".
		level (int, optional): The minimum logging level to capture (e.g., logging.INFO).
							   Defaults to logging.INFO.
		log_file (Optional[str], optional): The path to the file where logs should be saved.
											If None, logs are only printed to the console.
											Defaults to None.

	Returns:
		logging.Logger: The configured logger instance.
	"""
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.propagate = False  # Prevents log messages from being passed to the root logger.

	# Only add handlers if the logger doesn't already have them.
	if not logger.handlers:
		# Handler for printing logs to the console.
		stream_handler = logging.StreamHandler()
		stream_handler.setLevel(level)
		formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
		stream_handler.setFormatter(formatter)
		logger.addHandler(stream_handler)

		# Handler for saving logs to a file, if a path is provided.
		if log_file is not None:
			# Ensure the directory for the log file exists.
			os.makedirs(os.path.dirname(log_file), exist_ok=True)
			file_handler = logging.FileHandler(log_file)
			file_handler.setLevel(level)
			file_handler.setFormatter(formatter)
			logger.addHandler(file_handler)

	return logger


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	"""
	Calculates and returns a dictionary of common regression performance metrics.

	This function consolidates the calculation of evaluation metrics as required by the
	project plan (Weeks 3-5). It takes the true and predicted values and returns the
	Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the R-squared (R²) score.

	Args:
		y_true (np.ndarray): The ground truth target values.
		y_pred (np.ndarray): The predicted values from the model.

	Returns:
		Dict[str, float]: A dictionary containing the calculated 'mae', 'rmse', and 'r2'.
	"""
	# Ensure numeric arrays and drop non-finite pairs to keep SSE/SST stable
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)
	mask = np.isfinite(y_true) & np.isfinite(y_pred)
	y_true = y_true[mask]
	y_pred = y_pred[mask]

	if len(y_true) == 0:
		return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}

	mae = float(mean_absolute_error(y_true, y_pred))
	rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

	# Explicit R² = 1 - SSE/SST
	sse = np.sum((y_pred - y_true) ** 2)
	sst = np.sum((y_true - np.mean(y_true)) ** 2)
	r2 = float(1.0 - (sse / sst)) if sst > 0 else np.nan

	return {"mae": mae, "rmse": rmse, "r2": r2}


def compute_r2_linear_from_log(
	y_true_log: np.ndarray,
	y_pred_log: np.ndarray,
	y_true_linear_reference: np.ndarray,
) -> float:
	"""
	Computes R² in linear scale from log-scale predictions, using the true linear values
	as the reference for SST (total sum of squares).

	This function explicitly calculates R² = 1 - SSE/SST where:
	- y_pred_linear = 10^(y_pred_log)  [convert log predictions to linear scale]
	- SSE = Σ(y_pred_linear_i - y_true_linear_i)²  [sum of squared errors]
	- SST = Σ(y_true_linear_i - mean(y_true_linear))²  [total variance of true linear values]

	This gives the explanatory power of the log-scale model when evaluated in linear scale,
	using the true linear values' variance as the baseline.

	Args:
		y_true_log (np.ndarray): The ground truth target values in log10 scale.
		y_pred_log (np.ndarray): The predicted values in log10 scale.
		y_true_linear_reference (np.ndarray): The true values in linear scale (used for SST).

	Returns:
		float: The R² score in linear scale. By definition, 0 <= R² <= 1 when the model
			   performs at least as well as predicting the mean.
	"""
	# Convert log predictions to linear scale: y_linear = 10^(log_prediction)
	y_pred_linear = np.power(10.0, y_pred_log)

	# Ensure arrays are valid (no NaN/Inf)
	mask = (
		np.isfinite(y_pred_linear) &
		np.isfinite(y_true_linear_reference)
	)
	y_pred_linear = y_pred_linear[mask]
	y_true_linear = y_true_linear_reference[mask]

	if len(y_true_linear) == 0:
		return np.nan

	# Compute SSE: Sum of Squared Errors = Σ(y_pred_linear_i - y_true_linear_i)²
	sse = np.sum((y_pred_linear - y_true_linear) ** 2)

	# Compute SST: Total Sum of Squares = Σ(y_true_linear_i - mean(y_true_linear))²
	# This is the variance of the true linear values times n
	y_mean = np.mean(y_true_linear)
	sst = np.sum((y_true_linear - y_mean) ** 2)

	# Avoid division by zero
	if sst == 0:
		return np.nan

	# Compute R² = 1 - SSE/SST
	r2_linear = 1.0 - (sse / sst)

	return float(r2_linear)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
	"""
	Saves a pandas DataFrame to a CSV file.

	This is a simple helper function that includes an important feature: it automatically
	creates the destination directory if it doesn't already exist. This prevents errors
	when saving results to a new folder structure.

	Args:
		df (pd.DataFrame): The DataFrame to save.
		path (str): The full destination path for the CSV file.
	"""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	df.to_csv(path, index=False)


def seed_everything(seed: int = 42) -> None:
	"""
	Sets random seeds for major libraries to ensure experiment reproducibility.

	Reproducibility is a cornerstone of good scientific practice. This function sets the
	random seed for Python's built-in `random` module and `numpy`. This helps ensure that
	any process with a random component (like some model initializations) will produce the
	exact same result every time the code is run with the same seed.

	Args:
		seed (int, optional): The integer value for the seed. Defaults to 42.
	"""
	os.environ["PYTHONHASHSEED"] = str(seed)
	try:
		import random
		random.seed(seed)
	except Exception:
		pass
	try:
		np.random.seed(seed)
	except Exception:
		pass


def log_metrics(logger: logging.Logger, metrics: Dict[str, Any], prefix: str = "") -> None:
	"""
	Logs a dictionary of metrics using a provided logger.

	A helper function to produce clean and consistently formatted log output for a
	dictionary of metrics (e.g., {'r2': 0.65, 'rmse': 0.8}).

	Args:
		logger (logging.Logger): The logger instance to use.
		metrics (Dict[str, Any]): The dictionary of metrics to log.
		prefix (str, optional): A string to prepend to each metric name. Defaults to "".
	"""
	for key, value in metrics.items():
		log_key = f"{prefix}.{key}" if prefix else key
		logger.info(f"{log_key} = {value}")


def save_parity_plot(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	path: str,
	title: str = "Parity Plot",
	x_label: Optional[str] = None,
	y_label: Optional[str] = None,
	r2_linear_from_log: bool = False,
	r2_linear_values: Optional[Tuple[np.ndarray, np.ndarray]] = None,
	y_true_linear_reference: Optional[np.ndarray] = None,
	spearman_rho: Optional[float] = None,
) -> None:
	"""
	Generates and saves a parity plot (predicted vs. actual) to visualize model performance.

	A parity plot is a scatter plot that compares the model's predictions against the true
	values. For a perfect model, all points would lie on the 45-degree line (y=x). This
	visualization is a key deliverable for model evaluation.

	Args:
		y_true (np.ndarray): The ground truth target values.
		y_pred (np.ndarray): The predicted values from the model.
		path (str): The full destination path to save the plot image (e.g., 'results/plot.png').
		title (str, optional): The title for the plot. Defaults to "Parity Plot".
		r2_linear_from_log (bool): If True, compute R² in linear scale from log predictions.
		r2_linear_values (tuple): Optional tuple (y_true_lin, y_pred_lin) for direct linear R².
		y_true_linear_reference (np.ndarray): Optional true linear values for R² calculation.
			If provided with r2_linear_from_log=True, uses this for SST baseline.
			If not provided, will use 10^y_true as the linear reference.
	"""
	try:
		plt.figure(figsize=(8, 8))
		sns.set_theme(style="whitegrid")

		# Drop NaNs/Infs so plotting and metrics don't fail
		y_true = np.asarray(y_true, dtype=float)
		y_pred = np.asarray(y_pred, dtype=float)
		mask = np.isfinite(y_true) & np.isfinite(y_pred)
		y_true = y_true[mask]
		y_pred = y_pred[mask]

		# Create the scatter plot
		ax = sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=50, edgecolor="k", label="Predictions")

		# Determine the plot limits to make it square
		min_raw = float(min(np.nanmin(y_true), np.nanmin(y_pred)))
		max_raw = float(max(np.nanmax(y_true), np.nanmax(y_pred)))
		pad = 0.05 * (max_raw - min_raw) if max_raw > min_raw else 1.0
		min_val = min_raw - pad
		max_val = max_raw + pad
		ax.set_xlim(min_val, max_val)
		ax.set_ylim(min_val, max_val)

		# Plot the y=x line for reference (perfect prediction)
		ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")

		# Add labels and title
		ax.set_xlabel(x_label or r"Actual log10($\sigma$)", fontsize=14)
		ax.set_ylabel(y_label or r"Predicted log10($\sigma$)", fontsize=14)
		ax.set_title(title, fontsize=16)

		# Add R² score (and optionally linear-scale R² and Spearman ρ) to the plot
		sse = np.sum((y_pred - y_true) ** 2)
		sst = np.sum((y_true - np.mean(y_true)) ** 2)
		r2 = 1.0 - (sse / sst) if sst > 0 else np.nan
		text_lines = [f"$R^2 = {r2:.3f}$"]
		if r2_linear_values is not None:
			# Direct linear values provided - use explicit R² calculation
			try:
				y_true_lin, y_pred_lin = r2_linear_values
				y_true_lin = np.asarray(y_true_lin, dtype=float)
				y_pred_lin = np.asarray(y_pred_lin, dtype=float)
				mask_lin = np.isfinite(y_true_lin) & np.isfinite(y_pred_lin)
				y_true_lin = y_true_lin[mask_lin]
				y_pred_lin = y_pred_lin[mask_lin]
				if len(y_true_lin) > 0:
					# Explicit R² calculation: R² = 1 - SSE/SST
					sse = np.sum((y_pred_lin - y_true_lin) ** 2)
					sst = np.sum((y_true_lin - np.mean(y_true_lin)) ** 2)
					if sst > 0:
						r2_lin = 1.0 - (sse / sst)
						text_lines.append(f"$R^2_{{linear}} = {r2_lin:.3f}$")
			except Exception:
				pass
		elif r2_linear_from_log:
			# Convert log predictions to linear and compute R² explicitly
			try:
				# Convert predictions: y_linear = 10^(log_prediction)
				y_pred_lin = np.power(10.0, y_pred)
				
				# Use provided linear reference or convert from log
				if y_true_linear_reference is not None:
					y_true_lin = np.asarray(y_true_linear_reference, dtype=float)
				else:
					y_true_lin = np.power(10.0, y_true)
				
				# Filter out any NaN/Inf
				mask_lin = np.isfinite(y_true_lin) & np.isfinite(y_pred_lin)
				y_true_lin = y_true_lin[mask_lin]
				y_pred_lin = y_pred_lin[mask_lin]
				
				if len(y_true_lin) > 0:
					# Explicit R² calculation: R² = 1 - SSE/SST
					# SSE = Σ(y_pred_linear_i - y_true_linear_i)²
					sse = np.sum((y_pred_lin - y_true_lin) ** 2)
					# SST = Σ(y_true_linear_i - mean(y_true_linear))² = variance of test set
					sst = np.sum((y_true_lin - np.mean(y_true_lin)) ** 2)
					if sst > 0:
						r2_lin = 1.0 - (sse / sst)
						text_lines.append(f"$R^2_{{linear}} = {r2_lin:.3f}$")
			except Exception:
				pass
		if spearman_rho is not None and np.isfinite(spearman_rho):
			text_lines.append(r"$\rho = {:.3f}$".format(spearman_rho))

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

		# Ensure the directory exists and save the figure
		os.makedirs(os.path.dirname(path), exist_ok=True)
		plt.savefig(path, dpi=300)
		plt.close()  # Close the plot to free up memory
		logging.info(f"Parity plot saved to {path}")

	except ImportError:
		logging.warning("Matplotlib or Seaborn not installed. Skipping plot generation. Install with: pip install matplotlib seaborn")
	except Exception as e:
		logging.error(f"Could not generate or save parity plot: {e}")


def save_histogram(
	data: np.ndarray,
	path: str,
	title: str,
	x_label: str,
	bins: int = 50,
	vlines: Optional[Dict[float, str]] = None,
) -> None:
	"""
	Saves a histogram plot for exploratory data inspection.

	Args:
		data: 1D array-like data to plot.
		path: Destination path for the plot.
		title: Plot title.
		x_label: X-axis label.
		bins: Number of histogram bins.
		vlines: Optional mapping of x-position -> label for vertical reference lines.
	"""
	try:
		plt.figure(figsize=(8, 6))
		sns.set_theme(style="whitegrid")
		plt.hist(data, bins=bins, color="#4C72B0", alpha=0.85)
		if vlines:
			for x, label in vlines.items():
				plt.axvline(x=x, color="#C44E52", linestyle="--", linewidth=1.5, label=label)
			plt.legend()
		plt.title(title)
		plt.xlabel(x_label)
		plt.ylabel("Count")
		plt.tight_layout()
		os.makedirs(os.path.dirname(path), exist_ok=True)
		plt.savefig(path, dpi=300)
		plt.close()
		logging.info(f"Histogram saved to {path}")
	except ImportError:
		logging.warning("Matplotlib or Seaborn not installed. Skipping histogram generation.")
	except Exception as e:
		logging.error(f"Could not generate or save histogram: {e}")


def generate_interim_report(metrics: Dict[str, Dict[str, float]], path: str) -> None:
	"""
	Generates a Markdown report summarizing the Stage 1 hybrid experiments.

	Args:
		metrics (Dict[str, Dict[str, float]]): Mapping of experiment name -> metrics dict.
		path (str): Destination path for the Markdown report file.
	"""
	sorted_experiments = sorted(metrics.items(), key=lambda item: item[1]['r2'], reverse=True)
	best_experiment_name = sorted_experiments[0][0] if sorted_experiments else "N/A"
	best_r2 = metrics.get(best_experiment_name, {}).get('r2', 0)

	report_content = f"""# Interim Report: Stage 1 - Hybrid Structural Features

**Date:** {pd.to_datetime('today').strftime('%Y-%m-%d')}

## 1. Overview

Stage 1 augments the Stage 0 composition-only baseline with structural descriptors using a hybrid strategy:
- All samples receive lattice metadata parsed from the CSV (space-group number, lattice parameters, Z, derived volume).
- Samples with matching CIFs also receive rich geometric descriptors extracted via pymatgen (density, volume per atom, Li-Li/anion distances, etc.).
- CIF-only columns are zero-filled when a structure is missing, and a `has_cif_struct` indicator lets the model gate those features automatically.

## 1.5 Stage 0 Status (Composition-Only Baseline)

Stage 0 is established and includes:
- **Baseline features** derived from composition (elemental ratios + SMACT stoichiometry vector).
- **Embedding comparisons** on log10(sigma): Magpie, Mat2Vec, MegNet16, and all-embeddings combined.
- **Linear-scale sanity check** with Magpie-only embeddings against sigma (S cm-1).
- **Target inspection plots**: raw sigma and log10(sigma) histograms saved in `results/results_stage0/`.

## 2. Experiments, Metrics, and Parity Plots

All runs share the same 5-fold CV splits and `HistGradientBoostingRegressor`. Summary:

| Experiment          | **R^2 Score** | RMSE  | MAE   |
|---------------------|--------------|-------|-------|
"""
	for name, m in sorted_experiments:
		report_content += f"| {name:<20} | **{m['r2']:.4f}** | {m['rmse']:.4f} | {m['mae']:.4f} |\n"

	report_content += """

Parity plots (in `results/results_stage1/`) correspond to:
- `stage1_cv_parity_stage0_magpie.png` - Stage 0 composition-only baseline (ratios, SMACT, Magpie).
- `stage1_cv_parity_stage1_basic_struct.png` - adds basic structural scalars from CIF/CSV: `density`, `volume_per_atom`, `n_li_sites`, `n_total_atoms` (CIF-missing rows zero-filled; `has_cif_struct` gates them).
- `stage1_cv_parity_stage1_geometry.png` - builds on basic_struct with full lattice geometry and Li environment (all CIF-derived columns zero-filled when CIF is absent; `has_cif_struct` remains as the gate):
  - Lattice shape:  
    - `lattice_a`, `lattice_b`, `lattice_c` (Angstrom): cell edges from CIF.  
    - `lattice_alpha`, `lattice_beta`, `lattice_gamma` (degrees): inter-axial angles.
  - Anisotropy:  
    - `lattice_anisotropy_bc_a` = (b + c) / (2a): elongation vs. a.  
    - `lattice_anisotropy_max_min` = max(a,b,c) / min(a,b,c): distortion ratio.
  - Orthogonality:  
    - `angle_deviation_from_ortho` = mean(|alpha - 90|, |beta - 90|, |gamma - 90|).  
    - `is_cubic_like` = 1 if `lattice_anisotropy_max_min` < 1.05 and max angle dev < 2 degrees else 0.
  - Packing/composition:  
    - `li_fraction` = n_Li / n_total_atoms.  
    - `li_concentration` = n_Li / volume (Li per cubic Angstrom).  
    - `framework_density` = (n_total_atoms - n_Li) / volume (non-Li per cubic Angstrom).
  - Distances:  
    - `li_li_min_dist`: minimum Li-Li pair distance; `li_li_avg_dist`: mean of Li-Li distances < 5 Angstrom.  
    - `li_anion_min_dist`: minimum Li-anion distance (anion set {O,S,F,Cl,Br,I,N,Se,Te}).
  - Local environment:  
    - `li_coordination_avg`: mean Li coordination number via VoronoiNN (cutoff 5 Angstrom) over up to 10 Li sites.  
    - `li_site_avg_multiplicity`: average Wyckoff multiplicity parsed from atom-site records in CIF.
- `stage1_cv_parity_stage1_full_struct.png` - geometry plus space-group one-hot (`sg_1` ... `sg_230`), capturing symmetry classes.

### Feature Details (how each is computed)
- Lattice parameters/angles: read directly from CIF (`Structure.lattice.a/b/c`, `.alpha/.beta/.gamma`).
- Volume (used internally for densities): from CIF structure volume.
- Anisotropy:  
  - `lattice_anisotropy_bc_a` = (b + c) / (2a).  
  - `lattice_anisotropy_max_min` = max(a,b,c) / min(a,b,c).
- Orthogonality:  
  - `angle_deviation_from_ortho` = average absolute deviation of alpha, beta, gamma from 90 degrees.  
  - `is_cubic_like` = 1 if `lattice_anisotropy_max_min` < 1.05 AND max(|alpha - 90|, |beta - 90|, |gamma - 90|) < 2 degrees, else 0.
- Packing/composition:  
  - `li_fraction` = n_Li / n_atoms.  
  - `li_concentration` = n_Li / cell_volume.  
  - `framework_density` = (n_atoms - n_Li) / cell_volume.
- Distances:  
  - `li_li_min_dist`: minimum over all Li-Li pair distances.  
  - `li_li_avg_dist`: mean of Li-Li distances filtered to < 5 Angstrom (nearest-neighbor scale).  
  - `li_anion_min_dist`: minimum Li-anion distance considering up to 20 Li and 50 anion sites for speed.
- Coordination:  
  - `li_coordination_avg`: VoronoiNN coordination (cutoff 5 Angstrom) averaged over up to 10 Li sites.  
- Multiplicity:  
  - `li_site_avg_multiplicity`: average multiplicity of Li atom sites parsed from CIF atom-site records.
- Basic structural scalars (used in basic_struct and included in geometry):  
  - `density`: CIF mass density (g/cm-cube).  
  - `volume_per_atom`: cell_volume / n_atoms.  
  - `n_li_sites`: count of Li atoms (sites) in the cell.  
  - `n_total_atoms`: total atoms in the cell.
- Space-group one-hot (full_struct only):  
  - `sg_1` ... `sg_230` from CIF/CSV `spacegroup_number`; 1 for matching SG, else 0.

CIF-missing rows: all CIF-derived numeric columns are zero-filled, and `has_cif_struct` signals availability so the model can down-weight missing-structure rows automatically.

## 3. Key Findings

1. **Hybrid gating works:** Keeping all rows while selectively activating CIF descriptors yields a substantial R^2 gain over Stage 0 without shrinking the training set.
2. **CIF-derived metrics drive the lift:** Permutation importance highlights lattice-c, Li coordination average, framework density, and n_li_sites as leading contributors when CIFs are available.
3. **Space-group encoding adds value:** The full-struct experiment narrowly beats geometry-only, indicating symmetry-specific effects are meaningful.

### How permutation importance is computed

Permutation importance measures how much the validation score drops when a single feature column is randomly shuffled. For each feature we:

1. Record the baseline CV score.
2. Shuffle that feature (breaking its relationship with the target) while leaving all other columns intact.
3. Recompute the CV score and average the drop over several shuffles.

Large positive drops mean the model relied heavily on that feature; near-zero drops imply the feature added little unique signal.

## 4. Next Steps

- Continue acquiring CIFs (or Materials Project structures) for the remaining entries; the pipeline will automatically leverage the additional geometries.
- Stage 2 will extend this base with physics-informed descriptors and Optuna-based hyperparameter tuning.
"""
	try:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		with open(path, "w") as f:
			f.write(report_content)
		logging.info(f"Interim report generated and saved to {path}")
	except Exception as e:
		logging.error(f"Could not write interim report: {e}")


def generate_stage0_report(path: str) -> None:
	"""
	Static, instructional report for Stage 0.
	"""
	report_content = f"""# Interim Report: Stage 0 - Composition-Only Baseline

**Date:** {pd.to_datetime('today').strftime('%Y-%m-%d')}

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
"""
	try:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		with open(path, "w", encoding="utf-8") as f:
			f.write(report_content)
		logging.info(f"Stage 0 report generated and saved to {path}")
	except Exception as e:
		logging.error(f"Could not write Stage 0 report: {e}")


def generate_stage1_report(path: str, metrics: Optional[Dict[str, Dict[str, float]]] = None) -> None:
	"""
	Generate the Stage 1 interim report with optional metrics table.
	
	Args:
		path: Destination path for the Markdown report file.
		metrics: Optional dict mapping experiment name -> metrics dict with 'r2', 'rmse', 'mae'.
	"""
	report_content = f"""# Interim Report: Stage 1 - Structural Features

**Date:** {pd.to_datetime('today').strftime('%Y-%m-%d')}

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
"""
	# Add metrics table if provided
	if metrics:
		report_content += "\n## 7. Cross-Validation Results\n\n"
		report_content += "| Experiment | R² Score | RMSE | MAE |\n"
		report_content += "|------------|----------|------|-----|\n"
		for exp_name, m in metrics.items():
			r2 = m.get('r2', float('nan'))
			rmse = m.get('rmse', float('nan'))
			mae = m.get('mae', float('nan'))
			report_content += f"| {exp_name} | {r2:.4f} | {rmse:.4f} | {mae:.4f} |\n"
		report_content += "\n"
		
		# Add comparison note
		report_content += """**Note:** The stage0_magpie experiment should produce results consistent with
Stage 0 results from the baseline pipeline (`results/results_stage0/`). If there are
discrepancies, verify that no data leakage is occurring (e.g., the raw conductivity
column must not be included in features).

## 8. Key Observations

- **Baseline consistency:** Stage 0 Magpie results should match between this pipeline
  and the original Stage 0 pipeline to confirm no leakage.
- **Structural feature value:** Compare R² gains from adding structural features.
- **Feature importance:** Check the permutation importance plot to identify which
  structural features contribute most to prediction accuracy.
"""
	
	try:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		with open(path, "w", encoding="utf-8") as f:
			f.write(report_content)
		logging.info(f"Stage 1 report generated and saved to {path}")
	except Exception as e:
		logging.error(f"Could not write Stage 1 report: {e}")