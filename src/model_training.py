# =================================================================================================
# Module Imports
# =================================================================================================
# Import 'dataclasses' for creating simple classes to store data, and 'typing' for type hints.
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

# Import 'numpy' and 'pandas' for numerical operations and data manipulation.
import numpy as np
import pandas as pd

# Import specific machine learning components from scikit-learn.
# GroupKFold: A cross-validation strategy that ensures all samples from a specific group
#             (e.g., a chemical family) belong to the same fold.
from sklearn.model_selection import GroupKFold
# HistGradientBoostingRegressor: A modern, fast, and effective gradient boosting model.
from sklearn.ensemble import HistGradientBoostingRegressor
# ElasticNet: A linear regression model that combines L1 and L2 regularization.
from sklearn.linear_model import ElasticNet
# StandardScaler: A pre-processing step to standardize features by removing the mean and scaling to unit variance.
from sklearn.preprocessing import StandardScaler
# Pipeline: A utility to chain multiple steps (like scaling and modeling) together.
from sklearn.pipeline import Pipeline

# Import the custom metric calculation function from our own utilities module.
from .utils import compute_regression_metrics


# =================================================================================================
# Experiment Configuration
# =================================================================================================
# Using a dataclass is a clean way to group all experiment-related settings into a single object.
# This makes it easy to pass configuration around and to see all the settings in one place.
@dataclass
class ExperimentConfig:
	"""
	A data structure to hold all the configuration settings for a machine learning experiment.

	Attributes:
		model_name (str): The name of the model to use (e.g., "hgbt", "elasticnet").
		n_splits (int): The number of folds to use for cross-validation.
		random_state (int): The random seed for reproducibility.
		params (Optional[Dict[str, Any]]): A dictionary of hyperparameters to pass to the model.
											If None, the model's defaults will be used.
		group_col (Optional[str]): The name of the column to use for GroupKFold splitting.
								   If None, standard (random) KFold is effectively used.
	"""
	model_name: str = "hgbt"  # or "elasticnet"
	n_splits: int = 5
	random_state: int = 42
	params: Optional[Dict[str, Any]] = None
	group_col: Optional[str] = None
	# If True, clamp predictions to be non-negative in the model wrapper.
	non_negative: bool = False


class NonNegativeRegressor:
	"""
	Wraps a regressor to enforce non-negative predictions.

	This does not change the training objective; it applies a hard boundary
	condition at prediction time so outputs are >= 0.
	"""
	def __init__(self, estimator: Pipeline):
		self.estimator = estimator

	def fit(self, X: pd.DataFrame, y: pd.Series):
		self.estimator.fit(X, y)
		return self

	def predict(self, X: pd.DataFrame) -> np.ndarray:
		preds = self.estimator.predict(X)
		return np.clip(preds, a_min=0.0, a_max=None)


# =================================================================================================
# Model Building and Training Functions
# =================================================================================================

def _build_model(config: ExperimentConfig) -> Pipeline:
	"""
	Constructs a scikit-learn model pipeline based on the given experiment configuration.

	This is a factory function that returns a model pipeline ready for training. It supports
	different model types and automatically includes necessary pre-processing steps like
	scaling for linear models. This keeps the main training loop clean and modular.

	Args:
		config (ExperimentConfig): The configuration object specifying the model and its settings.

	Raises:
		ValueError: If an unsupported model_name is provided in the configuration.

	Returns:
		Pipeline: A scikit-learn Pipeline object containing the configured model.
	"""
	if config.model_name.lower() == "hgbt":
		# HistGradientBoostingRegressor is a tree-based model and does not require feature scaling.
		model = HistGradientBoostingRegressor(random_state=config.random_state, **(config.params or {}))
		pipeline = Pipeline(steps=[("model", model)])
	elif config.model_name.lower() == "elasticnet":
		# ElasticNet is a linear model and is sensitive to the scale of the features.
		# Therefore, we create a pipeline that first standardizes the data and then fits the model.
		model = ElasticNet(random_state=config.random_state, **(config.params or {}))
		pipeline = Pipeline(steps=[("scaler", StandardScaler(with_mean=True, with_std=True)), ("model", model)])
	else:
		raise ValueError(f"Unsupported model: {config.model_name}")

	# Enforce non-negative predictions if requested.
	if config.non_negative:
		return NonNegativeRegressor(pipeline)

	return pipeline


def run_group_kfold_cv(
	X: pd.DataFrame,
	y: pd.Series,
	config: ExperimentConfig,
) -> Tuple[List[float], Dict[str, float], np.ndarray]:
	"""
	Performs GroupKFold cross-validation for a given model configuration.

	This function is the core of our model evaluation strategy. It trains the model multiple
	times on different subsets of the data (folds) and evaluates it on the held-out parts.
	This provides a robust estimate of the model's performance on unseen data. It also
	generates "out-of-fold" (OOF) predictions, which can be used for ensembling or further analysis.

	Args:
		X (pd.DataFrame): The feature matrix.
		y (pd.Series): The target vector.
		config (ExperimentConfig): The experiment configuration object.

	Returns:
		Tuple[List[float], Dict[str, float], np.ndarray]:
			- A list of RMSE scores, one for each fold.
			- A dictionary of overall metrics (R², MAE, RMSE) calculated on the OOF predictions.
			- A numpy array of the out-of-fold predictions.
	"""
	# If a group column is specified, use it for splitting. Otherwise, use standard KFold.
	if config.group_col and config.group_col in X.columns:
		groups = X[config.group_col].values
	else:
		# If no group column, create a simple range of numbers, which makes GroupKFold behave like KFold.
		groups = np.arange(len(X))

	# The group column itself should not be used as a feature for the model.
	features = [c for c in X.columns if c != config.group_col]
	X_mat = X[features]
	
	gkf = GroupKFold(n_splits=config.n_splits)
	
	# Initialize an array to store the out-of-fold predictions.
	oof = np.zeros(len(X), dtype=float)
	fold_scores: List[float] = []

	# Loop through each fold defined by the GroupKFold splitter.
	for fold_idx, (trn_idx, val_idx) in enumerate(gkf.split(X_mat, y, groups=groups)):
		# Split the data into training and validation sets for the current fold.
		X_tr, X_val = X_mat.iloc[trn_idx], X_mat.iloc[val_idx]
		y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

		# Build a fresh model for each fold to prevent data leakage.
		m = _build_model(config)
		m.fit(X_tr, y_tr)
		
		# Make predictions on the validation set.
		preds = m.predict(X_val)
		oof[val_idx] = preds

		# Calculate and store the performance metrics for this fold.
		metrics = compute_regression_metrics(y_val.values, preds)
		fold_scores.append(metrics["rmse"])  # Use RMSE as the primary score for comparing folds.

	# After the loop, calculate the overall metrics using all the out-of-fold predictions.
	overall_metrics = compute_regression_metrics(y.values, oof)
	return fold_scores, overall_metrics, oof


def run_cv_with_predefined_splits(
	X: pd.DataFrame,
	y: pd.Series,
	splits: List[Tuple[np.ndarray, np.ndarray]],
	config: ExperimentConfig,
) -> Tuple[List[float], Dict[str, float], np.ndarray]:
	"""
	Performs cross-validation using a predefined list of training/validation splits.

	This function is essential for the embedding comparison experiment. It ensures that each
	model (using a different feature set) is trained and evaluated on the exact same folds
	of the data, making the comparison of their performance fair and reliable.

	Args:
		X (pd.DataFrame): The feature matrix.
		y (pd.Series): The target vector.
		splits (List[Tuple[np.ndarray, np.ndarray]]): A list of (train_indices, validation_indices) tuples.
		config (ExperimentConfig): The experiment configuration object.

	Returns:
		Tuple[List[float], Dict[str, float], np.ndarray]:
			- A list of RMSE scores, one for each fold.
			- A dictionary of overall metrics (R², MAE, RMSE) calculated on the OOF predictions.
			- A numpy array of the out-of-fold predictions.
	"""
	features = [c for c in X.columns if c != config.group_col]
	X_mat = X[features]
	
	oof = np.zeros(len(X), dtype=float)
	fold_scores: List[float] = []

	for fold_idx, (trn_idx, val_idx) in enumerate(splits):
		X_tr, X_val = X_mat.iloc[trn_idx], X_mat.iloc[val_idx]
		y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

		m = _build_model(config)
		m.fit(X_tr, y_tr)
		preds = m.predict(X_val)
		oof[val_idx] = preds

		metrics = compute_regression_metrics(y_val.values, preds)
		fold_scores.append(metrics["rmse"])

	overall_metrics = compute_regression_metrics(y.values, oof)
	return fold_scores, overall_metrics, oof


def run_cv_with_predefined_splits_and_gaps(
	X: pd.DataFrame,
	y: pd.Series,
	splits: List[Tuple[np.ndarray, np.ndarray]],
	config: ExperimentConfig,
	metric: str = "rmse",
) -> Tuple[List[float], Dict[str, float], np.ndarray, List[float]]:
	"""
	Same as run_cv_with_predefined_splits but also returns train metric per fold.
	Used for Optuna objectives that penalize large train-val gaps.

	Args:
		metric: "rmse" or "mae" for per-fold scores (default "rmse").

	Returns:
		Tuple of (fold_val_scores, overall_metrics, oof, train_metric_per_fold).
	"""
	features = [c for c in X.columns if c != config.group_col]
	X_mat = X[features]

	oof = np.zeros(len(X), dtype=float)
	fold_val_scores: List[float] = []
	train_metric_per_fold: List[float] = []

	for fold_idx, (trn_idx, val_idx) in enumerate(splits):
		X_tr, X_val = X_mat.iloc[trn_idx], X_mat.iloc[val_idx]
		y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

		m = _build_model(config)
		m.fit(X_tr, y_tr)
		val_preds = m.predict(X_val)
		oof[val_idx] = val_preds

		val_metrics = compute_regression_metrics(y_val.values, val_preds)
		fold_val_scores.append(val_metrics[metric])

		train_preds = m.predict(X_tr)
		train_metrics = compute_regression_metrics(y_tr.values, train_preds)
		train_metric_per_fold.append(train_metrics[metric])

	overall_metrics = compute_regression_metrics(y.values, oof)
	return fold_val_scores, overall_metrics, oof, train_metric_per_fold


def fit_full_and_predict(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_test: pd.DataFrame,
	config: ExperimentConfig,
) -> np.ndarray:
	"""
	Trains the model on the entire training dataset and makes predictions on the test set.

	After evaluating the model's performance with cross-validation, the final step is to
	train it on all available training data to build the best possible model. This model is
	then used to predict the outcomes for the unseen test data.

	Args:
		X_train (pd.DataFrame): The full training feature matrix.
		y_train (pd.Series): The full training target vector.
		X_test (pd.DataFrame): The test feature matrix.
		config (ExperimentConfig): The experiment configuration.

	Returns:
		np.ndarray: An array of predictions for the test set.
	"""
	# Ensure the group column is not included in the features used for training.
	features = [c for c in X_train.columns if c != config.group_col]
	model = _build_model(config)
	
	# Fit the model on all the training data.
	model.fit(X_train[features], y_train)
	
	# Predict on the test data.
	return model.predict(X_test[features])
