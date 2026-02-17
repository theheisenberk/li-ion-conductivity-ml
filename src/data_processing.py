# =================================================================================================
# Module Imports
# =================================================================================================
# Import 'os' for operating system interactions (e.g., file paths) and 'typing' for type hints.
import os
from typing import Tuple, Optional

# Import 'numpy' and 'pandas' for numerical operations and data manipulation.
import numpy as np
import pandas as pd


# =================================================================================================
# Constants
# =================================================================================================
# Define a global constant for the target column name. This is good practice as it avoids
# hardcoding the string in multiple places, making the code easier to maintain. If the target
# column name ever changes, we only need to update it here.
TARGET_COL = "log10_sigma"


# =================================================================================================
# Data Loading and Processing Functions
# =================================================================================================

def load_data(data_dir: str, train_filename: str = "train.csv", test_filename: str = "test.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Loads the training and testing data from CSV files into pandas DataFrames.

	This function abstracts the data loading process. It constructs the full file paths,
	reads the CSVs, and returns the two dataframes, which is the first step in our
	data familiarization phase (Weeks 1-2).

	Args:
		data_dir (str): The directory path where the raw data is stored.
		train_filename (str, optional): The filename for the training data. Defaults to "train.csv".
		test_filename (str, optional): The filename for the testing data. Defaults to "test.csv".

	Returns:
		Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing DataFrames.
	"""
	train_path = os.path.join(data_dir, train_filename)
	test_path = os.path.join(data_dir, test_filename)
	
	# Use pandas to read the CSV files.
	train_df = pd.read_csv(train_path)
	test_df = pd.read_csv(test_path)
	
	return train_df, test_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Performs basic data cleaning operations on a DataFrame.

	This function handles common data quality issues as part of the initial data cleanup
	(Weeks 1-2). The steps are:
	1. Make a copy to avoid modifying the original DataFrame (side-effect prevention).
	2. Strip leading/trailing whitespace from column names to prevent potential errors.
	3. Replace infinite values (positive and negative) with NaN (Not a Number) for consistency.
	4. Drop any rows that are exact duplicates of other rows.

	Args:
		df (pd.DataFrame): The input DataFrame to clean.

	Returns:
		pd.DataFrame: The cleaned DataFrame.
	"""
	df = df.copy()
	df.columns = [c.strip() for c in df.columns]
	df.replace([np.inf, -np.inf], np.nan, inplace=True)
	df.drop_duplicates(inplace=True)
	return df


def coerce_sigma_series(series: pd.Series, return_mask: bool = False):
	"""
	Coerces an ionic conductivity series to numeric, handling strings like '<1E-10'
	by converting them to the threshold value (e.g., 1e-10).
	
	Args:
		series: The ionic conductivity series to coerce.
		return_mask: If True, also return a boolean mask indicating which values were coerced.
	
	Returns:
		If return_mask is False: pd.Series of numeric values.
		If return_mask is True: Tuple of (pd.Series of numeric values, pd.Series of bool mask).
	"""
	s = series.copy()
	s_str = s.astype(str)
	pattern = r"^\s*<\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*$"
	mask = s_str.str.match(pattern, na=False)
	extracted = s_str.str.extract(pattern, expand=False)
	s_numeric = pd.to_numeric(s, errors="coerce")
	s_numeric.loc[mask] = pd.to_numeric(extracted[mask], errors="coerce")
	
	if return_mask:
		return s_numeric, mask
	return s_numeric


def add_target_log10_sigma(df: pd.DataFrame, target_sigma_col: str) -> pd.DataFrame:
	"""
	Creates the target variable `log10_sigma` from the raw ionic conductivity column.

	This function is a key part of the preprocessing tasks for Weeks 1-2. It performs the
	log10 transformation as specified in the project plan. The process is:
	1. Check if the raw conductivity column exists in the dataframe.
	2. Convert the column to a numeric type, mapping strings like '<1E-10' to 1e-10.
	3. Add a binary indicator `sigma_is_coerced` (1 if value was coerced from threshold string).
	4. Store the numeric conductivity back into the dataframe so cleaned data is numeric.
	5. To avoid taking the logarithm of zero or a negative number, `np.clip` is used to
	   set a very small positive minimum value (1e-30).
	6. Compute the log10 of the clipped conductivity values and store it in the `TARGET_COL`.

	Args:
		df (pd.DataFrame): The input DataFrame (typically the training set).
		target_sigma_col (str): The name of the column containing the raw ionic conductivity values.

	Returns:
		pd.DataFrame: The DataFrame with the new `log10_sigma` target column and
		              `sigma_is_coerced` indicator column added.
	"""
	df = df.copy()
	if target_sigma_col in df.columns:
		# Convert to numeric, coercing errors to NaN, and mapping '<1E-10' -> 1e-10
		# Also get the mask indicating which values were coerced from threshold strings
		sigma_numeric, coerced_mask = coerce_sigma_series(df[target_sigma_col], return_mask=True)
		
		# Add binary indicator for coerced sigma values (model can learn to weight these differently)
		df["sigma_is_coerced"] = coerced_mask.astype(int)
		
		# Keep the cleaned numeric conductivity values in the dataframe
		df[target_sigma_col] = sigma_numeric
		# Avoid log of non-positive numbers by clipping to a tiny positive value.
		# This is a standard practice to ensure numerical stability.
		clipped_sigma = np.clip(sigma_numeric, a_min=1e-30, a_max=None)
		df[TARGET_COL] = np.log10(clipped_sigma)
	return df


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COL, drop_cols: Optional[list] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
	"""
	Splits a DataFrame into a feature matrix (X) and a target vector (y).

	This utility function separates the independent variables (features) from the dependent
	variable (target). It is designed to work for both training data (which has a target)
	and testing data (which does not).

	Args:
		df (pd.DataFrame): The DataFrame to split.
		target_col (str, optional): The name of the target column. Defaults to TARGET_COL.
		drop_cols (Optional[list], optional): A list of any additional columns to drop
											 from the feature matrix. Defaults to None.

	Returns:
		Tuple[pd.DataFrame, Optional[pd.Series]]: A tuple containing the feature matrix (X)
												 and the target series (y). `y` will be `None`
												 if the target column is not found in the DataFrame.
	"""
	drop_cols = drop_cols or []
	# Create a list of columns to keep as features.
	feature_cols = [c for c in df.columns if c not in drop_cols]
	
	# If the target column is present, separate it from the features.
	if target_col in feature_cols:
		feature_cols.remove(target_col)
		X = df[feature_cols]
		y = df[target_col]
		return X, y
	# If the target column is not present (e.g., in the test set), return only the features.
	else:
		X = df[feature_cols]
		return X, None


