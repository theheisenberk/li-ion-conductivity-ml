# =================================================================================================
# Module Imports
# =================================================================================================
# Import 'typing' for type hints to improve code readability and maintainability.
import os
from typing import Tuple, Optional, Dict, Any, List

# Import 'numpy' and 'pandas' for numerical operations and data manipulation.
import numpy as np
import pandas as pd


# =================================================================================================
# Stage 0: Composition-Only Feature Engineering Functions
# =================================================================================================
# The functions in this section correspond to the feature engineering tasks for Weeks 3-5.
# They are designed to generate features based solely on the chemical composition of materials,
# forming the input for our baseline model.

def stage0_element_embeddings(
    df: pd.DataFrame, 
    composition_col: str = "composition",
    embedding_names: List[str] = None
) -> pd.DataFrame:
	"""
	Generates composition-weighted average element embeddings for each material.

	This is a key feature generation step for Weeks 3-5. Element embeddings represent each
	element as a dense vector in a high-dimensional space, capturing complex chemical properties
	and relationships. This function calculates the average embedding for a material by weighting
	each element's embedding vector by its atomic fraction in the composition.

	It uses the 'elementembeddings' library and generates features for three different,
	well-known embedding schemes: Mat2Vec, MegNet16, and Magpie.

	Args:
		df (pd.DataFrame): The input DataFrame containing a column with chemical compositions.
		composition_col (str, optional): The name of the composition column. Defaults to "composition".
		embedding_names (List[str], optional): A list of embedding schemes to generate. 
											   If None, defaults to all three. Defaults to None.

	Returns:
		pd.DataFrame: The DataFrame with new columns for each dimension of each embedding type.
	"""
	df = df.copy()
	if composition_col not in df.columns:
		# If the composition column doesn't exist, there's nothing to do.
		return df
	
	try:
		# These libraries are required for this function. We import them here so that an
		# error is only raised if this specific function is called without them being installed.
		from elementembeddings.core import Embedding
		from pymatgen.core.composition import Composition
		
		# As specified in the project plan, we will compare these three embedding schemes.
		if embedding_names is None:
			embedding_names = ["mat2vec", "megnet16", "magpie"]
		
		for emb_name in embedding_names:
			try:
				# Load the pre-trained embedding data.
				emb = Embedding.load_data(emb_name)
				emb_df = emb.as_dataframe()  # Convert to a DataFrame for easy lookup.
				embedding_dim = emb_df.shape[1]
				
				# Define a helper function to process a single composition string.
				def _get_weighted_embedding(comp_str: str) -> np.ndarray:
					"""Computes the composition-weighted average of element embeddings."""
					try:
						comp = Composition(comp_str)
						total_atoms = comp.num_atoms
						if total_atoms == 0:
							return np.full(embedding_dim, np.nan)
						
						# Initialize a zero vector to accumulate the weighted embeddings.
						embedding_sum = np.zeros(embedding_dim)
						for el, count in comp.items():
							atomic_fraction = count / total_atoms
							el_symbol = el.symbol
							if el_symbol in emb_df.index:
								el_emb = emb_df.loc[el_symbol].values
								embedding_sum += atomic_fraction * el_emb
						
						return embedding_sum
					except Exception:
						# If parsing fails, return a vector of NaNs.
						return np.full(embedding_dim, np.nan)
				
				# Apply this function to every row in the composition column.
				embeddings = df[composition_col].astype(str).map(_get_weighted_embedding)
				
				# The result is a Series of numpy arrays. We need to expand this into separate
				# columns in the DataFrame (e.g., 'mat2vec_emb_0', 'mat2vec_emb_1', ...).
				emb_data = {f"{emb_name}_emb_{i}": embeddings.map(
					lambda x: x[i] if isinstance(x, np.ndarray) and len(x) > i else np.nan
				) for i in range(embedding_dim)}
				
				# Create a new DataFrame from the expanded embedding data and concatenate it.
				emb_df_new = pd.DataFrame(emb_data, index=df.index)
				df = pd.concat([df, emb_df_new], axis=1)
			except Exception as e:
				import warnings
				warnings.warn(f"Could not load or process '{emb_name}' embedding: {e}")
				continue
				
	except ImportError:
		import warnings
		warnings.warn("ElementEmbeddings or pymatgen not installed. Skipping embedding features. Install with: pip install elementembeddings pymatgen")
	except Exception as e:
		import warnings
		warnings.warn(f"Element embeddings could not be generated due to an unexpected error: {e}")
	
	return df


def stage0_elemental_statistics(df: pd.DataFrame, composition_col: str = "composition") -> pd.DataFrame:
	"""
	Calculates composition-weighted statistics of elemental properties.

	As required for Weeks 3-5, this function adds features based on simple elemental
	statistics. For each chemical composition, it computes the weighted average and
	weighted variance (a measure of spread) of key elemental properties.

	Features generated:
	- Average and variance of atomic mass.
	- Average and variance of Pauling electronegativity.
	- Average and variance of covalent radius.

	Args:
		df (pd.DataFrame): The input DataFrame.
		composition_col (str, optional): The name of the composition column. Defaults to "composition".

	Returns:
		pd.DataFrame: The DataFrame with new columns for the elemental statistics.
	"""
	df = df.copy()
	if composition_col not in df.columns:
		return df

	try:
		from pymatgen.core import Element, Composition

		def _get_elemental_stats(comp_str: str) -> tuple:
			"""Helper function to compute stats for a single composition."""
			try:
				comp = Composition(comp_str)
				elements = list(comp.elements)
				weights = [comp.get_atomic_fraction(el) for el in elements]

				# Get elemental properties, handling cases where a property might be missing (e.g., covalent radius for some elements)
				atomic_masses = [el.atomic_mass for el in elements]
				electronegativities = [el.X if el.X is not None else 0 for el in elements]
				covalent_radii = [el.covalent_radius if el.covalent_radius is not None else 0 for el in elements]

				# Calculate weighted average for each property
				avg_atomic_mass = np.average(atomic_masses, weights=weights)
				avg_electronegativity = np.average(electronegativities, weights=weights)
				avg_covalent_radius = np.average(covalent_radii, weights=weights)

				# Calculate weighted variance for each property
				var_atomic_mass = np.average((np.array(atomic_masses) - avg_atomic_mass)**2, weights=weights)
				var_electronegativity = np.average((np.array(electronegativities) - avg_electronegativity)**2, weights=weights)
				var_covalent_radius = np.average((np.array(covalent_radii) - avg_covalent_radius)**2, weights=weights)
				
				return (
					avg_atomic_mass, var_atomic_mass,
					avg_electronegativity, var_electronegativity,
					avg_covalent_radius, var_covalent_radius
				)
			except Exception:
				return (np.nan,) * 6

		# Apply the function to the composition column
		stats = df[composition_col].astype(str).apply(_get_elemental_stats)
		
		# Create new columns from the returned tuples
		stat_cols = [
			"avg_atomic_mass", "var_atomic_mass",
			"avg_electronegativity", "var_electronegativity",
			"avg_covalent_radius", "var_covalent_radius"
		]
		df[stat_cols] = pd.DataFrame(stats.tolist(), index=df.index)

	except ImportError:
		import warnings
		warnings.warn("Pymatgen not installed. Skipping elemental statistics features. Install with: pip install pymatgen")
	except Exception as e:
		import warnings
		warnings.warn(f"Elemental statistics features could not be generated: {e}")

	return df


def stage0_smact_features(df: pd.DataFrame, composition_col: str = "composition") -> pd.DataFrame:
	"""
	Generates the SMACT fractional stoichiometry vector for each material.

	This is another key feature for Stage 0 (Weeks 1-5). The SMACT (Semiconducting Materials
	from Analogy and Chemical Theory) library provides a way to represent a chemical composition
	as a fixed-length vector of length 103 (one for each element up to Lr). The value at
	each position in the vector is the atomic fraction of that element in the composition.
	This provides a standardized numerical representation of stoichiometry.

	Args:
		df (pd.DataFrame): The input DataFrame containing a column with chemical compositions.
		composition_col (str, optional): The name of the composition column. Defaults to "composition".

	Returns:
		pd.DataFrame: The DataFrame with 103 new columns for the stoichiometry vector.
	"""
	df = df.copy()
	if composition_col not in df.columns:
		return df
	
	try:
		# Import required libraries here to avoid hard dependencies.
		from smact.screening import ml_rep_generator
		from pymatgen.core.composition import Composition
		
		# Helper function to process one composition string.
		def _get_stoichiometry_vector(comp_str: str) -> np.ndarray:
			"""Gets the 103-element stoichiometry vector from SMACT."""
			try:
				comp = Composition(comp_str)
				elements = [str(el) for el in comp.keys()]
				counts = [comp[el] for el in comp.keys()]
				# `ml_rep_generator` creates the normalized 103-element vector.
				ml_rep = ml_rep_generator(elements, counts)
				return ml_rep
			except Exception:
				return np.full(103, np.nan)
		
		# Apply the function to the composition column.
		stoich_vectors = df[composition_col].astype(str).map(_get_stoichiometry_vector)
		
		# Expand the resulting vectors into 103 new columns in the DataFrame.
		stoich_data = {
			f"smact_stoich_{i}": stoich_vectors.map(
				lambda x: x[i] if isinstance(x, np.ndarray) and len(x) > i else np.nan
			) for i in range(103)
		}
		
		stoich_df = pd.DataFrame(stoich_data, index=df.index)
		df = pd.concat([df, stoich_df], axis=1)
	except ImportError:
		import warnings
		warnings.warn("SMACT or pymatgen not installed. Skipping SMACT features. Install with: pip install smact pymatgen")
	except Exception as e:
		import warnings
		warnings.warn(f"SMACT stoichiometry features could not be generated: {e}")
	
	return df


def stage0_elemental_ratios(df: pd.DataFrame, composition_col: str = "composition") -> pd.DataFrame:
	"""
	Calculates simple elemental ratios and counts based on the chemical composition.

	As part of the Stage 0 feature set (Weeks 1-5), the project plan calls for simple,
	interpretable features like the fraction of lithium, the fraction of anions, and the
	total number of distinct elements in the formula. These features are easy to compute
	and can capture important high-level chemical information.

	Args:
		df (pd.DataFrame): The input DataFrame.
		composition_col (str, optional): The name of the composition column. Defaults to "composition".

	Returns:
		pd.DataFrame: DataFrame with new columns for 'li_fraction', 'anion_fraction', 'total_elements'.
	"""
	df = df.copy()
	if composition_col not in df.columns:
		return df
	
	try:
		# Import pymatgen here.
		from pymatgen.core.composition import Composition
		
		# Helper function to get the atomic fraction of Lithium.
		def _get_li_fraction(comp_str: str) -> float:
			try:
				comp = Composition(comp_str)
				return comp.get_atomic_fraction("Li") if comp.num_atoms > 0 else 0.0
			except Exception:
				return np.nan
		
		# Helper function to get the total atomic fraction of common anions.
		def _get_anion_fraction(comp_str: str) -> float:
			try:
				comp = Composition(comp_str)
				if comp.num_atoms == 0:
					return 0.0
				# Define a list of common anion-forming elements.
				anions = ["O", "S", "F", "Cl", "Br", "I", "N", "P"]
				return sum(comp.get_atomic_fraction(el) for el in anions if el in comp)
			except Exception:
				return np.nan
		
		# Helper function to count the number of distinct elements.
		def _get_total_elements(comp_str: str) -> int:
			try:
				comp = Composition(comp_str)
				return len(comp.elements)
			except Exception:
				return np.nan
		
		# Apply these functions to create the new feature columns.
		df["li_fraction"] = df[composition_col].astype(str).map(_get_li_fraction)
		df["anion_fraction"] = df[composition_col].astype(str).map(_get_anion_fraction)
		df["total_elements"] = df[composition_col].astype(str).map(_get_total_elements)
		
	except ImportError:
		import warnings
		warnings.warn("Pymatgen not installed. Skipping elemental ratio features. Install with: pip install pymatgen")
	except Exception as e:
		import warnings
		warnings.warn(f"Elemental ratio features could not be generated: {e}")
	
	return df


def stage0_basic_features(df: pd.DataFrame, composition_col: str = "composition") -> pd.DataFrame:
	"""
	A wrapper function that generates all Stage 0 composition-only baseline features.

	This function orchestrates the creation of all features required for Weeks 1-5. It calls
	the individual feature generation functions in sequence, making it a single entry point
	for creating the complete Stage 0 feature set.

	This modular design makes the main script cleaner and the feature generation process
	easier to understand and modify.

	Args:
		df (pd.DataFrame): The input DataFrame.
		composition_col (str, optional): The name of the composition column. Defaults to "composition".

	Returns:
		pd.DataFrame: The DataFrame augmented with all Stage 0 features.
	"""
	df = df.copy()
	
	# Apply all Stage 0 feature generators in order.
	df = stage0_elemental_ratios(df, composition_col=composition_col)
	df = stage0_smact_features(df, composition_col=composition_col)
	df = stage0_element_embeddings(df, composition_col=composition_col)
	
	return df


# =================================================================================================
# Stage 1: Structural Feature Engineering Functions (CSV-based)
# =================================================================================================
# These functions extract structural information directly from CSV columns.
# The CSV contains lattice parameters (a, b, c), angles (alpha, beta, gamma), 
# space group number, and formula units Z.

import math

AVOGADRO_NUMBER = 6.02214076e23  # mol^-1
ANGSTROM_TO_CM = 1e-8


def stage1_csv_structural_features(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Extracts structural features from CSV columns.
	
	The CSV file contains the following structural columns:
	- a, b, c: Lattice parameters (Å)
	- alpha, beta, gamma: Lattice angles (degrees)
	- Space group #: International Tables space group number
	- Z: Formula units per unit cell
	
	This function creates:
	- lattice_a, lattice_b, lattice_c: Renamed from a, b, c
	- lattice_alpha, lattice_beta, lattice_gamma: Renamed from alpha, beta, gamma
	- spacegroup_number: Renamed from "Space group #"
	- formula_units_z: Renamed from Z
	- lattice_volume: Unit cell volume computed from lattice parameters
	- lattice_anisotropy: Ratio of max to min lattice parameter
	- angle_deviation_from_ortho: Average deviation from 90 degrees
	- is_cubic_like: 1 if nearly cubic, 0 otherwise
	- density: Mass density (g/cm^3)
	- volume_per_atom: Å^3 per atom in the unit cell
	- n_li_sites: Number of Li atoms per unit cell (from composition)
	
	Args:
		df: DataFrame with structural columns from CSV
	
	Returns:
		DataFrame with added structural feature columns
	"""
	df = df.copy()
	
	# Rename columns to consistent names
	column_mapping = {
		'a': 'lattice_a',
		'b': 'lattice_b', 
		'c': 'lattice_c',
		'alpha': 'lattice_alpha',
		'beta': 'lattice_beta',
		'gamma': 'lattice_gamma',
		'Space group #': 'spacegroup_number',
		'Z': 'formula_units_z'
	}
	
	for old_col, new_col in column_mapping.items():
		if old_col in df.columns:
			df[new_col] = pd.to_numeric(df[old_col], errors='coerce')

	# Compute derived features
	if all(col in df.columns for col in ['lattice_a', 'lattice_b', 'lattice_c',
										 'lattice_alpha', 'lattice_beta', 'lattice_gamma']):
		# Convert angles to radians
		alpha = np.deg2rad(df['lattice_alpha'])
		beta = np.deg2rad(df['lattice_beta'])
		gamma = np.deg2rad(df['lattice_gamma'])

		# Volume formula for a triclinic cell
		volume = (
			df['lattice_a'] * df['lattice_b'] * df['lattice_c'] *
			np.sqrt(
				1
				+ 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
				- np.cos(alpha)**2
				- np.cos(beta)**2
				- np.cos(gamma)**2
			)
		)
		df['lattice_volume'] = volume
	
	# Lattice anisotropy (max/min ratio)
	if all(col in df.columns for col in ['lattice_a', 'lattice_b', 'lattice_c']):
		lattice_params = df[['lattice_a', 'lattice_b', 'lattice_c']]
		df['lattice_anisotropy'] = lattice_params.max(axis=1) / lattice_params.min(axis=1).replace(0, np.nan)
	
	# Angle deviation from orthogonal (90 degrees)
	if all(col in df.columns for col in ['lattice_alpha', 'lattice_beta', 'lattice_gamma']):
		angle_devs = df[['lattice_alpha', 'lattice_beta', 'lattice_gamma']].sub(90).abs()
		df['angle_deviation_from_ortho'] = angle_devs.mean(axis=1)
	
	# Is cubic-like (all params within 5% and angles near 90)
	if 'lattice_anisotropy' in df.columns and 'angle_deviation_from_ortho' in df.columns:
		df['is_cubic_like'] = ((df['lattice_anisotropy'] < 1.05) & 
							   (df['angle_deviation_from_ortho'] < 2)).astype(int)
	
	# Composition-based counts (Li per unit cell, total atoms, density)
	try:
		from pymatgen.core.composition import Composition

		def _composition_stats(comp_str: str):
			try:
				comp = Composition(comp_str)
				li_count = comp.get("Li", 0)
				total_atoms = comp.num_atoms
				formula_mass = comp.weight  # g/mol
				return li_count, total_atoms, formula_mass
			except Exception:
				return np.nan, np.nan, np.nan

		if 'Reduced Composition' in df.columns:
			stats = df['Reduced Composition'].astype(str).apply(_composition_stats)
			df[['li_per_formula', 'atoms_per_formula', 'formula_mass']] = pd.DataFrame(stats.tolist(), index=df.index)

			# Unit cell counts
			if 'formula_units_z' in df.columns:
				df['li_per_unit_cell'] = df['li_per_formula'] * df['formula_units_z']
				df['atoms_per_unit_cell'] = df['atoms_per_formula'] * df['formula_units_z']

			# Density (g/cm^3)
			if all(col in df.columns for col in ['formula_mass', 'formula_units_z', 'lattice_volume']):
				unit_cell_mass_g = (df['formula_mass'] * df['formula_units_z']) / AVOGADRO_NUMBER
				volume_cm3 = df['lattice_volume'] * (ANGSTROM_TO_CM ** 3)
				df['density'] = unit_cell_mass_g / volume_cm3.replace(0, np.nan)

			# Volume per atom (Å^3)
			if all(col in df.columns for col in ['lattice_volume', 'atoms_per_unit_cell']):
				df['volume_per_atom'] = df['lattice_volume'] / df['atoms_per_unit_cell'].replace(0, np.nan)

			# Number of Li sites (Li atoms per unit cell)
			if 'li_per_unit_cell' in df.columns:
				df['n_li_sites'] = df['li_per_unit_cell']

	except ImportError:
		import warnings
		warnings.warn("pymatgen not available. Skipping composition-derived structural features.")
	except Exception as e:
		import warnings
		warnings.warn(f"Could not compute composition-derived features: {e}")

	return df


def stage1_spacegroup_onehot(df: pd.DataFrame, spacegroup_col: str = "spacegroup_number") -> pd.DataFrame:
	"""
	Creates a one-hot encoding vector for space groups (1-230).
	
	Args:
		df: Input DataFrame with a space group number column.
		spacegroup_col: Name of the space group column.
	
	Returns:
		DataFrame with 230 new binary columns (sg_1, sg_2, ..., sg_230).
	"""
	df = df.copy()
	
	if spacegroup_col not in df.columns:
		import warnings
		warnings.warn(f"Column '{spacegroup_col}' not found. Skipping space group encoding.")
		return df
	
	# Create one-hot encoding for space groups 1-230
	sg_data = {f"sg_{sg}": (df[spacegroup_col] == sg).astype(int) for sg in range(1, 231)}
	sg_df = pd.DataFrame(sg_data, index=df.index)
	df = pd.concat([df, sg_df], axis=1)
	
	return df


# =================================================================================================
# Legacy CIF-based Functions (kept for reference, not used in main pipeline)
# =================================================================================================

def _parse_cif_raw(cif_path: str) -> dict:
	"""
	Parses a CIF file directly from text to extract basic structural parameters.
	This is a fallback when pymatgen parsing fails.
	
	Args:
		cif_path (str): Path to the CIF file.
	
	Returns:
		dict: Dictionary with extracted values, or empty dict if parsing fails.
	"""
	import re
	result = {}
	try:
		with open(cif_path, 'r', encoding='utf-8', errors='ignore') as f:
			cif_content = f.read()
		
		# Extract lattice parameters
		patterns = {
			'lattice_a': r'_cell_length_a\s+([\d.]+)',
			'lattice_b': r'_cell_length_b\s+([\d.]+)',
			'lattice_c': r'_cell_length_c\s+([\d.]+)',
			'lattice_alpha': r'_cell_angle_alpha\s+([\d.]+)',
			'lattice_beta': r'_cell_angle_beta\s+([\d.]+)',
			'lattice_gamma': r'_cell_angle_gamma\s+([\d.]+)',
			'spacegroup_number': r'_symmetry_Int_Tables_number\s+(\d+)',
			'cell_volume': r'_cell_volume\s+([\d.]+)',
			'formula_units_z': r'_cell_formula_units_Z\s+(\d+)',
		}
		
		for key, pattern in patterns.items():
			match = re.search(pattern, cif_content)
			if match:
				result[key] = float(match.group(1))
		
		# Count symmetry operations
		sym_ops = re.findall(r"^\s*\d+\s+'[^']+'\s*$", cif_content, re.MULTILINE)
		if sym_ops:
			result['n_symmetry_ops'] = len(sym_ops)
		
		# Count Li sites from atom_site entries
		li_sites = re.findall(r'^\s*Li\s+\w+\s+(\d+)\s+', cif_content, re.MULTILINE)
		if li_sites:
			result['li_multiplicities'] = [int(m) for m in li_sites]
			result['n_li_wyckoff_sites'] = len(li_sites)
		
		# Count total atoms from atom_site entries (each line with coordinates)
		atom_lines = re.findall(r'^\s*\w+\s+\w+\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+', cif_content, re.MULTILINE)
		if atom_lines:
			result['n_atom_types'] = len(atom_lines)
		
	except Exception:
		pass
	
	return result


def _parse_cif_structure(cif_path: str, suppress_warnings: bool = True):
	"""
	Parses a CIF file and returns a pymatgen Structure object.
	Uses multiple fallback strategies for robust parsing.
	
	Args:
		cif_path (str): Path to the CIF file.
		suppress_warnings (bool): Whether to suppress pymatgen warnings.
	
	Returns:
		Structure or None: The pymatgen Structure object, or None if parsing fails.
	"""
	import warnings
	
	try:
		from pymatgen.core import Structure
		from pymatgen.io.cif import CifParser
		
		# Try multiple parsing strategies
		structure = None
		
		# Strategy 1: Direct from_file (fastest)
		try:
			if suppress_warnings:
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					structure = Structure.from_file(cif_path)
			else:
				structure = Structure.from_file(cif_path)
			return structure
		except Exception:
			pass
		
		# Strategy 2: CifParser with occupancy tolerance
		try:
			if suppress_warnings:
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					parser = CifParser(cif_path, occupancy_tolerance=1.0)
					structures = parser.parse_structures(primitive=False)
					if structures:
						return structures[0]
			else:
				parser = CifParser(cif_path, occupancy_tolerance=1.0)
				structures = parser.parse_structures(primitive=False)
				if structures:
					return structures[0]
		except Exception:
			pass
		
		# Strategy 3: CifParser with site_tolerance
		try:
			if suppress_warnings:
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					parser = CifParser(cif_path, occupancy_tolerance=1.0, site_tolerance=0.01)
					structures = parser.parse_structures(primitive=False)
					if structures:
						return structures[0]
		except Exception:
			pass
		
		return None
		
	except Exception:
		return None


def _extract_structural_features_from_cif(cif_path: str, extended: bool = True) -> dict:
	"""
	Extracts comprehensive structural features from a single CIF file.
	
	Features extracted (basic):
	- density: Mass density of the structure (g/cm³)
	- volume_per_atom: Unit cell volume divided by total atoms (Å³/atom)
	- spacegroup_number: International Tables space group number (1-230)
	- n_li_sites: Number of distinct Li positions in the structure
	- n_total_atoms: Total number of atoms in the unit cell
	
	Features extracted (extended - geometric descriptors):
	- a, b, c: Lattice parameters (Å)
	- alpha, beta, gamma: Lattice angles (degrees)
	- lattice_anisotropy_bc_a: ratio of (b+c)/2 to a, measures elongation
	- lattice_anisotropy_max_min: ratio of max to min lattice parameter
	- angle_deviation_from_ortho: average deviation of angles from 90°
	- is_cubic_like: 1 if nearly cubic (all params within 5%), 0 otherwise
	- n_symmetry_ops: number of symmetry operations
	- li_fraction: fraction of atoms that are Li
	- li_concentration: Li atoms per Å³
	- framework_density: non-Li atoms per Å³
	- li_li_min_dist: minimum Li-Li distance (Å) - critical for hopping
	- li_li_avg_dist: average Li-Li nearest neighbor distance (Å)
	- li_anion_min_dist: minimum Li-anion distance (Å) - bottleneck
	- li_coordination_avg: average coordination number of Li sites
	- li_site_avg_multiplicity: average Wyckoff multiplicity of Li sites
	
	Args:
		cif_path (str): Path to the CIF file.
		extended (bool): Whether to extract extended geometric features.
	
	Returns:
		dict: Dictionary containing the extracted structural features.
	"""
	import re
	
	# Initialize all features with NaN
	result = {
		# Basic features
		"density": np.nan,
		"volume_per_atom": np.nan,
		"spacegroup_number": np.nan,
		"n_li_sites": np.nan,
		"n_total_atoms": np.nan,
	}
	
	if extended:
		result.update({
			# Lattice parameters
			"lattice_a": np.nan,
			"lattice_b": np.nan,
			"lattice_c": np.nan,
			"lattice_alpha": np.nan,
			"lattice_beta": np.nan,
			"lattice_gamma": np.nan,
			# Derived lattice features
			"lattice_anisotropy_bc_a": np.nan,
			"lattice_anisotropy_max_min": np.nan,
			"angle_deviation_from_ortho": np.nan,
			"is_cubic_like": np.nan,
			# Symmetry
			"n_symmetry_ops": np.nan,
			# Li-specific geometric features
			"li_fraction": np.nan,
			"li_concentration": np.nan,
			"framework_density": np.nan,
			"li_li_min_dist": np.nan,
			"li_li_avg_dist": np.nan,
			"li_anion_min_dist": np.nan,
			"li_coordination_avg": np.nan,
			"li_site_avg_multiplicity": np.nan,
		})
	
	# First, try to get raw CIF data as fallback
	raw_cif_data = _parse_cif_raw(cif_path)
	
	# Try to parse structure with pymatgen
	structure = _parse_cif_structure(cif_path, suppress_warnings=True)
	
	# If pymatgen fails completely, use raw CIF data for basic features
	if structure is None:
		if raw_cif_data:
			# Populate from raw CIF data
			if 'spacegroup_number' in raw_cif_data:
				result["spacegroup_number"] = raw_cif_data['spacegroup_number']
			
			if extended:
				for key in ['lattice_a', 'lattice_b', 'lattice_c', 
							'lattice_alpha', 'lattice_beta', 'lattice_gamma']:
					if key in raw_cif_data:
						result[key] = raw_cif_data[key]
				
				if 'n_symmetry_ops' in raw_cif_data:
					result["n_symmetry_ops"] = raw_cif_data['n_symmetry_ops']
				
				if 'li_multiplicities' in raw_cif_data:
					result["li_site_avg_multiplicity"] = np.mean(raw_cif_data['li_multiplicities'])
				
				# Calculate derived features from raw data
				if all(k in raw_cif_data for k in ['lattice_a', 'lattice_b', 'lattice_c']):
					a, b, c = raw_cif_data['lattice_a'], raw_cif_data['lattice_b'], raw_cif_data['lattice_c']
					params = [a, b, c]
					if min(params) > 0:
						result["lattice_anisotropy_bc_a"] = (b + c) / (2 * a)
						result["lattice_anisotropy_max_min"] = max(params) / min(params)
				
				if all(k in raw_cif_data for k in ['lattice_alpha', 'lattice_beta', 'lattice_gamma']):
					alpha = raw_cif_data['lattice_alpha']
					beta = raw_cif_data['lattice_beta']
					gamma = raw_cif_data['lattice_gamma']
					angle_devs = [abs(alpha - 90), abs(beta - 90), abs(gamma - 90)]
					result["angle_deviation_from_ortho"] = np.mean(angle_devs)
					
					if 'lattice_anisotropy_max_min' in result and not np.isnan(result.get("lattice_anisotropy_max_min", np.nan)):
						param_ratio = result["lattice_anisotropy_max_min"]
						max_angle_dev = max(angle_devs)
						result["is_cubic_like"] = 1 if (param_ratio < 1.05 and max_angle_dev < 2) else 0
		
		return result
	
	try:
		# =========================================================================
		# Basic features from pymatgen Structure
		# =========================================================================
		# Density (g/cm³)
		result["density"] = structure.density
		
		# Volume per atom (Å³/atom)
		n_atoms = len(structure)
		result["n_total_atoms"] = n_atoms
		if n_atoms > 0:
			result["volume_per_atom"] = structure.volume / n_atoms
		
		# Count Li sites
		li_indices = [i for i, site in enumerate(structure) if site.specie.symbol == "Li"]
		n_li = len(li_indices)
		result["n_li_sites"] = n_li
		
		# Space group number - use raw CIF data first (more reliable)
		if 'spacegroup_number' in raw_cif_data:
			result["spacegroup_number"] = raw_cif_data['spacegroup_number']
		
		# Symmetry operations from raw CIF
		n_sym_ops_from_cif = raw_cif_data.get('n_symmetry_ops', None)
		
		# Try SpacegroupAnalyzer as backup
		from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
		try:
			sga = SpacegroupAnalyzer(structure, symprec=0.1)
			if np.isnan(result.get("spacegroup_number", np.nan)):
				result["spacegroup_number"] = sga.get_space_group_number()
			if extended:
				result["n_symmetry_ops"] = len(sga.get_symmetry_operations())
		except Exception:
			if extended and n_sym_ops_from_cif:
				result["n_symmetry_ops"] = n_sym_ops_from_cif
		
		if not extended:
			return result
		
		# =========================================================================
		# Extended geometric features - extract each independently for robustness
		# =========================================================================
		try:
			lattice = structure.lattice
			
			# Lattice parameters (a, b, c in Å)
			a, b, c = lattice.a, lattice.b, lattice.c
			result["lattice_a"] = a
			result["lattice_b"] = b
			result["lattice_c"] = c
			
			# Lattice angles (α, β, γ in degrees)
			alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
			result["lattice_alpha"] = alpha
			result["lattice_beta"] = beta
			result["lattice_gamma"] = gamma
			
			# Lattice anisotropy: how elongated/distorted is the cell?
			params = [a, b, c]
			if min(params) > 0:
				result["lattice_anisotropy_bc_a"] = (b + c) / (2 * a)
				result["lattice_anisotropy_max_min"] = max(params) / min(params)
			
			# Angle deviation from orthogonal (90°)
			angle_deviations = [abs(alpha - 90), abs(beta - 90), abs(gamma - 90)]
			result["angle_deviation_from_ortho"] = np.mean(angle_deviations)
			
			# Is the cell cubic-like? (all params within 5% and angles near 90°)
			param_ratio = max(params) / min(params) if min(params) > 0 else 999
			max_angle_dev = max(angle_deviations)
			result["is_cubic_like"] = 1 if (param_ratio < 1.05 and max_angle_dev < 2) else 0
		except Exception:
			pass
		
		# Li fraction and concentration
		try:
			if n_atoms > 0:
				result["li_fraction"] = n_li / n_atoms
			
			volume = structure.volume
			if volume > 0:
				result["li_concentration"] = n_li / volume  # Li per Å³
				result["framework_density"] = (n_atoms - n_li) / volume  # non-Li per Å³
		except Exception:
			pass
		
		# =========================================================================
		# Li-Li and Li-anion distance analysis
		# =========================================================================
		try:
			if n_li >= 2:
				# Get Li-Li distances
				li_li_distances = []
				for i, idx1 in enumerate(li_indices):
					for idx2 in li_indices[i+1:]:
						try:
							dist = structure.get_distance(idx1, idx2)
							li_li_distances.append(dist)
						except Exception:
							pass
				
				if li_li_distances:
					result["li_li_min_dist"] = min(li_li_distances)
					# Average of nearest neighbor distances (within 5 Å)
					nn_dists = [d for d in li_li_distances if d < 5.0]
					if nn_dists:
						result["li_li_avg_dist"] = np.mean(nn_dists)
		except Exception:
			pass
		
		# Li-anion minimum distance (bottleneck for migration)
		try:
			anion_symbols = {"O", "S", "F", "Cl", "Br", "I", "N", "Se", "Te"}
			anion_indices = [i for i, site in enumerate(structure) if site.specie.symbol in anion_symbols]
			
			if n_li > 0 and len(anion_indices) > 0:
				li_anion_dists = []
				for li_idx in li_indices[:min(20, n_li)]:  # Limit for speed
					for anion_idx in anion_indices[:min(50, len(anion_indices))]:
						try:
							dist = structure.get_distance(li_idx, anion_idx)
							li_anion_dists.append(dist)
						except Exception:
							pass
				if li_anion_dists:
					result["li_anion_min_dist"] = min(li_anion_dists)
		except Exception:
			pass
		
		# =========================================================================
		# Li coordination analysis
		# =========================================================================
		try:
			if n_li > 0:
				from pymatgen.analysis.local_env import VoronoiNN
				vnn = VoronoiNN(cutoff=5.0, allow_pathological=True)
				coordination_numbers = []
				
				for li_idx in li_indices[:min(10, n_li)]:  # Sample up to 10 Li sites for speed
					try:
						cn = vnn.get_cn(structure, li_idx)
						coordination_numbers.append(cn)
					except Exception:
						pass
				
				if coordination_numbers:
					result["li_coordination_avg"] = np.mean(coordination_numbers)
		except Exception:
			pass
		
		# =========================================================================
		# Li site multiplicity from CIF
		# =========================================================================
		try:
			with open(cif_path, 'r') as f:
				cif_content = f.read()
			
			# Parse atom site data to get Li multiplicities
			# Look for patterns like: Li  Li0  4  ... (where 4 is multiplicity)
			li_mult_pattern = r'Li\s+\w+\s+(\d+)\s+'
			li_multiplicities = [int(m) for m in re.findall(li_mult_pattern, cif_content)]
			
			if li_multiplicities:
				result["li_site_avg_multiplicity"] = np.mean(li_multiplicities)
		except Exception:
			pass
		
	except Exception:
		pass
	
	return result


def stage1_structural_features(
	df: pd.DataFrame, 
	id_col: str = "ID", 
	cif_dir: str = None,
	extended: bool = True,
	verbose: bool = True
) -> Tuple[pd.DataFrame, dict]:
	"""
	Extracts structural features from CIF files and adds them to the DataFrame.
	
	This function maps material IDs to their corresponding CIF files and extracts
	key structural descriptors that are relevant for predicting ionic conductivity.
	
	Basic features:
	- density: Crystal density (g/cm³)
	- volume_per_atom: Average volume available per atom (Å³/atom)
	- spacegroup_number: Used for one-hot encoding of crystal symmetry
	- n_li_sites: Number of lithium sites - directly relevant to Li+ conduction
	- n_total_atoms: Total atoms in unit cell
	
	Extended features (when extended=True):
	- Lattice parameters: a, b, c (Å) and angles alpha, beta, gamma (°)
	- Lattice anisotropy ratios: measure of cell elongation/distortion
	- Angle deviations: how far from orthogonal (90°)
	- is_cubic_like: binary flag for nearly cubic cells
	- n_symmetry_ops: number of symmetry operations
	- Li-specific: li_fraction, li_concentration, framework_density
	- Li distances: li_li_min_dist, li_li_avg_dist, li_anion_min_dist
	- Li coordination: li_coordination_avg, li_site_avg_multiplicity
	
	Args:
		df (pd.DataFrame): Input DataFrame with an ID column that matches CIF filenames.
		id_col (str, optional): Name of the column containing material IDs. Defaults to "ID".
		cif_dir (str, optional): Directory containing CIF files. Must be provided.
		extended (bool, optional): Whether to extract extended features. Defaults to True.
		verbose (bool, optional): Whether to print progress. Defaults to True.
	
	Returns:
		Tuple[pd.DataFrame, dict]: DataFrame with structural features and parsing statistics.
	"""
	df = df.copy()
	
	# Parsing statistics
	stats = {
		'total': 0,
		'parsed_full': 0,
		'parsed_partial': 0,
		'failed': 0,
		'failed_ids': [],
		'missing_cif': 0,
		'missing_cif_ids': [],
	}
	
	if cif_dir is None:
		import warnings
		warnings.warn("No CIF directory provided. Skipping structural features.")
		return df, stats
	
	if id_col not in df.columns:
		import warnings
		warnings.warn(f"Column '{id_col}' not found. Skipping structural features.")
		return df, stats
	
	# Define all feature columns based on extended flag
	basic_cols = ["density", "volume_per_atom", "spacegroup_number", "n_li_sites", "n_total_atoms"]
	extended_cols = [
		"lattice_a", "lattice_b", "lattice_c",
		"lattice_alpha", "lattice_beta", "lattice_gamma",
		"lattice_anisotropy_bc_a", "lattice_anisotropy_max_min",
		"angle_deviation_from_ortho", "is_cubic_like",
		"n_symmetry_ops",
		"li_fraction", "li_concentration", "framework_density",
		"li_li_min_dist", "li_li_avg_dist", "li_anion_min_dist",
		"li_coordination_avg", "li_site_avg_multiplicity",
	]
	
	feature_cols = basic_cols + (extended_cols if extended else [])
	
	# Initialize feature columns with NaN
	for col in feature_cols:
		df[col] = np.nan
	
	stats['total'] = len(df)
	
	# Process each material
	for idx, row in df.iterrows():
		material_id = row[id_col]
		cif_path = os.path.join(cif_dir, f"{material_id}.cif")
		
		if not os.path.exists(cif_path):
			stats['missing_cif'] += 1
			stats['missing_cif_ids'].append(material_id)
			continue
		
		features = _extract_structural_features_from_cif(cif_path, extended=extended)
		
		# Count non-NaN features extracted
		valid_features = sum(1 for v in features.values() if not (isinstance(v, float) and np.isnan(v)))
		
		if valid_features == 0:
			stats['failed'] += 1
			stats['failed_ids'].append(material_id)
		elif valid_features >= len(basic_cols):
			stats['parsed_full'] += 1
		else:
			stats['parsed_partial'] += 1
		
		for col, value in features.items():
			if col in df.columns:
				df.at[idx, col] = value
	
	return df, stats


def stage1_matminer_features(
	df: pd.DataFrame, 
	composition_col: str = "composition",
	id_col: str = "ID",
	cif_dir: str = None
) -> pd.DataFrame:
	"""
	Main entry point for Stage 1 structural feature engineering.
	
	This function coordinates the extraction of all Stage 1 features:
	1. Basic structural properties from CIF files (density, volume per atom, Li sites)
	2. Space group one-hot encoding (230 binary features for crystal symmetry)
	
	The structural features capture geometry-dependent information that cannot be
	derived from composition alone, such as:
	- Crystal packing efficiency (density, volume per atom)
	- Symmetry-specific ion transport pathways (space group)
	- Availability of Li+ conducting sites (n_li_sites)
	
	Args:
		df (pd.DataFrame): Input DataFrame.
		composition_col (str, optional): Composition column name. Defaults to "composition".
		id_col (str, optional): ID column for CIF file matching. Defaults to "ID".
		cif_dir (str, optional): Directory containing CIF files.
	
	Returns:
		pd.DataFrame: DataFrame augmented with all Stage 1 structural features.
	"""
	df = df.copy()
	
	# Extract basic structural features from CIF files
	df = stage1_structural_features(df, id_col=id_col, cif_dir=cif_dir)
	
	# Create one-hot encoding for space groups
	df = stage1_spacegroup_onehot(df, spacegroup_col="spacegroup_number")
	
	return df


# =================================================================================================
# Stage 2: Placeholder for Physics-Based Feature Engineering
# =================================================================================================

def stage2_smact_features(df: pd.DataFrame, composition_col: str = "composition") -> pd.DataFrame:
	"""
	Placeholder for Stage 2 features (physics-based descriptors).

	(Weeks 8-9) This function will be implemented to compute advanced physical descriptors
	such as bond-valence mismatch, Ewald energy, and Voronoi coordination numbers, which
	require detailed structural analysis.
	"""
	df = df.copy()
	# TODO: Implement Stage 2 features (charge balance, electronegativity spread, etc.) as per project plan.
	return df


# =================================================================================================
# Main Feature Building Function
# =================================================================================================

def build_feature_matrix(
	df: pd.DataFrame,
	composition_col: str = "composition",
	id_col: str = "ID",
	cif_dir: str = None,
	use_stage0: bool = True,
	use_stage1: bool = True,
	use_stage2: bool = True,
	embedding_schemes: List[str] = None,
) -> pd.DataFrame:
	"""
	Constructs the final feature matrix by applying the selected feature generation stages.

	This is the main entry point for the feature engineering pipeline. It allows the user
	to easily enable or disable different stages of feature generation.

	Stage 0: Composition-only features (elemental ratios, SMACT vectors, element embeddings)
	Stage 1: Structural features (density, volume per atom, space group encoding, Li sites)
	Stage 2: Physics-based features (placeholder for future implementation)

	It also includes a robust post-processing step to handle any missing values (NaNs)
	that may have been created during feature generation, ensuring the model receives a
	clean, complete feature matrix.

	Args:
		df (pd.DataFrame): The input DataFrame.
		composition_col (str, optional): The name of the composition column. Defaults to "composition".
		id_col (str, optional): The name of the ID column for CIF matching. Defaults to "ID".
		cif_dir (str, optional): Directory containing CIF files for Stage 1 features.
		use_stage0 (bool, optional): Whether to apply Stage 0 features. Defaults to True.
		use_stage1 (bool, optional): Whether to apply Stage 1 features. Defaults to True.
		use_stage2 (bool, optional): Whether to apply Stage 2 features. Defaults to True.
		embedding_schemes (List[str], optional): List of embedding schemes to use in Stage 0.
			Options are "mat2vec", "megnet16", "magpie". Defaults to ["magpie"] for best performance.

	Returns:
		pd.DataFrame: The final, feature-augmented DataFrame ready for model training.
	"""
	out = df.copy()
	
	# Default to Magpie embeddings only (best performing from baseline experiments)
	if embedding_schemes is None:
		embedding_schemes = ["magpie"]
	
	if use_stage0:
		# Apply baseline features: elemental ratios and SMACT stoichiometry
		out = stage0_elemental_ratios(out, composition_col=composition_col)
		out = stage0_smact_features(out, composition_col=composition_col)
		# Apply only the specified embedding schemes
		out = stage0_element_embeddings(out, composition_col=composition_col, embedding_names=embedding_schemes)
	
	if use_stage1:
		out = stage1_matminer_features(out, composition_col=composition_col, id_col=id_col, cif_dir=cif_dir)
	
	if use_stage2:
		out = stage2_smact_features(out, composition_col=composition_col)
	
	# --- Post-processing: Handle Missing Values ---
	# Replace any infinite values that might have been generated.
	out.replace([np.inf, -np.inf], np.nan, inplace=True)
	
	# Use forward-fill and then backward-fill to propagate the last valid observation.
	# This is a reasonable strategy for some types of data, but should be used with care.
	out = out.ffill()
	out = out.bfill()
	
	# As a final safeguard, fill any remaining NaNs with 0. This ensures no missing values
	# are passed to the model, which would cause an error.
	out = out.fillna(0)
	
	return out
