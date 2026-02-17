# =================================================================================================
# Li-ion Mobility Prediction - Stage 2: Partial Occupancy Solutions
# =================================================================================================
# This script implements four experiments to handle partial occupancy in CIF files:
#
# 1. BASELINE: Current approach (physics features fail for partial occupancy)
# 2. FULL_OCC_ONLY: Only use physics features from full-occupancy structures
# 3. ORDERED_APPROX: Create ordered approximations of disordered structures
# 4. OCC_WEIGHTED: Weight physics features by site occupancy
#
# The goal is to maximize physics feature coverage while maintaining physical relevance.
# =================================================================================================

import os
import sys
import shutil
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def clear_pycache(root_path: Path):
    """Recursively removes all __pycache__ directories."""
    for pycache in root_path.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
        except Exception:
            pass


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
)
from src.data_processing import load_data, clean_data, add_target_log10_sigma, TARGET_COL
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
# Constants
# =================================================================================================

# Sentinel value for missing physics features - physically impossible value
# This resolves ambiguity: 0 could be a valid physics value, but -999 is clearly "missing"
# The gating indicator variables (has_bv_mismatch, etc.) tell the model when to ignore this
MISSING_PHYSICS_SENTINEL = -999.0


# =================================================================================================
# Partial Occupancy Handling Utilities
# =================================================================================================

def check_partial_occupancy(structure) -> Tuple[bool, float]:
    """
    Check if a structure has partial occupancy and return minimum occupancy.
    
    Returns:
        Tuple of (has_partial_occupancy, min_occupancy)
    """
    if structure is None:
        return False, 1.0
    
    min_occ = 1.0
    has_partial = False
    
    for site in structure:
        if hasattr(site, 'species') and hasattr(site.species, 'as_dict'):
            species_dict = site.species.as_dict()
            for el, occ in species_dict.items():
                if occ < 0.99:  # Allow small tolerance
                    has_partial = True
                    min_occ = min(min_occ, occ)
    
    return has_partial, min_occ


def create_ordered_approximation(structure, logger=None):
    """
    Create an ordered approximation of a disordered structure.
    
    Uses pymatgen's OrderDisorderedStructureTransformation to enumerate
    possible orderings and select the lowest-energy one.
    
    For Li-ion conductors, this is physically meaningful because:
    - We want to analyze a representative Li configuration
    - The ordered approximation captures average local environment
    
    Args:
        structure: pymatgen Structure (possibly disordered)
        logger: Optional logger
    
    Returns:
        Ordered Structure or None if transformation fails
    """
    if structure is None:
        return None
    
    has_partial, min_occ = check_partial_occupancy(structure)
    if not has_partial:
        return structure  # Already ordered
    
    try:
        from pymatgen.transformations.standard_transformations import (
            OrderDisorderedStructureTransformation
        )
        
        # For very low occupancy sites, we need to be careful
        # Only keep sites with reasonable occupancy for ordering
        if min_occ < 0.1:
            # Create a structure with only high-occupancy sites
            from pymatgen.core import Structure, Lattice
            
            new_species = []
            new_coords = []
            
            for site in structure:
                if hasattr(site.species, 'as_dict'):
                    species_dict = site.species.as_dict()
                    # Keep site if any species has occupancy > 0.3
                    max_occ = max(species_dict.values())
                    if max_occ >= 0.3:
                        # Use the majority species
                        majority_species = max(species_dict, key=species_dict.get)
                        new_species.append(majority_species)
                        new_coords.append(site.frac_coords)
                else:
                    new_species.append(site.species)
                    new_coords.append(site.frac_coords)
            
            if len(new_species) < 3:
                return None  # Too few atoms remaining
            
            ordered = Structure(structure.lattice, new_species, new_coords)
            return ordered
        
        # Use OrderDisorderedStructureTransformation for moderate disorder
        transformation = OrderDisorderedStructureTransformation(
            algo=2,  # Use enumlib algorithm
            symmetrized_structures=False,
            no_oxi_states=True,  # Don't require oxidation states
        )
        
        # This can be slow, so we limit the search
        try:
            ordered_structures = transformation.apply_transformation(
                structure, 
                return_ranked_list=1  # Only get best structure
            )
            if ordered_structures:
                return ordered_structures[0]['structure']
        except Exception:
            pass
        
        # Fallback: Simple majority species approach
        from pymatgen.core import Structure
        
        new_species = []
        new_coords = []
        
        for site in structure:
            if hasattr(site.species, 'as_dict'):
                species_dict = site.species.as_dict()
                majority_species = max(species_dict, key=species_dict.get)
                new_species.append(majority_species)
            else:
                new_species.append(site.species)
            new_coords.append(site.frac_coords)
        
        ordered = Structure(structure.lattice, new_species, new_coords)
        return ordered
        
    except Exception as e:
        if logger:
            logger.debug(f"Ordering transformation failed: {e}")
        return None


def get_occupancy_weights(structure) -> Dict[int, float]:
    """
    Get occupancy weights for each site in a structure.
    
    For Li-ion conductors, this allows us to weight contributions
    by how likely a site is to be occupied.
    
    Returns:
        Dict mapping site index to occupancy weight
    """
    if structure is None:
        return {}
    
    weights = {}
    for i, site in enumerate(structure):
        if hasattr(site.species, 'as_dict'):
            species_dict = site.species.as_dict()
            # For mixed sites, use the total occupancy
            weights[i] = sum(species_dict.values())
        else:
            weights[i] = 1.0
    
    return weights


# =================================================================================================
# Physics Feature Extraction with Partial Occupancy Handling
# =================================================================================================

def extract_physics_features_baseline(structure, verbose: bool = False) -> Dict:
    """
    BASELINE: Original physics feature extraction (fails for partial occupancy).
    """
    result = {
        "bv_mismatch_avg": np.nan,
        "bv_mismatch_std": np.nan,
        "ewald_energy_avg": np.nan,
        "ewald_energy_std": np.nan,
        "li_voronoi_cn_avg": np.nan,
        # Separate indicators for each physics feature type (critical for gating!)
        "has_bv_mismatch": 0,
        "has_ewald_energy": 0,
        "has_voronoi_cn": 0,
        "has_physics_features": 0,  # Combined indicator (any physics available)
    }
    
    if structure is None:
        return result
    
    # Find Li site indices
    li_indices = [i for i, site in enumerate(structure) if "Li" in str(site.species)]
    if len(li_indices) == 0:
        return result
    
    features_calculated = 0
    
    # Bond-Valence Mismatch
    try:
        from pymatgen.analysis.local_env import CrystalNN
        
        cnn = CrystalNN()
        bv_params = {
            "O": 1.466, "S": 2.052, "F": 1.36, "Cl": 1.91,
            "Br": 2.07, "I": 2.34, "N": 1.61, "Se": 2.22,
        }
        B = 0.37
        
        li_bvs_mismatches = []
        
        for li_idx in li_indices[:min(10, len(li_indices))]:
            try:
                neighbors = cnn.get_nn_info(structure, li_idx)
                if neighbors:
                    bv_sum = 0.0
                    for neighbor in neighbors:
                        neighbor_site = neighbor["site"]
                        if hasattr(neighbor_site.species, 'elements'):
                            neighbor_element = str(neighbor_site.species.elements[0])
                        else:
                            neighbor_element = str(neighbor_site.species).split(":")[0]
                        
                        distance = neighbor["site"].distance(structure[li_idx])
                        
                        r0 = None
                        for anion, r0_val in bv_params.items():
                            if anion in neighbor_element:
                                r0 = r0_val
                                break
                        
                        if r0 is not None and distance > 0:
                            bv = np.exp((r0 - distance) / B)
                            bv_sum += bv
                    
                    if bv_sum > 0:
                        mismatch = abs(1.0 - bv_sum)
                        li_bvs_mismatches.append(mismatch)
            except Exception:
                pass
        
        if li_bvs_mismatches:
            result["bv_mismatch_avg"] = float(np.mean(li_bvs_mismatches))
            result["bv_mismatch_std"] = float(np.std(li_bvs_mismatches)) if len(li_bvs_mismatches) > 1 else 0.0
            result["has_bv_mismatch"] = 1  # Set individual indicator
            features_calculated += 1
    except Exception:
        pass
    
    # Ewald Site Energy
    try:
        from pymatgen.analysis.ewald import EwaldSummation
        from pymatgen.analysis.bond_valence import BVAnalyzer
        
        oxi_struct = None
        try:
            bva = BVAnalyzer()
            oxi_struct = bva.get_oxi_state_decorated_structure(structure)
        except Exception:
            try:
                oxi_struct = structure.copy()
                oxi_struct.add_oxidation_state_by_guess()
            except Exception:
                pass
        
        if oxi_struct is not None:
            ewald = EwaldSummation(oxi_struct)
            li_ewald_energies = []
            
            for li_idx in li_indices:
                if li_idx < len(oxi_struct):
                    try:
                        site_energy = ewald.get_site_energy(li_idx)
                        li_ewald_energies.append(site_energy)
                    except Exception:
                        pass
            
            if li_ewald_energies:
                result["ewald_energy_avg"] = float(np.mean(li_ewald_energies))
                result["ewald_energy_std"] = float(np.std(li_ewald_energies)) if len(li_ewald_energies) > 1 else 0.0
                result["has_ewald_energy"] = 1  # Set individual indicator
                features_calculated += 1
    except Exception:
        pass
    
    # Voronoi Coordination Number
    try:
        from pymatgen.analysis.local_env import VoronoiNN
        
        vnn = VoronoiNN(cutoff=5.0, allow_pathological=True)
        li_cns = []
        
        for li_idx in li_indices[:min(10, len(li_indices))]:
            try:
                cn = vnn.get_cn(structure, li_idx)
                li_cns.append(cn)
            except Exception:
                pass
        
        if li_cns:
            result["li_voronoi_cn_avg"] = float(np.mean(li_cns))
            result["has_voronoi_cn"] = 1  # Set individual indicator
            features_calculated += 1
    except Exception:
        pass
    
    if features_calculated > 0:
        result["has_physics_features"] = 1
    
    return result


def extract_physics_features_full_occ_only(structure, verbose: bool = False) -> Dict:
    """
    OPTION 1: Only extract physics features for full-occupancy structures.
    
    For partial occupancy structures, return NaN (to be zero-filled later).
    This ensures physics features are only from physically meaningful structures.
    """
    result = {
        "bv_mismatch_avg": np.nan,
        "bv_mismatch_std": np.nan,
        "ewald_energy_avg": np.nan,
        "ewald_energy_std": np.nan,
        "li_voronoi_cn_avg": np.nan,
        # Separate indicators for each physics feature type (critical for gating!)
        "has_bv_mismatch": 0,
        "has_ewald_energy": 0,
        "has_voronoi_cn": 0,
        "has_physics_features": 0,
        "is_full_occupancy": 0,
    }
    
    if structure is None:
        return result
    
    # Check for partial occupancy
    has_partial, min_occ = check_partial_occupancy(structure)
    
    if has_partial:
        # Skip physics features for disordered structures
        result["is_full_occupancy"] = 0
        return result
    
    result["is_full_occupancy"] = 1
    
    # Use baseline extraction for full-occupancy structures
    baseline_result = extract_physics_features_baseline(structure, verbose)
    result.update(baseline_result)
    
    return result


def extract_physics_features_ordered_approx(structure, logger=None, verbose: bool = False) -> Dict:
    """
    OPTION 2: Create ordered approximation before extracting physics features.
    
    For partial occupancy structures:
    1. Create an ordered approximation using majority species
    2. Extract physics features from the ordered structure
    
    This is physically meaningful because it captures the average local environment.
    """
    result = {
        "bv_mismatch_avg": np.nan,
        "bv_mismatch_std": np.nan,
        "ewald_energy_avg": np.nan,
        "ewald_energy_std": np.nan,
        "li_voronoi_cn_avg": np.nan,
        # Separate indicators for each physics feature type (critical for gating!)
        "has_bv_mismatch": 0,
        "has_ewald_energy": 0,
        "has_voronoi_cn": 0,
        "has_physics_features": 0,
        "used_ordered_approx": 0,
    }
    
    if structure is None:
        return result
    
    # Check for partial occupancy
    has_partial, min_occ = check_partial_occupancy(structure)
    
    if has_partial:
        # Create ordered approximation
        ordered_struct = create_ordered_approximation(structure, logger)
        if ordered_struct is None:
            return result
        result["used_ordered_approx"] = 1
        structure = ordered_struct
    
    # Extract physics features from (potentially ordered) structure
    baseline_result = extract_physics_features_baseline(structure, verbose)
    result.update(baseline_result)
    
    return result


def extract_physics_features_occ_weighted(structure, verbose: bool = False) -> Dict:
    """
    OPTION 3: Weight physics feature contributions by site occupancy.
    
    For Li-ion conductors, this is physically meaningful because:
    - Higher occupancy sites contribute more to average properties
    - Low occupancy sites (which may be artifacts) contribute less
    
    Implementation:
    - For BV mismatch: Weight each Li site's contribution by its occupancy
    - For Ewald: Weight site energies by occupancy
    - For Voronoi: Weight coordination numbers by occupancy
    """
    result = {
        "bv_mismatch_avg": np.nan,
        "bv_mismatch_std": np.nan,
        "ewald_energy_avg": np.nan,
        "ewald_energy_std": np.nan,
        "li_voronoi_cn_avg": np.nan,
        # Separate indicators for each physics feature type (critical for gating!)
        "has_bv_mismatch": 0,
        "has_ewald_energy": 0,
        "has_voronoi_cn": 0,
        "has_physics_features": 0,
        "avg_li_occupancy": np.nan,
    }
    
    if structure is None:
        return result
    
    # Get occupancy weights
    occ_weights = get_occupancy_weights(structure)
    
    # Find Li site indices and their occupancies
    li_data = []
    for i, site in enumerate(structure):
        if "Li" in str(site.species):
            occ = occ_weights.get(i, 1.0)
            if occ > 0.01:  # Only consider sites with >1% occupancy
                li_data.append((i, occ))
    
    if len(li_data) == 0:
        return result
    
    li_indices = [x[0] for x in li_data]
    li_occupancies = [x[1] for x in li_data]
    result["avg_li_occupancy"] = float(np.mean(li_occupancies))
    
    features_calculated = 0
    
    # Bond-Valence Mismatch (occupancy-weighted)
    try:
        from pymatgen.analysis.local_env import CrystalNN
        
        cnn = CrystalNN()
        bv_params = {
            "O": 1.466, "S": 2.052, "F": 1.36, "Cl": 1.91,
            "Br": 2.07, "I": 2.34, "N": 1.61, "Se": 2.22,
        }
        B = 0.37
        
        weighted_mismatches = []
        weights = []
        
        for li_idx, li_occ in zip(li_indices[:min(15, len(li_indices))], 
                                   li_occupancies[:min(15, len(li_occupancies))]):
            try:
                neighbors = cnn.get_nn_info(structure, li_idx)
                if neighbors:
                    bv_sum = 0.0
                    valid_neighbors = 0
                    
                    for neighbor in neighbors:
                        neighbor_site = neighbor["site"]
                        if hasattr(neighbor_site.species, 'elements'):
                            neighbor_element = str(neighbor_site.species.elements[0])
                        else:
                            neighbor_element = str(neighbor_site.species).split(":")[0]
                        
                        distance = neighbor["site"].distance(structure[li_idx])
                        
                        r0 = None
                        for anion, r0_val in bv_params.items():
                            if anion in neighbor_element:
                                r0 = r0_val
                                break
                        
                        if r0 is not None and distance > 0.5:  # Minimum reasonable distance
                            bv = np.exp((r0 - distance) / B)
                            # Weight neighbor contribution by neighbor occupancy
                            neighbor_idx = None
                            for k, s in enumerate(structure):
                                if np.allclose(s.frac_coords, neighbor_site.frac_coords, atol=0.01):
                                    neighbor_idx = k
                                    break
                            neighbor_occ = occ_weights.get(neighbor_idx, 1.0) if neighbor_idx else 1.0
                            bv_sum += bv * neighbor_occ
                            valid_neighbors += 1
                    
                    if bv_sum > 0 and valid_neighbors > 0:
                        mismatch = abs(1.0 - bv_sum)
                        weighted_mismatches.append(mismatch)
                        weights.append(li_occ)
            except Exception:
                pass
        
        if weighted_mismatches and sum(weights) > 0:
            # Weighted average
            result["bv_mismatch_avg"] = float(np.average(weighted_mismatches, weights=weights))
            if len(weighted_mismatches) > 1:
                # Weighted standard deviation
                avg = result["bv_mismatch_avg"]
                variance = np.average((np.array(weighted_mismatches) - avg)**2, weights=weights)
                result["bv_mismatch_std"] = float(np.sqrt(variance))
            else:
                result["bv_mismatch_std"] = 0.0
            result["has_bv_mismatch"] = 1  # Set individual indicator
            features_calculated += 1
    except Exception:
        pass
    
    # Ewald Site Energy (occupancy-weighted)
    try:
        from pymatgen.analysis.ewald import EwaldSummation
        from pymatgen.analysis.bond_valence import BVAnalyzer
        
        oxi_struct = None
        try:
            bva = BVAnalyzer()
            oxi_struct = bva.get_oxi_state_decorated_structure(structure)
        except Exception:
            try:
                oxi_struct = structure.copy()
                oxi_struct.add_oxidation_state_by_guess()
            except Exception:
                pass
        
        if oxi_struct is not None:
            ewald = EwaldSummation(oxi_struct)
            
            weighted_energies = []
            weights = []
            
            for li_idx, li_occ in zip(li_indices, li_occupancies):
                if li_idx < len(oxi_struct):
                    try:
                        site_energy = ewald.get_site_energy(li_idx)
                        weighted_energies.append(site_energy)
                        weights.append(li_occ)
                    except Exception:
                        pass
            
            if weighted_energies and sum(weights) > 0:
                result["ewald_energy_avg"] = float(np.average(weighted_energies, weights=weights))
                if len(weighted_energies) > 1:
                    avg = result["ewald_energy_avg"]
                    variance = np.average((np.array(weighted_energies) - avg)**2, weights=weights)
                    result["ewald_energy_std"] = float(np.sqrt(variance))
                else:
                    result["ewald_energy_std"] = 0.0
                result["has_ewald_energy"] = 1  # Set individual indicator
                features_calculated += 1
    except Exception:
        pass
    
    # Voronoi Coordination Number (occupancy-weighted)
    try:
        from pymatgen.analysis.local_env import VoronoiNN
        
        vnn = VoronoiNN(cutoff=5.0, allow_pathological=True)
        
        weighted_cns = []
        weights = []
        
        for li_idx, li_occ in zip(li_indices[:min(15, len(li_indices))],
                                   li_occupancies[:min(15, len(li_occupancies))]):
            try:
                cn = vnn.get_cn(structure, li_idx)
                weighted_cns.append(cn)
                weights.append(li_occ)
            except Exception:
                pass
        
        if weighted_cns and sum(weights) > 0:
            result["li_voronoi_cn_avg"] = float(np.average(weighted_cns, weights=weights))
            result["has_voronoi_cn"] = 1  # Set individual indicator
            features_calculated += 1
    except Exception:
        pass
    
    if features_calculated > 0:
        result["has_physics_features"] = 1
    
    return result


def extract_physics_features_for_method(
    df: pd.DataFrame,
    method: str,
    id_col: str = "ID",
    cif_dir: str = None,
    logger=None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extract physics features using the specified method.
    
    Args:
        df: Input DataFrame
        method: One of "baseline", "full_occ_only", "ordered_approx", "occ_weighted"
        id_col: ID column name
        cif_dir: Directory containing CIF files
        logger: Logger instance
    
    Returns:
        Tuple of (DataFrame with physics features, extraction statistics)
    """
    df = df.copy()
    
    # Define physics feature columns based on method
    # CRITICAL: Include separate indicators for each physics feature type for proper gating!
    base_physics_cols = [
        "bv_mismatch_avg", "bv_mismatch_std", 
        "ewald_energy_avg", "ewald_energy_std",
        "li_voronoi_cn_avg",
        # Individual gating indicators (allow model to learn when to trust each feature)
        "has_bv_mismatch", "has_ewald_energy", "has_voronoi_cn",
        "has_physics_features"  # Combined indicator
    ]
    
    if method == "baseline":
        physics_cols = base_physics_cols
        extract_fn = extract_physics_features_baseline
    elif method == "full_occ_only":
        physics_cols = base_physics_cols + ["is_full_occupancy"]
        extract_fn = extract_physics_features_full_occ_only
    elif method == "ordered_approx":
        physics_cols = base_physics_cols + ["used_ordered_approx"]
        extract_fn = extract_physics_features_ordered_approx
    elif method == "occ_weighted":
        physics_cols = base_physics_cols + ["avg_li_occupancy"]
        extract_fn = extract_physics_features_occ_weighted
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Initialize columns
    for col in physics_cols:
        df[col] = np.nan
    
    stats = {
        "total": len(df),
        "processed": 0,
        "with_bv": 0,
        "with_ewald": 0,
        "with_voronoi": 0,
        "with_any_physics": 0,
    }
    
    if cif_dir is None or not os.path.exists(cif_dir):
        return df, stats
    
    # Import pymatgen
    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifParser
    except ImportError:
        if logger:
            logger.warning("pymatgen not available")
        return df, stats
    
    # Process each material
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
        if method == "ordered_approx":
            features = extract_fn(structure, logger=logger)
        else:
            features = extract_fn(structure)
        
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
        if features.get("has_physics_features", 0) == 1:
            stats["with_any_physics"] += 1
    
    return df, stats


# =================================================================================================
# Main Execution Block
# =================================================================================================

def main():
    """
    Run four experiments comparing partial occupancy handling methods.
    """
    # -----------------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------------
    seed_everything(42)
    
    results_dir = os.path.join(PROJECT_ROOT, "results", "results_partial_occ")
    os.makedirs(results_dir, exist_ok=True)
    
    log_path = os.path.join(results_dir, "partial_occ_experiments.log")
    logger = setup_logger("partial_occ", log_file=log_path)
    logger.info("=" * 80)
    logger.info("Stage 2: Partial Occupancy Solutions Experiments")
    logger.info("=" * 80)
    
    # -----------------------------------------------------------------------------------------
    # Data Loading
    # -----------------------------------------------------------------------------------------
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    train_cif_dir = os.path.join(data_dir, "train_cifs")
    test_cif_dir = os.path.join(data_dir, "test_cifs")
    
    logger.info(f"Loading data from: {data_dir}")
    train_df_full, test_df_full = load_data(data_dir)
    
    # Clean and transform
    train_df = add_target_log10_sigma(clean_data(train_df_full), target_sigma_col="Ionic conductivity (S cm-1)")
    test_df = add_target_log10_sigma(clean_data(test_df_full), target_sigma_col="Ionic conductivity (S cm-1)")
    
    train_df.dropna(subset=[TARGET_COL], inplace=True)
    
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # -----------------------------------------------------------------------------------------
    # Stage 0 + Stage 1 Features (common to all experiments)
    # -----------------------------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("Generating Stage 0 + Stage 1 baseline features...")
    
    # Stage 0
    base_train = stage0_elemental_ratios(train_df.copy(), "Reduced Composition")
    base_train = stage0_smact_features(base_train, "Reduced Composition")
    base_train = stage0_element_embeddings(base_train, "Reduced Composition", embedding_names=["magpie"])
    
    base_test = stage0_elemental_ratios(test_df.copy(), "Reduced Composition")
    base_test = stage0_smact_features(base_test, "Reduced Composition")
    base_test = stage0_element_embeddings(base_test, "Reduced Composition", embedding_names=["magpie"])
    
    # Stage 1 CSV structural features
    base_train = stage1_csv_structural_features(base_train)
    base_test = stage1_csv_structural_features(base_test)
    
    # Stage 1 CIF structural features
    struct_train, train_stats = stage1_structural_features(
        base_train, id_col="ID", cif_dir=train_cif_dir, extended=True, verbose=False
    )
    struct_test, test_stats = stage1_structural_features(
        base_test, id_col="ID", cif_dir=test_cif_dir, extended=True, verbose=False
    )
    
    # Add CIF availability indicator
    train_cif_files = [f.replace(".cif", "") for f in os.listdir(train_cif_dir) if f.endswith(".cif")]
    test_cif_files = [f.replace(".cif", "") for f in os.listdir(test_cif_dir) if f.endswith(".cif")]
    
    struct_train["has_cif_struct"] = struct_train["ID"].isin(train_cif_files).astype(int)
    struct_test["has_cif_struct"] = struct_test["ID"].isin(test_cif_files).astype(int)
    
    # Zero-fill CIF-only structural features
    cif_only_cols = [
        "li_fraction", "li_concentration", "framework_density",
        "li_li_min_dist", "li_li_avg_dist", "li_anion_min_dist",
        "li_coordination_avg", "li_site_avg_multiplicity",
        "lattice_anisotropy_bc_a", "lattice_anisotropy_max_min"
    ]
    for col in cif_only_cols:
        if col in struct_train.columns:
            struct_train[col] = struct_train[col].fillna(0)
        if col in struct_test.columns:
            struct_test[col] = struct_test[col].fillna(0)
    
    # Space group one-hot
    full_train = stage1_spacegroup_onehot(struct_train.copy(), spacegroup_col="spacegroup_number")
    full_test = stage1_spacegroup_onehot(struct_test.copy(), spacegroup_col="spacegroup_number")
    
    # Define feature columns
    metadata_cols = [
        "Space group", "Space group #", "a", "b", "c", "alpha", "beta", "gamma", "Z",
        "IC (Total)", "IC (Bulk)", "ID", "Family", "DOI", "Checked", "Ref", "Cif ID",
        "Cif ref_1", "Cif ref_2", "note", "close match", "close match DOI", "ICSD ID",
        "Laskowski ID", "Liion ID", "True Composition", "Reduced Composition",
        "Ionic conductivity (S cm-1)",
    ]
    
    numeric_cols = full_train.select_dtypes(include=np.number).columns.tolist()
    stage0_feature_cols = [c for c in numeric_cols if c != TARGET_COL and c not in metadata_cols]
    
    geometry_cols = [
        "density", "volume_per_atom", "n_li_sites", "n_total_atoms",
        "lattice_a", "lattice_b", "lattice_c",
        "lattice_alpha", "lattice_beta", "lattice_gamma",
        "lattice_anisotropy_bc_a", "lattice_anisotropy_max_min",
        "angle_deviation_from_ortho", "is_cubic_like",
        "li_fraction", "li_concentration", "framework_density",
        "li_li_min_dist", "li_li_avg_dist", "li_anion_min_dist",
        "li_coordination_avg", "li_site_avg_multiplicity",
        "has_cif_struct"
    ]
    
    spacegroup_cols = [f"sg_{i}" for i in range(1, 231)]
    
    baseline_feature_cols = stage0_feature_cols + geometry_cols + spacegroup_cols
    baseline_feature_cols = [c for c in baseline_feature_cols if c in full_train.columns]
    
    logger.info(f"Baseline feature count: {len(baseline_feature_cols)}")
    
    # -----------------------------------------------------------------------------------------
    # Cross-Validation Setup
    # -----------------------------------------------------------------------------------------
    y = full_train[TARGET_COL]
    groups = np.arange(len(full_train))
    
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X=full_train, y=y, groups=groups))
    logger.info(f"Created {len(splits)} CV splits.")
    
    # -----------------------------------------------------------------------------------------
    # Define Experiments
    # -----------------------------------------------------------------------------------------
    # We focus on two key experiments:
    # 1. no_physics: TRUE BASELINE (Stage 1 full_struct replica) - no physics features
    # 2. full_occ_only: Physics features ONLY from full-occupancy structures (most reliable)
    #
    # Key improvements in full_occ_only:
    # - Separate gating indicators (has_bv_mismatch, has_ewald_energy, has_voronoi_cn)
    # - Sentinel value (-999) for missing physics values (instead of 0) to resolve ambiguity
    # - Physics features only from physically meaningful full-occupancy structures
    experiments = {
        "no_physics": {
            "description": "Stage 1 full_struct replica (no physics features) - TRUE BASELINE",
            "method": None,  # None means skip physics feature extraction
        },
        "full_occ_only": {
            "description": "Physics features ONLY from full-occupancy structures (with gating + sentinel)",
            "method": "full_occ_only",
        },
    }
    
    # -----------------------------------------------------------------------------------------
    # Run Experiments
    # -----------------------------------------------------------------------------------------
    all_results = {}
    
    for exp_name, exp_config in experiments.items():
        logger.info("=" * 80)
        logger.info(f"Experiment: {exp_name}")
        logger.info(f"Description: {exp_config['description']}")
        logger.info("-" * 80)
        
        method = exp_config["method"]
        
        # Handle "no_physics" experiment (method=None): skip physics features entirely
        # This replicates stage1_full_struct exactly for baseline comparison
        if method is None:
            logger.info("Skipping physics feature extraction (no_physics baseline)")
            train_with_physics = full_train.copy()
            test_with_physics = full_test.copy()
            train_physics_stats = {
                "total": len(full_train),
                "processed": 0,
                "with_bv": 0,
                "with_ewald": 0,
                "with_voronoi": 0,
                "with_any_physics": 0,
            }
            physics_feature_cols = []  # No physics features for this experiment
            full_feature_cols = baseline_feature_cols  # Use ONLY baseline features
        else:
            # Extract physics features using this method
            logger.info(f"Extracting physics features using '{method}' method...")
            
            train_with_physics, train_physics_stats = extract_physics_features_for_method(
                full_train.copy(), method=method, id_col="ID", cif_dir=train_cif_dir, logger=logger
            )
            test_with_physics, test_physics_stats = extract_physics_features_for_method(
                full_test.copy(), method=method, id_col="ID", cif_dir=test_cif_dir, logger=logger
            )
            
            logger.info(f"Physics feature extraction results (training):")
            logger.info(f"  Processed CIFs: {train_physics_stats['processed']}")
            logger.info(f"  With BV mismatch: {train_physics_stats['with_bv']} ({100*train_physics_stats['with_bv']/max(1,train_physics_stats['processed']):.1f}%)")
            logger.info(f"  With Ewald energy: {train_physics_stats['with_ewald']} ({100*train_physics_stats['with_ewald']/max(1,train_physics_stats['processed']):.1f}%)")
            logger.info(f"  With Voronoi CN: {train_physics_stats['with_voronoi']} ({100*train_physics_stats['with_voronoi']/max(1,train_physics_stats['processed']):.1f}%)")
            logger.info(f"  With ANY physics: {train_physics_stats['with_any_physics']} ({100*train_physics_stats['with_any_physics']/len(train_with_physics):.1f}% of total)")
            
            # Physics feature columns
            physics_value_cols = ["bv_mismatch_avg", "bv_mismatch_std",
                                 "ewald_energy_avg", "ewald_energy_std", "li_voronoi_cn_avg"]
            # CRITICAL: Separate indicators for each physics feature type for proper gating!
            physics_indicator_cols = [
                "has_bv_mismatch", "has_ewald_energy", "has_voronoi_cn",  # Individual
                "has_physics_features"  # Combined
            ]
            
            # Add method-specific indicators
            if method == "full_occ_only":
                physics_indicator_cols.append("is_full_occupancy")
            elif method == "ordered_approx":
                physics_indicator_cols.append("used_ordered_approx")
            elif method == "occ_weighted":
                physics_indicator_cols.append("avg_li_occupancy")
            
            # Fill missing physics VALUE columns with sentinel (physically impossible value)
            # This resolves ambiguity: the model can learn that -999 means "not available"
            # Combined with gating indicators (has_bv_mismatch, etc.), model knows when to ignore
            for col in physics_value_cols:
                if col in train_with_physics.columns:
                    train_with_physics[col] = train_with_physics[col].fillna(MISSING_PHYSICS_SENTINEL)
                if col in test_with_physics.columns:
                    test_with_physics[col] = test_with_physics[col].fillna(MISSING_PHYSICS_SENTINEL)
            
            # Fill indicator columns with 0 (meaning "not available")
            for col in physics_indicator_cols:
                if col in train_with_physics.columns:
                    train_with_physics[col] = train_with_physics[col].fillna(0)
                if col in test_with_physics.columns:
                    test_with_physics[col] = test_with_physics[col].fillna(0)
            
            # Define full feature set (baseline + physics)
            physics_feature_cols = [c for c in physics_value_cols + physics_indicator_cols if c in train_with_physics.columns]
            full_feature_cols = baseline_feature_cols + physics_feature_cols
        
        logger.info(f"Total features: {len(full_feature_cols)} (baseline: {len(baseline_feature_cols)}, physics: {len(physics_feature_cols)})")
        
        # Prepare feature matrix
        available_cols = [c for c in full_feature_cols if c in train_with_physics.columns]
        X = train_with_physics[available_cols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.fillna(0)
        
        # Model configuration
        config = ExperimentConfig(
            model_name="hgbt",
            n_splits=5,
            random_state=42,
            params={},
            group_col=None
        )
        
        # Run cross-validation
        fold_scores, overall_metrics, oof = run_cv_with_predefined_splits(X, y, splits, config)
        
        logger.info(f"[{exp_name}] CV Results:")
        logger.info(f"  R²:   {overall_metrics['r2']:.4f}")
        logger.info(f"  RMSE: {overall_metrics['rmse']:.4f}")
        logger.info(f"  MAE:  {overall_metrics['mae']:.4f}")
        
        all_results[exp_name] = {
            "metrics": overall_metrics,
            "physics_stats": train_physics_stats,
        }
        
        # Save CV parity plot
        parity_path = os.path.join(results_dir, f"{exp_name}_cv_parity.png")
        save_parity_plot(y.values, oof, parity_path, title=f"CV Parity: {exp_name}")
        
        # Train final model and predict on test
        X_test = test_with_physics[[c for c in available_cols if c in test_with_physics.columns]].copy()
        X_test = X_test.reindex(columns=X.columns, fill_value=0)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test = X_test.fillna(0)
        
        preds = fit_full_and_predict(X, y, X_test, config)
        
        # Save predictions
        pred_path = os.path.join(results_dir, f"{exp_name}_predictions.csv")
        save_dataframe(pd.DataFrame({"ID": test_df["ID"], "prediction": preds}), pred_path)
        
        # Test parity plot
        if TARGET_COL in test_df.columns:
            test_parity_path = os.path.join(results_dir, f"{exp_name}_test_parity.png")
            save_parity_plot(test_df[TARGET_COL].values, preds, test_parity_path, 
                           title=f"Test Parity: {exp_name}")
    
    # -----------------------------------------------------------------------------------------
    # Summary and Report
    # -----------------------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("SUMMARY: Partial Occupancy Experiments")
    logger.info("=" * 80)
    
    # Reference metrics from Stage 1 (stage1_full_struct) for validation
    stage1_ref = {
        "r2": 0.7370,
        "rmse": 1.3734,
        "mae": 0.8253,
    }
    
    # Validate no_physics experiment matches Stage 1
    if "no_physics" in all_results:
        no_physics_metrics = all_results["no_physics"]["metrics"]
        r2_diff = abs(no_physics_metrics['r2'] - stage1_ref['r2'])
        rmse_diff = abs(no_physics_metrics['rmse'] - stage1_ref['rmse'])
        mae_diff = abs(no_physics_metrics['mae'] - stage1_ref['mae'])
        
        logger.info("-" * 80)
        logger.info("VALIDATION: no_physics vs Stage 1 (stage1_full_struct)")
        logger.info(f"  Reference: R²={stage1_ref['r2']:.4f}, RMSE={stage1_ref['rmse']:.4f}, MAE={stage1_ref['mae']:.4f}")
        logger.info(f"  no_physics: R²={no_physics_metrics['r2']:.4f}, RMSE={no_physics_metrics['rmse']:.4f}, MAE={no_physics_metrics['mae']:.4f}")
        
        if r2_diff < 0.001 and rmse_diff < 0.001 and mae_diff < 0.001:
            logger.info("  [PASS] Results match Stage 1 within tolerance!")
        else:
            logger.warning(f"  [WARN] Results differ from Stage 1 (R² diff: {r2_diff:.4f}, RMSE diff: {rmse_diff:.4f}, MAE diff: {mae_diff:.4f})")
        logger.info("-" * 80)
    
    # Create summary table
    summary_data = []
    for exp_name, result in all_results.items():
        m = result["metrics"]
        ps = result["physics_stats"]
        summary_data.append({
            "Experiment": exp_name,
            "R²": f"{m['r2']:.4f}",
            "RMSE": f"{m['rmse']:.4f}",
            "MAE": f"{m['mae']:.4f}",
            "Physics Coverage": f"{100*ps['with_any_physics']/len(full_train):.1f}%",
            "BV Coverage": f"{100*ps['with_bv']/max(1,ps['processed']):.1f}%",
            "Ewald Coverage": f"{100*ps['with_ewald']/max(1,ps['processed']):.1f}%",
            "Voronoi Coverage": f"{100*ps['with_voronoi']/max(1,ps['processed']):.1f}%",
        })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info("\n" + summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(os.path.join(results_dir, "experiment_summary.csv"), index=False)
    
    # Generate markdown report
    report_content = f"""# Partial Occupancy Solutions: Experiment Report

**Date:** {pd.to_datetime('today').strftime('%Y-%m-%d')}

## Problem Statement

Physics-informed features (Bond-Valence mismatch, Ewald energy, Voronoi coordination)
fail for structures with partial site occupancy. Analysis showed:

- **76.8%** of CIF files have partial occupancy
- Physics feature success rate: **4.6-31.8%** for partial occupancy vs **91-98%** for full occupancy

## Key Improvements in This Version

1. **Separate gating indicators** for each physics feature type:
   - `has_bv_mismatch`, `has_ewald_energy`, `has_voronoi_cn`
   - Allows model to learn when to trust each specific feature

2. **Sentinel value (-999)** for missing physics features:
   - Resolves ambiguity: 0 could be a valid physics value
   - Model can learn that -999 means "not available"

3. **Focused experiments**: Only testing the most reliable approach (full_occ_only)
   against the true baseline (no physics features)

## Experiments

### 1. No Physics (TRUE BASELINE)
**Stage 1 full_struct replica** - Uses only Stage 0 + Stage 1 structural features.
No physics features. This validates pipeline consistency with Stage 1 results.

### 2. Full Occupancy Only (with improved gating)
Only extract physics features from structures with full occupancy (~23% of CIFs).
- More reliable physics calculations
- Separate gating indicators for each feature type
- Sentinel value for missing data

## Results

| Experiment | R² | RMSE | MAE | Physics Coverage |
|------------|-----|------|-----|------------------|
"""
    
    for exp_name, result in all_results.items():
        m = result["metrics"]
        ps = result["physics_stats"]
        coverage = f"{100*ps['with_any_physics']/len(full_train):.1f}%"
        report_content += f"| {exp_name} | {m['r2']:.4f} | {m['rmse']:.4f} | {m['mae']:.4f} | {coverage} |\n"
    
    report_content += f"""
## Physics Feature Coverage by Method

| Method | BV Mismatch | Ewald Energy | Voronoi CN |
|--------|------------|--------------|------------|
"""
    
    for exp_name, result in all_results.items():
        ps = result["physics_stats"]
        processed = max(1, ps['processed'])
        report_content += f"| {exp_name} | {100*ps['with_bv']/processed:.1f}% | {100*ps['with_ewald']/processed:.1f}% | {100*ps['with_voronoi']/processed:.1f}% |\n"
    
    report_content += """
## Key Findings

- **No Physics (TRUE BASELINE)** should match Stage 1 `stage1_full_struct` results exactly
- **Full Occupancy Only** tests whether reliable physics features improve generalization
- Separate gating indicators allow the model to learn meta-patterns about feature availability
- Sentinel values (-999) resolve ambiguity between "missing" and "zero" physics values

## Validation

The `no_physics` experiment serves as a pipeline validation check. It should produce
identical R², RMSE, and MAE values as the Stage 1 `stage1_full_struct` experiment
(R²=0.7370, RMSE=1.3734, MAE=0.8253). Any discrepancy indicates a pipeline issue.

## Methodology Notes

**Gating Indicators:** Each physics feature has its own indicator variable:
- `has_bv_mismatch` = 1 when BV mismatch was calculated, 0 otherwise
- `has_ewald_energy` = 1 when Ewald energy was calculated, 0 otherwise
- `has_voronoi_cn` = 1 when Voronoi CN was calculated, 0 otherwise

**Sentinel Value:** When a physics feature is not available, its value is set to -999
instead of 0. This allows tree-based models to learn splits like:
- "If has_bv_mismatch=1 AND bv_mismatch_avg < 0.5, then..."
- "If has_bv_mismatch=0 (value is -999), use composition features instead"

## Artifacts

- `*_cv_parity.png` - Cross-validation parity plots
- `*_test_parity.png` - Test set parity plots  
- `*_predictions.csv` - Test set predictions
- `experiment_summary.csv` - Summary statistics
"""
    
    report_path = os.path.join(results_dir, "partial_occupancy_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    logger.info(f"\nReport saved to: {report_path}")
    logger.info("=" * 80)
    logger.info("Experiments completed successfully!")


if __name__ == "__main__":
    main()
