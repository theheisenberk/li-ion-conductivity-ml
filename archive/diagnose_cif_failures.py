"""
Diagnostic script to understand why CIF parsing and physics feature extraction fails.

This script analyzes:
1. CIF parsing failures (pymatgen Structure.from_file)
2. Bond-valence mismatch calculation failures
3. Ewald energy calculation failures
4. Voronoi coordination number failures
"""

import os
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CIF_DIR = os.path.join(DATA_DIR, "train_cifs")

def diagnose_cif_parsing(cif_path):
    """
    Diagnose why a CIF file might fail to parse.
    Returns a dict with diagnostic information.
    """
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifParser
    
    result = {
        "file": os.path.basename(cif_path),
        "parse_success": False,
        "parse_method": None,
        "error_type": None,
        "error_message": None,
        "has_partial_occupancy": False,
        "has_li": False,
        "n_atoms": None,
        "n_li_sites": None,
        "composition": None,
        "spacegroup": None,
    }
    
    structure = None
    
    # Try Method 1: Direct from_file
    try:
        structure = Structure.from_file(cif_path)
        result["parse_success"] = True
        result["parse_method"] = "Structure.from_file"
    except Exception as e1:
        result["error_type"] = type(e1).__name__
        result["error_message"] = str(e1)[:100]
        
        # Try Method 2: CifParser with tolerances
        try:
            parser = CifParser(cif_path, occupancy_tolerance=1.0)
            structures = parser.parse_structures(primitive=False)
            if structures:
                structure = structures[0]
                result["parse_success"] = True
                result["parse_method"] = "CifParser(occupancy_tolerance=1.0)"
                result["error_type"] = None
                result["error_message"] = None
        except Exception as e2:
            result["error_type"] = type(e2).__name__
            result["error_message"] = str(e2)[:100]
    
    # Analyze structure if we got one
    if structure is not None:
        result["n_atoms"] = len(structure)
        result["composition"] = str(structure.composition.reduced_formula)
        
        # Check for Li
        li_sites = [i for i, site in enumerate(structure) if "Li" in str(site.species)]
        result["n_li_sites"] = len(li_sites)
        result["has_li"] = len(li_sites) > 0
        
        # Check for partial occupancy
        for site in structure:
            if hasattr(site, 'species') and hasattr(site.species, 'as_dict'):
                species_dict = site.species.as_dict()
                for el, occ in species_dict.items():
                    if occ < 1.0:
                        result["has_partial_occupancy"] = True
                        break
        
        # Get spacegroup
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            result["spacegroup"] = sga.get_space_group_number()
        except:
            pass
    
    return result, structure


def diagnose_bv_mismatch(structure, verbose=False):
    """
    Diagnose why bond-valence mismatch calculation might fail.
    """
    result = {
        "bv_success": False,
        "bv_error_type": None,
        "bv_error_message": None,
        "n_li_with_neighbors": 0,
        "n_li_with_bv_params": 0,
        "neighbor_elements": set(),
        "missing_bv_params": set(),
    }
    
    if structure is None:
        result["bv_error_type"] = "NoStructure"
        return result
    
    # Find Li sites
    li_indices = [i for i, site in enumerate(structure) if "Li" in str(site.species)]
    if not li_indices:
        result["bv_error_type"] = "NoLiSites"
        return result
    
    # BV parameters available
    bv_params = {"O", "S", "F", "Cl", "Br", "I", "N", "Se"}
    
    try:
        from pymatgen.analysis.local_env import CrystalNN
        cnn = CrystalNN()
        
        for li_idx in li_indices[:min(5, len(li_indices))]:  # Check up to 5 Li sites
            try:
                neighbors = cnn.get_nn_info(structure, li_idx)
                if neighbors:
                    result["n_li_with_neighbors"] += 1
                    
                    has_valid_neighbor = False
                    for neighbor in neighbors:
                        neighbor_site = neighbor["site"]
                        # Get element symbol
                        if hasattr(neighbor_site.species, 'elements'):
                            el = str(neighbor_site.species.elements[0])
                        else:
                            el = str(neighbor_site.species).split(":")[0]
                        
                        result["neighbor_elements"].add(el)
                        
                        # Check if BV params exist for this element
                        if any(anion in el for anion in bv_params):
                            has_valid_neighbor = True
                        else:
                            result["missing_bv_params"].add(el)
                    
                    if has_valid_neighbor:
                        result["n_li_with_bv_params"] += 1
            except Exception as e:
                if verbose:
                    print(f"  Neighbor finding failed for Li site {li_idx}: {e}")
        
        if result["n_li_with_bv_params"] > 0:
            result["bv_success"] = True
        else:
            result["bv_error_type"] = "NoBVParams"
            result["bv_error_message"] = f"Missing params for: {result['missing_bv_params']}"
            
    except Exception as e:
        result["bv_error_type"] = type(e).__name__
        result["bv_error_message"] = str(e)[:100]
    
    return result


def diagnose_ewald_energy(structure):
    """
    Diagnose why Ewald energy calculation might fail.
    """
    result = {
        "ewald_success": False,
        "ewald_error_type": None,
        "ewald_error_message": None,
        "oxidation_assigned": False,
        "oxidation_method": None,
    }
    
    if structure is None:
        result["ewald_error_type"] = "NoStructure"
        return result
    
    # Try to assign oxidation states
    oxi_struct = None
    
    # Method 1: BVAnalyzer
    try:
        from pymatgen.analysis.bond_valence import BVAnalyzer
        bva = BVAnalyzer()
        oxi_struct = bva.get_oxi_state_decorated_structure(structure)
        result["oxidation_assigned"] = True
        result["oxidation_method"] = "BVAnalyzer"
    except Exception as e1:
        # Method 2: add_oxidation_state_by_guess
        try:
            oxi_struct = structure.copy()
            oxi_struct.add_oxidation_state_by_guess()
            result["oxidation_assigned"] = True
            result["oxidation_method"] = "add_oxidation_state_by_guess"
        except Exception as e2:
            result["ewald_error_type"] = "OxidationFailed"
            result["ewald_error_message"] = f"BVA: {str(e1)[:50]}, Guess: {str(e2)[:50]}"
            return result
    
    # Try Ewald summation
    try:
        from pymatgen.analysis.ewald import EwaldSummation
        ewald = EwaldSummation(oxi_struct)
        
        # Try to get Li site energy
        li_indices = [i for i, site in enumerate(oxi_struct) if "Li" in str(site.species)]
        if li_indices:
            energy = ewald.get_site_energy(li_indices[0])
            result["ewald_success"] = True
        else:
            result["ewald_error_type"] = "NoLiInOxiStruct"
    except Exception as e:
        result["ewald_error_type"] = type(e).__name__
        result["ewald_error_message"] = str(e)[:100]
    
    return result


def diagnose_voronoi(structure):
    """
    Diagnose why Voronoi coordination number calculation might fail.
    """
    result = {
        "voronoi_success": False,
        "voronoi_error_type": None,
        "voronoi_error_message": None,
    }
    
    if structure is None:
        result["voronoi_error_type"] = "NoStructure"
        return result
    
    li_indices = [i for i, site in enumerate(structure) if "Li" in str(site.species)]
    if not li_indices:
        result["voronoi_error_type"] = "NoLiSites"
        return result
    
    try:
        from pymatgen.analysis.local_env import VoronoiNN
        vnn = VoronoiNN(cutoff=5.0, allow_pathological=True)
        
        cn = vnn.get_cn(structure, li_indices[0])
        result["voronoi_success"] = True
    except Exception as e:
        result["voronoi_error_type"] = type(e).__name__
        result["voronoi_error_message"] = str(e)[:100]
    
    return result


def main():
    print("=" * 80)
    print("CIF PARSING AND PHYSICS FEATURE DIAGNOSIS")
    print("=" * 80)
    
    # Get list of CIF files
    cif_files = sorted([f for f in os.listdir(CIF_DIR) if f.endswith('.cif')])
    print(f"\nFound {len(cif_files)} CIF files to analyze.\n")
    
    # Analyze each CIF
    all_results = []
    
    for i, cif_file in enumerate(cif_files):
        cif_path = os.path.join(CIF_DIR, cif_file)
        
        # 1. CIF Parsing
        parse_result, structure = diagnose_cif_parsing(cif_path)
        
        # 2. Bond-Valence Mismatch
        bv_result = diagnose_bv_mismatch(structure)
        
        # 3. Ewald Energy
        ewald_result = diagnose_ewald_energy(structure)
        
        # 4. Voronoi CN
        voronoi_result = diagnose_voronoi(structure)
        
        # Combine results
        combined = {**parse_result, **bv_result, **ewald_result, **voronoi_result}
        all_results.append(combined)
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(cif_files)} CIF files...")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\n1. CIF PARSING:")
    print(f"   - Successfully parsed: {df['parse_success'].sum()}/{len(df)} ({100*df['parse_success'].mean():.1f}%)")
    print(f"   - Has Li atoms: {df['has_li'].sum()}/{len(df)} ({100*df['has_li'].mean():.1f}%)")
    print(f"   - Has partial occupancy: {df['has_partial_occupancy'].sum()}/{len(df)} ({100*df['has_partial_occupancy'].mean():.1f}%)")
    
    print(f"\n2. BOND-VALENCE MISMATCH:")
    print(f"   - Successfully calculated: {df['bv_success'].sum()}/{len(df)} ({100*df['bv_success'].mean():.1f}%)")
    
    print(f"\n3. EWALD ENERGY:")
    print(f"   - Oxidation states assigned: {df['oxidation_assigned'].sum()}/{len(df)} ({100*df['oxidation_assigned'].mean():.1f}%)")
    print(f"   - Successfully calculated: {df['ewald_success'].sum()}/{len(df)} ({100*df['ewald_success'].mean():.1f}%)")
    
    print(f"\n4. VORONOI COORDINATION:")
    print(f"   - Successfully calculated: {df['voronoi_success'].sum()}/{len(df)} ({100*df['voronoi_success'].mean():.1f}%)")
    
    # Failure analysis
    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS")
    print("=" * 80)
    
    print("\n--- CIF Parsing Failures ---")
    parse_failures = df[~df['parse_success']]
    if len(parse_failures) > 0:
        print(f"Total failures: {len(parse_failures)}")
        print("Error types:")
        print(parse_failures['error_type'].value_counts().to_string())
    else:
        print("No parsing failures!")
    
    print("\n--- Bond-Valence Failures ---")
    bv_failures = df[~df['bv_success'] & df['parse_success']]
    if len(bv_failures) > 0:
        print(f"Total failures (among parsed CIFs): {len(bv_failures)}")
        print("Error types:")
        print(bv_failures['bv_error_type'].value_counts().to_string())
        
        # Collect all missing BV params
        all_missing = set()
        for params in bv_failures['missing_bv_params']:
            if isinstance(params, set):
                all_missing.update(params)
        print(f"\nElements missing BV parameters: {sorted(all_missing)}")
    
    print("\n--- Ewald Energy Failures ---")
    ewald_failures = df[~df['ewald_success'] & df['parse_success']]
    if len(ewald_failures) > 0:
        print(f"Total failures (among parsed CIFs): {len(ewald_failures)}")
        print("Error types:")
        print(ewald_failures['ewald_error_type'].value_counts().to_string())
    
    print("\n--- Voronoi Failures ---")
    voronoi_failures = df[~df['voronoi_success'] & df['parse_success']]
    if len(voronoi_failures) > 0:
        print(f"Total failures (among parsed CIFs): {len(voronoi_failures)}")
        print("Error types:")
        print(voronoi_failures['voronoi_error_type'].value_counts().to_string())
    
    # Partial occupancy analysis
    print("\n" + "=" * 80)
    print("PARTIAL OCCUPANCY ANALYSIS")
    print("=" * 80)
    partial_occ = df[df['has_partial_occupancy']]
    if len(partial_occ) > 0:
        print(f"\nCIFs with partial occupancy: {len(partial_occ)}/{len(df)} ({100*len(partial_occ)/len(df):.1f}%)")
        print(f"BV success rate for partial occupancy: {100*partial_occ['bv_success'].mean():.1f}%")
        print(f"Ewald success rate for partial occupancy: {100*partial_occ['ewald_success'].mean():.1f}%")
        print(f"Voronoi success rate for partial occupancy: {100*partial_occ['voronoi_success'].mean():.1f}%")
        
        non_partial = df[~df['has_partial_occupancy'] & df['parse_success']]
        print(f"\nBV success rate for FULL occupancy: {100*non_partial['bv_success'].mean():.1f}%")
        print(f"Ewald success rate for FULL occupancy: {100*non_partial['ewald_success'].mean():.1f}%")
        print(f"Voronoi success rate for FULL occupancy: {100*non_partial['voronoi_success'].mean():.1f}%")
    
    # Save detailed results
    output_path = os.path.join(PROJECT_ROOT, "results", "cif_diagnosis_results.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert sets to strings for CSV
    df['neighbor_elements'] = df['neighbor_elements'].apply(lambda x: str(sorted(x)) if isinstance(x, set) else str(x))
    df['missing_bv_params'] = df['missing_bv_params'].apply(lambda x: str(sorted(x)) if isinstance(x, set) else str(x))
    
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
