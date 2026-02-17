import pandas as pd
import pytest

import src.features as features_module
from src.features import (
	stage1_csv_structural_features,
	stage1_spacegroup_onehot,
	stage1_structural_features,
)


def test_stage1_csv_structural_features_computes_lattice_metrics():
	df = pd.DataFrame(
		{
			"a": [5.0, 3.0],
			"b": [5.0, 4.0],
			"c": [5.0, 5.0],
			"alpha": [90.0, 100.0],
			"beta": [90.0, 95.0],
			"gamma": [90.0, 110.0],
			"Space group #": [225, 62],
			"Z": [4, 8],
		}
	)

	result = stage1_csv_structural_features(df)

	assert pytest.approx(result.loc[0, "lattice_volume"], rel=1e-5) == 125.0
	assert result.loc[0, "is_cubic_like"] == 1
	assert result.loc[1, "is_cubic_like"] == 0
	assert result.loc[1, "spacegroup_number"] == 62
	assert "lattice_anisotropy" in result.columns
	assert "angle_deviation_from_ortho" in result.columns


def test_stage1_spacegroup_onehot_creates_binary_columns():
	df = pd.DataFrame({"spacegroup_number": [1, 230, 15]})

	result = stage1_spacegroup_onehot(df)

	assert result.loc[0, "sg_1"] == 1
	assert result.loc[1, "sg_230"] == 1
	assert result.filter(like="sg_").sum(axis=1).tolist() == [1, 1, 1]


def test_stage1_spacegroup_onehot_warns_when_missing_column():
	df = pd.DataFrame({"not_spacegroup": [1, 2]})

	with pytest.warns(UserWarning):
		result = stage1_spacegroup_onehot(df, spacegroup_col="missing")

	assert result.equals(df)


def test_stage1_structural_features_records_stats(monkeypatch, tmp_path):
	df = pd.DataFrame({"ID": ["full", "partial", "missing"]})

	cif_dir = tmp_path / "cifs"
	cif_dir.mkdir()
	for name in ["full", "partial"]:
		(cif_dir / f"{name}.cif").write_text(f"data for {name}", encoding="utf-8")

	def fake_extract(cif_path, extended=True):
		if "full" in cif_path:
			return {
				"density": 1.23,
				"volume_per_atom": 4.56,
				"spacegroup_number": 225,
				"n_li_sites": 10,
				"n_total_atoms": 40,
			}
		if "partial" in cif_path:
			return {"density": 0.9}
		return {}

	monkeypatch.setattr(features_module, "_extract_structural_features_from_cif", fake_extract)

	struct_df, stats = stage1_structural_features(
		df,
		id_col="ID",
		cif_dir=str(cif_dir),
		extended=True,
		verbose=False,
	)

	assert stats["parsed_full"] == 1
	assert stats["parsed_partial"] == 1
	assert stats["missing_cif"] == 1
	assert "full" in struct_df["ID"].values
	full_row = struct_df.loc[struct_df["ID"] == "full"].iloc[0]
	assert full_row["density"] == pytest.approx(1.23)
	assert full_row["n_li_sites"] == 10
	partial_row = struct_df.loc[struct_df["ID"] == "partial"].iloc[0]
	assert partial_row["density"] == pytest.approx(0.9)

