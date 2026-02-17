import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stage1_structural_features as pipeline
from src.data_processing import TARGET_COL


class DummyLogger:
	def __init__(self):
		self.messages = []

	def info(self, msg):
		self.messages.append(("info", msg))

	def warning(self, msg):
		self.messages.append(("warning", msg))


class DummyGroupKFold:
	def __init__(self, n_splits):
		self.n_splits = n_splits

	def split(self, X, y, groups):
		yield np.array([0, 1, 2]), np.array([3, 4])


def test_main_runs_experiments_with_mocks(monkeypatch, tmp_path):
	project_root = tmp_path / "project"
	project_root.mkdir()
	(project_root / "requirements.txt").write_text("", encoding="utf-8")

	data_dir = project_root / "data" / "raw"
	train_cif_dir = data_dir / "train_cifs"
	test_cif_dir = data_dir / "test_cifs"
	train_cif_dir.mkdir(parents=True)
	test_cif_dir.mkdir(parents=True)

	for name in ["mat1", "mat2"]:
		(train_cif_dir / f"{name}.cif").write_text("data", encoding="utf-8")
	for name in ["mat3"]:
		(test_cif_dir / f"{name}.cif").write_text("data", encoding="utf-8")

	train_df = pd.DataFrame(
		{
			"ID": ["mat1", "mat2", "mat4", "mat5", "mat6"],
			"Reduced Composition": ["Li2O"] * 5,
			"Ionic conductivity (S cm-1)": [1e-3, 2e-3, 3e-3, 4e-3, 5e-3],
		}
	)
	test_df = pd.DataFrame(
		{
			"ID": ["mat3", "mat7"],
			"Reduced Composition": ["Li3N", "LiF"],
		}
	)

	pipeline.PROJECT_ROOT = str(project_root)

	monkeypatch.setattr(pipeline, "setup_logger", lambda *args, **kwargs: DummyLogger())
	monkeypatch.setattr(pipeline, "GroupKFold", DummyGroupKFold)

	monkeypatch.setattr(
		pipeline,
		"seed_everything",
		lambda *args, **kwargs: None,
	)

	monkeypatch.setattr(
		pipeline,
		"load_data",
		lambda *_args, **_kwargs: (train_df.copy(), test_df.copy()),
	)
	monkeypatch.setattr(
		pipeline,
		"clean_data",
		lambda df: df.copy(),
	)

	def fake_add_target(df, target_sigma_col=None):
		out = df.copy()
		out[TARGET_COL] = np.arange(len(out))
		return out

	monkeypatch.setattr(
		pipeline,
		"add_target_log10_sigma",
		fake_add_target,
	)

	def _stage0_base(df, *_args, **_kwargs):
		df = df.copy()
		df["stage0_base"] = 1.0
		return df

	monkeypatch.setattr(pipeline, "stage0_elemental_ratios", _stage0_base)
	monkeypatch.setattr(pipeline, "stage0_smact_features", _stage0_base)

	def _stage0_embeddings(df, *_args, **_kwargs):
		df = df.copy()
		df["magpie_emb_0"] = 0.5
		return df

	monkeypatch.setattr(pipeline, "stage0_element_embeddings", _stage0_embeddings)

	def _stage1_csv(df):
		df = df.copy()
		df["density"] = 1.0
		df["volume_per_atom"] = 2.0
		df["spacegroup_number"] = 1
		df["n_li_sites"] = 3
		df["n_total_atoms"] = 6
		return df

	monkeypatch.setattr(pipeline, "stage1_csv_structural_features", _stage1_csv)

	def _stage1_struct(df, **_kwargs):
		df = df.copy()
		df["density"] = 1.1
		df["volume_per_atom"] = 2.2
		df["spacegroup_number"] = 12
		df["n_li_sites"] = 4
		df["n_total_atoms"] = 8
		return df, {
			"parsed_full": len(df),
			"parsed_partial": 0,
			"missing_cif": 0,
			"failed": 0,
			"failed_ids": [],
			"missing_cif_ids": [],
		}

	monkeypatch.setattr(pipeline, "stage1_structural_features", _stage1_struct)
	monkeypatch.setattr(
		pipeline,
		"stage1_spacegroup_onehot",
		lambda df, **_kwargs: df.assign(sg_1=1),
	)

	run_records = []

	def fake_run_cv(X, y, splits, config):
		run_records.append(
			{
				"features": list(X.columns),
				"rows": len(X),
				"splits": len(splits),
				"model_name": config.model_name,
			}
		)
		return [0.1], {"r2": 0.9, "rmse": 0.1, "mae": 0.05}, np.zeros(len(y))

	monkeypatch.setattr(pipeline, "run_cv_with_predefined_splits", fake_run_cv)
	monkeypatch.setattr(
		pipeline,
		"fit_full_and_predict",
		lambda X, y, X_test, config: np.full(len(X_test), 0.42),
	)
	monkeypatch.setattr(pipeline, "save_parity_plot", lambda *args, **kwargs: None)
	monkeypatch.setattr(pipeline, "generate_interim_report", lambda *args, **kwargs: None)

	pipeline.main()

	assert len(run_records) == 4
	assert all(record["model_name"] == "hgbt" for record in run_records)
	assert all(record["splits"] == 1 for record in run_records)

	results_dir = Path(pipeline.PROJECT_ROOT) / "results" / "results_stage1"
	for exp in [
		"stage0_magpie",
		"stage1_basic_struct",
		"stage1_geometry",
		"stage1_full_struct",
	]:
		pred_path = results_dir / f"stage1_predictions_{exp}.csv"
		assert pred_path.exists(), f"missing predictions for {exp}"




