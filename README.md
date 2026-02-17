# Predicting Li-Ion Conductivity with a 3-Stage ML Pipeline

A machine-learning pipeline that progressively incorporates composition, structural geometry, and physics-informed features to predict lithium-ion conductivity in solid-state electrolytes.

## Project Overview

This project implements a three-stage feature-engineering strategy on top of a `HistGradientBoostingRegressor`:

1. **Stage 0 -- Composition only:** elemental ratios, SMACT stoichiometry features, and Magpie element embeddings.
2. **Stage 1 -- Structural geometry:** adds lattice parameters, density, Li coordination metrics, and other CIF-derived features.
3. **Stage 2 -- Physics-informed:** adds bond-valence mismatch, Ewald site energy, and Voronoi coordination numbers computed via `pymatgen`.

Optuna is used for hyperparameter optimization in Stage 2.

## Project Structure

```
.
├── src/                              # Reusable pipeline modules
│   ├── data_processing.py            #   Data loading, cleaning, target transform
│   ├── features.py                   #   Feature engineering (Stages 0-2)
│   ├── model_training.py             #   Model training, CV, Optuna integration
│   └── utils.py                      #   Logging, plotting, metrics helpers
│
├── stage0_embedding_comparison.py    # Stage 0: composition-only baseline & embedding comparison
├── stage1_structural_features.py     # Stage 1: structural geometry features from CIFs
├── stage2_physics_optuna.py          # Stage 2: physics-informed features + Optuna optimization
├── stage3_final_comparison.py        # Stage 3: final 3-stage comparison & publication plots
│
├── results/                          # All experiment outputs
│   ├── results_stage0/               #   Stage 0 results
│   ├── results_stage1/               #   Stage 1 results
│   ├── results_stage2/               #   Stage 2 (Optuna-optimized) results
│   └── results_stage3_final/         #   Final comparison report, plots, and PDF
│
├── docs/                             # LaTeX report source and compiled PDF
├── archive/                          # Exploratory scripts kept for reference
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `scikit-learn`, `pymatgen`, `matminer`, `smact`, `optuna`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

## How to Run

Execute the scripts in order. Each stage builds on the previous one:

```bash
# Stage 0: composition-only baseline (embedding comparison)
python stage0_embedding_comparison.py

# Stage 1: add structural geometry features
python stage1_structural_features.py

# Stage 2: add physics features + Optuna hyperparameter optimization
python stage2_physics_optuna.py

# Stage 3: final comparison — combined parity plots and summary across all 3 stages
python stage3_final_comparison.py
```

## Key Results

| Stage | Description | CV R² | Test Spearman rho |
|-------|-------------|------:|------------------:|
| 0 | Composition (Magpie) | 0.746 | 0.710 |
| 1 | + Structural geometry | 0.737 | 0.711 |
| 2 | + Physics (BV, Ewald, Voronoi) | 0.752 | 0.748 |

The full report is available at [`docs/final_report.pdf`](docs/final_report.pdf).

## Data

This project uses the **OBELiX** dataset (Open solid Battery Electrolytes with Li: an eXperimental dataset), a curated collection of 599 synthesized solid-electrolyte materials with experimentally measured room-temperature ionic conductivities and crystallographic descriptions.

- **Repository:** <https://github.com/NRC-Mila/OBELiX>
- **Citation:** Therrien et al., "OBELiX: A Curated Dataset of Crystal Structures and Experimentally Measured Ionic Conductivities for Lithium Solid-State Electrolytes", *arXiv preprint arXiv:2502.14234*, 2025.

Raw data should be placed in `data/raw/`:

```
data/raw/
├── train.csv
├── test.csv
├── train_cifs/      # CIF structure files for training set
└── test_cifs/       # CIF structure files for test set
```

The files are included in this repository under `data/raw/` and were originally sourced from the [OBELiX GitHub repository](https://github.com/NRC-Mila/OBELiX/tree/main/data/downloads). The OBELiX dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
