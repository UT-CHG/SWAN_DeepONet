# SWAN_DeepONet

**Operator Learning for Predicting Bulk Wave Parameters of Spectral Wave Models**

This repository contains the code, example notebooks, and data-generation setup for a
DeepONet surrogate of the spectral wave model **SWAN**. The surrogate
predicts bulk wave quantities (significant wave height (*H*<sub>sig</sub>) and the
wave-induced radiation-stress forcing), and
is evaluated on idealized 1D cases and on the field-scale **DUCK** case.

> The large binary assets (training/validation **data**, trained **model** weights, and
> the sampling-sensitivity model set) are **not** stored in this repository. They are
> on the **DesignSafe-CI Data Depot** under project **PRJ-6235 —
> *Operator Learning for Predicting Bulk Wave Parameters of Spectral Wave Models*** and
> can be downloaded from there.

---

## Repository structure

```text
NeuralOperator-CoastalWaves/
├── analysis_tools/          # Shared Python modules (data extraction, scaling, metrics, plotting)
├── examples/                # End-to-end notebooks that reproduce the paper figures
│   ├── 1d/                  #   Idealized 1D cases
│   └── duck/                #   Field-scale DUCK case
├── baseline_mlp/            # MLP baseline for comparison against the neural operator
│   └── MLP/
├── cross_grid_analysis/     # Cross-grid / grid-refinement generalization study (DUCK)
├── runtime_benchmark/       # Execution-time comparison: surrogate inference vs. SWAN
├── sampling_sensitivity/    # How training-sample selection affects surrogate accuracy
├── swan_data_generation/    # SWAN control files (*.swn), bathymetry, data-gen scripts
│   └── CrossGridValidation/
├── Models/
│   └── scripts/             # Training scripts (trained weights hosted on DesignSafe)
├── README.md
└── gitpush.sh

# Downloaded from DesignSafe (PRJ-6235), placed at repo root
#   Data/                        SWAN input/output for training, validation, evaluation
#   Models/ (weights + scalers)  Trained neural-operator & MLP weights
#   sampling_sensitivity_models/ Model set for the sampling-sensitivity study
```


| Path | Description |
|------|-------------|
| `analysis_tools/` | Reusable Python modules: data extraction, scaling/preprocessing, error metrics, and plotting. Imported by the example notebooks. |
| `examples/` | End-to-end notebooks that reproduce the paper figures (`1d/`, `duck/`). Start here. |
| `baseline_mlp/` | Multilayer-perceptron (MLP) baseline used for comparison against the neural operator. |
| `cross_grid_analysis/` | Cross-grid / grid-refinement generalization study (DUCK). |
| `runtime_benchmark/` | Execution-time comparison: surrogate inference vs. running SWAN. |
| `sampling_sensitivity/` | Study of how training-sample selection affects surrogate accuracy. |
| `swan_data_generation/` | SWAN control files (`*.swn`), bathymetry, and scripts used to generate the training data. |
| `Models/scripts/` | Training scripts for the neural-operator models (trained weights live on DesignSafe). |

### Data on DesignSafe 

Download these from **PRJ-6235** and place them at the repository root. They are on the
> [**DesignSafe-CI Data Depot**](https://doi.org/10.17603/DS2-DKHW-PT40) under project
> [**PRJ-6235**](https://doi.org/10.17603/DS2-DKHW-PT40) —
> *Operator Learning for Predicting Bulk Wave Parameters of Spectral Wave Models*.

| Path | Contents |
|------|----------|
| `Data/` | SWAN input/output used for training, validation, and evaluation. |
| `Models/` | Trained neural-operator and MLP weights + fitted scalers. |
| `sampling_sensitivity_models/` | Model set for the sampling-sensitivity study. |

---

## Getting started

### Environment
The analysis code uses Python 3.11 with:

```
numpy  pandas  matplotlib  cmocean  scikit-learn
tensorflow  joblib  tqdm  netCDF4
```

We recommend a dedicated environment, e.g.:

```bash
conda create -n coastalwaves python=3.11
conda activate coastalwaves
pip install numpy pandas matplotlib cmocean scikit-learn tensorflow joblib tqdm netCDF4
```

### Reproducing the results
1. Download `Data/` and `Models/` from DesignSafe (PRJ-6235) into the repository root.
2. Open a notebook under `examples/` (e.g. `examples/duck/duck_analysis.ipynb`) and run
   it top to bottom. The notebooks import the shared modules in `analysis_tools/`.

---

## Citation

If you use this code or data, please cite the associated publication and the DesignSafe
dataset (PRJ-6235).

```bibtex
@article{
  title   = {Operator Learning for Predicting Bulk Wave Parameters of Spectral Wave Models},
  author  = {Shukai Cai, Sourav Dutta, Mark Loveland, Eirik Valseth, Peter Rivera-Casillas, Corey Trahan, Clint Dawson},
  journal = {Ocean Engineering},
  volume = {367},
  pages = {127866},
  year = {2026},
  issn = {0029-8018},
  doi = {https://doi.org/10.1016/j.oceaneng.2026.127866},
  url = {https://www.sciencedirect.com/science/article/pii/S0029801826037005}
}
```