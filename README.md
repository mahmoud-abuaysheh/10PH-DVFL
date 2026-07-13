# 10PH-DVFL: A Communication-Efficient Decoupled Vertical Federated Learning Architecture for Multimodal Precision Health (10P-Health)

This repository contains the code for the paper:

> **10PH-DVFL: A Communication-Efficient Decoupled Vertical Federated Learning Architecture for Multimodal Precision Health (10P-Health)**
> Submitted to the 14th International Workshop on e-Health Pervasive Wireless
> Applications and Services (e-HPWAS'26), held in conjunction with IEEE WiMob 2026,
---

## Overview

10PH-DVFL is a decoupled Vertical Federated Learning (VFL) architecture designed for communication-efficient multimodal precision health. The architecture separates encoder pre-training (Tier 1) from cross-silo fusion (Tier 2), eliminating per-batch gradient communication and enabling asynchronous participation across healthcare silos.

The repository contains three independent experiments across different medical domains:

| Experiment | Domain | Dataset | Silos | Task |
|------------|--------|---------|-------|------|
| `glioma/` | Neuro-oncology | TCGA Glioma (UCI) | 2 | Brain tumor grade classification |
| `diabetes/` | Endocrinology | Kaggle Diabetes Prediction | 2 | Diabetes diagnosis |
| `midas/` | Dermatology | MRA-MIDAS (Stanford AIMI) | 3 | Skin cancer malignancy classification |

---

## Repository Structure

```
10PH-DVFL/
├── glioma/          # Glioma experiment (tabular, CPU)
│   ├── README.md
│   ├── make_glioma_npz.py
│   ├── pyproject.toml
│   └── ...
├── diabetes/        # Diabetes experiment (tabular, CPU)
│   ├── README.md
│   ├── make_diabetes_npz_cv.py
│   ├── make_iid_partitions.py
│   ├── pyproject.toml
│   └── ...
└── midas/           # MIDAS experiment (images, GPU)
    ├── README.md
    ├── rebuild_aligned_canonical_table_validated.py
    ├── generate_midas_fold_npz.py
    ├── fold_npz/    # Pre-computed 5-fold CV splits
    ├── pyproject.toml
    └── ...
```

---

## Experimental Conditions

All three experiments evaluate the same five conditions:

| Condition | Description |
|-----------|-------------|
| 1 — SplitNN | Standard online VFL baseline (immediate gradient passing) |
| 2 — Decoupled VFL (SUP + DAE/SSL) | Active: supervised pre-training. Passive: self-supervised pre-training |
| 3 — Decoupled VFL (DAE/SSL + DAE/SSL) | Both silos: self-supervised pre-training (fully label-free) |
| 4 — Decoupled VFL (HFL passive) | Passive silo: federated horizontal FL pre-training (K=10, K=20) |
| 5 — Centralized | Upper bound: full joint feature space, no federation |

> **Note:** MIDAS has 4 conditions (no HFL passive pre-training). Self-supervised pre-training uses DAE for Glioma/Diabetes and BYOL for MIDAS.

---

## Architecture

The 10PH-DVFL architecture operates in two tiers:

**Tier 1 — Decoupled Pre-training:**
- Each silo pre-trains its encoder independently with no cross-silo communication
- Active silo: supervised or self-supervised pre-training
- Passive silo: self-supervised (DAE/BYOL) or federated HFL pre-training
- Encoders are frozen after Tier 1

**Tier 2 — Cross-Silo VFL Fusion:**
- Frozen encoders transmit embeddings once to the server (Stage A)
- Server trains a lightweight fusion head on cached embeddings (Stage B)
- No further client communication after Stage A

---

## Requirements

All experiments use:
- Python 3.10
- PyTorch 2.5.1
- Flower 1.26.1 (federated learning framework)
- Ubuntu 22.04.5 LTS via WSL2

**Glioma and Diabetes:** CPU only
**MIDAS:** GPU required for ResNet50 pre-training and feature extraction

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Mahmoud2231991/10PH-DVFL.git
cd 10PH-DVFL
```

### 2. Navigate to the experiment folder

```bash
cd glioma/   # or diabetes/ or midas/
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Follow the experiment-specific README

Each folder has a detailed `README.md` with:
- Dataset download instructions
- Data preparation steps
- Step-by-step running instructions
- Exact hyperparameters used in the paper

---

## Datasets

| Experiment | Dataset | Source |
|------------|---------|--------|
| Glioma | TCGA Glioma Grading Clinical and Mutation Features | [UCI ML Repository](https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset) |
| Diabetes | Diabetes Prediction Dataset | [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) |
| MIDAS | MRA-MIDAS Multimodal Skin Cancer Dataset | [Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/f4c2020f-801a-42dd-a477-a1a8357ef2a5) |

---

## Federated Learning Framework

All federated experiments use **Flower (flwr) version 1.26.1**. Before running any experiment, configure `~/.flwr/config.toml` with the correct number of supernodes:

```toml
[superlink.local-simulation]
options.num-supernodes = 2      # Glioma/Diabetes VFL experiments
# options.num-supernodes = 10   # Glioma/Diabetes HFL K=10
# options.num-supernodes = 20   # Glioma/Diabetes HFL K=20
# options.num-supernodes = 3    # MIDAS experiments
options.backend.name = "ray"
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0
```

See each experiment's `README.md` for experiment-specific Flower configuration.

---

## Reproducibility

- Fixed random seed: **42** across all experiments
- Pre-computed fold splits provided for exact reproduction of train/val/test indices
- All hyperparameters documented in each experiment's `README.md`
- Deterministic PyTorch settings enabled

---

## Environment

| Package | Version |
|---------|---------|
| Python | 3.10 |
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| numpy | 2.2.6 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| flwr | 1.26.1 |
| ray | 2.51.1 |
| scipy | 1.15.3 |
| pillow | 12.0.0 |

---

## Citation

If you use this code, please cite:

```bibtex
@article{10ph-dvfl-2025,
  title     = {10PH-DVFL: A Decoupled Vertical Federated Learning Architecture for Privacy-Preserving Multimodal Personalized Medicine},
  journal   = {Journal of Artificial Intelligence Research},
  year      = {2025}
}
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

This work uses the following publicly available datasets:
- Erdal Tasci (2022). Glioma Grading Clinical and Mutation Features Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5R62J
- Mustafa, M. (2023). Diabetes Prediction Dataset. Kaggle.
- Chiou, A., et al. (2024). MRA-MIDAS: Multimodal Image Dataset for AI-based Skin Cancer. Stanford AIMI. https://doi.org/10.71718/15NZ-JV40
