# Glioma Experiment — 10PH-DVFL

This folder contains the code for the Glioma experiment reported in:

> **10PH-DVFL: A Decoupled Vertical Federated Learning Architecture for Privacy-Preserving Multimodal Personalized Medicine**

---

## Dataset

The Glioma dataset is obtained from the UCI Machine Learning Repository, constructed from the TCGA-LGG and TCGA-GBM brain glioma projects.

- **Citation:** Erdal Tasci (2022). Glioma Grading Clinical and Mutation Features Dataset. UCI Machine Learning Repository.
  https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset
- **File needed:** `TCGA_InfoWithGrade.csv` (download from the link above)
- **Cohort:** 838 patients after removing one record with missing values
- **Task:** Binary classification — Low-Grade Glioma (LGG, class 0) vs. Glioblastoma Multiforme (GBM, class 1)

---

## Data Preparation

After downloading `TCGA_InfoWithGrade.csv`, run:

```bash
python make_glioma_npz.py --csv TCGA_InfoWithGrade.csv --out glioma_aligned_vfl_hfl_cv.npz
```

This generates `glioma_aligned_vfl_hfl_cv.npz`, which is required by all training scripts. The script performs:

- Vertical partitioning into two silos of 13 features each (active and passive)
- 5-fold stratified cross-validation splits with fixed seed=42
- IID partitioning for HFL pre-training with K=10 and K=20

**Notes:**
- Only IID partitioning was used in the paper (K=10 and K=20). Non-IID (Dirichlet) partitioning and K=5, K=15 are also supported but were not used in the reported results.
- DAE pre-training uses Gaussian noise σ=0.1, hardcoded in the pre-training scripts.

---

## Vertical Partition

| Silo | Features | Labels |
|------|----------|--------|
| Active (Silo 1) | Gender, Age at diagnosis, Race, IDH1, TP53, ATRX, PTEN, EGFR, CIC, MUC16, PIK3CA, NF1, PIK3R1 | Yes |
| Passive (Silo 2) | Gender, Age at diagnosis, Race, FUBP1, RB1, NOTCH1, BCOR, CSMD3, SMARCA4, GRIN2A, IDH2, FAT4, PDGFRA | No |

Both silos standardize features using training-fold means and standard deviations.

---

## Experimental Conditions

| Condition | Description | Tier 1 (Pre-training) | Tier 2 (VFL) |
|-----------|-------------|----------------------|--------------|
| 1 — SplitNN | Standard online VFL baseline (immediate gradient passing) | — | `serverapp_vfl_glioma_immediate.py` + `clientapp_vfl_glioma_immediate.py` |
| 2 — Decoupled VFL (SUP + DAE) | Active: supervised pre-training. Passive: standalone DAE | `pretrain_active_supervised.py` + `run_passive_ssl_pretrain_local.py` | `serverapp_vfl_glioma_router.py` + `clientapp_vfl_glioma_router_client_both_ssl.py` |
| 3 — Decoupled VFL (DAE + DAE) | Both silos: standalone DAE pre-training (fully label-free) | `run_active_ssl_pretrain_local.py` + `run_passive_ssl_pretrain_local.py` | `serverapp_vfl_glioma_router.py` + `clientapp_vfl_glioma_router_client_both_ssl.py` |
| 4 — Decoupled VFL (HFL passive) | Passive silo: federated HFL DAE pre-training (K=10 or K=20) | `serverapp_hfl_passive_glioma.py` + `clientapp_hfl_passive_glioma.py` | `serverapp_vfl_glioma_router.py` + `clientapp_vfl_glioma_router_client_both_ssl.py` |
| 5 — Centralized | Upper bound: full joint feature space, no federation | — | `train_centralized_glioma.py` |

> **Note:** Conditions 2, 3, and 4 all share the same Tier 2 client and server scripts (`clientapp_vfl_glioma_router_client_both_ssl.py` and `serverapp_vfl_glioma_router.py`). The difference between conditions lies in which pre-trained checkpoints are loaded at Tier 2 initialisation.

> **Note:** In this codebase, "immediate" refers to the SplitNN baseline (immediate gradient passing). Scripts named `_immediate_` implement the SplitNN protocol.

---

## Hyperparameters

All conditions use Adam optimizer, fixed seed=42, and 5-fold stratified cross-validation.

| Stage | Epochs/Rounds | Batch Size | Learning Rate |
|-------|--------------|------------|---------------|
| SplitNN (Condition 1) | 100 epochs | 64 | 1×10⁻³ |
| Supervised active pre-training (Condition 2) | 100 epochs | 64 | 1×10⁻³ |
| DAE pre-training (Conditions 2, 3) | 100 epochs | 64 | 1×10⁻³ |
| HFL passive pre-training (Condition 4) | 100 rounds, 1 local epoch | 64 | 1×10⁻³ |
| Decoupled Tier 2 — frozen encoders (Conditions 2, 3, 4) | 100 epochs | 64 | 1×10⁻³ |
| Centralized (Condition 5) | 100 epochs | 64 | 1×10⁻³ |

---

## Running the Experiments

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 2 — Configure Flower

Before running any Flower experiment, configure `~/.flwr/config.toml` with the correct number of supernodes for your experiment:

```toml
[superlink.local-simulation]
options.num-supernodes = 2      # VFL / SplitNN / Decoupled experiments
# options.num-supernodes = 10   # HFL K=10
# options.num-supernodes = 20   # HFL K=20
options.backend.name = "ray"
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0
```

### Step 3 — Update pyproject.toml

In `pyproject.toml`, replace `YOUR_SERVER_SCRIPT` and `YOUR_CLIENT_SCRIPT` with the scripts for your experiment (see Experimental Conditions table above). For example, for SplitNN:

```toml
[tool.flwr.app.components]
serverapp = "serverapp_vfl_glioma_immediate:app"
clientapp = "clientapp_vfl_glioma_immediate:app"
```

### Step 4 — Run per fold

```bash
for FOLD in 1 2 3 4 5; do
    FOLD=$FOLD flwr run . --federation local-simulation
done
```

For HFL passive pre-training (Condition 4), run with K=10 first, then K=20 by updating `num-supernodes` in `~/.flwr/config.toml` accordingly.

---

## File Descriptions

| File | Description |
|------|-------------|
| `make_glioma_npz.py` | Data preparation: vertical split, CV folds, HFL partitions |
| `train_centralized_glioma.py` | Centralized upper bound (Condition 5) |
| `pretrain_active_supervised.py` | Supervised pre-training for active silo (Condition 2) |
| `run_active_ssl_pretrain_local.py` | DAE pre-training for active silo (Condition 3) |
| `run_passive_ssl_pretrain_local.py` | DAE pre-training for passive silo (Conditions 2, 3) |
| `serverapp_hfl_passive_glioma.py` | Flower HFL server for passive silo pre-training (Condition 4) |
| `clientapp_hfl_passive_glioma.py` | Flower HFL client for passive silo pre-training (Condition 4) |
| `serverapp_vfl_glioma_immediate.py` | Flower VFL server — SplitNN baseline (Condition 1) |
| `clientapp_vfl_glioma_immediate.py` | Flower VFL client — SplitNN baseline (Condition 1) |
| `serverapp_vfl_glioma_router.py` | Flower VFL server — Decoupled architecture (Conditions 2, 3, 4) |
| `clientapp_vfl_glioma_router_client_both_ssl.py` | Flower VFL client — Decoupled architecture (Conditions 2, 3, 4) |
| `pyproject.toml` | Flower app configuration template |

---

## Environment

All experiments were run on **Ubuntu 22.04.5 LTS via WSL2 (Windows Subsystem for Linux 2)** on a Windows machine. The glioma experiment runs on **CPU only** — no GPU is required.

| Package | Version |
|---------|---------|
| Python | 3.10 |
| torch | 2.5.1+cu121 |
| numpy | 2.2.6 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| flwr | 1.26.1 |
| ray | 2.51.1 |
| scipy | 1.15.3 |

To install all dependencies:

```bash
pip install -e .
```

> **Note:** Although `torch` was installed with CUDA support (`+cu121`), the glioma experiment runs on CPU. CUDA is not required to reproduce the results.
