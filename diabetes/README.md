# Diabetes Experiment — 10PH-DVFL

This folder contains the code for the Diabetes experiment reported in:

> **10PH-DVFL: A Decoupled Vertical Federated Learning Architecture for Privacy-Preserving Multimodal Precision Health (10P-Health)**

---

## Dataset

The Diabetes dataset is derived from the publicly available Kaggle Diabetes Prediction Dataset.

- **Citation:** Mustafa, M. (2023). Diabetes Prediction Dataset. Kaggle.
  https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
- **File needed:** `diabetes_prediction_dataset.csv` (download from the link above)
- **Cohort:** 96,146 samples after removing duplicate records
- **Task:** Binary classification — Diabetes negative (class 0, 91.2%) vs. positive (class 1, 8.8%)
- **Class imbalance:** ~10:1 ratio, addressed through per-fold positive class weighting in all supervised loss computations

---

## Data Preparation

Data preparation is a two-step process:

### Step 1 — Create CV splits and vertical partition

```bash
python make_diabetes_npz_cv.py --csv diabetes_prediction_dataset.csv --out_npz diabetes_vfl_cv.npz
```

This generates `diabetes_vfl_cv.npz` with:
- Vertical partitioning into two silos (active and passive)
- 5-fold stratified cross-validation splits with fixed seed=42
- val_frac=0.2 from outer training fold

### Step 2 — Create HFL IID partitions (Condition 4 only)

```bash
python make_iid_partitions.py --npz diabetes_vfl_cv.npz --Ks 10 20 --out_dir .
```

This generates `partitions_K10_train.json` and `partitions_K20_train.json` in the current directory. Only IID partitioning was used in the paper.

---

## Vertical Partition

| Silo | Features | Labels |
|------|----------|--------|
| Active (Silo 1) | Age, Hypertension, Heart disease, Gender (3 binary encoded columns) — 6 features total | Yes |
| Passive (Silo 2) | BMI, HbA1c level, Blood glucose level, Smoking history (6 binary encoded columns) — 9 features total | No |

The passive silo has no access to labels at any stage.

---

## Experimental Conditions

| Condition | Description | Tier 1 (Pre-training) | Tier 2 (VFL) |
|-----------|-------------|----------------------|--------------|
| 1 — SplitNN | Standard online VFL baseline (immediate gradient passing) | — | `serverapp_vfl_diabetes_splitnn.py` + `clientapp_vfl_diabetes_splitnn.py` |
| 2 — Decoupled VFL (SUP + DAE) | Active: supervised pre-training. Passive: standalone DAE | `pretrain_active_supervised_diabetes.py` + `run_passive_ssl_pretrain_local_diabetes.py` | `serverapp_vfl_diabetes_decoupled.py` + `clientapp_vfl_diabetes_decoupled.py` |
| 3 — Decoupled VFL (DAE + DAE) | Both silos: standalone DAE pre-training (fully label-free) | `run_active_ssl_pretrain_local_diabetes.py` + `run_passive_ssl_pretrain_local_diabetes.py` | `serverapp_vfl_diabetes_decoupled.py` + `clientapp_vfl_diabetes_decoupled.py` |
| 4 — Decoupled VFL (HFL passive) | Passive silo: federated HFL DAE pre-training (K=10 or K=20) | `serverapp_hfl_passive_diabetes.py` + `clientapp_hfl_passive_diabetes.py` | `serverapp_vfl_diabetes_decoupled.py` + `clientapp_vfl_diabetes_decoupled.py` |
| 5 — Centralized | Upper bound: full joint feature space, no federation | — | `train_centralized.py` |

> **Note:** Conditions 2, 3, and 4 all share the same Tier 2 client and server scripts (`clientapp_vfl_diabetes_decoupled.py` and `serverapp_vfl_diabetes_decoupled.py`). The difference between conditions lies in which pre-trained checkpoints are loaded at Tier 2 initialisation.

---

## Hyperparameters

All conditions use Adam optimizer, BCEWithLogitsLoss with per-fold positive class weighting, fixed seed=42, and 5-fold stratified cross-validation.

| Stage | Epochs/Rounds | Batch Size | Learning Rate |
|-------|--------------|------------|---------------|
| SplitNN (Condition 1) | 100 epochs | 256 | 1×10⁻³ |
| Supervised active pre-training (Condition 2) | 100 epochs | 256 | 1×10⁻³ |
| DAE pre-training (Conditions 2, 3) | 100 epochs | 256 | 1×10⁻³ |
| HFL passive pre-training (Condition 4) | 100 rounds, 1 local epoch | 256 | 1×10⁻³ |
| Decoupled Tier 2 — frozen encoders (Conditions 2, 3, 4) | 100 epochs | 2048 | 1×10⁻³ |
| Centralized (Condition 5) | 100 epochs | 256 | 1×10⁻³ |

> **Note:** DAE pre-training uses Gaussian noise σ=0.1, hardcoded in the pre-training scripts.

---

## Running the Experiments

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 2 — Run Tier 1 pre-training

**Conditions 2 and 3 — standalone pre-training (no Flower required):**

```bash
# Condition 2 — supervised active pre-training
python pretrain_active_supervised_diabetes.py

# Condition 2 and 3 — DAE passive pre-training
python run_passive_ssl_pretrain_local_diabetes.py

# Condition 3 — DAE active pre-training
python run_active_ssl_pretrain_local_diabetes.py
```

**Condition 4 — HFL passive pre-training (Flower required):**

Before running, update `pyproject.toml` to point to the HFL scripts:

```toml
[tool.flwr.app.components]
serverapp = "serverapp_hfl_passive_diabetes:app"
clientapp = "clientapp_hfl_passive_diabetes:app"
```

Configure `~/.flwr/config.toml` with the correct number of supernodes:

```toml
[superlink.local-simulation-k10]
options.num-supernodes = 10
options.backend.name = "ray"
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0

[superlink.local-simulation-k20]
options.num-supernodes = 20
options.backend.name = "ray"
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0
```

Run HFL pre-training for all folds and both K values:

```bash
# K=10
for FOLD in 1 2 3 4 5; do
    FOLD=$FOLD flwr run . --federation local-simulation-k10
done

# K=20
for FOLD in 1 2 3 4 5; do
    FOLD=$FOLD flwr run . --federation local-simulation-k20
done
```

### Step 3 — Run Tier 2 VFL (Conditions 1, 2, 3, 4)

Update `pyproject.toml` with the correct server and client scripts for your condition (see Experimental Conditions table). Then configure `~/.flwr/config.toml` with `num-supernodes = 2` for all conditions, and run:

```bash
for FOLD in 1 2 3 4 5; do
    FOLD=$FOLD flwr run . --federation local-simulation
done
```

### Step 4 — Run Centralized baseline (Condition 5)

```bash
python train_centralized.py
```

---

## File Descriptions

| File | Description |
|------|-------------|
| `make_diabetes_npz_cv.py` | Data preparation: vertical split and CV folds |
| `make_iid_partitions.py` | HFL IID partition generation for K=10 and K=20 (Condition 4) |
| `train_centralized.py` | Centralized upper bound (Condition 5) |
| `pretrain_active_supervised_diabetes.py` | Supervised pre-training for active silo (Condition 2) |
| `run_active_ssl_pretrain_local_diabetes.py` | DAE pre-training for active silo (Condition 3) |
| `run_passive_ssl_pretrain_local_diabetes.py` | DAE pre-training for passive silo (Conditions 2, 3) |
| `serverapp_hfl_passive_diabetes.py` | Flower HFL server for passive silo pre-training (Condition 4) |
| `clientapp_hfl_passive_diabetes.py` | Flower HFL client for passive silo pre-training (Condition 4) |
| `serverapp_vfl_diabetes_splitnn.py` | Flower VFL server — SplitNN baseline (Condition 1) |
| `clientapp_vfl_diabetes_splitnn.py` | Flower VFL client — SplitNN baseline (Condition 1) |
| `serverapp_vfl_diabetes_decoupled.py` | Flower VFL server — Decoupled architecture (Conditions 2, 3, 4) |
| `clientapp_vfl_diabetes_decoupled.py` | Flower VFL client — Decoupled architecture (Conditions 2, 3, 4) |
| `pyproject.toml` | Flower app configuration template |
| `requirements.txt` | Python dependencies with exact versions |

---

## Environment

All experiments were run on **Ubuntu 22.04.5 LTS via WSL2 (Windows Subsystem for Linux 2)** on a Windows machine. The diabetes experiment runs on **CPU only** — no GPU is required.

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
pip install -r requirements.txt
pip install -e .
```

> **Note:** Although `torch` was installed with CUDA support (`+cu121`), the diabetes experiment runs on CPU. CUDA is not required to reproduce the results.
