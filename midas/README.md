# MIDAS Experiment — 10PH-DVFL

This folder contains the code for the MIDAS (Dermatology) experiment reported in:

> **10PH-DVFL: A Decoupled Vertical Federated Learning Architecture for Privacy-Preserving Multimodal Personalized Medicine**

---

## Dataset

The MIDAS dataset is obtained from the Melanoma Research Alliance Multimodal Image Dataset for AI-based Skin Cancer (MRA-MIDAS), provided through Stanford AIMI.

- **Citation:** Chiou, A., et al. (2024). MRA-MIDAS: Multimodal Image Dataset for AI-based Skin Cancer. Center for Artificial Intelligence in Medicine and Imaging. https://doi.org/10.71718/15NZ-JV40
- **Access:** Request access through Stanford AIMI: https://aimi.stanford.edu/midas
- **Cohort:** 660 aligned lesions (1,980 images total) after retaining only lesions with all three modalities present
- **Task:** Binary classification — Benign (class 0, n=354, 53.6%) vs. Malignant (class 1, n=306, 46.4%)
- **Three vertical silos:** Dermoscopy (dscope, active), 6-inch photography (6in, passive), 1-foot photography (1ft, passive)

---

## Pre-computed Fold Splits

The `fold_npz/` directory contains **pre-computed 5-fold cross-validation splits** used in the paper:

```
fold_npz/
├── active_dscope_fold{1-5}.npz   # Active silo: image paths + labels
├── passive_6in_fold{1-5}.npz     # Passive silo: image paths only
└── passive_1ft_fold{1-5}.npz     # Passive silo: image paths only
```

**These files contain only image filenames and binary labels — no actual image data.** They are provided to guarantee exact reproducibility of the train/val/test splits used in the paper:

| Split | Size |
|-------|------|
| Train | 423 lesions |
| Val | 105 lesions |
| Test | 132 lesions |

> **Important:** To use these fold files, download the MIDAS images from Stanford AIMI and point the scripts to your local image directory. The filenames in the NPZ files correspond exactly to the MIDAS image filenames.

---

## Data Preparation

If you prefer to rebuild the fold splits from scratch:

### Step 1 — Build aligned canonical table

```bash
python rebuild_aligned_canonical_table_validated.py \
    --excel midas_metadata.xlsx \
    --image_root /path/to/midas/images \
    --out_csv aligned_canonical_table_validated.csv
```

This reads the raw MIDAS Excel metadata file and resolves image paths, producing `aligned_canonical_table_validated.csv` (660 aligned lesion triples).

### Step 2 — Generate per-fold NPZ files

```bash
python generate_midas_fold_npz.py \
    --table_csv aligned_canonical_table_validated.csv \
    --fold_dir fold_npz \
    --out_dir fold_npz \
    --verify
```

> **Note:** If using the pre-computed `fold_npz/` files provided in this repository, you can skip Steps 1 and 2 entirely and proceed directly to pre-training.

---

## Vertical Partition

| Silo | Modality | Role | Labels |
|------|----------|------|--------|
| Active (Silo 1) | Dermoscopy (dscope) | Close-range dermoscopic images | Yes |
| Passive (Silo 2) | 6-inch photography (6in) | Mid-range clinical photography | No |
| Passive (Silo 3) | 1-foot photography (1ft) | Contextual photography | No |

---

## Experimental Conditions

| Condition | Description | Tier 1 (Pre-training) | Tier 2 (VFL) |
|-----------|-------------|----------------------|--------------|
| 1 — SplitNN | Standard online VFL baseline | — | `serverapp_vfl_midas_splitnn_head.py` + `clientapp_vfl_midas_splitnn_proj256.py` |
| 2 — Decoupled VFL (SSL+SSL) | All three silos: BYOL pre-training | `train_byol_resnet50_midas_trainonly.py` → `extract_features_byol_midas.py` | `serverapp_vfl_midas_decoupled_byol.py` + `clientapp_vfl_midas_decoupled_byol.py` |
| 3 — Decoupled VFL (SUP+SSL) | Active: supervised pre-training. Passive: BYOL | `run_active_supervised_pretrain_vfl_folds.py` → `extract_features_sup_active.py` + `extract_features_byol_midas.py` | `serverapp_vfl_midas_decoupled_sup.py` + `clientapp_vfl_midas_decoupled_sup.py` |
| 4 — Centralized | Upper bound: all encoders trained jointly | — | `train_centralized_midas_e2e.py` |

> **Note:** Conditions 2 and 3 share the same Tier 2 pattern but use different pre-trained features. The difference lies in which encoder checkpoints are loaded at Tier 2 initialisation.

---

## Hyperparameters

All conditions use fixed seed=42 and 5-fold stratified cross-validation.

| Stage | Epochs/Rounds | Batch Size | Optimizer | Learning Rate |
|-------|--------------|------------|-----------|---------------|
| BYOL pre-training — all silos (Condition 2) | 100 epochs | 64 | AdamW | 3×10⁻⁴ |
| Supervised active pre-training (Condition 3) | 100 epochs | 64 | AdamW | 1×10⁻⁴ |
| Decoupled Tier 2 — frozen encoders (Conditions 2, 3) | 100 rounds | 64 | Adam | 1×10⁻³ |
| SplitNN (Condition 1) | 100 rounds | 64 | AdamW | 1×10⁻⁴ |
| Centralized (Condition 4) | 100 epochs | 64 | AdamW | 1×10⁻⁴ |

**Additional settings:**
- BYOL: exponential moving average τ annealed 0.996→1.0, cosine annealing schedule
- PCA: 256 components, fitted on training fold only
- Embedding dimension: 256 per silo (768 total concatenation)
- Head architecture: TopMLP — 768→512→1 with ReLU and Dropout(0.2)

---

## Running the Experiments

### Step 1 — Install dependencies

```bash
pip install -e .
```

### Step 2 — Download MIDAS images

Request access and download images from Stanford AIMI: https://aimi.stanford.edu/midas

### Step 3 — Run BYOL pre-training (Condition 2)

```bash
# Pre-train all three silos with BYOL
python train_byol_resnet50_midas_trainonly.py

# Extract and compress features via PCA
python extract_features_byol_midas.py
```

### Step 4 — Run supervised pre-training (Condition 3 only)

```bash
# Pre-train active silo with supervised objective
python run_active_supervised_pretrain_vfl_folds.py

# Extract features
python extract_features_sup_active.py
```

### Step 5 — Configure Flower and run Tier 2

Configure `~/.flwr/config.toml`:

```toml
[superlink.local-simulation]
options.num-supernodes = 3
options.backend.name = "ray"
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0.3
```

Update `pyproject.toml` with the correct server and client scripts, then run:

```bash
for FOLD in 1 2 3 4 5; do
    FOLD_NUM=$FOLD flwr run . --federation local-simulation
done
```

### Step 6 — Run SplitNN baseline (Condition 1)

```bash
python run_vfl_splitnn_5fold.py \
    --fold_npz_dir fold_npz \
    --image_root /path/to/midas/images \
    --out_dir runs_vfl_splitnn
```

---

## File Descriptions

| File | Description |
|------|-------------|
| `rebuild_aligned_canonical_table_validated.py` | Step 1: Build aligned table from raw MIDAS Excel + images |
| `generate_midas_fold_npz.py` | Step 2: Generate per-fold NPZ files from aligned table |
| `train_byol_resnet50_midas_trainonly.py` | BYOL pre-training for all silos (Condition 2) |
| `extract_features_byol_midas.py` | PCA feature extraction after BYOL pre-training |
| `run_active_supervised_pretrain_vfl_folds.py` | Supervised pre-training for active silo (Condition 3) |
| `extract_features_sup_active.py` | PCA feature extraction after supervised pre-training |
| `serverapp_vfl_midas_decoupled_byol.py` | Flower VFL server — Decoupled SSL+SSL (Condition 2) |
| `clientapp_vfl_midas_decoupled_byol.py` | Flower VFL client — Decoupled SSL+SSL (Condition 2) |
| `serverapp_vfl_midas_decoupled_sup.py` | Flower VFL server — Decoupled SUP+SSL (Condition 3) |
| `clientapp_vfl_midas_decoupled_sup.py` | Flower VFL client — Decoupled SUP+SSL (Condition 3) |
| `serverapp_vfl_midas_splitnn_head.py` | Flower VFL server — SplitNN baseline (Condition 1) |
| `clientapp_vfl_midas_splitnn_proj256.py` | Flower VFL client — SplitNN baseline (Condition 1) |
| `run_vfl_splitnn_5fold.py` | Runner script for SplitNN across all 5 folds |
| `train_centralized_midas_e2e.py` | Centralized upper bound (Condition 4) |
| `pyproject.toml` | Flower app configuration template |
| `fold_npz/` | Pre-computed 5-fold CV splits (image paths + labels only, no image data) |

---

## Environment

All experiments were run on **Ubuntu 22.04.5 LTS via WSL2 (Windows Subsystem for Linux 2)** on a Windows machine. The MIDAS experiment requires **GPU** due to ResNet50 image encoders.

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
| pillow | 12.0.0 |

To install all dependencies:

```bash
pip install -e .
```

> **Note:** CUDA-capable GPU is required for BYOL pre-training and feature extraction. The Tier 2 Flower VFL session (frozen encoders + head training) can run on CPU.
