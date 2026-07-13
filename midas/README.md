# MIDAS Experiment — 10PH-DVFL

This folder contains the code for the MIDAS (Dermatology) experiment reported in:

> **10PH-DVFL: A Communication-Efficient Decoupled Vertical Federated Learning Architecture for Multimodal Precision Health (10P-Health)**

---

## Dataset

The MIDAS dataset is obtained from the Melanoma Research Alliance Multimodal Image Dataset for AI-based Skin Cancer (MRA-MIDAS), provided through Stanford AIMI.

- **Citation:** Chiou, A., et al. (2024). MRA-MIDAS: Multimodal Image Dataset for AI-based Skin Cancer. Center for Artificial Intelligence in Medicine and Imaging. https://doi.org/10.71718/15NZ-JV40
- **Access:** Download from Stanford AIMI: https://stanfordaimi.azurewebsites.net/datasets/f4c2020f-801a-42dd-a477-a1a8357ef2a5
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

> **Note on canonical selection:** Each patient may have multiple images per modality. This script selects one canonical image per modality per lesion deterministically by taking the first record after sorting by filename. Only lesions with all three modalities physically present and resolvable on disk are retained.

### Step 2 — Generate per-fold NPZ files (optional — for transparency only)

```bash
python generate_midas_fold_npz.py \
    --table_csv aligned_canonical_table_validated.csv \
    --fold_dir fold_splits_5cv \
    --out_dir fold_npz \
    --verify
```

> **Note:** The pre-computed `fold_npz/` files are already provided in this repository. You do NOT need to run this script to reproduce the paper results. It is provided for transparency only and requires `fold_splits_5cv/fold{N}_indices.npz` files which are not included in this repository.

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

| Stage | Epochs/Rounds | Batch Size | Optimizer | Learning Rate | Early Stopping |
|-------|--------------|------------|-----------|---------------|----------------|
| BYOL pre-training — all silos (Condition 2) | 20 epochs | 64 | AdamW | 3×10⁻⁴ | No — fixed schedule, final checkpoint used |
| Supervised active pre-training (Condition 3) | 20 epochs | 64 | AdamW | 1×10⁻⁴ | Yes — patience=7 |
| Decoupled Tier 2 — frozen encoders (Conditions 2, 3) | 20 rounds | 64 | Adam | 1×10⁻⁴ | Yes — patience=7 |
| SplitNN (Condition 1) | 20 rounds | 64 | AdamW | 1×10⁻⁴ | Yes — patience=7 |
| Centralized (Condition 4) | 20 epochs | 64 | AdamW | 1×10⁻⁴ | Yes — patience=7 |

**Additional settings:**
- BYOL: exponential moving average τ annealed 0.996→1.0, cosine annealing schedule, final epoch checkpoint used for feature extraction
- PCA: 256 components, fitted on training fold only
- Embedding dimension: 256 per silo (768 total concatenation)
- Head architecture: TopMLP — 768→512→1 with ReLU and Dropout(0.2)

---

## Running the Experiments

### Step 1 — Install dependencies

```bash
# Clone the repo and navigate to the midas folder
git clone https://github.com/Mahmoud2231991/10PH-DVFL.git
cd 10PH-DVFL/midas

pip install -r requirements.txt
pip install -e .
```

> **Important:** All commands in this README must be run from the `midas/` directory so that `fold_npz/` is found correctly.

### Step 2 — Download MIDAS images

Download images from Stanford AIMI: https://stanfordaimi.azurewebsites.net/datasets/f4c2020f-801a-42dd-a477-a1a8357ef2a5

### Step 3 — Run BYOL pre-training (Condition 2)

BYOL must be run separately for each modality and each fold (15 runs total):

```bash
for FOLD in 1 2 3 4 5; do
    for MODALITY in dscope 6in 1ft; do
        python train_byol_resnet50_midas_trainonly.py \
            --fold_npz_dir fold_npz \
            --image_root /path/to/midas/images \
            --modality $MODALITY \
            --fold $FOLD \
            --out_dir byol_checkpoints \
            --epochs 20 \
            --batch_size 64 \
            --lr 3e-4 \
            --amp
    done
done
```

Then extract and compress features via PCA (256 components, fitted on training fold only):

```bash
python extract_features_byol_midas.py \
    --byol_dir byol_checkpoints \
    --fold_npz_dir fold_npz \
    --image_root /path/to/midas/images \
    --out_dir byol_features_pca \
    --batch_size 64
```

### Step 4 — Run supervised pre-training (Condition 3 only)

Pre-train the active silo (dscope) with a supervised objective across all folds:

```bash
python run_active_supervised_pretrain_vfl_folds.py \
    --fold_npz_dir fold_npz \
    --image_root /path/to/midas/images \
    --out_dir sup_active_ckpts \
    --epochs 20 \
    --batch_size 64
```

Then extract features:

```bash
python extract_features_sup_active.py \
    --ckpt_dir sup_active_ckpts \
    --fold_npz_dir fold_npz \
    --image_root /path/to/midas/images \
    --out_dir features_sup_active \
    --batch_size 64
```

> **Note:** For Condition 3, passive silos (6in, 1ft) reuse the BYOL features from Step 3.

### Step 5 — Configure Flower and run Tier 2

Configure `~/.flwr/config.toml`:

```toml
[superlink.local-simulation]
options.num-supernodes = 3
options.backend.name = "ray"
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0.3
```

Update `pyproject.toml` with the correct server and client scripts, then run per fold:

```bash
for FOLD in 1 2 3 4 5; do
    FOLD_NUM=$FOLD flwr run . --federation local-simulation
done
```

### Step 6 — Run SplitNN baseline (Condition 1)

Update `pyproject.toml` with the SplitNN scripts:

```toml
[tool.flwr.app.components]
serverapp = "serverapp_vfl_midas_splitnn_head:app"
clientapp = "clientapp_vfl_midas_splitnn_proj256:app"
```

Then run per fold:

```bash
for FOLD in 1 2 3 4 5; do
    FOLD_NUM=$FOLD \
    IMAGE_ROOT=/path/to/midas/images \
    FOLD_NPZ_DIR=fold_npz \
    OUT_DIR=runs_midas_splitnn \
    BATCH_SIZE=64 \
    flwr run . --federation local-simulation
done
```

### Step 7 — Run Centralized baseline (Condition 4)

```bash
python train_centralized_midas_e2e.py \
    --fold_npz_dir fold_npz \
    --image_root /path/to/midas/images \
    --out_dir runs_centralized \
    --batch_size 64
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

> **Note on GPU requirements:**
> - **BYOL pre-training and feature extraction:** GPU required (ResNet50)
> - **Supervised active pre-training:** GPU required (ResNet50)
> - **SplitNN (Condition 1) and Centralized (Condition 4):** GPU required — ResNet50 backbones remain fully trainable during training
> - **Decoupled Tier 2 (Conditions 2, 3):** Can run on CPU — encoders are frozen and only the lightweight fusion head is trained on pre-extracted PCA features

> **Note on dimensionality reduction:**
> - **Decoupled VFL (Conditions 2, 3):** After pre-training, each frozen ResNet50 backbone extracts 2,048-dimensional features. **PCA** (256 components, fitted on training fold only) reduces these to 256 dimensions. Validation and test splits are transformed using the training-derived PCA projection. This replaces the trainable projection layer entirely.
> - **SplitNN (Condition 1) and Centralized (Condition 4):** No PCA is used. Instead, a trainable **ProjectionMLP** (2,048 → 512 → 256, with LayerNorm, GELU, and Dropout(0.2)) is trained end-to-end alongside the ResNet50 backbone during the experiment.
