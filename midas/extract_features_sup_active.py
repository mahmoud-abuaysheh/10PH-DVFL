#!/usr/bin/env python3
"""
extract_features_sup_active.py
-------------------------------
Extracts dscope features from the supervised pretrained active encoder
and saves them in EXACTLY the same format as features_byol NPZ files:
    X_train  (423, 256) float32  — PCA-reduced features
    X_val    (105, 256) float32
    X_test   (132, 256) float32
    pca_variance (1,)   float32  — explained variance ratio

Output: features_dscope_sup_fold{N}.npz  (one per fold)

These replace features_dscope_fold{N}.npz for the active silo only.
Passive silos (6in, 1ft) keep their original BYOL features unchanged.

Usage:
    python extract_features_sup_active.py \
        --ckpt_dir   sup_active_ckpts_vfl_folds \
        --npz_dir    fold_npz \
        --image_root "C:\path\to\images" \
        --out_dir    features_sup_active \
        --folds      1 2 3 4 5 \
        --device     cuda
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import models, transforms


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Image resolver ────────────────────────────────────────────────────────────
_lookup: dict = {}

def _build_lookup(image_root: str) -> None:
    global _lookup
    if _lookup:
        return
    for fname in os.listdir(image_root):
        _lookup[fname.lower()] = os.path.join(image_root, fname)

def resolve_image(image_root: str, filename: str) -> str:
    _build_lookup(image_root)
    base, _ = os.path.splitext(str(filename))
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        for candidate in [base + ext, base + "_cropped" + ext]:
            found = _lookup.get(candidate.lower())
            if found:
                return found
    raise FileNotFoundError(f"Image not found: {filename}")


# ── Backbone loader ───────────────────────────────────────────────────────────
def load_backbone(ckpt_path: Path, device: torch.device) -> nn.Module:
    """
    Load ResNet50 backbone from supervised pretrain checkpoint.
    Uses backbone_state_dict key — head is discarded.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    backbone     = models.resnet50(weights=None)
    backbone.fc  = nn.Identity()

    if "backbone_state_dict" in ckpt:
        backbone.load_state_dict(ckpt["backbone_state_dict"])
        print(f"  Loaded backbone_state_dict from {ckpt_path.name}")
    elif "state_dict" in ckpt:
        # Strip head keys, keep only backbone
        sd = {k.replace("backbone.", ""): v
              for k, v in ckpt["state_dict"].items()
              if k.startswith("backbone.")}
        backbone.load_state_dict(sd)
        print(f"  Loaded backbone from state_dict (stripped head) from {ckpt_path.name}")
    else:
        raise KeyError(f"Cannot find backbone weights in {ckpt_path.name}. "
                       f"Keys: {list(ckpt.keys())}")

    backbone = backbone.to(device)
    backbone.eval()
    return backbone


# ── Feature extraction ────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(
    backbone:   nn.Module,
    filenames:  np.ndarray,
    image_root: str,
    transform,
    device:     torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract 2048-D backbone features for a list of image filenames."""
    all_feats = []
    n = len(filenames)

    for start in range(0, n, batch_size):
        batch_files = filenames[start:start + batch_size]
        imgs = []
        for fname in batch_files:
            path = resolve_image(image_root, str(fname))
            imgs.append(transform(Image.open(path).convert("RGB")))

        x      = torch.stack(imgs).to(device)
        feats  = backbone(x)                         # (B, 2048)
        all_feats.append(feats.cpu().numpy())

        if (start // batch_size) % 5 == 0:
            print(f"    Extracted {min(start+batch_size, n)}/{n} images...")

    return np.concatenate(all_feats, axis=0).astype(np.float32)  # (N, 2048)


# ── PCA reduction ─────────────────────────────────────────────────────────────
def fit_pca_and_reduce(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
    n_components: int = 256,
) -> tuple:
    """
    Fit PCA on train features only, transform all splits.
    Matches BYOL feature extraction pipeline exactly.
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train).astype(np.float32)
    X_val_pca   = pca.transform(X_val).astype(np.float32)
    X_test_pca  = pca.transform(X_test).astype(np.float32)

    var_explained = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA: {X_train.shape[1]}D → {n_components}D | "
          f"variance explained={var_explained:.4f}")

    return X_train_pca, X_val_pca, X_test_pca, var_explained


# ── Per-fold extraction ───────────────────────────────────────────────────────
def run_fold(
    fold:       int,
    ckpt_dir:   Path,
    npz_dir:    Path,
    image_root: str,
    out_dir:    Path,
    args,
    device:     torch.device,
) -> None:
    print(f"\n{'='*60}")
    print(f"  FOLD {fold}")
    print(f"{'='*60}")

    # Load supervised backbone for this fold
    ckpt_path = ckpt_dir / f"pretrained_active_sup_fold{fold}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    backbone = load_backbone(ckpt_path, device)

    # Load image paths from fold NPZ (active dscope)
    npz_path = npz_dir / f"active_dscope_fold{fold}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Fold NPZ not found: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)

    paths_train = d["paths_train"].astype(str)
    paths_val   = d["paths_val"].astype(str)
    paths_test  = d["paths_test"].astype(str)

    print(f"  train={len(paths_train)} val={len(paths_val)} test={len(paths_test)}")

    # Eval transform — same as VFL/Centralized eval transform
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Extract 2048-D features
    print("  Extracting train features...")
    X_train_raw = extract_features(backbone, paths_train, image_root,
                                   tf, device, args.batch_size)
    print("  Extracting val features...")
    X_val_raw   = extract_features(backbone, paths_val,   image_root,
                                   tf, device, args.batch_size)
    print("  Extracting test features...")
    X_test_raw  = extract_features(backbone, paths_test,  image_root,
                                   tf, device, args.batch_size)

    print(f"  Raw features: train={X_train_raw.shape} "
          f"val={X_val_raw.shape} test={X_test_raw.shape}")

    # PCA to 256-D — fit on train only, same as BYOL pipeline
    X_train, X_val, X_test, var = fit_pca_and_reduce(
        X_train_raw, X_val_raw, X_test_raw, n_components=args.pca_dim
    )

    # Save in exact same format as BYOL feature NPZs
    out_path = out_dir / f"features_dscope_sup_fold{fold}.npz"
    np.savez(
        out_path,
        X_train      = X_train,
        X_val        = X_val,
        X_test       = X_test,
        pca_variance = np.array([var], dtype=np.float32),
    )
    print(f"  Saved → {out_path}")
    print(f"  Shape: X_train={X_train.shape} X_val={X_val.shape} "
          f"X_test={X_test.shape}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract supervised active encoder features (BYOL NPZ format)"
    )
    ap.add_argument("--ckpt_dir",   required=True,
                    help="Directory with pretrained_active_sup_fold{N}.pt files")
    ap.add_argument("--npz_dir",    required=True,
                    help="fold_npz directory with active_dscope_fold{N}.npz files")
    ap.add_argument("--image_root", required=True,
                    help="Root directory of MIDAS images")
    ap.add_argument("--out_dir",    required=True,
                    help="Output directory for feature NPZ files")
    ap.add_argument("--folds",      nargs="+", type=int, default=[1,2,3,4,5])
    ap.add_argument("--pca_dim",    type=int,   default=256,
                    help="PCA output dimension (must match BYOL features=256)")
    ap.add_argument("--batch_size", type=int,   default=32)
    ap.add_argument("--device",     default="auto")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SETUP] device={device} folds={args.folds} pca_dim={args.pca_dim}")
    print(f"[SETUP] Output: {out_dir}")
    print(f"[SETUP] Passive silos (6in, 1ft) keep original BYOL features unchanged")

    set_seed(args.seed)

    for fold in args.folds:
        run_fold(
            fold       = fold,
            ckpt_dir   = Path(args.ckpt_dir),
            npz_dir    = Path(args.npz_dir),
            image_root = args.image_root,
            out_dir    = out_dir,
            args       = args,
            device     = device,
        )

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")
    print(f"Generated files:")
    for fold in args.folds:
        p = out_dir / f"features_dscope_sup_fold{fold}.npz"
        print(f"  {p}")
    print(f"\nNext step: run BYOL server script pointing to:")
    print(f"  --art_dir {out_dir}  (for dscope)")
    print(f"  Keep original features_byol dir for 6in and 1ft")


if __name__ == "__main__":
    main()