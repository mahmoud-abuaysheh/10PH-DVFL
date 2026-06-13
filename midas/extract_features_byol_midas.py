#!/usr/bin/env python3
"""
extract_features_byol_midas.py
-------------------------------
Step 2: Extract BYOL features + PCA reduction for MIDAS decoupled VFL.

Reads the final epoch BYOL encoder checkpoint (byol_{modality}_fold{N}.pt)
and extracts 2048-D ResNet50 features, then reduces to 256-D via PCA
fitted on the training fold only.

Outputs per fold and modality:
    features_{modality}_fold{N}.npz
        X_train  (423, 256) float32
        X_val    (105, 256) float32
        X_test   (132, 256) float32
        pca_variance (1,)   float32

Usage:
    python extract_features_byol_midas.py \
        --byol_dir     byol_checkpoints \
        --fold_npz_dir fold_npz \
        --image_root   /path/to/midas/images \
        --out_dir      byol_features_pca \
        --pca_dim      256 \
        --folds        1 2 3 4 5 \
        --modalities   dscope 6in 1ft \
        --batch_size   64
"""
from __future__ import annotations
import os, argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Image lookup (WSL/NTFS fix — scan once)
# ---------------------------------------------------------------------------

def build_image_lookup(image_root: str) -> Dict[str, str]:
    lookup = {}
    for fname in os.listdir(image_root):
        lookup[fname.lower()] = os.path.join(image_root, fname)
    print(f"[DATA] Lookup built: {len(lookup)} files indexed")
    return lookup

def resolve_from_lookup(lookup: Dict[str, str], filename: str) -> str:
    base, _ = os.path.splitext(str(filename))
    for ext in [".jpg", ".jpeg"]:
        for candidate in [base + ext, base + "_cropped" + ext]:
            hit = lookup.get(candidate.lower())
            if hit:
                return hit
    raise FileNotFoundError(f"Not found in lookup: {filename}")

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

NPZ_NAME = {"dscope": "active_dscope", "6in": "passive_6in", "1ft": "passive_1ft"}

def load_encoder(byol_dir: str, modality: str, fold: int, device: torch.device) -> nn.Module:
    # Load final epoch checkpoint — used for feature extraction
    ckpt = Path(byol_dir) / f"byol_{modality}_fold{fold}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Not found: {ckpt}")
    base = models.resnet50(weights=None)
    encoder = nn.Sequential(*list(base.children())[:-1])
    encoder.load_state_dict(torch.load(ckpt, map_location=device))
    encoder.eval().to(device)
    for p in encoder.parameters(): p.requires_grad_(False)
    print(f"[ENCODER] Loaded {ckpt.name}")
    return encoder

# ---------------------------------------------------------------------------
# Dataset (eval transform, absolute paths pre-resolved)
# ---------------------------------------------------------------------------

class FeatureDataset(Dataset):
    def __init__(self, paths: List[str], lookup: Dict[str, str]):
        self.abs_paths = [resolve_from_lookup(lookup, p) for p in paths]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.abs_paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.abs_paths[idx]).convert("RGB"))

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(encoder, paths, lookup, device, batch_size=64, num_workers=0):
    ds     = FeatureDataset(paths, lookup)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type=="cuda"))
    feats = []
    for batch in loader:
        feats.append(torch.flatten(encoder(batch.to(device)), 1).cpu().numpy())
    return np.vstack(feats).astype(np.float32)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device={device} | PCA dim={args.pca_dim}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build image lookup ONCE for all modalities/folds
    lookup = build_image_lookup(args.image_root)

    for fold in args.folds:
        for modality in args.modalities:
            print(f"\n{'='*55}")
            print(f"[EXTRACT] fold={fold} | modality={modality}")
            print(f"{'='*55}")

            # Load fold paths
            npz_path = Path(args.fold_npz_dir) / f"{NPZ_NAME[modality]}_fold{fold}.npz"
            d = np.load(npz_path, allow_pickle=True)
            paths_train = d["paths_train"].astype(str).tolist()
            paths_val   = d["paths_val"].astype(str).tolist()
            paths_test  = d["paths_test"].astype(str).tolist()
            print(f"[DATA] train={len(paths_train)} val={len(paths_val)} test={len(paths_test)}")

            # Load encoder
            encoder = load_encoder(args.byol_dir, modality, fold, device)

            # Extract 2048-D features
            print("[EXTRACT] train..."); X_tr = extract_features(encoder, paths_train, lookup, device, args.batch_size, args.num_workers)
            print("[EXTRACT] val...");   X_va = extract_features(encoder, paths_val,   lookup, device, args.batch_size, args.num_workers)
            print("[EXTRACT] test...");  X_te = extract_features(encoder, paths_test,  lookup, device, args.batch_size, args.num_workers)
            print(f"[EXTRACT] Raw: train={X_tr.shape} val={X_va.shape} test={X_te.shape}")

            # PCA — fit on train only
            print(f"[PCA] Fitting on train → {args.pca_dim}-D ...")
            pca = PCA(n_components=args.pca_dim, random_state=42)
            pca.fit(X_tr)
            var = pca.explained_variance_ratio_.sum()
            print(f"[PCA] Variance explained: {var*100:.1f}%")

            X_tr_r = pca.transform(X_tr).astype(np.float32)
            X_va_r = pca.transform(X_va).astype(np.float32)
            X_te_r = pca.transform(X_te).astype(np.float32)

            # Save
            out_path = out_dir / f"features_{modality}_fold{fold}.npz"
            np.savez(out_path,
                     X_train=X_tr_r, X_val=X_va_r, X_test=X_te_r,
                     pca_variance=np.array([var]))
            print(f"[SAVE] {out_path} | train={X_tr_r.shape} val={X_va_r.shape} test={X_te_r.shape}")

            del encoder
            torch.cuda.empty_cache()

    # Verification
    print(f"\n{'='*55}")
    print("VERIFICATION")
    print(f"{'='*55}")
    for fold in args.folds:
        for modality in args.modalities:
            p = out_dir / f"features_{modality}_fold{fold}.npz"
            if not p.exists():
                print(f"  MISSING: {p.name}")
                continue
            d = np.load(p)
            var = float(d["pca_variance"][0])
            print(f"  fold{fold} {modality:8s} | train={d['X_train'].shape} "
                  f"val={d['X_val'].shape} test={d['X_test'].shape} | "
                  f"PCA variance={var*100:.1f}%")

    print(f"\n[DONE] All features saved to {out_dir}/")
    print(f"[DONE] Next: train decoupled VFL classifier")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--byol_dir",      required=True,
                   help="Directory containing byol_{modality}_fold{N}.pt files")
    p.add_argument("--fold_npz_dir",  required=True,
                   help="Directory containing fold NPZ files (fold_npz/)")
    p.add_argument("--image_root",    required=True,
                   help="Root directory of MIDAS images")
    p.add_argument("--out_dir",       required=True,
                   help="Output directory for feature NPZ files")
    p.add_argument("--pca_dim",       type=int,   default=256)
    p.add_argument("--folds",         nargs="+",  type=int, default=[1,2,3,4,5])
    p.add_argument("--modalities",    nargs="+",  default=["dscope","6in","1ft"])
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--num_workers",   type=int,   default=0)
    run_all(p.parse_args())

if __name__ == "__main__":
    main()
