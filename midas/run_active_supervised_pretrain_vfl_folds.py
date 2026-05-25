#!/usr/bin/env python3
"""
run_active_supervised_pretrain_vfl_folds.py
-------------------------------------------
Supervised pretraining of the active silo encoder (dscope ResNet50)
using EXACTLY the same 5-fold CV splits as VFL SplitNN and Centralized.

Fold N encoder is trained on fold N train+val indices ONLY.
Fold N test indices are NEVER seen during pretraining. ← No leakage.

For each fold produces:
    pretrained_active_sup_fold{N}.pt   — full model state dict
    sup_pretrain_fold{N}_log.csv       — per-epoch train/val loss+acc

Usage:
    python run_active_supervised_pretrain_vfl_folds.py \
        --table_csv  aligned_canonical_table_validated.csv \
        --fold_dir   fold_splits_5cv \
        --image_root /path/to/images \
        --out_dir    sup_active_ckpts_vfl_folds \
        --folds      1 2 3 4 5 \
        --epochs     20 \
        --batch_size 32 \
        --device     cuda
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Image path resolver ───────────────────────────────────────────────────────
import os
_image_lookup: dict = {}

def _build_lookup(image_root: str) -> None:
    global _image_lookup
    if _image_lookup:
        return
    for fname in os.listdir(image_root):
        _image_lookup[fname.lower()] = os.path.join(image_root, fname)

def resolve_image_path(image_root: str, filename: str) -> str:
    _build_lookup(image_root)
    filename = str(filename)
    base, _  = os.path.splitext(filename)
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        for candidate in [base + ext, base + "_cropped" + ext]:
            found = _image_lookup.get(candidate.lower())
            if found:
                return found
    raise FileNotFoundError(f"Image not found in {image_root}: {filename}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class FoldDataset(Dataset):
    """
    Loads images by row index from the aligned CSV table.
    Uses fold_indices.npz idx arrays directly — same splits as VFL.
    """
    def __init__(
        self,
        df:         pd.DataFrame,
        img_col:    str,
        label_col:  str,
        indices:    np.ndarray,   # row indices into df
        labels:     np.ndarray,   # float32 labels
        image_root: str,
        transform,
    ) -> None:
        self.image_root = image_root
        self.transform  = transform
        rows            = df.iloc[indices]
        self.filenames  = rows[img_col].astype(str).values
        self.labels     = labels.astype(np.float32)

        assert len(self.filenames) == len(self.labels), \
            f"Mismatch: {len(self.filenames)} paths vs {len(self.labels)} labels"

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        path  = resolve_image_path(self.image_root, self.filenames[idx])
        img   = Image.open(path).convert("RGB")
        x     = self.transform(img)
        y     = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y


# ── Model ─────────────────────────────────────────────────────────────────────
class ActiveEncoder(nn.Module):
    """
    ResNet50 (ImageNet V2) + MLP pretraining head.

    Head mirrors the downstream VFL architecture:
        Linear(2048→512) → ReLU → Dropout(0.2) → Linear(512→1)
    This matches the depth and hidden size of ProjectionMLP+OldMLP in VFL,
    so the backbone learns features compatible with the actual downstream task.

    After pretraining, only backbone_state_dict is kept.
    The head is discarded — VFL attaches ProjectionMLP+OldMLP instead.
    """
    def __init__(self) -> None:
        super().__init__()
        backbone      = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feat_dim = backbone.fc.in_features   # 2048
        backbone.fc   = nn.Identity()
        self.backbone = backbone
        # MLP head — mirrors VFL ProjectionMLP(2048→512) + OldMLP(→1)
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)   # (B, 2048)
        return self.head(z)    # (B, 1)


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> Tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score
    model.eval()
    losses, correct, total = [], 0, 0
    all_probs, all_labels  = [], []
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        losses.append(float(criterion(logits, y).item()))
        probs  = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        preds  = (probs >= 0.5).astype(float)
        labels = y.cpu().numpy().reshape(-1)
        correct += (preds == labels).sum()
        total   += len(labels)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
    val_loss  = float(np.mean(losses))
    val_acc   = correct / max(1, total)
    try:
        val_auroc = float(roc_auc_score(all_labels, all_probs))
    except Exception:
        val_auroc = 0.0
    return val_loss, val_acc, val_auroc


# ── Per-fold training ─────────────────────────────────────────────────────────
def run_fold(
    fold:       int,
    df:         pd.DataFrame,
    fold_dir:   Path,
    image_root: str,
    out_dir:    Path,
    args,
    device:     torch.device,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  FOLD {fold} / {max(args.folds)}")
    print(f"{'='*60}")

    # Load fold indices — SAME file used by VFL SplitNN & Centralized
    fold_path = fold_dir / f"fold{fold}_indices.npz"
    if not fold_path.exists():
        raise FileNotFoundError(f"Fold file not found: {fold_path}")
    d = np.load(fold_path, allow_pickle=True)

    tr_idx  = d["train_idx"].astype(np.int64)
    va_idx  = d["val_idx"].astype(np.int64)
    te_idx  = d["test_idx"].astype(np.int64)
    tr_lbl  = d["train_labels"].astype(np.float32)
    va_lbl  = d["val_labels"].astype(np.float32)

    print(f"[DATA] train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)} "
          f"(test indices NEVER used during pretraining)")
    print(f"[DATA] train pos={int(tr_lbl.sum())} neg={int((tr_lbl==0).sum())} | "
          f"val pos={int(va_lbl.sum())} neg={int((va_lbl==0).sum())}")

    # Transforms — same as VFL SplitNN / Centralized
    tf_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FoldDataset(df, args.img_col, args.label_col,
                           tr_idx, tr_lbl, image_root, tf_train)
    val_ds   = FoldDataset(df, args.img_col, args.label_col,
                           va_idx, va_lbl, image_root, tf_eval)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    set_seed(args.seed + fold)   # fold-specific seed for reproducibility

    model     = ActiveEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.wd)

    # Class-weighted loss — same approach as VFL/Centralized
    pos  = max(1, int(tr_lbl.sum()))
    neg  = max(1, int((tr_lbl == 0).sum()))
    pw   = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    print(f"[MODEL] ResNet50 + Linear head | pos_weight={pw.item():.4f}")

    best_val_auroc = -1.0
    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    best_epoch     = 0
    bad            = 0
    best_ckpt_path = out_dir / f"pretrained_active_sup_fold{fold}.pt"
    log_path       = out_dir / f"sup_pretrain_fold{fold}_log.csv"

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_auroc"])

    for ep in range(1, args.epochs + 1):
        tr_loss                   = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_auroc = eval_epoch(model,  val_loader,   criterion, device)

        print(f"  [Ep {ep:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"val_acc={va_acc:.4f} val_auroc={va_auroc:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, tr_loss, va_loss, va_acc, va_auroc])

        # Save on best val AUROC — consistent with VFL SplitNN and Centralized
        if va_auroc > best_val_auroc + 1e-4:
            best_val_auroc = va_auroc
            best_val_loss  = va_loss
            best_val_acc   = va_acc
            best_epoch     = ep
            bad            = 0
            torch.save({
                "state_dict":          model.state_dict(),
                "backbone_state_dict": model.backbone.state_dict(),
                "head_state_dict":     model.head.state_dict(),
                "fold":       fold,
                "epoch":      ep,
                "val_auroc":  float(va_auroc),
                "val_loss":   float(va_loss),
                "val_acc":    float(va_acc),
                "pos_weight": float(pw.item()),
                "feat_dim":   model.feat_dim,
                "split":      "train+val only — test indices never seen",
            }, best_ckpt_path)
            print(f"  → Saved best checkpoint (val_auroc={va_auroc:.4f})")
        else:
            bad += 1
            if args.patience > 0 and ep >= args.min_epochs and bad >= args.patience:
                print(f"  [EarlyStopping] patience={args.patience} at epoch {ep}")
                break

    print(f"\n[Fold {fold}] DONE | best_epoch={best_epoch} "
          f"val_auroc={best_val_auroc:.4f} val_loss={best_val_loss:.4f} val_acc={best_val_acc:.4f}")
    print(f"[Fold {fold}] Checkpoint → {best_ckpt_path}")

    return {
        "fold":           fold,
        "best_epoch":     best_epoch,
        "best_val_auroc": best_val_auroc,
        "best_val_loss":  best_val_loss,
        "best_val_acc":   best_val_acc,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Supervised active encoder pretraining — exact VFL fold splits"
    )
    ap.add_argument("--table_csv",  required=True,
                    help="aligned_canonical_table_validated.csv")
    ap.add_argument("--fold_dir",   required=True,
                    help="Directory with fold1_indices.npz ... fold5_indices.npz")
    ap.add_argument("--image_root", required=True,
                    help="Root directory of MIDAS images")
    ap.add_argument("--out_dir",    required=True,
                    help="Output directory for checkpoints and logs")
    ap.add_argument("--folds",      nargs="+", type=int, default=[1,2,3,4,5])
    ap.add_argument("--img_col",    default="dscope_file",
                    help="CSV column for dscope image filenames")
    ap.add_argument("--label_col",  default="label",
                    help="CSV column for binary labels")
    ap.add_argument("--epochs",     type=int,   default=20)
    ap.add_argument("--min_epochs", type=int,   default=5,
                    help="Min epochs before early stopping kicks in")
    ap.add_argument("--patience",   type=int,   default=7)
    ap.add_argument("--batch_size", type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--wd",         type=float, default=1e-4)
    ap.add_argument("--num_workers",type=int,   default=0,
                    help="0 for Windows/WSL")
    ap.add_argument("--device",     default="auto")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SETUP] device={device} folds={args.folds}")
    print(f"[SETUP] epochs={args.epochs} patience={args.patience} "
          f"batch={args.batch_size} lr={args.lr} wd={args.wd} seed={args.seed}")
    print(f"[SETUP] Using fold splits from: {args.fold_dir}")
    print(f"[SETUP] ✓ Test indices are NEVER used during pretraining")

    set_seed(args.seed)
    df = pd.read_csv(args.table_csv).reset_index(drop=True)
    print(f"[DATA] Table: {len(df)} rows | columns: {list(df.columns)}")

    for col in [args.img_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    results = []
    for fold in args.folds:
        r = run_fold(fold, df, Path(args.fold_dir),
                     args.image_root, out_dir, args, device)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    summary_path = out_dir / "sup_pretrain_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold","best_epoch","best_val_auroc","best_val_loss","best_val_acc"])
        w.writeheader()
        w.writerows(results)

    for r in results:
        print(f"  Fold {r['fold']}: best_epoch={r['best_epoch']} "
              f"val_auroc={r['best_val_auroc']:.4f} val_loss={r['best_val_loss']:.4f} val_acc={r['best_val_acc']:.4f}")

    mean_auroc = np.mean([r['best_val_auroc'] for r in results])
    print(f"\n  Mean val_auroc = {mean_auroc:.4f}")
    print(f"\n[DONE] Checkpoints saved to {out_dir}")
    print(f"[DONE] Use pretrained_active_sup_fold{{N}}.pt as active encoder in VFL")
    print(f"       Load key: 'backbone_state_dict' for feature extraction")


if __name__ == "__main__":
    main()