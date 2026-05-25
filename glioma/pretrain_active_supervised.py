"""
pretrain_active_supervised.py
==============================
Standalone supervised pretraining for the ACTIVE silo encoder.

Trains BottomMLP (27→32→16) + TopMLP (16→16→8→1) jointly on X1 features
using BCE loss with pos_weight (matching the VFL downstream setup).

Saves per-fold checkpoint containing:
  - bottom_state  : encoder weights  (used by Experiment 1 VFL)
  - head_state    : classifier weights (used by Experiment 2 distillation as teacher)

Usage:
    python pretrain_active_supervised.py [--epochs 100] [--out_dir runs_sup_pretrain]

Outputs (one per fold):
    <out_dir>/pretrained_active_bottom_sup_fold{1..5}.pt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Architecture  — must exactly match clientapp_vfl_glioma_router_client_both_ssl.py
# ---------------------------------------------------------------------------

class BottomMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopMLP(nn.Module):
    """Supervised pretraining head: takes only z1 (out_dim=16) as input.
    NOTE: VFL head takes in_dim=32 (z1+z2 concat) — different size.
    This head is saved as the distillation teacher for Experiment 2."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(1)


# ---------------------------------------------------------------------------
# Data loading  — same as client
# ---------------------------------------------------------------------------

def load_npz(npz_path: str):
    d     = np.load(npz_path, allow_pickle=True)
    X1    = d["X1"].astype(np.float32)
    X2    = d["X2"].astype(np.float32)
    y     = d["y"].astype(np.int64)
    folds = list(d["folds"])
    return X1, X2, y, folds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fold(
    fold: int,
    X1: np.ndarray,
    y: np.ndarray,
    folds: list,
    epochs: int,
    batch_size: int,
    lr: float,
    out_dim: int,
    seed: int,
    out_dir: Path,
    npz_path: str,
    device: torch.device,
):
    split_obj = folds[fold - 1]
    split     = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr_idx    = split["train"].astype(np.int64)
    va_idx    = split["val"].astype(np.int64)

    # Normalize using training set stats — same as client _init_if_needed
    mu = X1[tr_idx].mean(axis=0, keepdims=True)
    sd = X1[tr_idx].std(axis=0, keepdims=True) + 1e-8
    X1s = (X1 - mu) / sd

    X1t = torch.from_numpy(X1s).float().to(device)
    yt  = torch.from_numpy(y.astype(np.float32)).to(device)

    # pos_weight — same as _ensure_vfl_head and pretrain_supervised in client
    y_tr      = yt[torch.from_numpy(tr_idx).long().to(device)]
    pos       = float(y_tr.sum().item())
    neg       = float(y_tr.numel() - y_tr.sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    np.random.seed(seed); torch.manual_seed(seed)
    bottom = BottomMLP(in_dim=X1.shape[1], out_dim=out_dim).to(device)
    head   = TopMLP(in_dim=out_dim).to(device)
    opt    = torch.optim.Adam(
        list(bottom.parameters()) + list(head.parameters()), lr=lr
    )

    rng = np.random.default_rng(seed)
    steps_per_epoch = int(np.ceil(len(tr_idx) / batch_size))

    print(f"\n[Fold {fold}] Starting supervised pretraining for {epochs} epochs...")
    for ep in range(1, epochs + 1):
        bottom.train(); head.train()
        tr_copy = tr_idx.copy()
        rng.shuffle(tr_copy)
        loss_sum = 0.0; n = 0
        for s in range(steps_per_epoch):
            b  = tr_copy[s * batch_size : (s + 1) * batch_size]
            if b.size == 0:
                continue
            bt = torch.from_numpy(b).long().to(device)
            xb = X1t.index_select(0, bt)
            yb = yt.index_select(0, bt)
            opt.zero_grad(set_to_none=True)
            z1     = bottom(xb)
            logits = head(z1)
            loss   = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(b); n += len(b)
        avg_loss = loss_sum / max(n, 1)

        # Validation AUROC every 10 epochs and on last epoch
        if ep % 10 == 0 or ep == epochs:
            from sklearn.metrics import roc_auc_score, average_precision_score
            bottom.eval(); head.eval()
            with torch.no_grad():
                va_bt    = torch.from_numpy(va_idx).long().to(device)
                z1_val   = bottom(X1t.index_select(0, va_bt))
                logits_v = head(z1_val)
                probs_v  = torch.sigmoid(logits_v).cpu().numpy()
                y_val    = y[va_idx]
            try:
                auroc = roc_auc_score(y_val, probs_v)
                prauc = average_precision_score(y_val, probs_v)
            except Exception:
                auroc = prauc = float("nan")
            print(f"  ep={ep:3d}/{epochs}  train_loss={avg_loss:.4f}  val_AUROC={auroc:.4f}  val_PR-AUC={prauc:.4f}")
        else:
            print(f"  ep={ep:3d}/{epochs}  train_loss={avg_loss:.4f}")

    # Save checkpoint — bottom_state + head_state
    ckpt = {
        "bottom_state": bottom.state_dict(),
        "head_state":   head.state_dict(),      # for distillation teacher (Experiment 2)
        "fold":         fold,
        "seed":         seed,
        "out_dim":      out_dim,
        "npz":          str(npz_path),
        "x1_mu":        mu.astype(np.float32),
        "x1_sd":        sd.astype(np.float32),
        "supervised":   True,
        "epochs":       epochs,
    }
    save_path = out_dir / f"pretrained_active_bottom_sup_fold{fold}.pt"
    torch.save(ckpt, save_path)
    print(f"[Fold {fold}] Saved -> {save_path}  (bottom_state + head_state)")
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Supervised pretraining for active encoder")
    parser.add_argument("--npz",        default=str(here / "glioma_aligned_vfl_hfl_cv.npz"))
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--out_dim",    type=int,   default=16)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--folds",      type=int,   nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--out_dir",    default=str(here / "runs_sup_pretrain"))
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    X1, X2, y, folds = load_npz(args.npz)

    print(f"Supervised pretraining — Active encoder")
    print(f"  NPZ: {args.npz}")
    print(f"  X1 shape: {X1.shape}  |  out_dim={args.out_dim}")
    print(f"  epochs={args.epochs}  lr={args.lr}  batch_size={args.batch_size}")
    print(f"  folds: {args.folds}")
    print(f"  out_dir: {out_dir}")
    print(f"  Architecture: BottomMLP({X1.shape[1]}->32->{args.out_dim}) + TopMLP({args.out_dim}->16->8->1)")
    print()

    for fold in args.folds:
        train_fold(
            fold=fold,
            X1=X1, y=y, folds=folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            out_dim=args.out_dim,
            seed=args.seed,
            out_dir=out_dir,
            npz_path=args.npz,
            device=device,
        )

    print("\nAll folds done.")
    print(f"Checkpoints in: {out_dir}")
    print("Each file contains: bottom_state (for VFL) + head_state (for distillation teacher)")


if __name__ == "__main__":
    main()