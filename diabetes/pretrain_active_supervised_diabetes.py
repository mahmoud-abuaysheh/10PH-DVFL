#!/usr/bin/env python3
"""
Supervised pretraining for the *active* (label-holding) silo in the diabetes decoupled VFL setup.

Trains:
  BottomMLP_Paper (in_dim -> 16 -> 8) + local linear head (8 -> 1)

on ACTIVE features X1 and labels y, using the train split of the selected fold.

Saves a "teacher" checkpoint containing BOTH bottom encoder + supervised head
so the active client can later serve teacher logits to the server (knowledge distillation),
while keeping VFL decoupled (no backprop to clients).

Output checkpoint (per fold):
  <out_dir>/pretrained_active_sup_teacher_fold{fold}.pt

Checkpoint format:
  {
    "bottom_state": bottom.state_dict(),
    "head_state": head.state_dict(),
    "meta": meta
  }

- bottom_state is compatible with your client's _unwrap_state_dict() logic
- head_state will be used by the active client to compute teacher logits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn


# Must match your client architecture
class BottomMLP_Paper(nn.Module):
    """Baseline bottom: in_dim -> 16 -> 8 with ReLU."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    X1 = d["X1"].astype(np.float32)
    y = d["y"].astype(np.int64)
    folds = list(d["folds"])
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return X1, y, folds, meta


def standardize_using_train(X: np.ndarray, tr_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X[tr_idx].mean(axis=0, keepdims=True)
    sd = X[tr_idx].std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict_prob(bottom: nn.Module, head: nn.Module, X: torch.Tensor, batch: int = 2048) -> np.ndarray:
    bottom.eval()
    head.eval()
    out = []
    for i in range(0, X.shape[0], batch):
        xb = X[i : i + batch]
        logits = head(bottom(xb)).view(-1)
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_one_fold(
    X1: np.ndarray,
    y: np.ndarray,
    split: Dict[str, Any],
    device: torch.device,
    seed: int,
    batch_size: int,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Tuple[nn.Module, nn.Module]:
    set_seed(seed)

    tr = split["train"].astype(np.int64)
    va = split.get("val", None)
    va = va.astype(np.int64) if va is not None else None

    Xs, _, _ = standardize_using_train(X1, tr)

    X_tr = torch.from_numpy(Xs[tr]).float().to(device)
    y_tr = torch.from_numpy(y[tr]).float().to(device)  # BCE expects float

    X_va: Optional[torch.Tensor]
    y_va_np: Optional[np.ndarray]
    if va is not None:
        X_va = torch.from_numpy(Xs[va]).float().to(device)
        y_va_np = y[va].astype(np.int64)  # keep numpy for sklearn metrics
    else:
        X_va = None
        y_va_np = None

    bottom = BottomMLP_Paper(in_dim=int(X_tr.shape[1])).to(device)
    head = nn.Linear(8, 1).to(device)

    # pos_weight computed on train only
    pos = float(y[tr].sum())
    neg = float(len(tr) - y[tr].sum())
    pw = neg / max(pos, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    opt = torch.optim.Adam(
        list(bottom.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_bottom_state: Optional[Dict[str, torch.Tensor]] = None
    best_head_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("-inf")
    no_improve = 0

    n = X_tr.shape[0]

    for epoch in range(1, max_epochs + 1):
        bottom.train()
        head.train()

        perm = torch.randperm(n, device=device)
        Xb = X_tr[perm]
        yb = y_tr[perm]

        for i in range(0, n, batch_size):
            xb = Xb[i : i + batch_size]
            y_true = yb[i : i + batch_size].view(-1, 1)

            logits = head(bottom(xb))
            loss = criterion(logits, y_true)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Early stopping on val AUROC if val exists
        if X_va is not None and y_va_np is not None and len(np.unique(y_va_np)) == 2:
            prob_va = predict_prob(bottom, head, X_va, batch=2048)
            from sklearn.metrics import roc_auc_score
            val_score = float(roc_auc_score(y_va_np, prob_va))
        else:
            # fallback: -train loss (only if no validation split exists)
            bottom.eval()
            head.eval()
            with torch.no_grad():
                loss_tr = criterion(head(bottom(X_tr)), y_tr.view(-1, 1)).item()
            val_score = -float(loss_tr)

        if val_score > best_val + 1e-6:
            best_val = val_score
            no_improve = 0
            best_bottom_state = {k: v.detach().cpu().clone() for k, v in bottom.state_dict().items()}
            best_head_state   = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                break

    # Restore best
    if best_bottom_state is not None:
        bottom.load_state_dict(best_bottom_state)
    if best_head_state is not None:
        head.load_state_dict(best_head_state)

    return bottom, head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to diabetes_vfl_cv.npz")
    ap.add_argument("--fold", type=int, required=True, help="Fold index (1-based)")
    ap.add_argument("--out_dir", required=True, help="Directory to write checkpoints")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15, help="Early stopping patience (0 disables)")
    args = ap.parse_args()

    X1, y, folds, meta = load_npz(args.npz)
    split_obj = folds[args.fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bottom, head = train_one_fold(
        X1=X1,
        y=y,
        split=split,
        device=device,
        seed=args.seed,
        batch_size=args.batch,
        max_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        patience=args.patience,
    )

    out_path = out_dir / f"pretrained_active_sup_teacher_fold{args.fold}.pt"
    torch.save(
        {"bottom_state": bottom.state_dict(), "head_state": head.state_dict(), "meta": meta},
        out_path,
    )
    print(f"[OK] Saved supervised active teacher (bottom+head) to: {out_path}")


if __name__ == "__main__":
    main()