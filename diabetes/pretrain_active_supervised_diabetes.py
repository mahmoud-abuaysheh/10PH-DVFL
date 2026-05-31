#!/usr/bin/env python3
"""
pretrain_active_supervised_diabetes.py

Train the active-silo encoder for the diabetes decoupled VFL experiment.

This script implements the supervised pre-training stage (Tier 1) for the
active silo encoder. It trains the bottom MLP encoder together with a temporary
local classification head using only the active feature view X1 and the labels
y from the selected fold's training split.

Only the trained bottom encoder checkpoint is saved and used later to generate
active-silo embeddings for the Tier 2 decoupled VFL fusion stage. The temporary
local classification head provides the supervised training signal but is
discarded after pre-training and is never transmitted to any other party.

Output:
    pretrained_active_sup_fold{fold}.pt
        Contains 'bottom_state' (encoder weights) and 'head_state' (temporary
        head weights, retained for reference but not used in downstream VFL).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn


class BottomMLP_Paper(nn.Module):
    """
    Bottom encoder used by the active silo in the decoupled VFL architecture.

    Projects active-silo input features through two linear layers with
    ReLU activations to produce an 8-dimensional embedding vector.
    Architecture: input -> 16 -> ReLU -> 8 -> ReLU
    """
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
    """
    Load the pre-computed cross-validation dataset from the NPZ file.
    Returns active-silo features X1, labels y, fold splits, and metadata.
    Only X1 and y are used during active-silo supervised pre-training.
    """
    d = np.load(npz_path, allow_pickle=True)
    X1 = d["X1"].astype(np.float32)
    y = d["y"].astype(np.int64)
    folds = list(d["folds"])
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return X1, y, folds, meta


def standardize_using_train(
    X: np.ndarray, tr_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using mean and standard deviation computed from
    the training split only, preventing any leakage from validation or test sets.
    Returns the standardized array along with the training mean and std.
    """
    mu = X[tr_idx].mean(axis=0, keepdims=True)
    sd = X[tr_idx].std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict_prob(
    bottom: nn.Module, head: nn.Module, X: torch.Tensor, batch: int = 2048
) -> np.ndarray:
    """
    Compute predicted probabilities for a feature tensor using the
    encoder and temporary classification head.
    Used for validation AUROC computation during early stopping.
    """
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
    """
    Train the active-silo bottom encoder and temporary classification head
    for a single cross-validation fold using supervised pre-training.

    Training uses BCEWithLogitsLoss with per-fold positive class weighting
    to handle the class imbalance in the diabetes dataset. Early stopping
    is applied based on validation AUROC to select the best checkpoint.

    Returns the trained bottom encoder and temporary head with the best
    validation checkpoint restored.
    """
    set_seed(seed)

    tr = split["train"].astype(np.int64)
    va = split.get("val", None)
    va = va.astype(np.int64) if va is not None else None

    Xs, _, _ = standardize_using_train(X1, tr)

    X_tr = torch.from_numpy(Xs[tr]).float().to(device)
    y_tr = torch.from_numpy(y[tr]).float().to(device)

    X_va: Optional[torch.Tensor]
    y_va_np: Optional[np.ndarray]
    if va is not None:
        X_va = torch.from_numpy(Xs[va]).float().to(device)
        y_va_np = y[va].astype(np.int64)
    else:
        X_va = None
        y_va_np = None

    bottom = BottomMLP_Paper(in_dim=int(X_tr.shape[1])).to(device)
    head = nn.Linear(8, 1).to(device)

    # Compute positive class weight from the training split only.
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

        # Use validation AUROC for early stopping when a validation split is available.
        if X_va is not None and y_va_np is not None and len(np.unique(y_va_np)) == 2:
            prob_va = predict_prob(bottom, head, X_va, batch=2048)
            from sklearn.metrics import roc_auc_score
            val_score = float(roc_auc_score(y_va_np, prob_va))
        else:
            # Fall back to negative training loss if no validation split is available.
            bottom.eval()
            head.eval()
            with torch.no_grad():
                loss_tr = criterion(head(bottom(X_tr)), y_tr.view(-1, 1)).item()
            val_score = -float(loss_tr)

        if val_score > best_val + 1e-6:
            best_val = val_score
            no_improve = 0
            best_bottom_state = {k: v.detach().cpu().clone() for k, v in bottom.state_dict().items()}
            best_head_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                break

    # Restore the best validation checkpoint before returning.
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

    # Save the encoder checkpoint with the updated naming convention.
    # The checkpoint name reflects the pre-training method (supervised)
    # and fold index for unambiguous identification.
    out_path = out_dir / f"pretrained_active_sup_fold{args.fold}.pt"
    torch.save(
        {"bottom_state": bottom.state_dict(), "head_state": head.state_dict(), "meta": meta},
        out_path,
    )
    print(f"[OK] Saved supervised active pre-training checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
