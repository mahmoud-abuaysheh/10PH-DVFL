# pretrain_active_supervised_glioma.py
#
# Supervised pre-training for the active-silo encoder in the glioma
# decoupled VFL experiment.
#
# This script implements the supervised pre-training stage (Tier 1) for the
# active silo encoder. It trains the bottom MLP encoder together with a
# temporary local classification head using only the active-silo feature
# view X1 and the labels y from the selected fold's training split.
#
# Only the trained bottom encoder checkpoint is saved and used later to
# generate active-silo embeddings for the Tier 2 decoupled VFL fusion stage.
# The temporary classification head provides the supervised training signal
# but is never transmitted to any other party. It is also saved alongside the
# encoder for reference but plays no role in downstream VFL fusion.
#
# Architecture:
#   BottomMLP: input -> 32 -> ReLU -> Dropout(0.0) -> 16 -> ReLU
#   TopMLP:    16 -> 16 -> ReLU -> 8 -> ReLU -> 1
#
# The Dropout layer is included at dropout=0.0 to maintain checkpoint
# compatibility with the HFL and VFL client scripts which use the same
# BottomMLP definition.
#
# Output:
#     pretrained_active_bottom_sup_fold{fold}.pt
#         Contains 'bottom_state', 'head_state', and fold-specific
#         standardization statistics for reproducibility.

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class BottomMLP(nn.Module):
    """
    Bottom encoder used by the active silo in the decoupled VFL architecture.

    Projects active-silo input features through two linear layers with
    ReLU activations to produce a 16-dimensional embedding vector.
    Architecture: input -> 32 -> ReLU -> Dropout -> 16 -> ReLU

    The Dropout layer is included at dropout=0.0 (disabled) to maintain
    checkpoint key compatibility with the HFL and VFL client scripts which
    use the same architecture definition with Dropout present.
    """
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
    """
    Temporary local classification head used only during supervised
    pre-training of the active-silo encoder.

    Takes only z1 (out_dim=16) as input — note this differs from the
    VFL fusion head which takes concatenated z1+z2 (in_dim=32). This
    head is discarded after pre-training and is never used in Tier 2.
    Architecture: 16 -> 16 -> ReLU -> 8 -> ReLU -> 1
    """
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
# Data loading
# ---------------------------------------------------------------------------

def load_npz(npz_path: str):
    """
    Load the pre-computed cross-validation dataset from the NPZ file.
    Returns active-silo features X1, passive-silo features X2 (unused),
    labels y, and fold splits.
    """
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
) -> Path:
    """
    Train the active-silo bottom encoder and temporary classification head
    for a single cross-validation fold using supervised pre-training.

    Training uses BCEWithLogitsLoss with per-fold positive class weighting
    to handle the class imbalance in the glioma dataset. Standardization
    is computed from the training split only to prevent leakage.

    Returns the path to the saved checkpoint.
    """
    split_obj = folds[fold - 1]
    split     = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr_idx    = split["train"].astype(np.int64)
    va_idx    = split["val"].astype(np.int64)

    # Standardize using training split statistics only to prevent leakage.
    mu  = X1[tr_idx].mean(axis=0, keepdims=True)
    sd  = X1[tr_idx].std(axis=0, keepdims=True) + 1e-8
    X1s = (X1 - mu) / sd

    X1t = torch.from_numpy(X1s).float().to(device)
    yt  = torch.from_numpy(y.astype(np.float32)).to(device)

    # Compute positive class weight from the training split only.
    y_tr       = yt[torch.from_numpy(tr_idx).long().to(device)]
    pos        = float(y_tr.sum().item())
    neg        = float(y_tr.numel() - y_tr.sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    np.random.seed(seed)
    torch.manual_seed(seed)

    bottom = BottomMLP(in_dim=X1.shape[1], out_dim=out_dim).to(device)
    head   = TopMLP(in_dim=out_dim).to(device)
    opt    = torch.optim.Adam(
        list(bottom.parameters()) + list(head.parameters()), lr=lr
    )

    rng             = np.random.default_rng(seed)
    steps_per_epoch = int(np.ceil(len(tr_idx) / batch_size))

    print(f"\n[Fold {fold}] Starting supervised pre-training for {epochs} epochs...")

    for ep in range(1, epochs + 1):
        bottom.train()
        head.train()
        tr_copy = tr_idx.copy()
        rng.shuffle(tr_copy)
        loss_sum = 0.0
        n = 0

        for s in range(steps_per_epoch):
            b = tr_copy[s * batch_size : (s + 1) * batch_size]
            if b.size == 0:
                continue
            bt     = torch.from_numpy(b).long().to(device)
            xb     = X1t.index_select(0, bt)
            yb     = yt.index_select(0, bt)
            opt.zero_grad(set_to_none=True)
            z1     = bottom(xb)
            logits = head(z1)
            loss   = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(b)
            n        += len(b)

        avg_loss = loss_sum / max(n, 1)

        # Evaluate validation AUROC every 10 epochs and on the last epoch.
        if ep % 10 == 0 or ep == epochs:
            from sklearn.metrics import roc_auc_score, average_precision_score
            bottom.eval()
            head.eval()
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
            print(
                f"  ep={ep:3d}/{epochs}  train_loss={avg_loss:.4f}  "
                f"val_AUROC={auroc:.4f}  val_PR-AUC={prauc:.4f}"
            )
        else:
            print(f"  ep={ep:3d}/{epochs}  train_loss={avg_loss:.4f}")

    # Save the encoder and head checkpoint.
    # The bottom encoder is used in Tier 2 to generate active-silo embeddings.
    # The head is retained for reference but is not used in downstream VFL.
    ckpt = {
        "bottom_state": bottom.state_dict(),
        "head_state":   head.state_dict(),
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
    print(f"[Fold {fold}] Saved -> {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        description="Supervised pre-training for the glioma active-silo encoder."
    )
    ap.add_argument("--npz",        default=str(here / "glioma_aligned_vfl_hfl_cv.npz"),
                    help="Path to glioma_aligned_vfl_hfl_cv.npz")
    ap.add_argument("--epochs",     type=int,   default=100,
                    help="Number of pre-training epochs")
    ap.add_argument("--batch_size", type=int,   default=64,
                    help="Training batch size")
    ap.add_argument("--lr",         type=float, default=1e-3,
                    help="Adam learning rate")
    ap.add_argument("--out_dim",    type=int,   default=16,
                    help="Encoder output dimensionality")
    ap.add_argument("--seed",       type=int,   default=42,
                    help="Random seed for reproducibility")
    ap.add_argument("--folds",      type=int,   nargs="+", default=[1, 2, 3, 4, 5],
                    help="Fold indices to train (1-based)")
    ap.add_argument("--out_dir",    default=str(here / "runs_sup_pretrain"),
                    help="Directory to write encoder checkpoints")
    ap.add_argument("--device",     default="cpu",
                    help="Device to use: cpu or cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    X1, X2, y, folds = load_npz(args.npz)

    print(f"Supervised pre-training — Active encoder")
    print(f"  NPZ:          {args.npz}")
    print(f"  X1 shape:     {X1.shape}  |  out_dim={args.out_dim}")
    print(f"  epochs={args.epochs}  lr={args.lr}  batch_size={args.batch_size}  seed={args.seed}")
    print(f"  folds:        {args.folds}")
    print(f"  out_dir:      {out_dir}")
    print(f"  Architecture: BottomMLP({X1.shape[1]}->32->{args.out_dim}) + "
          f"TopMLP({args.out_dim}->16->8->1)")

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
    print(f"Checkpoints written to: {out_dir}")


if __name__ == "__main__":
    main()
