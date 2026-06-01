# run_passive_ssl_pretrain_local_glioma.py
#
# Self-supervised pre-training for the passive-silo encoder (X2) in the
# glioma decoupled VFL experiment.
#
# This script implements the self-supervised Tier 1 pre-training stage for
# the passive silo using a Denoising Autoencoder (DAE) objective. The passive
# silo has no access to labels at any point, so self-supervised pre-training
# is the only available local pre-training strategy for this silo.
#
# The DAE objective trains the bottom encoder to reconstruct clean input
# features from a corrupted version, encouraging the encoder to learn
# compact and robust representations of the passive-silo features without
# using any labels. After pre-training, the reconstruction head is discarded
# and only the trained bottom encoder checkpoint is retained for Tier 2.
#
# For the alternative intra-silo HFL pre-training mode for the passive silo,
# see serverapp_hfl_passive_glioma.py and clientapp_hfl_passive_glioma.py.
#
# Architecture:
#   BottomMLP: input -> 32 -> ReLU -> Dropout(0.0) -> 16 -> ReLU
#   ReconHead: 16 -> 32 -> ReLU -> input_dim
#
# The Dropout layer is included at dropout=0.0 to maintain checkpoint
# compatibility with the HFL and VFL client scripts.
#
# Output:
#     pretrained_passive_bottom_ssl_fold{fold}.pt
#         Contains 'bottom_state' and fold-specific standardization
#         statistics for reproducibility.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class BottomMLP(nn.Module):
    """
    Bottom encoder used by the passive silo in the decoupled VFL architecture.

    Projects passive-silo input features through two linear layers with
    ReLU activations to produce a 16-dimensional embedding vector.
    Architecture: input -> 32 -> ReLU -> Dropout -> 16 -> ReLU

    The Dropout layer is included at dropout=0.0 (disabled) to maintain
    checkpoint key compatibility with the HFL and VFL client scripts.
    Must match the architecture defined in clientapp_hfl_passive_glioma.py
    and clientapp_vfl_glioma_decoupled.py.
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


class ReconHead(nn.Module):
    """
    Reconstruction head used only during DAE self-supervised pre-training.

    Decodes the 16-dimensional encoder output back to the original input
    dimensionality. This head is local to the pre-training script and is
    discarded after training. It is never used in Tier 2.
    Architecture: 16 -> 32 -> ReLU -> input_dim
    """
    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_npz(npz_path: str):
    """
    Load the pre-computed cross-validation dataset from the NPZ file.
    Returns passive-silo features X2, labels y (unused during SSL),
    and fold splits. Labels are never used during passive silo pre-training.
    """
    d     = np.load(npz_path, allow_pickle=True)
    X2    = d["X2"].astype(np.float32)
    y     = d["y"].astype(np.int64)
    folds = list(d["folds"])
    return X2, y, folds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Self-supervised DAE pre-training for the glioma passive-silo encoder (X2)."
    )
    ap.add_argument("--npz",        required=True,
                    help="Path to glioma_aligned_vfl_hfl_cv.npz")
    ap.add_argument("--fold",       type=int,   default=1,
                    help="Fold index (1-based)")
    ap.add_argument("--epochs",     type=int,   default=100,
                    help="Number of pre-training epochs")
    ap.add_argument("--batch-size", type=int,   default=64,
                    help="Training batch size")
    ap.add_argument("--lr",         type=float, default=1e-3,
                    help="Adam learning rate")
    ap.add_argument("--noise-std",  type=float, default=0.1,
                    help="Gaussian noise std for DAE corruption")
    ap.add_argument("--out-dim",    type=int,   default=16,
                    help="Encoder output dimensionality")
    ap.add_argument("--dropout",    type=float, default=0.0,
                    help="Dropout probability (0.0 disables)")
    ap.add_argument("--seed",       type=int,   default=42,
                    help="Random seed for reproducibility")
    ap.add_argument("--device",     default="cpu",
                    help="Device to use: cpu or cuda")
    ap.add_argument("--out-dir",    default="runs_passive_ssl_glioma",
                    help="Directory to write the pre-trained encoder checkpoint")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X2, _, folds = load_npz(args.npz)
    split_obj = folds[args.fold - 1]
    split     = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr        = split["train"].astype(np.int64)

    # Standardize using training split statistics only to prevent leakage.
    mu2 = X2[tr].mean(axis=0, keepdims=True)
    sd2 = X2[tr].std(axis=0, keepdims=True) + 1e-8
    X2s = (X2 - mu2) / sd2

    dev = torch.device(args.device)
    X2t = torch.from_numpy(X2s).float().to(dev)

    # Initialise the bottom encoder and reconstruction head.
    # The reconstruction head is a local-only component used solely during
    # pre-training and discarded before Tier 2.
    bottom = BottomMLP(
        in_dim=X2.shape[1], out_dim=args.out_dim, dropout=args.dropout
    ).to(dev)
    recon  = ReconHead(emb_dim=args.out_dim, out_dim=X2.shape[1]).to(dev)
    opt    = torch.optim.Adam(
        list(bottom.parameters()) + list(recon.parameters()), lr=args.lr
    )
    mse_fn = nn.MSELoss()

    rng = np.random.default_rng(args.seed)
    idx = tr.copy()

    # DAE pre-training loop.
    # Each epoch shuffles the training indices, corrupts each batch with
    # Gaussian noise, and trains the encoder to reconstruct the clean input.
    # No labels are used at any point, consistent with the passive silo's
    # absence of label access in the 10PH-DVFL architecture.
    for ep in range(1, args.epochs + 1):
        rng.shuffle(idx)
        loss_sum = 0.0
        n        = 0
        steps    = int(np.ceil(len(idx) / args.batch_size))

        bottom.train()
        recon.train()

        for s in range(steps):
            b = idx[s * args.batch_size : (s + 1) * args.batch_size]
            if b.size == 0:
                continue
            bt   = torch.from_numpy(b).long().to(dev)
            xb   = X2t.index_select(0, bt)
            xn   = xb + args.noise_std * torch.randn_like(xb)  # DAE corruption.
            opt.zero_grad(set_to_none=True)
            z    = bottom(xn)       # Encode the corrupted input.
            xr   = recon(z)         # Reconstruct the clean input.
            loss = mse_fn(xr, xb)   # MSE against the original clean input.
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * int(b.size)
            n        += int(b.size)

        print(
            f"[Passive SSL] fold={args.fold} epoch={ep}/{args.epochs} "
            f"avg_mse={loss_sum / max(n, 1):.6f}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the encoder checkpoint. The reconstruction head is not saved
    # as it is not used in any downstream stage.
    ckpt = {
        "bottom_state": {k: v.detach().cpu() for k, v in bottom.state_dict().items()},
        "fold":   int(args.fold),
        "seed":   int(args.seed),
        "out_dim": int(args.out_dim),
        "npz":    str(Path(args.npz).resolve()),
        "x2_mu":  mu2.astype(np.float32),
        "x2_sd":  sd2.astype(np.float32),
        "ssl":    True,
    }
    save_path = out_dir / f"pretrained_passive_bottom_ssl_fold{args.fold}.pt"
    torch.save(ckpt, save_path)
    print(f"[OK] saved -> {save_path}")


if __name__ == "__main__":
    main()
