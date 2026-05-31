# run_passive_ssl_pretrain_local_diabetes.py
#
# Self-supervised pre-training for the passive-silo encoder (X2) in the
# diabetes decoupled VFL experiment.
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
# This script trains locally on a single machine using only the passive-silo
# feature matrix X2. No cross-silo communication occurs during this stage.
# For the alternative intra-silo HFL pre-training mode for the passive silo,
# see serverapp_vfl_diabetes_decoupled.py (_run_passive_hfl_ssl) and
# clientapp_vfl_diabetes_decoupled.py (hfl_fit handler).
#
# Output:
#     pretrained_passive_bottom_ssl_fold{fold}.pt
#         Contains 'bottom_state' (encoder weights) and fold-specific
#         standardization statistics for reproducibility.

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn


class BottomMLP_Paper(nn.Module):
    """
    Bottom encoder used by the passive silo in the decoupled VFL architecture.

    Projects passive-silo input features through two linear layers with
    ReLU activations to produce an 8-dimensional embedding vector.
    Architecture: input -> 16 -> ReLU -> 8 -> ReLU

    This architecture is shared across all passive-silo pre-training modes
    (local SSL and intra-silo HFL) and must match the encoder architecture
    used in the client application.
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


class ReconHead(nn.Module):
    """
    Reconstruction head used only during DAE self-supervised pre-training.

    Decodes the 8-dimensional encoder output back to the original input
    dimensionality. This head is local to the pre-training script and is
    discarded after training. It is never transmitted to any other party
    and plays no role in the Tier 2 cross-silo fusion stage.
    Architecture: 8 -> 16 -> ReLU -> input_dim
    """
    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def load_npz(npz_path: str):
    """
    Load the pre-computed cross-validation dataset from the NPZ file.
    Returns passive-silo features X2, labels y (unused during SSL),
    fold splits, and metadata.
    Labels are loaded but not used during self-supervised pre-training,
    reflecting the passive silo's absence of label access in the VFL setting.
    """
    d = np.load(npz_path, allow_pickle=True)
    X2 = d["X2"].astype(np.float32)
    y = d["y"].astype(np.int64)
    folds = list(d["folds"])
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return X2, y, folds, meta


def main():
    ap = argparse.ArgumentParser(
        description="Self-supervised DAE pre-training for the passive silo encoder (X2)."
    )
    ap.add_argument("--npz", required=True, help="Path to diabetes_vfl_cv.npz")
    ap.add_argument("--fold", type=int, default=1, help="Fold index (1-based)")
    ap.add_argument("--epochs", type=int, default=100, help="Number of pre-training epochs")
    ap.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    ap.add_argument(
        "--noise-std", type=float, default=0.1,
        help="Standard deviation of Gaussian noise added to inputs for DAE corruption"
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--device", default="cpu", help="Device to use: cpu or cuda")
    ap.add_argument(
        "--out-dir", default="runs_passive_ssl_diabetes",
        help="Directory to write the pre-trained encoder checkpoint"
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X2, _, folds, meta = load_npz(args.npz)
    split_obj = folds[args.fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)

    # Standardize features using training split statistics only to prevent
    # any leakage from validation or test sets into the pre-training stage.
    mu = X2[tr].mean(axis=0, keepdims=True)
    sd = X2[tr].std(axis=0, keepdims=True) + 1e-8
    X2s = (X2 - mu) / sd

    dev = torch.device(args.device)
    X = torch.from_numpy(X2s).float().to(dev)

    # Initialise the bottom encoder and reconstruction head.
    # The reconstruction head is a local-only component used solely during
    # pre-training and discarded before Tier 2.
    bottom = BottomMLP_Paper(in_dim=int(X.shape[1])).to(dev)
    recon = ReconHead(emb_dim=8, out_dim=int(X.shape[1])).to(dev)

    opt = torch.optim.Adam(
        list(bottom.parameters()) + list(recon.parameters()), lr=args.lr
    )
    mse = nn.MSELoss()

    rng = np.random.default_rng(args.seed)
    idx = tr.copy()

    # DAE pre-training loop.
    # Each epoch shuffles the training indices, corrupts each batch with
    # Gaussian noise, and trains the encoder to reconstruct the clean input.
    # The MSE reconstruction loss is minimised over clean targets.
    # No labels are used at any point, consistent with the passive silo's
    # absence of label access in the 10PH-DVFL architecture.
    for ep in range(1, args.epochs + 1):
        rng.shuffle(idx)
        loss_sum, n = 0.0, 0
        steps = int(np.ceil(len(idx) / args.batch_size))

        bottom.train()
        recon.train()

        for s in range(steps):
            b = idx[s * args.batch_size : (s + 1) * args.batch_size]
            if b.size == 0:
                continue

            bt = torch.from_numpy(b).long().to(dev)
            xb = X.index_select(0, bt)

            # Corrupt the input by adding Gaussian noise (DAE corruption step).
            # The encoder is trained to reconstruct the original clean input from
            # this corrupted version, encouraging robust feature learning.
            xn = xb + args.noise_std * torch.randn_like(xb)

            opt.zero_grad(set_to_none=True)
            z = bottom(xn)       # Encode the corrupted input.
            xr = recon(z)        # Reconstruct the clean input from the encoding.
            loss = mse(xr, xb)   # MSE loss against the original clean input.
            loss.backward()
            opt.step()

            bs = int(b.size)
            loss_sum += float(loss.item()) * bs
            n += bs

        print(
            f"[Passive SSL] fold={args.fold} epoch={ep}/{args.epochs} "
            f"avg_mse={loss_sum / max(n, 1):.6f}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the encoder checkpoint. The reconstruction head is not saved as it
    # is not used in any downstream stage. Standardization statistics are
    # included for reproducibility and documentation purposes.
    ckpt = {
        "bottom_state": {k: v.detach().cpu() for k, v in bottom.state_dict().items()},
        "fold": int(args.fold),
        "seed": int(args.seed),
        "out_dim": 8,
        "npz": str(Path(args.npz).resolve()),
        "x2_mu": mu.astype(np.float32),
        "x2_sd": sd.astype(np.float32),
        "ssl": True,
        "meta": meta,
    }
    save_path = out_dir / f"pretrained_passive_bottom_ssl_fold{args.fold}.pt"
    torch.save(ckpt, save_path)
    print(f"[OK] saved -> {save_path}")


if __name__ == "__main__":
    main()
