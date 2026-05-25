# run_passive_ssl_pretrain_local_diabetes.py
# SSL (denoising autoencoder) pretrain for PASSIVE silo (X2) on Diabetes npz.
# Matches your VFL BottomMLP_Paper output dim = 8 (in_dim -> 16 -> 8).
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn


class BottomMLP_Paper(nn.Module):
    """in_dim -> 16 -> 8 with ReLU (matches client BottomMLP_Paper)."""
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
    """Decoder: 8 -> 16 -> in_dim (local-only)."""
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
    d = np.load(npz_path, allow_pickle=True)
    X2 = d["X2"].astype(np.float32)
    y = d["y"].astype(np.int64)
    folds = list(d["folds"])
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return X2, y, folds, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--fold", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise-std", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--out-dir", default="runs_passive_ssl_diabetes")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X2, _, folds, meta = load_npz(args.npz)
    split_obj = folds[args.fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)

    # standardize using TRAIN stats (fold-specific)
    mu = X2[tr].mean(axis=0, keepdims=True)
    sd = X2[tr].std(axis=0, keepdims=True) + 1e-8
    X2s = (X2 - mu) / sd

    dev = torch.device(args.device)
    X = torch.from_numpy(X2s).float().to(dev)

    bottom = BottomMLP_Paper(in_dim=int(X.shape[1])).to(dev)
    recon = ReconHead(emb_dim=8, out_dim=int(X.shape[1])).to(dev)

    opt = torch.optim.Adam(list(bottom.parameters()) + list(recon.parameters()), lr=args.lr)
    mse = nn.MSELoss()

    rng = np.random.default_rng(args.seed)
    idx = tr.copy()

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
            xn = xb + args.noise_std * torch.randn_like(xb)

            opt.zero_grad(set_to_none=True)
            z = bottom(xn)
            xr = recon(z)
            loss = mse(xr, xb)
            loss.backward()
            opt.step()

            bs = int(b.size)
            loss_sum += float(loss.item()) * bs
            n += bs

        print(f"[Passive SSL] fold={args.fold} epoch={ep}/{args.epochs} avg_mse={loss_sum/max(n,1):.6f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
