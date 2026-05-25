# run_active_ssl_pretrain_local.py
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

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
    def forward(self, x): return self.net(x)

class ReconHead(nn.Module):
    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )
    def forward(self, z): return self.net(z)

def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    X1 = d["X1"].astype(np.float32)
    y = d["y"].astype(np.int64)
    folds = list(d["folds"])
    return X1, y, folds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise-std", type=float, default=0.1)
    ap.add_argument("--out-dim", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="runs_active_ssl")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X1, y, folds = load_npz(args.npz)
    split_obj = folds[args.fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)

    # standardize using train stats
    mu1 = X1[tr].mean(axis=0, keepdims=True)
    sd1 = X1[tr].std(axis=0, keepdims=True) + 1e-8
    X1s = (X1 - mu1) / sd1

    dev = torch.device(args.device)
    X1t = torch.from_numpy(X1s).float().to(dev)

    bottom = BottomMLP(in_dim=X1.shape[1], out_dim=args.out_dim, dropout=args.dropout).to(dev)
    recon = ReconHead(emb_dim=args.out_dim, out_dim=X1.shape[1]).to(dev)
    opt = torch.optim.Adam(list(bottom.parameters()) + list(recon.parameters()), lr=args.lr)
    mse = nn.MSELoss()

    rng = np.random.default_rng(args.seed)
    idx = tr.copy()

    for ep in range(1, args.epochs + 1):
        rng.shuffle(idx)
        loss_sum, n = 0.0, 0
        steps = int(np.ceil(len(idx) / args.batch_size))
        for s in range(steps):
            b = idx[s * args.batch_size : (s + 1) * args.batch_size]
            if b.size == 0: 
                continue
            bt = torch.from_numpy(b).long().to(dev)
            xb = X1t.index_select(0, bt)
            xn = xb + args.noise_std * torch.randn_like(xb)

            opt.zero_grad(set_to_none=True)
            z = bottom(xn)
            xr = recon(z)
            loss = mse(xr, xb)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * int(b.size)
            n += int(b.size)

        print(f"[Active SSL] epoch={ep}/{args.epochs} avg_mse={loss_sum/max(n,1):.6f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "bottom_state": {k: v.detach().cpu() for k, v in bottom.state_dict().items()},
        "fold": int(args.fold),
        "seed": int(args.seed),
        "out_dim": int(args.out_dim),
        "npz": str(Path(args.npz).resolve()),
        "x1_mu": mu1.astype(np.float32),
        "x1_sd": sd1.astype(np.float32),
        "ssl": True,
    }
    save_path = out_dir / f"pretrained_active_bottom_ssl_fold{args.fold}.pt"
    torch.save(ckpt, save_path)
    print(f"[OK] saved -> {save_path}")

if __name__ == "__main__":
    main()
