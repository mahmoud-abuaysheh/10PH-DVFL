#!/usr/bin/env python3
"""
train_centralized_midas_e2e.py

Centralized end-to-end upper-bound baseline for MIDAS VFL comparison.

Architecture (mirrors VFL split-NN exactly):
  Per modality — 3 separate encoders (one per modality, same as 3 VFL clients):
    ResNet50 (ImageNet V2, TRAINABLE) → ProjectionMLP(2048→512→256) → 256-D
  Fusion (same as VFL server):
    concat(dscope_256, 6in_256, 1ft_256) → 768-D → OldMLP(768→512→1)

  ProjectionMLP: Linear(2048,512)→LayerNorm→GELU→Dropout(0.2)→Linear(512,256)
    IDENTICAL to serverapp_vfl_midas_splitnn_head.py

Hyperparams (IDENTICAL to VFL splitnn):
  lr=1e-4, wd=1e-4, batch=64, epochs=20, patience=7, min_delta=1e-4, seed=42

Splits: same 5-fold CV from fold_npz/ used in VFL splitnn
  active_dscope_fold{N}.npz  → paths + labels (train/val/test)
  passive_6in_fold{N}.npz    → paths (train/val/test)
  passive_1ft_fold{N}.npz    → paths (train/val/test)

Outputs per fold (out_dir/fold{N}_centralized/):
  history.csv                 — per-epoch train_loss, val_auroc, val_prauc
  threshold_metrics_val.csv   — full threshold sweep on val
  threshold_metrics_test.csv  — full threshold sweep on test
  summary.csv                 — auroc, prauc, best thresholds, best_epoch

After all folds:
  test_summary_mean_std.csv   — mean±std across folds

Usage:
  python train_centralized_midas_e2e.py \
      --image_root /path/to/midas/images \
      --fold_npz_dir fold_npz \
      --out_dir runs_centralized_e2e \
      --folds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch.utils.checkpoint as torch_checkpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── Hyperparams — IDENTICAL to VFL splitnn ───────────────────────────────────
LR          = 1e-4
WD          = 1e-4
MAX_EPOCHS  = 20
PATIENCE    = 7
MIN_ROUNDS  = 5
MIN_DELTA   = 1e-4
SEED        = 42
EMB_DIM     = 256
PROJ_HIDDEN = 512
PROJ_DROP   = 0.2
HEAD_HIDDEN = 512
HEAD_DROP   = 0.2
CONCAT_DIM  = EMB_DIM * 3  # 768

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Models — copied verbatim from VFL scripts ─────────────────────────────────

class ProjectionMLP(nn.Module):
    """
    Identical to clientapp_vfl_midas_splitnn_proj256.py ProjectionMLP.
    ResNet2048 → PROJ_HIDDEN → EMB_DIM with LayerNorm/GELU/Dropout.
    """
    def __init__(self, in_dim: int = 2048, hidden: int = PROJ_HIDDEN,
                 out_dim: int = EMB_DIM, dropout: float = PROJ_DROP):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OldMLP(nn.Module):
    """
    Identical to serverapp_vfl_midas_splitnn_head.py OldMLP.
    Linear(768→512) → ReLU → Dropout(0.2) → Linear(512→1)
    """
    def __init__(self, in_dim: int = CONCAT_DIM, hidden: int = HEAD_HIDDEN,
                 dropout: float = HEAD_DROP):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CentralizedModel(nn.Module):
    """
    Full centralized model = 3x (ResNet50 + ProjectionMLP) + OldMLP head.
    Equivalent to VFL splitnn with no communication split.
    """
    def __init__(self, use_checkpointing: bool = False):
        super().__init__()
        self.encoders          = nn.ModuleList([self._make_encoder() for _ in range(3)])
        self.head              = OldMLP(in_dim=CONCAT_DIM)
        self.use_checkpointing = use_checkpointing

    @staticmethod
    def _make_encoder() -> nn.Module:
        backbone     = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc  = nn.Identity()
        proj         = ProjectionMLP()
        return nn.Sequential(backbone, proj)

    def forward(self, imgs: List[torch.Tensor]) -> torch.Tensor:
        if self.use_checkpointing:
            embs = [torch_checkpoint.checkpoint(enc, x, use_reentrant=False)
                    for enc, x in zip(self.encoders, imgs)]
        else:
            embs = [enc(x) for enc, x in zip(self.encoders, imgs)]
        fused = torch.cat(embs, dim=1)
        return self.head(fused)


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiModalDataset(Dataset):
    def __init__(self, paths_list: List[List[str]], labels: np.ndarray,
                 image_root: str, transform):
        assert len(paths_list) == 3
        self.paths_list = paths_list
        self.labels     = labels
        self.transform  = transform
        self._lookup: Dict[str, Path] = {}
        for p in Path(image_root).rglob("*"):
            if p.is_file():
                self._lookup[p.name.lower()] = p

    def __len__(self):
        return len(self.labels)

    def _load(self, path_str: str) -> torch.Tensor:
        fname = Path(path_str).name.lower()
        fpath = self._lookup.get(fname)
        if fpath is None:
            base, ext = os.path.splitext(fname)
            fpath = self._lookup.get(base + "_cropped" + ext)
        if fpath is None:
            raise FileNotFoundError(f"Image not found: {path_str}")
        return self.transform(Image.open(fpath).convert("RGB"))

    def __getitem__(self, idx):
        imgs  = [self._load(self.paths_list[m][idx]) for m in range(3)]
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return imgs[0], imgs[1], imgs[2], label


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    return 1.0 / (1.0 + np.exp(-x))


def write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, step=0.01) -> List[dict]:
    y_true = y_true.astype(int).reshape(-1)
    y_prob = y_prob.astype(float).reshape(-1)
    rows = []
    for thr in np.arange(0.0, 1.0001, step):
        yp   = (y_prob >= thr).astype(int)
        tp   = int(((y_true==1)&(yp==1)).sum())
        tn   = int(((y_true==0)&(yp==0)).sum())
        fp   = int(((y_true==0)&(yp==1)).sum())
        fn   = int(((y_true==1)&(yp==0)).sum())
        acc  = (tp+tn)/max(1,tp+tn+fp+fn)
        prec = tp/max(1,tp+fp)
        rec  = tp/max(1,tp+fn)
        spec = tn/max(1,tn+fp)
        f1   = 2*prec*rec/max(1e-12,prec+rec)
        rows.append(dict(
            threshold=float(thr), acc=float(acc), f1=float(f1),
            precision=float(prec), recall=float(rec), specificity=float(spec),
            youdenJ=float(rec+spec-1), tp=tp, tn=tn, fp=fp, fn=fn,
        ))
    return rows


def pick_best(rows: List[dict], key: str) -> dict:
    return sorted(rows, key=lambda r: (-r[key], -r["f1"], -r["acc"], r["threshold"]))[0]


@torch.no_grad()
def eval_split(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for d0, d1, d2, lbl in loader:
        imgs   = [d0.to(device), d1.to(device), d2.to(device)]
        logits = model(imgs).detach().cpu().numpy().reshape(-1)
        probs  = sigmoid_np(logits).astype(np.float32)
        all_probs.append(probs)
        all_labels.append(lbl.numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    auroc  = float(roc_auc_score(labels, probs))
    ap     = float(average_precision_score(labels, probs))
    return auroc, ap, probs, labels


# ── Per-fold training ─────────────────────────────────────────────────────────

def run_fold(fold: int, args, device: torch.device) -> dict:
    print(f"\n{'='*60}")
    print(f"FOLD {fold} / {max(args.folds)}")
    print(f"{'='*60}")
    set_seed(SEED)

    fold_npz_dir = Path(args.fold_npz_dir)
    out_dir      = Path(args.out_dir) / f"fold{fold}_centralized"
    out_dir.mkdir(parents=True, exist_ok=True)

    ad = np.load(fold_npz_dir / f"active_dscope_fold{fold}.npz", allow_pickle=True)
    p6 = np.load(fold_npz_dir / f"passive_6in_fold{fold}.npz",   allow_pickle=True)
    p1 = np.load(fold_npz_dir / f"passive_1ft_fold{fold}.npz",   allow_pickle=True)

    y_train = ad["y_train"].astype(np.int64)
    y_val   = ad["y_val"].astype(np.int64)
    y_test  = ad["y_test"].astype(np.int64)

    paths = {
        "train": [ad["paths_train"].astype(str).tolist(),
                  p6["paths_train"].astype(str).tolist(),
                  p1["paths_train"].astype(str).tolist()],
        "val":   [ad["paths_val"].astype(str).tolist(),
                  p6["paths_val"].astype(str).tolist(),
                  p1["paths_val"].astype(str).tolist()],
        "test":  [ad["paths_test"].astype(str).tolist(),
                  p6["paths_test"].astype(str).tolist(),
                  p1["paths_test"].astype(str).tolist()],
    }

    print(f"[DATA] train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    tf_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = MultiModalDataset(paths["train"], y_train, args.image_root, tf_train)
    val_ds   = MultiModalDataset(paths["val"],   y_val,   args.image_root, tf_eval)
    test_ds  = MultiModalDataset(paths["test"],  y_test,  args.image_root, tf_eval)

    use_ckpt = args.batch_size >= 48 or args.grad_checkpointing

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)

    model     = CentralizedModel(use_checkpointing=use_ckpt).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = -1.0
    best_epoch     = 0
    bad            = 0
    best_path      = out_dir / "model_best.pt"
    history        = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0        = time.time()
        tr_losses = []

        for d0, d1, d2, lbl in train_loader:
            imgs   = [d0.to(device), d1.to(device), d2.to(device)]
            labels = lbl.to(device).view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        avg_loss                = float(np.mean(tr_losses))
        val_auroc, val_ap, _, _ = eval_split(model, val_loader, device)

        history.append(dict(epoch=epoch, train_loss=avg_loss,
                            val_auroc=val_auroc, val_prauc=val_ap))
        print(f"[Epoch {epoch:02d}/{MAX_EPOCHS}] loss={avg_loss:.4f} "
              f"val_auroc={val_auroc:.4f} ({time.time()-t0:.1f}s)")

        if val_auroc > best_val_auroc + MIN_DELTA:
            best_val_auroc = val_auroc
            best_epoch     = epoch
            bad            = 0
            torch.save(model.state_dict(), best_path)
        else:
            if epoch >= MIN_ROUNDS:
                bad += 1
                if bad >= PATIENCE:
                    print(f"[EarlyStopping] patience={PATIENCE} at epoch {epoch}.")
                    break

    write_csv(out_dir / "history.csv", history)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False))

    val_auroc,  val_ap,  val_probs,  y_val_np  = eval_split(model, val_loader,  device)
    test_auroc, test_ap, test_probs, y_test_np = eval_split(model, test_loader, device)

    print(f"[RESULT] fold={fold} test_auroc={test_auroc:.4f} test_prauc={test_ap:.4f}")

    thr_val  = threshold_sweep(y_val_np,  val_probs)
    thr_test = threshold_sweep(y_test_np, test_probs)
    write_csv(out_dir / "threshold_metrics_val.csv",  thr_val)
    write_csv(out_dir / "threshold_metrics_test.csv", thr_test)

    best_you = pick_best(thr_val, "youdenJ")
    best_f1  = pick_best(thr_val, "f1")

    summary = {
        "fold":                fold,
        "best_epoch":          best_epoch,
        "val_auroc":           val_auroc,
        "val_pr_auc":          val_ap,
        "test_auroc":          test_auroc,
        "test_pr_auc":         test_ap,
        "val_thr_best_youden": best_you["threshold"],
        "val_thr_best_f1":     best_f1["threshold"],
    }
    write_csv(out_dir / "summary.csv", [summary])
    return summary


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(summaries: List[dict], out_dir: Path):
    metrics = ["test_auroc", "test_pr_auc", "val_auroc", "val_pr_auc",
               "best_epoch", "val_thr_best_youden", "val_thr_best_f1"]
    results = {}
    for k in metrics:
        vals = [float(s[k]) for s in summaries if k in s]
        if vals:
            results[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    print(f"\n{'='*60}")
    print("CENTRALIZED E2E — 5-FOLD RESULTS")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k:30s}: {v['mean']:.4f} ± {v['std']:.4f}")

    agg_path = out_dir / "test_summary_mean_std.csv"
    write_csv(agg_path, [{"metric": k, "mean": v["mean"], "std": v["std"]}
                          for k, v in results.items()])
    print(f"Saved → {agg_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Centralized e2e baseline — mirrors VFL splitnn architecture")
    ap.add_argument("--image_root",   required=True)
    ap.add_argument("--fold_npz_dir", required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[1,2,3,4,5])
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--grad_checkpointing", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[SETUP] device={device} folds={args.folds}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    summaries = []
    for fold in args.folds:
        summaries.append(run_fold(fold, args, device))

    aggregate(summaries, Path(args.out_dir))


if __name__ == "__main__":
    main()
