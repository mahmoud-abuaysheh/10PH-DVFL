#!/usr/bin/env python3
"""
run_vfl_splitnn_5fold.py
========================
Runs all 5 folds of VFL SplitNN then aggregates results.

Usage (Windows):
  python run_vfl_splitnn_5fold.py ^
      --fold_npz_dir "C:\Thesis\RAD\MIDAS_SkinCancer\resnet\scripts\fold_npz" ^
      --image_root "C:\Thesis\RAD\MIDAS_SkinCancer\midasmultimodalimagedatasetforaibasedskincancer" ^
      --out_dir "runs_vfl_splitnn_standalone" ^
      --batch_size 32

Usage (Linux/WSL):
  python run_vfl_splitnn_5fold.py \
      --fold_npz_dir fold_npz \
      --image_root /mnt/c/Thesis/RAD/MIDAS_SkinCancer/... \
      --out_dir runs_vfl_splitnn_standalone \
      --batch_size 32
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run_fold(fold: int, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "train_vfl_splitnn_midas.py",
        "--fold",         str(fold),
        "--fold_npz_dir", args.fold_npz_dir,
        "--image_root",   args.image_root,
        "--out_dir",      args.out_dir,
        "--rounds",       str(args.rounds),
        "--patience",     str(args.patience),
        "--min_rounds",   str(args.min_rounds),
        "--batch_size",   str(args.batch_size),
        "--emb_dim",      str(args.emb_dim),
        "--proj_hidden",  str(args.proj_hidden),
        "--head_hidden",  str(args.head_hidden),
        "--client_lr",    str(args.client_lr),
        "--head_lr",      str(args.head_lr),
        "--seed",         str(args.seed),
    ]
    if args.freeze_backbone:
        cmd.append("--freeze_backbone")

    print(f"\n{'='*70}")
    print(f"  FOLD {fold}")
    print(f"{'='*70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Fold {fold} failed with return code {result.returncode}")
        sys.exit(result.returncode)


def aggregate(out_dir: str, folds: list) -> None:
    results = []
    for fold in folds:
        summary_path = Path(out_dir) / f"fold{fold}_vfl_splitnn" / "summary.csv"
        if not summary_path.exists():
            print(f"[WARN] Missing summary: {summary_path}")
            continue
        with summary_path.open() as f:
            row = list(csv.DictReader(f))[0]
            results.append(row)

    if not results:
        print("[WARN] No results to aggregate.")
        return

    import numpy as np
    metrics = ["test_auroc", "test_pr_auc", "val_auroc", "val_pr_auc",
               "best_round", "total_rounds",
               "comm_cost_per_round_MB", "comm_cost_total_MB"]

    agg_rows = []
    for m in metrics:
        vals = [float(r[m]) for r in results if m in r]
        if vals:
            agg_rows.append({
                "metric": m,
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals)),
                "values": str(vals),
            })
            print(f"  {m:35s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    out_path = Path(out_dir) / "test_summary_mean_std.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "values"])
        w.writeheader()
        w.writerows(agg_rows)

    print(f"\n[AGG] Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold_npz_dir",    required=True)
    p.add_argument("--image_root",      required=True)
    p.add_argument("--out_dir",         default="runs_vfl_splitnn_standalone")
    p.add_argument("--folds",           nargs="+", type=int, default=[1,2,3,4,5])
    p.add_argument("--rounds",          type=int,   default=20)
    p.add_argument("--patience",        type=int,   default=7)
    p.add_argument("--min_rounds",      type=int,   default=5)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--emb_dim",         type=int,   default=256)
    p.add_argument("--proj_hidden",     type=int,   default=512)
    p.add_argument("--head_hidden",     type=int,   default=512)
    p.add_argument("--client_lr",       type=float, default=1e-4)
    p.add_argument("--head_lr",         type=float, default=1e-4)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--seed",            type=int,   default=42)
    args = p.parse_args()

    for fold in args.folds:
        run_fold(fold, args)

    print(f"\n{'='*70}")
    print("  AGGREGATING RESULTS")
    print(f"{'='*70}")
    aggregate(args.out_dir, args.folds)


if __name__ == "__main__":
    main()