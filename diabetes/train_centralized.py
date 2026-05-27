# train_centralized_diabetes_paper.py
# Centralized baseline for Diabetes (tabular) aligned with PPVFL-SplitNN paper:
# - MLP: 16 -> 8 -> 1 (Linear-ReLU, Linear-ReLU, Linear)
# - BCEWithLogitsLoss
# - Standardize using TRAIN only
# - pos_weight computed from TRAIN only
# - Early stopping on VAL AUROC
# - Save BEST checkpoint by VAL AUROC + save LAST
# - Save per-fold curves + per-fold history CSV + overall summary CSV
# - ALSO: threshold metrics (threshold chosen on VAL only; reported on TEST)

import argparse
import csv
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CentralMLP_Paper(nn.Module):
    """
    Paper-style architecture for Diabetes/Breast Cancer:
    FC units: 16, 8, 1 with Linear-ReLU activations.
    (Paper uses Linear-Sigmoid at output; we use logits + BCEWithLogitsLoss.)
    """
    def __init__(self, in_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    X = np.concatenate([d["X1"].astype(np.float32), d["X2"].astype(np.float32)], axis=1)
    y = d["y"].astype(np.int64)
    folds = list(d["folds"])  # each element is already a dict in your file
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return X, y, folds, meta


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xb = torch.from_numpy(X).to(device)
    logits = model(xb)
    return torch.sigmoid(logits).detach().cpu().numpy()


@torch.no_grad()
def evaluate_metrics(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device):
    probs = predict_proba(model, X, device)
    auroc = roc_auc_score(y, probs)
    prauc = average_precision_score(y, probs)
    return float(auroc), float(prauc)


@torch.no_grad()
def compute_loss(model: nn.Module, X: np.ndarray, y: np.ndarray, criterion, device: torch.device, batch_size: int = 4096):
    model.eval()
    n = len(X)
    tot = 0.0
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        yb = torch.from_numpy(y[i:i + batch_size]).to(device).float()
        logits = model(xb)
        loss = criterion(logits, yb)
        tot += float(loss.item()) * (len(xb) / n)
    return float(tot)


def get_lr(opt: torch.optim.Optimizer) -> float:
    return float(opt.param_groups[0]["lr"])


def _confusion(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(fp), int(tn), int(fn)


def threshold_metrics(y_true: np.ndarray, probs: np.ndarray, thr: float):
    y_pred = (probs >= thr).astype(np.int64)
    tp, fp, tn, fn = _confusion(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # sensitivity / TPR
    spec = tn / (tn + fp + 1e-12)  # TNR
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = 0.5 * (rec + spec)

    return {
        "thr": float(thr),
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "bal_acc": float(bal_acc),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def find_best_threshold_youden(y_true: np.ndarray, probs: np.ndarray) -> float:
    # maximize Youden J = TPR + TNR - 1
    thresholds = np.unique(probs)
    best_t, best_j = 0.5, -1e18
    for t in thresholds:
        y_pred = (probs >= t).astype(np.int64)
        tp, fp, tn, fn = _confusion(y_true, y_pred)
        tpr = tp / (tp + fn + 1e-12)
        tnr = tn / (tn + fp + 1e-12)
        j = tpr + tnr - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def find_best_threshold_maxf1(y_true: np.ndarray, probs: np.ndarray) -> float:
    thresholds = np.unique(probs)
    best_t, best_f1 = 0.5, -1e18
    for t in thresholds:
        y_pred = (probs >= t).astype(np.int64)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _mean_std(vals):
    vals = np.array(vals, dtype=float)
    return float(vals.mean()), float(vals.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="diabetes_vfl_cv.npz")
    ap.add_argument("--out_dir", default="./ckpts_diabetes_central_paperMLP")

    # training
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.0)

    # reproducibility / device
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")

    # early stop + scheduler
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--scheduler", choices=["none", "plateau"], default="plateau")
    ap.add_argument("--plateau_patience", type=int, default=5)
    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    # logging loss efficiently
    ap.add_argument("--loss_eval_batch", type=int, default=4096)

    # thresholding mode (we always compute both youden and maxf1; this just controls printing)
    ap.add_argument("--print_thresholds", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    X, y, folds, meta = load_npz(args.npz)

    summary_rows = []
    summary_path = os.path.join(args.out_dir, "central_cv_summary.csv")

    header = [
        "fold",
        "best_val_auroc", "best_val_prauc",
        "test_auroc_at_best", "test_prauc_at_best",
        "best_round", "epochs_used",
        "final_lr",
        "pos_weight_train",

        # thresholds chosen on VAL
        "thr_val_youden", "thr_val_maxf1",

        # TEST metrics @ VAL-Youden
        "test_acc_youden", "test_prec_youden", "test_rec_youden", "test_spec_youden", "test_f1_youden", "test_balacc_youden",
        "test_tp_youden", "test_fp_youden", "test_tn_youden", "test_fn_youden",

        # TEST metrics @ VAL-MaxF1
        "test_acc_maxf1", "test_prec_maxf1", "test_rec_maxf1", "test_spec_maxf1", "test_f1_maxf1", "test_balacc_maxf1",
        "test_tp_maxf1", "test_fp_maxf1", "test_tn_maxf1", "test_fn_maxf1",

        # TEST metrics @ 0.5
        "test_acc_05", "test_prec_05", "test_rec_05", "test_spec_05", "test_f1_05", "test_balacc_05",
        "test_tp_05", "test_fp_05", "test_tn_05", "test_fn_05",
    ]

    for fold_i, split in enumerate(folds, start=1):
        tr = split["train"].astype(np.int64)
        va = split["val"].astype(np.int64)
        te = split["test"].astype(np.int64)

        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]
        Xte, yte = X[te], y[te]

        # ---- standardize using TRAIN only ----
        mu = Xtr.mean(0, keepdims=True)
        sd = Xtr.std(0, keepdims=True) + 1e-6
        Xtrn = (Xtr - mu) / sd
        Xvan = (Xva - mu) / sd
        Xten = (Xte - mu) / sd

        model = CentralMLP_Paper(in_dim=X.shape[1], dropout=args.dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        scheduler = None
        if args.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max",
                factor=args.plateau_factor,
                patience=args.plateau_patience,
                min_lr=args.min_lr,
            )

        # ---- pos_weight from TRAIN only ----
        pos = float(ytr.sum())
        neg = float(len(ytr) - ytr.sum())
        pw = neg / max(pos, 1.0)
        pos_weight = torch.tensor([pw], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_auroc, best_val_prauc, best_round = -1.0, -1.0, -1
        best_path = os.path.join(args.out_dir, f"central_best_fold{fold_i}.pt")
        last_path = os.path.join(args.out_dir, f"central_last_fold{fold_i}.pt")

        n = len(Xtrn)
        steps = math.ceil(n / args.batch)

        hist_round, hist_train_loss, hist_val_loss = [], [], []
        hist_val_auroc, hist_val_prauc, hist_lr = [], [], []

        no_improve, epochs_used = 0, 0

        for rnd in range(1, args.rounds + 1):
            model.train()
            perm = np.random.permutation(n)
            Xb = Xtrn[perm]
            yb = ytr[perm]

            for s in range(steps):
                a = s * args.batch
                b = min(n, (s + 1) * args.batch)
                xb = torch.from_numpy(Xb[a:b]).to(device)
                yy = torch.from_numpy(yb[a:b]).to(device).float()

                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yy)
                loss.backward()
                opt.step()

            # validate (AUROC/PR-AUC)
            val_auroc, val_prauc = evaluate_metrics(model, Xvan, yva, device)
            if scheduler is not None:
                scheduler.step(val_auroc)

            train_loss = compute_loss(model, Xtrn, ytr, criterion, device, batch_size=args.loss_eval_batch)
            val_loss = compute_loss(model, Xvan, yva, criterion, device, batch_size=args.loss_eval_batch)

            hist_round.append(rnd)
            hist_train_loss.append(train_loss)
            hist_val_loss.append(val_loss)
            hist_val_auroc.append(val_auroc)
            hist_val_prauc.append(val_prauc)
            hist_lr.append(get_lr(opt))

            # ---- best checkpoint on VAL AUROC ----
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_val_prauc = val_prauc
                best_round = rnd
                torch.save(model.state_dict(), best_path)
                no_improve = 0
            else:
                no_improve += 1

            epochs_used = rnd

            if rnd % 20 == 0 or rnd == 1 or rnd == args.rounds:
                print(
                    f"[fold {fold_i}] round {rnd:03d}/{args.rounds}  "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}  "
                    f"val_AUROC={val_auroc:.4f} val_PR-AUC={val_prauc:.4f}  "
                    f"best={best_val_auroc:.4f}@{best_round}  lr={get_lr(opt):.2e}"
                )

            if args.patience > 0 and no_improve >= args.patience:
                print(f"[fold {fold_i}] early stop at round {rnd} (no val AUROC improvement for {args.patience} rounds)")
                break

        # save last
        torch.save(model.state_dict(), last_path)

        # save curves
        curves_path = os.path.join(args.out_dir, f"central_fold{fold_i}_curves.npz")
        np.savez(
            curves_path,
            round=np.array(hist_round, dtype=np.int32),
            train_loss=np.array(hist_train_loss, dtype=np.float32),
            val_loss=np.array(hist_val_loss, dtype=np.float32),
            val_auroc=np.array(hist_val_auroc, dtype=np.float32),
            val_prauc=np.array(hist_val_prauc, dtype=np.float32),
            lr=np.array(hist_lr, dtype=np.float32),
        )

        # save history CSV
        hist_csv = os.path.join(args.out_dir, f"central_fold{fold_i}_history.csv")
        with open(hist_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "train_loss", "val_loss", "val_auroc", "val_prauc", "lr"])
            for i in range(len(hist_round)):
                w.writerow([hist_round[i], hist_train_loss[i], hist_val_loss[i],
                            hist_val_auroc[i], hist_val_prauc[i], hist_lr[i]])

        # ---- test at BEST (val AUROC) ----
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_auroc, test_prauc = evaluate_metrics(model, Xten, yte, device)

        # probs for thresholds
        val_probs = predict_proba(model, Xvan, device)
        test_probs = predict_proba(model, Xten, device)

        thr_youden = find_best_threshold_youden(yva, val_probs)
        thr_maxf1 = find_best_threshold_maxf1(yva, val_probs)

        m_youden = threshold_metrics(yte, test_probs, thr_youden)
        m_maxf1 = threshold_metrics(yte, test_probs, thr_maxf1)
        m_05 = threshold_metrics(yte, test_probs, 0.5)

        final_lr = get_lr(opt)
        msg = (
            f"[fold {fold_i}] DONE. best_val_AUROC={best_val_auroc:.4f} (round {best_round}) | "
            f"test_AUROC={test_auroc:.4f} test_PR-AUC={test_prauc:.4f} | "
            f"epochs_used={epochs_used} | final_lr={final_lr:.2e}"
        )
        if args.print_thresholds:
            msg += (
                f" | thr_youden={thr_youden:.4f} testF1_youden={m_youden['f1']:.4f}"
                f" | thr_maxf1={thr_maxf1:.4f} testF1_maxf1={m_maxf1['f1']:.4f}"
            )
        print(msg)

        summary_rows.append([
            fold_i,
            best_val_auroc, best_val_prauc,
            test_auroc, test_prauc,
            best_round, epochs_used,
            final_lr,
            pw,

            thr_youden, thr_maxf1,

            m_youden["acc"], m_youden["precision"], m_youden["recall"], m_youden["specificity"], m_youden["f1"], m_youden["bal_acc"],
            m_youden["tp"], m_youden["fp"], m_youden["tn"], m_youden["fn"],

            m_maxf1["acc"], m_maxf1["precision"], m_maxf1["recall"], m_maxf1["specificity"], m_maxf1["f1"], m_maxf1["bal_acc"],
            m_maxf1["tp"], m_maxf1["fp"], m_maxf1["tn"], m_maxf1["fn"],

            m_05["acc"], m_05["precision"], m_05["recall"], m_05["specificity"], m_05["f1"], m_05["bal_acc"],
            m_05["tp"], m_05["fp"], m_05["tn"], m_05["fn"],
        ])

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(summary_rows)

    # ---- aggregate reporting ----
    test_aurocs = [r[3] for r in summary_rows]
    test_praucs = [r[4] for r in summary_rows]
    m_auc = _mean_std(test_aurocs)
    m_pr = _mean_std(test_praucs)
    print("\n[CV TEST] AUROC mean±std:", m_auc[0], m_auc[1])
    print("[CV TEST] PR-AUC mean±std:", m_pr[0], m_pr[1])

    # Aggregates for Youden threshold metrics
    # indices in row: locate by header names for safety
    h = {name: i for i, name in enumerate(header)}
    youden_f1 = [r[h["test_f1_youden"]] for r in summary_rows]
    youden_rec = [r[h["test_rec_youden"]] for r in summary_rows]
    youden_spec = [r[h["test_spec_youden"]] for r in summary_rows]
    youden_bal = [r[h["test_balacc_youden"]] for r in summary_rows]
    thr_y = [r[h["thr_val_youden"]] for r in summary_rows]

    mf1_f1 = [r[h["test_f1_maxf1"]] for r in summary_rows]
    thr_m = [r[h["thr_val_maxf1"]] for r in summary_rows]

    print("\n[CV TEST @ VAL-Youden threshold]")
    print("  thr mean±std:", *_mean_std(thr_y))
    print("  F1 mean±std:", *_mean_std(youden_f1))
    print("  Recall mean±std:", *_mean_std(youden_rec))
    print("  Specificity mean±std:", *_mean_std(youden_spec))
    print("  BalAcc mean±std:", *_mean_std(youden_bal))

    print("\n[CV TEST @ VAL-MaxF1 threshold]")
    print("  thr mean±std:", *_mean_std(thr_m))
    print("  F1 mean±std:", *_mean_std(mf1_f1))

    print(f"\n[OK] wrote summary: {summary_path}")
    if meta:
        print("[meta]", meta)


if __name__ == "__main__":
    main()
