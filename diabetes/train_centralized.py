# train_centralized_diabetes.py
#
# Centralized upper-bound baseline for the diabetes experiment.
#
# This script trains a single MLP on the full concatenated feature space
# (active silo X1 + passive silo X2) without any federation constraints.
# It serves as the centralized reference baseline against which the
# 10PH-DVFL decoupled architecture and SplitNN baseline are compared.
#
# Since all features are available at a single location, this baseline
# represents the performance ceiling that federated approaches aim to
# approach while preserving data locality.
#
# Architecture (matches the two-silo VFL architecture at inference):
#   Input (15 features) -> 16 -> ReLU -> 8 -> ReLU -> 1
#   BCEWithLogitsLoss with per-fold positive class weighting.
#
# Evaluation protocol:
#   - Stratified 5-fold cross-validation with fixed seed=42
#   - Standardization using training split statistics only
#   - Early stopping based on validation AUROC
#   - Classification threshold selected on validation set using max F1
#   - Primary metrics: AUROC and PR-AUC (reported in the paper)
#   - Secondary metrics: accuracy, balanced accuracy, precision, recall,
#     specificity, and F1 (computed at both Youden and max-F1 thresholds)

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


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility across numpy, torch, and CUDA."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class CentralMLP_Paper(nn.Module):
    """
    Centralized MLP baseline for the diabetes experiment.

    Trained on the full concatenated feature space (X1 + X2) to provide
    the performance upper bound for the federated experiments. The
    architecture matches the combined VFL architecture at inference time:
    input features -> 16 -> ReLU -> dropout -> 8 -> ReLU -> 1 (logit).

    BCEWithLogitsLoss is used with per-fold positive class weighting to
    handle the class imbalance in the diabetes dataset.
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_npz(npz_path: str):
    """
    Load the pre-computed cross-validation dataset from the NPZ file.
    Concatenates active-silo features X1 and passive-silo features X2
    into a single full feature matrix for centralized training.
    Returns the full feature matrix, labels, fold splits, and metadata.
    """
    d = np.load(npz_path, allow_pickle=True)
    X = np.concatenate(
        [d["X1"].astype(np.float32), d["X2"].astype(np.float32)], axis=1
    )
    y = d["y"].astype(np.int64)
    folds = list(d["folds"])
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return X, y, folds, meta


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_proba(
    model: nn.Module, X: np.ndarray, device: torch.device
) -> np.ndarray:
    """Compute predicted probabilities for a feature matrix."""
    model.eval()
    xb = torch.from_numpy(X).to(device)
    logits = model(xb)
    return torch.sigmoid(logits).detach().cpu().numpy()


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device
) -> tuple:
    """
    Compute AUROC and PR-AUC for a feature matrix and label array.
    These are the primary metrics reported in the paper.
    """
    probs = predict_proba(model, X, device)
    auroc = roc_auc_score(y, probs)
    prauc = average_precision_score(y, probs)
    return float(auroc), float(prauc)


@torch.no_grad()
def compute_loss(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    criterion,
    device: torch.device,
    batch_size: int = 4096,
) -> float:
    """
    Compute the average BCEWithLogitsLoss over a dataset in batches.
    Used for training history logging without loading the full dataset at once.
    """
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
    """Return the current learning rate from the first parameter group."""
    return float(opt.param_groups[0]["lr"])


# ---------------------------------------------------------------------------
# Threshold selection and threshold-dependent metrics
# ---------------------------------------------------------------------------

def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Extract TP, FP, TN, FN from a confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(fp), int(tn), int(fn)


def threshold_metrics(
    y_true: np.ndarray, probs: np.ndarray, thr: float
) -> dict:
    """
    Compute threshold-dependent classification metrics at a fixed threshold.
    Includes accuracy, balanced accuracy, precision, recall, specificity,
    F1 score, and raw confusion matrix counts.
    """
    y_pred = (probs >= thr).astype(np.int64)
    tp, fp, tn, fn = _confusion(y_true, y_pred)
    acc     = accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    spec    = tn / (tn + fp + 1e-12)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = 0.5 * (rec + spec)
    return {
        "thr": float(thr),
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "bal_acc": float(bal_acc),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def find_best_threshold_youden(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Select the threshold that maximises Youden's J statistic (TPR + TNR - 1)
    on the validation set. Provides a balanced operating point that accounts
    for both sensitivity and specificity.
    """
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
    """
    Select the threshold that maximises F1 score on the validation set.
    Used as the primary threshold selection method for consistency with
    the SplitNN and decoupled VFL scripts.
    """
    thresholds = np.unique(probs)
    best_t, best_f1 = 0.5, -1e18
    for t in thresholds:
        y_pred = (probs >= t).astype(np.int64)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _mean_std(vals: list) -> tuple:
    """Compute mean and standard deviation of a list of values."""
    vals = np.array(vals, dtype=float)
    return float(vals.mean()), float(vals.std())


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Centralized upper-bound baseline for the diabetes experiment."
    )
    ap.add_argument("--npz", default="diabetes_vfl_cv.npz",
                    help="Path to diabetes_vfl_cv.npz")
    ap.add_argument("--out_dir", default="./runs_diabetes_centralized",
                    help="Directory to write checkpoints and results")
    ap.add_argument("--rounds", type=int, default=100,
                    help="Maximum number of training epochs")
    ap.add_argument("--batch", type=int, default=256,
                    help="Training batch size")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Adam learning rate")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout probability (0.0 disables dropout)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
    ap.add_argument("--device", default="cpu",
                    help="Device to use: cpu or cuda")

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
        "thr_val_youden", "thr_val_maxf1",
        "test_acc_youden", "test_prec_youden", "test_rec_youden",
        "test_spec_youden", "test_f1_youden", "test_balacc_youden",
        "test_tp_youden", "test_fp_youden", "test_tn_youden", "test_fn_youden",
        "test_acc_maxf1", "test_prec_maxf1", "test_rec_maxf1",
        "test_spec_maxf1", "test_f1_maxf1", "test_balacc_maxf1",
        "test_tp_maxf1", "test_fp_maxf1", "test_tn_maxf1", "test_fn_maxf1",
        "test_acc_05", "test_prec_05", "test_rec_05",
        "test_spec_05", "test_f1_05", "test_balacc_05",
        "test_tp_05", "test_fp_05", "test_tn_05", "test_fn_05",
    ]

    for fold_i, split in enumerate(folds, start=1):
        tr = split["train"].astype(np.int64)
        va = split["val"].astype(np.int64)
        te = split["test"].astype(np.int64)

        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]
        Xte, yte = X[te], y[te]

        # Standardize features using training split statistics only to prevent
        # any leakage from validation or test sets into the training stage.
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

        # Compute positive class weight from the training split only.
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
        epochs_used = 0

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

            # Evaluate validation AUROC and PR-AUC.
            val_auroc, val_prauc = evaluate_metrics(model, Xvan, yva, device)
            if scheduler is not None:
                scheduler.step(val_auroc)

            train_loss = compute_loss(
                model, Xtrn, ytr, criterion, device,
                batch_size=args.loss_eval_batch
            )
            val_loss = compute_loss(
                model, Xvan, yva, criterion, device,
                batch_size=args.loss_eval_batch
            )

            hist_round.append(rnd)
            hist_train_loss.append(train_loss)
            hist_val_loss.append(val_loss)
            hist_val_auroc.append(val_auroc)
            hist_val_prauc.append(val_prauc)
            hist_lr.append(get_lr(opt))

            # Save the best checkpoint based on validation AUROC.
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_val_prauc = val_prauc
                best_round = rnd
                torch.save(model.state_dict(), best_path)
            else:

            epochs_used = rnd

            if rnd % 20 == 0 or rnd == 1 or rnd == args.rounds:
                print(
                    f"[fold {fold_i}] round {rnd:03d}/{args.rounds}  "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}  "
                    f"val_AUROC={val_auroc:.4f} val_PR-AUC={val_prauc:.4f}  "
                    f"best={best_val_auroc:.4f}@{best_round}  lr={get_lr(opt):.2e}"
                )

                print(
                    f"[fold {fold_i}] early stop at round {rnd} "
                )
                break

        # Save the final model state.
        torch.save(model.state_dict(), last_path)

        # Save per-fold training curves as a compressed NumPy archive.
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

        # Save per-round training history as a CSV file.
        hist_csv = os.path.join(args.out_dir, f"central_fold{fold_i}_history.csv")
        with open(hist_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "train_loss", "val_loss", "val_auroc", "val_prauc", "lr"])
            for i in range(len(hist_round)):
                w.writerow([
                    hist_round[i], hist_train_loss[i], hist_val_loss[i],
                    hist_val_auroc[i], hist_val_prauc[i], hist_lr[i],
                ])

        # Restore the best checkpoint and evaluate on the held-out test set.
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_auroc, test_prauc = evaluate_metrics(model, Xten, yte, device)

        # Compute predicted probabilities on validation and test sets
        # for threshold selection and threshold-dependent metric evaluation.
        val_probs  = predict_proba(model, Xvan, device)
        test_probs = predict_proba(model, Xten, device)

        # Select thresholds on the validation set only; apply to the test set.
        thr_youden = find_best_threshold_youden(yva, val_probs)
        thr_maxf1  = find_best_threshold_maxf1(yva, val_probs)

        m_youden = threshold_metrics(yte, test_probs, thr_youden)
        m_maxf1  = threshold_metrics(yte, test_probs, thr_maxf1)
        m_05     = threshold_metrics(yte, test_probs, 0.5)

        final_lr = get_lr(opt)
        msg = (
            f"[fold {fold_i}] DONE. best_val_AUROC={best_val_auroc:.4f} "
            f"(round {best_round}) | "
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
            final_lr, pw,
            thr_youden, thr_maxf1,
            m_youden["acc"], m_youden["precision"], m_youden["recall"],
            m_youden["specificity"], m_youden["f1"], m_youden["bal_acc"],
            m_youden["tp"], m_youden["fp"], m_youden["tn"], m_youden["fn"],
            m_maxf1["acc"], m_maxf1["precision"], m_maxf1["recall"],
            m_maxf1["specificity"], m_maxf1["f1"], m_maxf1["bal_acc"],
            m_maxf1["tp"], m_maxf1["fp"], m_maxf1["tn"], m_maxf1["fn"],
            m_05["acc"], m_05["precision"], m_05["recall"],
            m_05["specificity"], m_05["f1"], m_05["bal_acc"],
            m_05["tp"], m_05["fp"], m_05["tn"], m_05["fn"],
        ])

    # Save the cross-validation summary CSV with per-fold results.
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(summary_rows)

    # Print aggregated cross-validation results.
    test_aurocs = [r[3] for r in summary_rows]
    test_praucs = [r[4] for r in summary_rows]
    m_auc = _mean_std(test_aurocs)
    m_pr  = _mean_std(test_praucs)
    print("\n[CV TEST] AUROC mean±std:", f"{m_auc[0]:.4f} ± {m_auc[1]:.4f}")
    print("[CV TEST] PR-AUC mean±std:", f"{m_pr[0]:.4f} ± {m_pr[1]:.4f}")

    h = {name: i for i, name in enumerate(header)}
    youden_f1   = [r[h["test_f1_youden"]]      for r in summary_rows]
    youden_rec  = [r[h["test_rec_youden"]]      for r in summary_rows]
    youden_spec = [r[h["test_spec_youden"]]     for r in summary_rows]
    youden_bal  = [r[h["test_balacc_youden"]]   for r in summary_rows]
    thr_y       = [r[h["thr_val_youden"]]       for r in summary_rows]
    mf1_f1      = [r[h["test_f1_maxf1"]]        for r in summary_rows]
    thr_m       = [r[h["thr_val_maxf1"]]        for r in summary_rows]

    print("\n[CV TEST @ VAL-Youden threshold]")
    print(f"  thr mean±std: {_mean_std(thr_y)[0]:.4f} ± {_mean_std(thr_y)[1]:.4f}")
    print(f"  F1 mean±std: {_mean_std(youden_f1)[0]:.4f} ± {_mean_std(youden_f1)[1]:.4f}")
    print(f"  Recall mean±std: {_mean_std(youden_rec)[0]:.4f} ± {_mean_std(youden_rec)[1]:.4f}")
    print(f"  Specificity mean±std: {_mean_std(youden_spec)[0]:.4f} ± {_mean_std(youden_spec)[1]:.4f}")
    print(f"  BalAcc mean±std: {_mean_std(youden_bal)[0]:.4f} ± {_mean_std(youden_bal)[1]:.4f}")

    print("\n[CV TEST @ VAL-MaxF1 threshold]")
    print(f"  thr mean±std: {_mean_std(thr_m)[0]:.4f} ± {_mean_std(thr_m)[1]:.4f}")
    print(f"  F1 mean±std: {_mean_std(mf1_f1)[0]:.4f} ± {_mean_std(mf1_f1)[1]:.4f}")

    print(f"\n[OK] wrote summary: {summary_path}")
    if meta:
        print("[meta]", meta)


if __name__ == "__main__":
    main()
