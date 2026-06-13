# train_centralized_glioma.py
#
# Centralized upper-bound baseline for the glioma experiment.
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
# Architecture (matches the combined VFL architecture at inference):
#   Input -> 32 -> ReLU -> 16 -> ReLU -> 8 -> ReLU -> 1
#   BCEWithLogitsLoss with per-fold positive class weighting.
#
# Evaluation protocol:
#   - Stratified 5-fold cross-validation with fixed seed=42
#   - Standardization using training split statistics only
#   - Best checkpoint selected by validation AUROC
#   - Learning rate scheduler disabled to match VFL fixed LR
#   - Primary metrics: AUROC and PR-AUC (reported in the paper)
#   - Secondary threshold-dependent metrics: accuracy, sensitivity, specificity,
#     precision, F1 at Youden, max-F1, and fixed 0.5 thresholds

import argparse
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score


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

class CentralMLP(nn.Module):
    """
    Centralized MLP baseline for the glioma experiment.

    Trained on the full concatenated feature space (X1 + X2) to provide
    the performance upper bound for the federated experiments. The
    architecture mirrors the combined VFL architecture at inference:
    input -> 32 -> ReLU -> 16 -> ReLU -> 8 -> ReLU -> 1 (logit).

    This matches the glioma VFL bottom encoder output dimension (16 per silo,
    32 concatenated) followed by the same 3-layer fusion head used in the
    decoupled and SplitNN VFL configurations.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
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
    Returns the full feature matrix, labels, and fold splits.
    """
    data = np.load(npz_path, allow_pickle=True)
    X = np.concatenate(
        [data["X1"].astype(np.float32), data["X2"].astype(np.float32)], axis=1
    )
    y     = data["y"].astype(np.int64)
    folds = list(data["folds"])
    return X, y, folds


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device
) -> tuple:
    """
    Compute AUROC and PR-AUC for a feature matrix and label array.
    These are the primary metrics reported in the paper.
    """
    model.eval()
    xb     = torch.from_numpy(X).to(device)
    logits = model(xb)
    probs  = torch.sigmoid(logits).detach().cpu().numpy()
    auroc  = roc_auc_score(y, probs)
    prauc  = average_precision_score(y, probs)
    return float(auroc), float(prauc)


@torch.no_grad()
def compute_loss(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    criterion,
    device: torch.device,
    batch_size: int = 1024,
) -> float:
    """
    Compute the average BCEWithLogitsLoss over a dataset in batches.
    Used for training history logging without loading the full dataset at once.
    """
    model.eval()
    n   = len(X)
    tot = 0.0
    for i in range(0, n, batch_size):
        xb     = torch.from_numpy(X[i:i + batch_size]).to(device)
        yb     = torch.from_numpy(y[i:i + batch_size]).to(device).float()
        logits = model(xb)
        loss   = criterion(logits, yb)
        tot   += float(loss.item()) * (len(xb) / n)
    return float(tot)


def get_lr(opt: torch.optim.Optimizer) -> float:
    """Return the current learning rate from the first parameter group."""
    return float(opt.param_groups[0]["lr"])


# ---------------------------------------------------------------------------
# Threshold selection and threshold-dependent metrics
# ---------------------------------------------------------------------------

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Extract TN, FP, FN, TP from binary prediction arrays."""
    y_true = y_true.astype(np.int64).reshape(-1)
    y_pred = y_pred.astype(np.int64).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tn, fp, fn, tp


def metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> tuple:
    """
    Compute threshold-dependent classification metrics from confusion matrix counts.
    Returns accuracy, sensitivity (recall), specificity, precision, and F1.
    """
    eps  = 1e-12
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1   = (2 * prec * sens) / max(prec + sens, eps)
    return acc, sens, spec, prec, f1


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Select classification thresholds on the validation set using two criteria:
      - Youden's J statistic (maximises sensitivity + specificity - 1)
      - Maximum F1 score
    Returns a dict mapping rule name to threshold value, including fixed 0.5.
    All thresholds are selected on the validation set and applied to the test
    set without further tuning.
    """
    y_true = y_true.astype(np.int64).reshape(-1)
    y_prob = y_prob.astype(np.float64).reshape(-1)
    thr_c  = np.unique(y_prob)
    if thr_c.size == 0:
        return {"0.5": 0.5, "youden": 0.5, "maxf1": 0.5}
    best_youden = {"thr": 0.5, "score": -1e9}
    best_f1     = {"thr": 0.5, "score": -1e9}
    for thr in thr_c:
        y_pred = (y_prob >= thr).astype(np.int64)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        _, sens, _, _, f1 = metrics_from_counts(tn, fp, fn, tp)
        fpr    = fp / max(fp + tn, 1)
        youden = sens - fpr
        if youden > best_youden["score"]:
            best_youden = {"thr": float(thr), "score": float(youden)}
        if f1 > best_f1["score"]:
            best_f1 = {"thr": float(thr), "score": float(f1)}
    return {"0.5": 0.5, "youden": best_youden["thr"], "maxf1": best_f1["thr"]}


def write_threshold_report(csv_path: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """
    Write threshold-dependent metrics for all three threshold rules to a CSV file.
    Thresholds are selected on the validation set and applied to test predictions.
    """
    thr_map = find_best_thresholds(y_true, y_prob)
    rows = []
    for rule, thr in thr_map.items():
        y_pred = (y_prob >= thr).astype(np.int64)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        acc, sens, spec, prec, f1 = metrics_from_counts(tn, fp, fn, tp)
        rows.append([rule, thr, acc, sens, spec, prec, f1, tn, fp, fn, tp])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "thr", "acc", "sens", "spec", "prec", "f1", "tn", "fp", "fn", "tp"])
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Centralized upper-bound baseline for the glioma experiment."
    )
    ap.add_argument("--npz",      required=True,
                    help="Path to glioma_aligned_vfl_hfl_cv.npz")
    ap.add_argument("--out_dir",  default="./runs_centralized_glioma",
                    help="Directory to write checkpoints and results")
    ap.add_argument("--rounds",   type=int,   default=100,
                    help="Number of training epochs (matches VFL epochs=100)")
    ap.add_argument("--batch",    type=int,   default=64,
                    help="Training batch size (matches VFL batch_size=64)")
    ap.add_argument("--lr",       type=float, default=1e-3,
                    help="Adam learning rate")
    ap.add_argument("--folds",    type=int,   default=5,
                    help="Number of cross-validation folds")
    ap.add_argument("--seed",     type=int,   default=42,
                    help="Random seed for reproducibility")
    ap.add_argument("--device",   default="cpu",
                    help="Device to use: cpu or cuda")
    ap.add_argument("--scheduler", choices=["none", "plateau"], default="none",
                    help="Learning rate scheduler (disabled to match VFL fixed LR)")
    ap.add_argument("--plateau_patience", type=int,   default=5,
                    help="ReduceLROnPlateau patience (only used if scheduler=plateau)")
    ap.add_argument("--plateau_factor",   type=float, default=0.5,
                    help="ReduceLROnPlateau reduction factor")
    ap.add_argument("--min_lr",           type=float, default=1e-6,
                    help="Minimum learning rate for ReduceLROnPlateau")
    ap.add_argument("--loss_eval_batch",  type=int,   default=1024,
                    help="Batch size for loss evaluation (does not affect training)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    X, y, folds = load_npz(args.npz)
    assert args.folds == len(folds), \
        f"--folds {args.folds} does not match npz folds {len(folds)}"

    summary_rows = []
    summary_path = os.path.join(args.out_dir, "central_cv_summary.csv")
    header = [
        "fold", "best_val_auroc", "best_val_prauc",
        "test_auroc_at_best", "test_prauc_at_best",
        "best_round", "epochs_used", "final_lr",
    ]

    for fold_i, split_obj in enumerate(folds, start=1):
        split = split_obj.item() if hasattr(split_obj, "item") else split_obj

        tr = split["train"].astype(np.int64)
        va = split["val"].astype(np.int64)
        te = split["test"].astype(np.int64)

        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]
        Xte, yte = X[te], y[te]

        # Standardize features using training split statistics only to prevent
        # any leakage from validation or test sets into the training stage.
        mu   = Xtr.mean(0, keepdims=True)
        sd   = Xtr.std(0, keepdims=True) + 1e-6
        Xtrn = (Xtr - mu) / sd
        Xvan = (Xva - mu) / sd
        Xten = (Xte - mu) / sd

        model = CentralMLP(X.shape[1]).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

        scheduler = None
        if args.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max",
                factor=args.plateau_factor,
                min_lr=args.min_lr,
            )

        # Compute positive class weight from the training split only.
        pos        = float(ytr.sum())
        neg        = float(len(ytr) - ytr.sum())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_auroc = -1.0
        best_val_prauc = -1.0
        best_round     = -1
        best_path      = os.path.join(args.out_dir, f"central_best_fold{fold_i}.pt")
        last_path      = os.path.join(args.out_dir, f"central_last_fold{fold_i}.pt")

        n     = len(Xtrn)
        steps = math.ceil(n / args.batch)

        hist_round      = []
        hist_train_loss = []
        hist_val_loss   = []
        hist_val_auroc  = []
        hist_val_prauc  = []
        hist_lr         = []

        epochs_used = 0

        for rnd in range(1, args.rounds + 1):
            model.train()
            perm = np.random.permutation(n)
            Xb   = Xtrn[perm]
            yb   = ytr[perm]

            for s in range(steps):
                a  = s * args.batch
                b  = min(n, (s + 1) * args.batch)
                xb = torch.from_numpy(Xb[a:b]).to(device)
                yy = torch.from_numpy(yb[a:b]).to(device).float()
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss   = criterion(logits, yy)
                loss.backward()
                opt.step()

            # Evaluate validation AUROC and PR-AUC for checkpoint selection.
            val_auroc, val_prauc = evaluate(model, Xvan, yva, device)
            if scheduler is not None:
                scheduler.step(val_auroc)

            # Compute and log train and validation loss for history tracking.
            train_loss = compute_loss(
                model, Xtrn, ytr, criterion, device,
                batch_size=args.loss_eval_batch,
            )
            val_loss = compute_loss(
                model, Xvan, yva, criterion, device,
                batch_size=args.loss_eval_batch,
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
                best_round     = rnd
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

        # Save the final model state at end of training.
        torch.save(model.state_dict(), last_path)

        # Save per-fold training curves as a compressed NumPy archive.
        curves_path = os.path.join(args.out_dir, f"central_fold{fold_i}_curves.npz")
        np.savez(
            curves_path,
            round=np.array(hist_round,      dtype=np.int32),
            train_loss=np.array(hist_train_loss, dtype=np.float32),
            val_loss=np.array(hist_val_loss,   dtype=np.float32),
            val_auroc=np.array(hist_val_auroc,  dtype=np.float32),
            val_prauc=np.array(hist_val_prauc,  dtype=np.float32),
            lr=np.array(hist_lr,           dtype=np.float32),
        )

        # Save per-round training history as a CSV file.
        hist_csv = os.path.join(args.out_dir, f"central_fold{fold_i}_history.csv")
        hist_arr = np.column_stack([
            np.array(hist_round,      dtype=np.int32),
            np.array(hist_train_loss, dtype=np.float32),
            np.array(hist_val_loss,   dtype=np.float32),
            np.array(hist_val_auroc,  dtype=np.float32),
            np.array(hist_val_prauc,  dtype=np.float32),
            np.array(hist_lr,         dtype=np.float32),
        ])
        with open(hist_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "train_loss", "val_loss", "val_auroc", "val_prauc", "lr"])
            w.writerows(hist_arr.tolist())

        # Restore the best checkpoint and evaluate on the held-out test set.
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_auroc, test_prauc = evaluate(model, Xten, yte, device)

        # Compute test set predicted probabilities for threshold-dependent metrics.
        # Thresholds are selected on the validation set and applied to test predictions.
        model.eval()
        with torch.no_grad():
            test_logits = model(
                torch.from_numpy(Xten).to(device)
            ).detach().cpu().numpy()
        test_prob = 1.0 / (1.0 + np.exp(-test_logits))  # Numerically stable sigmoid.

        thr_csv = os.path.join(args.out_dir, f"test_thresholds_fold{fold_i}.csv")
        write_threshold_report(thr_csv, yte, test_prob)

        final_lr = get_lr(opt)
        print(
            f"[fold {fold_i}] DONE. best_val_AUROC={best_val_auroc:.4f} "
            f"(round {best_round}) | "
            f"test_AUROC={test_auroc:.4f} test_PR-AUC={test_prauc:.4f} | "
            f"epochs_used={epochs_used} | final_lr={final_lr:.2e}"
        )

        summary_rows.append([
            fold_i, best_val_auroc, best_val_prauc,
            test_auroc, test_prauc,
            best_round, epochs_used, final_lr,
        ])

    # Save the cross-validation summary CSV with per-fold results.
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(summary_rows)

    test_aurocs = np.array([r[3] for r in summary_rows], dtype=float)
    test_praucs = np.array([r[4] for r in summary_rows], dtype=float)
    print(f"\n[CV TEST] AUROC mean±std: {test_aurocs.mean():.4f} ± {test_aurocs.std():.4f}")
    print(f"[CV TEST] PR-AUC mean±std: {test_praucs.mean():.4f} ± {test_praucs.std():.4f}")
    print(f"[OK] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
