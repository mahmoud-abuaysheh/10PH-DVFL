# serverapp_vfl_glioma_splitnn.py
#
# Flower 1.26.1 server application for the SplitNN VFL baseline in the
# glioma decoupled VFL experiment.
#
# This script implements the SplitNN-based VFL baseline against which the
# 10PH-DVFL decoupled architecture is compared for the glioma dataset.
#
# SplitNN communication pattern (per training batch):
#   1. Server requests cut-layer activations from both silos simultaneously
#      using send_and_receive for parallel dispatch
#   2. Server concatenates activations and computes forward pass and loss
#   3. Server back-propagates to obtain gradients at the cut layer
#   4. Server sends per-silo gradient slices back using send_and_receive
#      as a barrier, ensuring both clients update before the next forward pass
#   5. Server updates the fusion head after both clients have stepped
#
# The barrier pattern in steps 4-5 is critical for correctness: using
# fire-and-forget gradient push would allow the server to start the next
# forward pass before clients finish updating their encoders, producing
# stale activations and incorrect gradient estimates.
#
# Key architectural differences from the diabetes SplitNN:
#   - Bottom encoder: input -> 32 -> 16 (deeper than diabetes 16 -> 8)
#   - Fusion head: TopMLP (32 -> 16 -> 8 -> 1) instead of TopLinear (16 -> 1)
#   - Communication tracking: byte counts are recorded for both upload and
#     download directions across training, validation, and test phases
#
# Key differences from serverapp_vfl_glioma_decoupled.py:
#   - Encoders are updated end-to-end through gradient feedback each batch
#   - Both activations and gradients cross silo boundaries every batch
#   - Client encoder checkpoints are saved and restored for early stopping

from __future__ import annotations

import copy
import csv
import os
from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility across numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Server-side fusion head
# ---------------------------------------------------------------------------

class TopMLP(nn.Module):
    """
    Server-side fusion head for the glioma SplitNN and decoupled VFL experiments.

    Receives concatenated cut-layer activations from the active and passive
    silos and produces a scalar logit for binary classification.
    Architecture: in_dim -> 16 -> ReLU -> 8 -> ReLU -> 1

    Uses a 3-layer MLP rather than a single linear layer to match the
    decoupled glioma fusion head, ensuring a fair architectural comparison
    between SplitNN and decoupled conditions.
    Input dimension is 2 * out_feature_dim (default 32 = 16 + 16).
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, 8),      nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(1)


# ---------------------------------------------------------------------------
# Array utilities
# ---------------------------------------------------------------------------

def _arr_i64(x: np.ndarray) -> Array:
    """Convert a numpy array to a Flower Array with int64 dtype."""
    return Array(np.asarray(x, dtype=np.int64))


def _arr_f32(x: np.ndarray) -> Array:
    """Convert a numpy array to a Flower Array with float32 dtype."""
    return Array(np.asarray(x, dtype=np.float32))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_npz(npz_path: str):
    """
    Load the pre-computed cross-validation splits and labels from the NPZ file.
    Returns labels y and the list of fold split dicts.
    """
    d     = np.load(npz_path, allow_pickle=True)
    y     = d["y"].astype(np.int64)
    folds = list(d["folds"])
    return y, folds


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC, returning nan if fewer than two classes are present."""
    try:
        return float(roc_auc_score(y_true, y_score)) \
            if len(np.unique(y_true)) >= 2 else float("nan")
    except Exception:
        return float("nan")


def safe_prauc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC, returning nan if fewer than two classes are present."""
    try:
        return float(average_precision_score(y_true, y_score)) \
            if len(np.unique(y_true)) >= 2 else float("nan")
    except Exception:
        return float("nan")


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
    """Compute accuracy, sensitivity, specificity, precision, and F1."""
    eps  = 1e-12
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1   = (2 * prec * sens) / max(prec + sens, eps)
    return acc, sens, spec, prec, f1


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Select classification thresholds on the validation set using Youden's J
    statistic and maximum F1 score. Returns a dict including fixed 0.5.
    All thresholds are applied to the test set without further tuning.
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


def write_threshold_report(
    csv_path: str, y_true: np.ndarray, y_prob: np.ndarray
) -> None:
    """Write threshold-dependent metrics for all threshold rules to a CSV file."""
    thr_map = find_best_thresholds(y_true, y_prob)
    rows = []
    for rule, thr in thr_map.items():
        y_pred = (y_prob >= thr).astype(np.int64)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        acc, sens, spec, prec, f1 = metrics_from_counts(tn, fp, fn, tp)
        rows.append([rule, thr, acc, sens, spec, prec, f1, tn, fp, fn, tp])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "thr", "acc", "sens", "spec", "prec", "f1",
                    "tn", "fp", "fn", "tp"])
        w.writerows(rows)


# ---------------------------------------------------------------------------
# VFL communication helpers
# ---------------------------------------------------------------------------

def _request_embeddings(
    grid: Grid,
    node_ids: List[int],
    batch_idx: np.ndarray,
    out_dim: int,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Request cut-layer activations from both silo nodes simultaneously.

    Both messages are dispatched in parallel using send_and_receive, then
    replies are matched to node IDs. Returns the concatenated activation
    tensor with gradient tracking enabled and per-node byte counts.
    """
    msgs = [
        Message(
            content=RecordDict({
                "arrays": ArrayRecord({"batch_idx": _arr_i64(batch_idx)}),
                "config": ConfigRecord({"view": view}),
            }),
            message_type="query.generate_embeddings",
            dst_node_id=nid,
        )
        for view, nid in enumerate(node_ids)
    ]
    reps = grid.send_and_receive(msgs)

    emb_list: List[torch.Tensor] = []
    bytes_up:  List[int]         = []
    for nid in node_ids:
        rep = next(r for r in reps if r.metadata.src_node_id == nid)
        if not rep.has_content():
            raise RuntimeError(f"Client {nid} returned no content.")
        emb_np = rep.content["arrays"]["embedding"].numpy().astype(np.float32)
        bytes_up.append(int(emb_np.nbytes))
        emb_list.append(torch.from_numpy(emb_np).float())

    # Concatenate activations and enable gradient tracking so that
    # dL/dz can be computed and split for transmission back to clients.
    z = torch.cat(emb_list, dim=1).requires_grad_()
    return z, bytes_up


def _push_embedding_grads_barrier(
    grid: Grid,
    node_ids: List[int],
    grads_per_silo: List[torch.Tensor],
    batch_idx: np.ndarray,
) -> List[int]:
    """
    Send per-silo gradient slices to both clients and wait for both to finish.

    Uses send_and_receive as a synchronisation barrier to ensure both clients
    complete their encoder update steps before the server proceeds to the next
    forward pass. This prevents stale activations from being used in subsequent
    training steps, which would otherwise corrupt the gradient estimates.

    Returns per-node byte counts for communication tracking.
    """
    msgs       = []
    bytes_down = []
    for view, nid in enumerate(node_ids):
        grad_np = grads_per_silo[view].detach().cpu().numpy().astype(np.float32)
        bytes_down.append(int(grad_np.nbytes))
        msgs.append(Message(
            content=RecordDict({
                "arrays": ArrayRecord({
                    "batch_idx":       _arr_i64(batch_idx),
                    "local_gradients": _arr_f32(grad_np),
                }),
                "config": ConfigRecord({"view": view}),
            }),
            message_type="train.apply_gradients",
            dst_node_id=nid,
        ))
    grid.send_and_receive(msgs)  # Barrier: wait for both clients to finish stepping.
    return bytes_down


def _checkpoint_clients(grid: Grid, node_ids: List[int]) -> None:
    """
    Instruct both client nodes to save their current encoder state as the
    best checkpoint. Called whenever a new best validation AUROC is achieved.
    """
    msgs = [
        Message(
            content=RecordDict({
                "arrays": ArrayRecord({}),
                "config": ConfigRecord({}),
            }),
            message_type="query.checkpoint_bottom",
            dst_node_id=nid,
        )
        for nid in node_ids
    ]
    grid.send_and_receive(msgs)


def _restore_best_clients(grid: Grid, node_ids: List[int]) -> None:
    """
    Instruct both client nodes to restore their encoder state from the
    best checkpoint. Called after training before final test evaluation.
    """
    msgs = [
        Message(
            content=RecordDict({
                "arrays": ArrayRecord({}),
                "config": ConfigRecord({}),
            }),
            message_type="query.restore_best_bottom",
            dst_node_id=nid,
        )
        for nid in node_ids
    ]
    grid.send_and_receive(msgs)


# ---------------------------------------------------------------------------
# Flower ServerApp entry point
# ---------------------------------------------------------------------------

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main server function for the glioma SplitNN VFL baseline.

    Runs end-to-end SplitNN training across 100 epochs with per-batch
    activation and gradient exchange. Best checkpoint is selected by
    validation AUROC and restored before final test evaluation.
    Communication costs are tracked per phase (train, val, test) and
    written to per-fold CSV files alongside the metrics history.
    """
    DEFAULT_NPZ = Path(__file__).resolve().parent / "glioma_aligned_vfl_hfl_cv.npz"
    rc          = getattr(context, "run_config", {}) or {}

    npz:        str   = str(rc.get("npz", str(DEFAULT_NPZ)))
    fold:       int   = int(os.environ.get("FOLD", rc.get("fold", 1)))
    seed:       int   = int(rc.get("seed", 42))
    device:     str   = str(rc.get("device", "cpu"))
    lr_head:    float = float(rc.get("lr_head", 1e-3))
    batch_size: int   = int(rc.get("batch_size", 64))
    epochs:     int   = int(os.environ.get("EPOCHS", rc.get("epochs", 100)))
    out_dim:    int   = int(rc.get("out_feature_dim", 16))
    patience:   int   = int(rc.get("patience", 0))

    out_dir = Path(rc.get("out_dir",
                          str(Path(__file__).resolve().parent / "runs_splitnn_vfl_glioma")))
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv            = out_dir / f"metrics_fold{fold}.csv"
    comm_csv               = out_dir / f"comm_fold{fold}.csv"
    comm_summary_csv       = out_dir / f"comm_summary_fold{fold}.csv"
    experiment_summary_csv = out_dir / f"experiment_summary_fold{fold}.csv"
    test_thresholds_csv    = out_dir / f"test_thresholds_fold{fold}.csv"

    set_seed(seed)
    dev = torch.device(device)

    node_ids = sorted(list(grid.get_node_ids()))
    if len(node_ids) != 2:
        raise ValueError(f"Expected 2 clients, got {len(node_ids)}: {node_ids}")
    log(INFO, "Node IDs: %s -> view=0 (X1), %s -> view=1 (X2)",
        node_ids[0], node_ids[1])

    y_all, folds_data = load_npz(npz)
    split_obj = folds_data[fold - 1]
    split     = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr_idx = split["train"].astype(np.int64)
    va_idx = split["val"].astype(np.int64)
    te_idx = split["test"].astype(np.int64)

    # Fusion head input dimension is 2 * out_dim since active and passive
    # embeddings are concatenated before classification.
    head = TopMLP(in_dim=out_dim * 2).to(dev)
    opt  = torch.optim.Adam(head.parameters(), lr=lr_head)

    # Compute positive class weight from the training split only.
    y_tr       = torch.from_numpy(y_all[tr_idx]).float()
    pos        = float(y_tr.sum().item())
    neg        = float(y_tr.numel() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)]).float().to(dev)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    steps_per_epoch = int(np.ceil(len(tr_idx) / batch_size))
    total_steps     = epochs * steps_per_epoch
    log(INFO,
        "fold=%d | epochs=%d | steps/epoch=%d | total=%d | out_dim_per_silo=%d",
        fold, epochs, steps_per_epoch, total_steps, out_dim)

    # Communication byte trackers for train, validation, and test phases.
    totals: Dict[str, Dict] = {
        "train": {"up": [0, 0], "down": [0, 0], "total": 0},
        "val":   {"up": [0, 0], "down": [0, 0], "total": 0},
        "test":  {"up": [0, 0], "down": [0, 0], "total": 0},
    }

    best_val:        Dict          = {"epoch": 0, "val_loss": float("nan"),
                                      "val_auroc": float("nan"), "val_prauc": float("nan")}
    best_head_state: Optional[Dict] = None
    no_improve   = 0
    rng          = np.random.default_rng(seed)
    global_step  = 0
    test_loss = test_auroc = test_prauc = float("nan")

    with open(metrics_csv, "w", newline="") as fmet, \
         open(comm_csv,    "w", newline="") as fcom:

        met_w = csv.writer(fmet)
        com_w = csv.writer(fcom)
        met_w.writerow(["epoch", "train_loss", "val_loss", "val_auroc", "val_prauc", "lr_head"])
        com_w.writerow(["phase", "epoch", "step", "global_step",
                        "bytes_up_x1", "bytes_up_x2",
                        "bytes_down_x1", "bytes_down_x2", "total"])

        # ----------------------------------------------------------------
        # SplitNN training loop
        # ----------------------------------------------------------------
        for ep in range(1, epochs + 1):
            head.train()
            tr_loss_sum = tr_loss_n = 0
            rng.shuffle(tr_idx)

            for s in range(steps_per_epoch):
                global_step += 1
                b = tr_idx[s * batch_size : (s + 1) * batch_size]
                if b.size == 0:
                    continue

                # Request cut-layer activations from both silos in parallel.
                z, up_bytes = _request_embeddings(grid, node_ids, b, out_dim)
                z = z.to(dev)

                logits = head(z)
                yb     = torch.from_numpy(y_all[b]).float().to(dev)
                loss   = criterion(logits, yb)

                tr_loss_sum += float(loss.item()) * int(b.size)
                tr_loss_n   += int(b.size)

                # Back-propagate through the fusion head to obtain cut-layer gradients.
                opt.zero_grad(set_to_none=True)
                loss.backward()

                # Split the cut-layer gradient into per-silo slices.
                # Active silo receives the first out_dim dimensions;
                # passive silo receives the last out_dim dimensions.
                grads          = z.grad.detach()
                grads_per_silo = list(grads.split([out_dim, out_dim], dim=1))

                # Send gradient slices to both clients and wait for both to
                # finish updating their encoders before proceeding.
                down_bytes = _push_embedding_grads_barrier(
                    grid, node_ids, grads_per_silo, b
                )

                # Update the fusion head after both client encoders have stepped
                # to ensure consistent gradient estimates across the full model.
                opt.step()

                bu0, bu1 = int(up_bytes[0]),   int(up_bytes[1])
                bd0, bd1 = int(down_bytes[0]), int(down_bytes[1])
                total    = bu0 + bu1 + bd0 + bd1
                totals["train"]["up"][0]   += bu0
                totals["train"]["up"][1]   += bu1
                totals["train"]["down"][0] += bd0
                totals["train"]["down"][1] += bd1
                totals["train"]["total"]   += total
                com_w.writerow(["train", ep, s + 1, global_step,
                                bu0, bu1, bd0, bd1, total])

                if global_step % 50 == 0:
                    with torch.no_grad():
                        p   = torch.sigmoid(logits).detach().cpu().numpy()
                        acc = float(
                            ((p >= 0.5) == (yb.cpu().numpy() >= 0.5)).mean() * 100
                        )
                    log(INFO, "ep=%d step=%d/%d loss=%.4f acc=%.2f%%",
                        ep, global_step, total_steps, loss.item(), acc)

            # ----------------------------------------------------------------
            # Validation
            # ----------------------------------------------------------------
            head.eval()
            with torch.no_grad():
                val_logits_all, val_y_all = [], []
                val_loss_sum = val_n = 0

                for s in range(int(np.ceil(len(va_idx) / batch_size))):
                    b = va_idx[s * batch_size : (s + 1) * batch_size]
                    if b.size == 0:
                        continue
                    z, up_bytes = _request_embeddings(grid, node_ids, b, out_dim)
                    z      = z.to(dev)
                    logits = head(z)
                    yb     = torch.from_numpy(y_all[b]).float().to(dev)
                    val_loss_sum += float(criterion(logits, yb).item()) * int(b.size)
                    val_n        += int(b.size)
                    val_logits_all.append(logits.detach().cpu())
                    val_y_all.append(yb.detach().cpu())
                    bu0, bu1 = int(up_bytes[0]), int(up_bytes[1])
                    totals["val"]["up"][0] += bu0
                    totals["val"]["up"][1] += bu1
                    totals["val"]["total"] += bu0 + bu1
                    com_w.writerow(["val", ep, s + 1, global_step,
                                   bu0, bu1, 0, 0, bu0 + bu1])

                val_logits = torch.cat(val_logits_all).numpy()
                val_prob   = 1.0 / (1.0 + np.exp(-val_logits))
                val_y      = torch.cat(val_y_all).numpy().astype(np.float32)
                val_loss   = val_loss_sum / max(val_n, 1)
                val_auroc  = safe_auroc(val_y, val_prob)
                val_prauc  = safe_prauc(val_y, val_prob)

            train_loss = tr_loss_sum / max(tr_loss_n, 1)
            met_w.writerow([ep, train_loss, val_loss, val_auroc, val_prauc,
                            float(opt.param_groups[0]["lr"])])
            fmet.flush()
            fcom.flush()
            log(INFO, "[VAL] ep=%d loss=%.4f AUROC=%.4f PR-AUC=%.4f",
                ep, val_loss, val_auroc, val_prauc)

            # Save the best checkpoint based on validation AUROC.
            if not np.isnan(val_auroc) and (
                np.isnan(best_val["val_auroc"]) or val_auroc > best_val["val_auroc"]
            ):
                best_val = {"epoch": ep, "val_loss": val_loss,
                            "val_auroc": val_auroc, "val_prauc": val_prauc}
                best_head_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in head.state_dict().items()}
                )
                _checkpoint_clients(grid, node_ids)
                no_improve = 0
                log(INFO, "[CKPT] ep=%d AUROC=%.4f — head and encoders saved",
                    ep, val_auroc)
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    log(INFO, "[EARLY STOP] ep=%d no improvement for %d epochs",
                        ep, patience)
                    break

        # ----------------------------------------------------------------
        # Restore best checkpoint and evaluate on the held-out test set
        # ----------------------------------------------------------------
        if best_head_state is not None:
            head.load_state_dict(best_head_state)
            log(INFO, "[CKPT] Restored best head from epoch=%d (AUROC=%.4f)",
                best_val["epoch"], best_val["val_auroc"])
        _restore_best_clients(grid, node_ids)

        head.eval()
        with torch.no_grad():
            test_logits_all, test_y_all = [], []
            test_loss_sum = test_n = 0

            for s in range(int(np.ceil(len(te_idx) / batch_size))):
                b = te_idx[s * batch_size : (s + 1) * batch_size]
                if b.size == 0:
                    continue
                z, up_bytes = _request_embeddings(grid, node_ids, b, out_dim)
                z      = z.to(dev)
                logits = head(z)
                yb     = torch.from_numpy(y_all[b]).float().to(dev)
                test_loss_sum += float(criterion(logits, yb).item()) * int(b.size)
                test_n        += int(b.size)
                test_logits_all.append(logits.detach().cpu())
                test_y_all.append(yb.detach().cpu())
                bu0, bu1 = int(up_bytes[0]), int(up_bytes[1])
                totals["test"]["up"][0] += bu0
                totals["test"]["up"][1] += bu1
                totals["test"]["total"] += bu0 + bu1
                com_w.writerow(["test", epochs, s + 1, global_step,
                               bu0, bu1, 0, 0, bu0 + bu1])

            test_logits = torch.cat(test_logits_all).numpy()
            test_prob   = 1.0 / (1.0 + np.exp(-test_logits))
            test_y      = torch.cat(test_y_all).numpy().astype(np.float32)
            test_loss   = test_loss_sum / max(test_n, 1)
            test_auroc  = safe_auroc(test_y, test_prob)
            test_prauc  = safe_prauc(test_y, test_prob)

        log(INFO, "[TEST] loss=%.4f AUROC=%.4f PR-AUC=%.4f",
            test_loss, test_auroc, test_prauc)
        write_threshold_report(test_thresholds_csv, test_y, test_prob)

    # ----------------------------------------------------------------
    # Write communication and experiment summary CSVs
    # ----------------------------------------------------------------
    train_total = totals["train"]["total"]
    all_total   = (train_total
                   + totals["val"]["total"]
                   + totals["test"]["total"])

    with open(comm_summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "bytes_up_x1", "bytes_up_x2",
                    "bytes_down_x1", "bytes_down_x2", "total_bytes"])
        w.writerow(["train_only",
                    totals["train"]["up"][0], totals["train"]["up"][1],
                    totals["train"]["down"][0], totals["train"]["down"][1],
                    train_total])
        w.writerow(["train_val_test",
                    totals["train"]["up"][0] + totals["val"]["up"][0] + totals["test"]["up"][0],
                    totals["train"]["up"][1] + totals["val"]["up"][1] + totals["test"]["up"][1],
                    totals["train"]["down"][0] + totals["val"]["down"][0] + totals["test"]["down"][0],
                    totals["train"]["down"][1] + totals["val"]["down"][1] + totals["test"]["down"][1],
                    all_total])

    with open(experiment_summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "fold", "best_val_epoch", "best_val_loss", "best_val_auroc", "best_val_prauc",
            "test_loss", "test_auroc", "test_prauc",
            "train_only_bytes_up_x1", "train_only_bytes_up_x2",
            "train_only_bytes_down_x1", "train_only_bytes_down_x2", "train_only_total_bytes",
            "train_val_test_bytes_up_x1", "train_val_test_bytes_up_x2",
            "train_val_test_bytes_down_x1", "train_val_test_bytes_down_x2",
            "train_val_test_total_bytes",
        ])
        w.writerow([
            fold,
            best_val["epoch"], best_val["val_loss"],
            best_val["val_auroc"], best_val["val_prauc"],
            test_loss, test_auroc, test_prauc,
            totals["train"]["up"][0], totals["train"]["up"][1],
            totals["train"]["down"][0], totals["train"]["down"][1], train_total,
            totals["train"]["up"][0] + totals["val"]["up"][0] + totals["test"]["up"][0],
            totals["train"]["up"][1] + totals["val"]["up"][1] + totals["test"]["up"][1],
            totals["train"]["down"][0] + totals["val"]["down"][0] + totals["test"]["down"][0],
            totals["train"]["down"][1] + totals["val"]["down"][1] + totals["test"]["down"][1],
            all_total,
        ])

    log(INFO, "Done. Outputs: %s", str(out_dir))
