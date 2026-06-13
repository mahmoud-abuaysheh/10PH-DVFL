# serverapp_vfl_diabetes_splitnn.py
#
# Flower 1.26.1 server application for the SplitNN VFL baseline in the
# diabetes decoupled VFL experiment.
#
# This script implements the SplitNN-based VFL baseline (Condition 1) against
# which the 10PH-DVFL decoupled architecture is compared. It serves as the
# primary communication and performance reference point in the evaluation.
#
# SplitNN communication pattern (per training batch):
#   1. Server requests cut-layer activations from the active silo (MSG_EMB)
#   2. Server requests cut-layer activations from the passive silo (MSG_EMB)
#   3. Server requests labels from the active silo (MSG_Y)
#   4. Server concatenates activations, computes forward pass and loss
#   5. Server back-propagates loss to obtain gradients at the cut layer
#   6. Server sends the active silo's gradient slice back (MSG_BWD)
#   7. Server sends the passive silo's gradient slice back (MSG_BWD)
#   8. Each client applies the received gradient to update its local encoder
#
# Unlike the decoupled architecture, SplitNN requires repeated activation and
# gradient exchange across silos for every training batch and every epoch.
# This creates communication overhead that grows linearly with the number of
# training rounds, which is the primary bottleneck addressed by 10PH-DVFL.
#
# Key differences from serverapp_vfl_diabetes_decoupled.py:
#   - Encoders are updated end-to-end through gradient feedback (steps 6-8)
#   - Both activation and gradient tensors are transmitted every batch
#   - Client encoder checkpoints are saved and restored for test evaluation

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

INFO = 20

# Message type constants must match the handler names registered in the
# client application (clientapp_vfl_diabetes_splitnn.py).
MSG_EMB  = "query.generate_embeddings"   # Request cut-layer activations
MSG_Y    = "query.get_labels"            # Request labels (active silo only)
MSG_CKPT = "query.checkpoint_bottom"     # Save client encoder checkpoint
MSG_RST  = "query.restore_best_bottom"   # Restore best client encoder checkpoint
MSG_BWD  = "train.backward"              # Send gradient to client for encoder update


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Flower grid utilities
# ---------------------------------------------------------------------------

def _arr_i64(x: np.ndarray) -> Array:
    """Convert a numpy array to a Flower Array with int64 dtype."""
    return Array(np.asarray(x, dtype=np.int64))


def _arr_f32(x: np.ndarray) -> Array:
    """Convert a numpy array to a Flower Array with float32 dtype."""
    return Array(np.asarray(x, dtype=np.float32))


def _cfg(d: Dict[str, object] | None = None) -> ConfigRecord:
    """Wrap a plain dict as a Flower ConfigRecord."""
    return ConfigRecord(d or {})


def _grid_request(grid: Grid, msg: Message, timeout: int = 300) -> Message:
    """
    Send a single message to a client node and return the reply.
    Raises RuntimeError if the reply is empty or has no content,
    which typically indicates a client crash or handler name mismatch.
    """
    reps = grid.send_and_receive([msg], timeout=timeout)
    if not reps:
        raise RuntimeError("grid.send_and_receive returned empty list")
    rep = reps[0]
    if not rep.has_content():
        dst = getattr(msg, "dst_node_id", "<?>")
        raise RuntimeError(
            f"Client reply has no content (dst_node_id={dst}). "
            "Usually the client crashed or the handler name mismatched."
        )
    return rep


def get_node_ids(grid: Grid) -> List[int]:
    """
    Retrieve sorted node IDs from the Flower grid.
    Tries multiple API access patterns to support different Flower versions.
    """
    if hasattr(grid, "get_node_ids") and callable(getattr(grid, "get_node_ids")):
        return sorted([int(x) for x in grid.get_node_ids()])
    if hasattr(grid, "node_ids"):
        v = getattr(grid, "node_ids")
        if callable(v):
            return sorted([int(x) for x in v()])
        try:
            return sorted([int(x) for x in v])
        except TypeError:
            pass
    for attr in ("_node_ids", "_nodes", "nodes", "_node_registry", "_nodes_by_id"):
        if hasattr(grid, attr):
            obj = getattr(grid, attr)
            if isinstance(obj, dict):
                return sorted([int(k) for k in obj.keys()])
            if isinstance(obj, (list, tuple, set)):
                return sorted([int(x) for x in obj])
    node_like = [a for a in dir(grid) if "node" in a.lower()]
    raise AttributeError(
        "Could not obtain node ids from grid. "
        f"Grid type={type(grid)}; node-like attrs={node_like}"
    )


# ---------------------------------------------------------------------------
# Server-side top model
# ---------------------------------------------------------------------------

class TopLinear(nn.Module):
    """
    Server-side top model for SplitNN VFL.

    Receives concatenated cut-layer activations from the active and passive
    silos and produces a scalar logit for binary classification.
    Input: 8 (active) + 8 (passive) = 16 dimensions.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(1)


# ---------------------------------------------------------------------------
# Client encoder checkpoint management
# ---------------------------------------------------------------------------

def _checkpoint_clients(
    grid: Grid, active_id: int, passive_id: int, fold: int, out_dir: str
) -> None:
    """
    Instruct both client nodes to save their current encoder state as the
    best checkpoint. Called whenever a new best validation AUROC is achieved.
    """
    for node_id, view in [(active_id, 0), (passive_id, 1)]:
        _grid_request(grid, Message(
            content=RecordDict({
                "config": ConfigRecord({"view": view, "fold": fold, "out_dir": out_dir})
            }),
            dst_node_id=node_id,
            message_type=MSG_CKPT,
        ))


def _restore_best_clients(
    grid: Grid, active_id: int, passive_id: int, fold: int, out_dir: str
) -> None:
    """
    Instruct both client nodes to restore their encoder state from the
    best checkpoint saved during training. Called after training completes
    before final evaluation on the test set.
    """
    for node_id, view in [(active_id, 0), (passive_id, 1)]:
        _grid_request(grid, Message(
            content=RecordDict({
                "config": ConfigRecord({"view": view, "fold": fold, "out_dir": out_dir})
            }),
            dst_node_id=node_id,
            message_type=MSG_RST,
        ))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_npz(npz_path: str):
    """
    Load the pre-computed cross-validation splits and labels from the NPZ file.
    Returns labels y, the list of fold split dicts, and optional metadata.
    """
    d = np.load(npz_path, allow_pickle=True)
    y = d["y"].astype(np.int64)
    folds = list(d["folds"])
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return y, folds, meta


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def _pick_threshold_maxf1(y_true: np.ndarray, p: np.ndarray) -> float:
    """
    Select the classification threshold that maximises F1 score on the
    validation set. Applied to the test set without further tuning.
    """
    ts = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _threshold_metrics(
    y_true: np.ndarray, p: np.ndarray, thr: float
) -> Dict[str, float]:
    """Compute threshold-dependent classification metrics at a fixed threshold."""
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    bacc = 0.5 * (rec + spec)
    return {
        "thr": float(thr),
        "acc": float(acc),
        "balanced_acc": float(bacc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
    }


@torch.no_grad()
def _eval_probs(
    grid: Grid,
    top: nn.Module,
    device: torch.device,
    idx: np.ndarray,
    active_id: int,
    passive_id: int,
    timeout: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the top model over a set of sample indices.

    Requests cut-layer activations from both silos and labels from the active
    silo, concatenates activations, and runs the top model forward pass.
    No gradients are computed or returned to clients during evaluation.
    Returns arrays of true labels and predicted probabilities.
    """
    top.eval()
    y_all: List[np.ndarray] = []
    p_all: List[np.ndarray] = []

    bs = 4096
    for s in range(0, len(idx), bs):
        bidx = idx[s:s + bs].astype(np.int64)

        rep_x = _grid_request(
            grid,
            Message(
                content=RecordDict({
                    "config": _cfg({"view": 0}),
                    "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                }),
                dst_node_id=active_id,
                message_type=MSG_EMB,
            ),
            timeout=timeout,
        )
        rep_p = _grid_request(
            grid,
            Message(
                content=RecordDict({
                    "config": _cfg({"view": 1}),
                    "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                }),
                dst_node_id=passive_id,
                message_type=MSG_EMB,
            ),
            timeout=timeout,
        )

        ex = rep_x.content["arrays"]["embedding"].numpy().astype(np.float32)
        ep = rep_p.content["arrays"]["embedding"].numpy().astype(np.float32)

        rep_y = _grid_request(
            grid,
            Message(
                content=RecordDict({
                    "config": _cfg({}),
                    "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                }),
                dst_node_id=active_id,
                message_type=MSG_Y,
            ),
            timeout=timeout,
        )
        yb = rep_y.content["arrays"]["y"].numpy().astype(np.int64)

        z = torch.from_numpy(np.concatenate([ex, ep], axis=1)).to(device)
        logits = top(z)
        pb = torch.sigmoid(logits).cpu().numpy()

        y_all.append(yb)
        p_all.append(pb)

    return np.concatenate(y_all), np.concatenate(p_all)


# ---------------------------------------------------------------------------
# Main server logic
# ---------------------------------------------------------------------------

def server_main(grid: Grid, context: Context) -> None:
    """
    Main server function for the SplitNN VFL baseline.

    Runs end-to-end SplitNN training where:
      - The server requests cut-layer activations from both silos each batch
      - The server computes the forward pass and loss
      - The server back-propagates and sends per-silo gradient slices back
      - Each client updates its local encoder using the received gradient
      - The top model is updated on the server using the same backward pass
    Early stopping is applied based on validation AUROC with checkpoint
    saving and restoration for both the server-side top model and the
    client-side encoders.
    """
    run = context.run_config
    seed = int(run.get("seed", 42))
    set_seed(seed)

    device = torch.device(str(run.get("device", "cpu")))
    DEFAULT_NPZ = Path(__file__).resolve().parent / "diabetes_vfl_cv.npz"
    npz = str(run.get("npz", str(DEFAULT_NPZ)))
    fold = int(os.environ.get("FOLD", run.get("fold", 1)))

    out_dir = str(run.get("out_dir", "./runs_vfl_diabetes_splitnn"))
    os.makedirs(out_dir, exist_ok=True)

    rounds   = int(run.get("rounds", 100))
    batch    = int(run.get("batch", 256))
    lr_top   = float(run.get("lr_top", 1e-3))

    # Identify which grid nodes act as the active and passive silos.
    # Node IDs are sorted deterministically; active silo is always node 0.
    node_ids = get_node_ids(grid)
    if len(node_ids) < 2:
        raise RuntimeError(f"Need 2 supernodes, got {node_ids}")
    active_id, passive_id = node_ids[0], node_ids[1]
    log(INFO, f"Node IDs: active={active_id}, passive={passive_id}")

    y, folds, meta = load_npz(npz)
    split_obj = folds[fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)
    va = split["val"].astype(np.int64)
    te = split["test"].astype(np.int64)

    # Compute positive class weight from the training split to handle
    # the class imbalance in the diabetes dataset (~10:1 ratio).
    pos = float(y[tr].sum())
    neg = float(len(tr) - y[tr].sum())
    pw = neg / max(pos, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    # Initialise the server-side top model and optimizer.
    top = TopLinear().to(device)
    opt = torch.optim.Adam(top.parameters(), lr=lr_top)

    best_val_auroc = -1.0
    best_round = -1

    best_path = os.path.join(out_dir, f"splitnn_best_fold{fold}.pt")
    hist_path = os.path.join(out_dir, f"splitnn_fold{fold}_history.csv")

    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["round", "train_loss", "val_auroc", "val_prauc", "lr"])

    n = len(tr)
    steps = math.ceil(n / batch)

    # SplitNN training loop.
    # Each round iterates over all training batches. For each batch:
    #   1. Request activations from both silos
    #   2. Forward pass and loss computation on the server
    #   3. Backward pass to obtain cut-layer gradients
    #   4. Send per-silo gradient slices back to each client for encoder update
    #   5. Update the server-side top model
    for rnd in range(1, rounds + 1):
        top.train()
        perm = np.random.permutation(tr)
        train_loss_acc = 0.0

        for s in range(steps):
            bidx = perm[s * batch : min(n, (s + 1) * batch)].astype(np.int64)

            # Request cut-layer activations from the active silo.
            rep_x = _grid_request(
                grid,
                Message(
                    content=RecordDict({
                        "config": _cfg({"view": 0}),
                        "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                    }),
                    dst_node_id=active_id,
                    message_type=MSG_EMB,
                ),
            )

            # Request cut-layer activations from the passive silo.
            rep_p = _grid_request(
                grid,
                Message(
                    content=RecordDict({
                        "config": _cfg({"view": 1}),
                        "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                    }),
                    dst_node_id=passive_id,
                    message_type=MSG_EMB,
                ),
            )

            ex = rep_x.content["arrays"]["embedding"].numpy().astype(np.float32)
            ep = rep_p.content["arrays"]["embedding"].numpy().astype(np.float32)

            # Request labels from the active silo.
            # The passive silo has no access to labels at any point.
            rep_y = _grid_request(
                grid,
                Message(
                    content=RecordDict({
                        "config": _cfg({}),
                        "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                    }),
                    dst_node_id=active_id,
                    message_type=MSG_Y,
                ),
            )
            yb = rep_y.content["arrays"]["y"].numpy().astype(np.int64)

            # Concatenate activations and enable gradient tracking so that
            # dL/dz can be computed for transmission back to the clients.
            z = torch.from_numpy(
                np.concatenate([ex, ep], axis=1)
            ).to(device).requires_grad_(True)
            yy = torch.from_numpy(yb).float().to(device)

            opt.zero_grad(set_to_none=True)
            logits = top(z)
            loss = criterion(logits, yy)
            loss.backward()

            # Extract the gradient at the concatenation point and split it
            # into per-silo gradient slices for transmission back to clients.
            # Active silo receives the first 8 dimensions; passive receives the last 8.
            dz = z.grad.detach().cpu().numpy().astype(np.float32)
            dx = dz[:, :8].copy()   # Active silo gradient slice.
            dp = dz[:, 8:].copy()   # Passive silo gradient slice.

            # Send the active silo's gradient slice back for encoder update.
            _grid_request(
                grid,
                Message(
                    content=RecordDict({
                        "config": _cfg({"view": 0}),
                        "arrays": ArrayRecord({
                            "batch_idx": _arr_i64(bidx),
                            "local_gradients": _arr_f32(dx),
                        }),
                    }),
                    dst_node_id=active_id,
                    message_type=MSG_BWD,
                ),
            )

            # Send the passive silo's gradient slice back for encoder update.
            _grid_request(
                grid,
                Message(
                    content=RecordDict({
                        "config": _cfg({"view": 1}),
                        "arrays": ArrayRecord({
                            "batch_idx": _arr_i64(bidx),
                            "local_gradients": _arr_f32(dp),
                        }),
                    }),
                    dst_node_id=passive_id,
                    message_type=MSG_BWD,
                ),
            )

            opt.step()
            train_loss_acc += float(loss.item()) * (len(bidx) / n)

        # Evaluate on the validation set using current encoder states from both silos.
        y_va, p_va = _eval_probs(grid, top, device, va, active_id, passive_id)
        val_auroc = float(roc_auc_score(y_va, p_va))
        val_prauc = float(average_precision_score(y_va, p_va))
        lr_now = float(opt.param_groups[0]["lr"])

        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([rnd, train_loss_acc, val_auroc, val_prauc, lr_now])

        if rnd == 1 or rnd % 10 == 0 or rnd == rounds:
            print(
                f"[fold {fold}] round {rnd:03d}/{rounds} train_loss={train_loss_acc:.4f} "
                f"val_AUROC={val_auroc:.4f} val_PR-AUC={val_prauc:.4f} "
                f"best={best_val_auroc:.4f}@{best_round} lr={lr_now:.2e}"
            )

        # Save the best checkpoint for both the server top model and client encoders.
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_round = rnd
            torch.save({"top": top.state_dict()}, best_path)
            _checkpoint_clients(grid, active_id, passive_id, fold, out_dir)

    # Restore the best checkpoint for both the server and clients before
    # final evaluation on the held-out test set.
    top.load_state_dict(torch.load(best_path, map_location=device)["top"])
    _restore_best_clients(grid, active_id, passive_id, fold, out_dir)

    y_va, p_va = _eval_probs(grid, top, device, va, active_id, passive_id)
    y_te, p_te = _eval_probs(grid, top, device, te, active_id, passive_id)

    # Select classification threshold on the validation set using max F1,
    # then apply it to the test set without further tuning.
    thr = _pick_threshold_maxf1(y_va, p_va)
    tm_val = _threshold_metrics(y_va, p_va, thr)
    tm_te  = _threshold_metrics(y_te, p_te, thr)

    test_auroc = float(roc_auc_score(y_te, p_te))
    test_prauc = float(average_precision_score(y_te, p_te))

    summary_path = os.path.join(out_dir, f"splitnn_fold{fold}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "best_round", "best_val_auroc", "test_auroc", "test_prauc", "pos_weight_train"])
        w.writerow([fold, best_round, best_val_auroc, test_auroc, test_prauc, float(pw)])
        w.writerow([])
        w.writerow(["VAL (maxF1 threshold)"])
        for k, v in tm_val.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["TEST (apply VAL threshold)"])
        for k, v in tm_te.items():
            w.writerow([k, v])

    print(
        f"[fold {fold}] DONE best_val_AUROC={best_val_auroc:.4f}@{best_round} | "
        f"test_AUROC={test_auroc:.4f} test_PR-AUC={test_prauc:.4f} | thr(val,maxF1)={thr:.3f}"
    )
    print(f"[OK] wrote: {summary_path}")
    if meta:
        print("[meta]", meta)


# ---------------------------------------------------------------------------
# Flower ServerApp entry point
# ---------------------------------------------------------------------------

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    server_main(grid, context)
