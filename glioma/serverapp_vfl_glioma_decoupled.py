# serverapp_vfl_glioma_decoupled.py
#
# Flower 1.26.1 server application for the decoupled VFL glioma experiment.
#
# This server implements Tier 2 of the 10PH-DVFL architecture for the glioma
# dataset:
#   - Requests frozen embeddings from the active and passive silos once per batch
#   - Trains only the lightweight fusion head (TopMLP) on the server side
#   - No gradients are returned to any silo at any point
#
# Communication pattern (Tier 2):
#   For each training batch:
#     1. Request active embeddings from active silo (MSG_EMB, view=0)
#     2. Request passive embeddings from passive silo (MSG_EMB, view=1)
#     3. Request labels from active silo (MSG_Y)
#     4. Concatenate embeddings, compute loss, update fusion head locally
#     No gradients are sent back to either silo.
#
# Key architectural differences from the diabetes decoupled server:
#   - Fusion head: TopMLP (32 -> 16 -> 8 -> 1) instead of TopLinear (16 -> 1)
#   - Embedding dimension: 16 per silo (32 concatenated) instead of 8 (16)
#   - Threshold metrics: three rules (Youden, max-F1, fixed 0.5) written
#     to a per-fold CSV alongside AUROC and PR-AUC

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
# client application.
MSG_EMB = "query.generate_embeddings"
MSG_Y   = "query.get_labels"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Flower grid utilities
# ---------------------------------------------------------------------------

def _arr_i64(x: np.ndarray) -> Array:
    """Convert a numpy array to a Flower Array with int64 dtype."""
    return Array(np.asarray(x, dtype=np.int64))


def _cfg(d: Dict[str, object] | None = None) -> ConfigRecord:
    """Wrap a plain dict as a Flower ConfigRecord."""
    return ConfigRecord(d or {})


def _grid_request(grid: Grid, msg: Message, timeout: int = 300) -> Message:
    """
    Send a single message to a client node and return the reply.
    Raises RuntimeError if the reply is empty or has no content.
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


def _rcfg(run, key: str, default):
    """
    Read a value from the Flower run configuration.
    Flower converts underscore keys to hyphens internally, so both are checked.
    """
    hyphen_key = key.replace("_", "-")
    if key in run:
        return run[key]
    if hyphen_key in run:
        return run[hyphen_key]
    return default


# ---------------------------------------------------------------------------
# Server-side fusion head
# ---------------------------------------------------------------------------

class TopMLP(nn.Module):
    """
    Lightweight fusion head trained on the server in Tier 2.

    Receives concatenated embeddings from the active and passive silos
    and produces a scalar logit for binary classification.
    Architecture: in_dim -> 16 -> ReLU -> 8 -> ReLU -> 1

    Input dimensionality must equal emb_dim_active + emb_dim_passive (32).
    Uses a 3-layer MLP rather than a single linear layer to match the
    glioma architecture used in the SplitNN baseline for fair comparison.
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
# Data loading
# ---------------------------------------------------------------------------

def load_npz(npz_path: str):
    """
    Load the pre-computed cross-validation splits and labels from the NPZ file.
    Returns labels y, the list of fold split dicts, and optional metadata.
    """
    d     = np.load(npz_path, allow_pickle=True)
    y     = d["y"].astype(np.int64)
    folds = list(d["folds"])
    meta  = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return y, folds, meta


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def _pick_threshold_maxf1(y_true: np.ndarray, p: np.ndarray) -> float:
    """
    Select the classification threshold that maximises F1 on the validation set.
    Applied to the test set without further tuning.
    """
    ts = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = float(t)
    return best_t


def _threshold_metrics(
    y_true: np.ndarray, p: np.ndarray, thr: float
) -> Dict[str, float]:
    """Compute threshold-dependent classification metrics at a fixed threshold."""
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    bacc = 0.5 * (rec + spec)
    return {
        "thr": float(thr), "acc": float(acc), "balanced_acc": float(bacc),
        "precision": float(prec), "recall": float(rec),
        "specificity": float(spec), "f1": float(f1),
    }


@torch.no_grad()
def _eval_probs(
    grid: Grid,
    top: nn.Module,
    device: torch.device,
    idx: np.ndarray,
    active_id: int,
    passive_id: int,
    emb_dim: int,
    timeout: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the fusion head over a set of sample indices.

    Requests frozen embeddings from both silos and labels from the active
    silo, concatenates embeddings, and runs the fusion head forward pass.
    Returns arrays of true labels and predicted probabilities.
    """
    top.eval()
    y_all: List[np.ndarray] = []
    p_all: List[np.ndarray] = []

    bs = 1024
    for s in range(0, len(idx), bs):
        bidx = idx[s : s + bs].astype(np.int64)

        rep_x = _grid_request(grid, Message(
            content=RecordDict({
                "config": _cfg({"view": 0, "role": "active"}),
                "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
            }),
            dst_node_id=active_id,
            message_type=MSG_EMB,
        ), timeout=timeout)

        rep_p = _grid_request(grid, Message(
            content=RecordDict({
                "config": _cfg({"view": 1, "role": "passive"}),
                "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
            }),
            dst_node_id=passive_id,
            message_type=MSG_EMB,
        ), timeout=timeout)

        ex = rep_x.content["arrays"]["embedding"].numpy().astype(np.float32)
        ep = rep_p.content["arrays"]["embedding"].numpy().astype(np.float32)
        if ex.shape[1] != emb_dim or ep.shape[1] != emb_dim:
            raise RuntimeError(
                f"Embedding dim mismatch: ex={ex.shape}, ep={ep.shape}, "
                f"expected emb_dim={emb_dim}"
            )

        rep_y = _grid_request(grid, Message(
            content=RecordDict({
                "config": _cfg({"role": "active"}),
                "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
            }),
            dst_node_id=active_id,
            message_type=MSG_Y,
        ), timeout=timeout)
        yb = rep_y.content["arrays"]["y"].numpy().astype(np.int64)

        z      = torch.from_numpy(np.concatenate([ex, ep], axis=1)).to(device)
        logits = top(z)
        pb     = torch.sigmoid(logits).cpu().numpy()

        y_all.append(yb)
        p_all.append(pb)

    return np.concatenate(y_all), np.concatenate(p_all)


# ---------------------------------------------------------------------------
# Main server logic
# ---------------------------------------------------------------------------

def server_main(grid: Grid, context: Context) -> None:
    """
    Main server function for the glioma decoupled VFL experiment.

    Runs the Tier 2 fusion head training loop: requests frozen embeddings
    from both silos each batch, trains only the server-side TopMLP, and
    applies early stopping based on validation AUROC. After training,
    restores the best checkpoint and evaluates on the held-out test set.
    """
    run  = context.run_config
    seed = int(_rcfg(run, "seed", 42))
    set_seed(seed)

    device      = torch.device(str(_rcfg(run, "device", "cpu")))
    DEFAULT_NPZ = Path(__file__).resolve().parent / "glioma_aligned_vfl_hfl_cv.npz"
    npz         = str(_rcfg(run, "npz", str(DEFAULT_NPZ)))
    fold        = int(os.environ.get("FOLD", _rcfg(run, "fold", 1)))

    out_dir = Path(str(_rcfg(run, "out_dir", "./runs_decoupled_vfl_glioma"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds   = int(_rcfg(run, "rounds", 100))
    batch    = int(_rcfg(run, "batch",  64))
    lr_top   = float(_rcfg(run, "lr_top", 1e-3))
    patience = int(_rcfg(run, "patience", 15))
    emb_dim  = int(_rcfg(run, "emb_dim", 16))

    # Identify which grid nodes act as the active and passive silos.
    node_ids = get_node_ids(grid)
    if len(node_ids) < 2:
        raise RuntimeError(f"Need 2 supernodes, got {node_ids}")
    active_id, passive_id = node_ids[0], node_ids[1]
    log(INFO, f"Node IDs: active={active_id}, passive={passive_id}")

    y, folds, meta = load_npz(npz)
    split_obj = folds[fold - 1]
    split     = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)
    va = split["val"].astype(np.int64)
    te = split["test"].astype(np.int64)

    # Compute positive class weight from the training split only.
    pos = float(y[tr].sum())
    neg = float(len(tr) - y[tr].sum())
    pw  = neg / max(pos, 1.0)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], device=device)
    )

    # Initialise the server-side fusion head.
    # Input dimension is 2 * emb_dim since active and passive embeddings
    # are concatenated before classification.
    top = TopMLP(in_dim=2 * emb_dim).to(device)
    opt = torch.optim.Adam(top.parameters(), lr=lr_top)

    best_val_auroc = -1.0
    best_round     = -1
    no_improve     = 0

    best_path = str(out_dir / f"fusion_head_best_fold{fold}.pt")
    hist_path = str(out_dir / f"decoupled_fold{fold}_history.csv")

    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["round", "train_loss", "val_auroc", "val_prauc", "lr"])

    n     = len(tr)
    steps = math.ceil(n / batch)

    # Tier 2 training loop.
    # Each round requests frozen embeddings from both silos, concatenates
    # them on the server, and updates only the fusion head.
    # No gradients are returned to any silo at any point.
    for rnd in range(1, rounds + 1):
        top.train()
        perm           = np.random.permutation(tr)
        train_loss_acc = 0.0

        for s in range(steps):
            bidx = perm[s * batch : min(n, (s + 1) * batch)].astype(np.int64)

            # Request frozen active-silo embeddings.
            rep_x = _grid_request(grid, Message(
                content=RecordDict({
                    "config": _cfg({"view": 0, "role": "active"}),
                    "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                }),
                dst_node_id=active_id,
                message_type=MSG_EMB,
            ))

            # Request frozen passive-silo embeddings.
            rep_p = _grid_request(grid, Message(
                content=RecordDict({
                    "config": _cfg({"view": 1, "role": "passive"}),
                    "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                }),
                dst_node_id=passive_id,
                message_type=MSG_EMB,
            ))

            ex = rep_x.content["arrays"]["embedding"].numpy().astype(np.float32)
            ep = rep_p.content["arrays"]["embedding"].numpy().astype(np.float32)
            if ex.shape[1] != emb_dim or ep.shape[1] != emb_dim:
                raise RuntimeError(
                    f"Embedding dim mismatch in train: ex={ex.shape}, "
                    f"ep={ep.shape}, expected emb_dim={emb_dim}"
                )

            # Request labels from the active silo (passive silo has no labels).
            rep_y = _grid_request(grid, Message(
                content=RecordDict({
                    "config": _cfg({"role": "active"}),
                    "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                }),
                dst_node_id=active_id,
                message_type=MSG_Y,
            ))
            yb = rep_y.content["arrays"]["y"].numpy().astype(np.int64)

            # Concatenate embeddings and update the fusion head.
            z      = torch.from_numpy(np.concatenate([ex, ep], axis=1)).to(device)
            yy     = torch.from_numpy(yb).float().to(device)
            opt.zero_grad(set_to_none=True)
            logits = top(z)
            loss   = criterion(logits, yy)
            loss.backward()
            opt.step()

            train_loss_acc += float(loss.item()) * (len(bidx) / n)

        # Evaluate on the validation set using frozen embeddings.
        y_va, p_va = _eval_probs(
            grid, top, device, va, active_id, passive_id, emb_dim
        )
        val_auroc = float(roc_auc_score(y_va, p_va))
        val_prauc = float(average_precision_score(y_va, p_va))
        lr_now    = float(opt.param_groups[0]["lr"])

        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [rnd, train_loss_acc, val_auroc, val_prauc, lr_now]
            )

        if rnd == 1 or rnd % 10 == 0 or rnd == rounds:
            print(
                f"[fold {fold}] round {rnd:03d}/{rounds} "
                f"loss={train_loss_acc:.4f} "
                f"val_AUROC={val_auroc:.4f} val_PR-AUC={val_prauc:.4f} "
                f"best={best_val_auroc:.4f}@{best_round} lr={lr_now:.2e}"
            )

        # Save the best fusion head checkpoint based on validation AUROC.
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_round     = rnd
            torch.save({"top": top.state_dict()}, best_path)
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(
                    f"[fold {fold}] early stop at round {rnd} "
                    f"(no val AUROC improvement for {patience} rounds)"
                )
                break

    # Restore the best fusion head and evaluate on the held-out test set.
    top.load_state_dict(torch.load(best_path, map_location=device)["top"])
    y_va, p_va = _eval_probs(
        grid, top, device, va, active_id, passive_id, emb_dim
    )
    y_te, p_te = _eval_probs(
        grid, top, device, te, active_id, passive_id, emb_dim
    )

    # Select the classification threshold on the validation set using max F1,
    # then apply it to the test set without further tuning.
    thr    = _pick_threshold_maxf1(y_va, p_va)
    tm_val = _threshold_metrics(y_va, p_va, thr)
    tm_te  = _threshold_metrics(y_te, p_te, thr)

    test_auroc = float(roc_auc_score(y_te, p_te))
    test_prauc = float(average_precision_score(y_te, p_te))

    summary_path = str(out_dir / f"decoupled_fold{fold}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "best_round", "best_val_auroc",
                    "test_auroc", "test_prauc", "pos_weight_train"])
        w.writerow([fold, best_round, best_val_auroc,
                    test_auroc, test_prauc, float(pw)])
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
        f"test_AUROC={test_auroc:.4f} test_PR-AUC={test_prauc:.4f} | "
        f"thr(val,maxF1)={thr:.3f}"
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
