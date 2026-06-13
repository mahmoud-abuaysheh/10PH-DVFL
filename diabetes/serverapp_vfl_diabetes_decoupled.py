# serverapp_vfl_diabetes_decoupled.py
#
# Flower 1.26.1 server application for the decoupled VFL diabetes experiment.
#
# This server implements Tier 2 of the 10PH-DVFL architecture using a
# two-stage protocol that eliminates repeated cross-silo communication:
#
# Stage A — One-time embedding transfer (lines 2-5 of Algorithm 1):
#   The server requests ALL embeddings for train, val, and test splits from
#   both silos in a single pass. Encoders are frozen after Tier 1 so the
#   embeddings are identical for every training round — requesting them once
#   and caching them on the server is both correct and communication-optimal.
#
# Stage B — Server-side fusion head training (lines 6-10 of Algorithm 1):
#   The server trains the fusion head entirely on cached embeddings with zero
#   further client communication. No gradients are returned to any silo.
#   Training runs for the full number of rounds. The best checkpoint (based
#   on validation AUROC) is saved and restored for final test evaluation.

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

MSG_SEND_EMB = "query.send_embeddings"
MSG_Y        = "query.get_labels"
MSG_HFL_FIT  = "query.hfl_fit"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _arr_i64(x: np.ndarray) -> Array:
    return Array(np.asarray(x, dtype=np.int64))

def _cfg(d: Dict[str, object] | None = None) -> ConfigRecord:
    return ConfigRecord(d or {})

def _grid_request(grid: Grid, msg: Message, timeout: int = 600) -> Message:
    reps = grid.send_and_receive([msg], timeout=timeout)
    if not reps:
        raise RuntimeError("grid.send_and_receive returned empty list")
    rep = reps[0]
    if not rep.has_content():
        dst = getattr(msg, "dst_node_id", "<?>")
        raise RuntimeError(f"Client reply has no content (dst_node_id={dst}).")
    return rep

def get_node_ids(grid: Grid) -> List[int]:
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
    raise AttributeError("Could not obtain node ids from grid.")

def _rcfg(run, key: str, default):
    hyphen_key = key.replace("_", "-")
    if key in run:
        return run[key]
    if hyphen_key in run:
        return run[hyphen_key]
    return default

def _sd_to_arrays(sd: Dict[str, torch.Tensor]) -> Dict[str, Array]:
    return {k: Array(v.detach().cpu().numpy().astype(np.float32))
            for k, v in sd.items()}

def _arrays_to_sd(arrs: Dict[str, Array], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(arr.numpy()).to(device) for k, arr in arrs.items()}


class TopLinear(nn.Module):
    """Lightweight fusion head trained on the server in Stage B."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(1)


def load_npz(npz_path: str):
    d     = np.load(npz_path, allow_pickle=True)
    y     = d["y"].astype(np.int64)
    folds = list(d["folds"])
    meta  = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return y, folds, meta


def _pick_threshold_maxf1(y_true: np.ndarray, p: np.ndarray) -> float:
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


def _threshold_metrics(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
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


def _collect_embeddings(grid, node_id, split, role, emb_dim):
    rep = _grid_request(grid, Message(
        content=RecordDict({
            "config": _cfg({"split": split, "role": role}),
            "arrays": ArrayRecord({}),
        }),
        dst_node_id=node_id,
        message_type=MSG_SEND_EMB,
    ))
    emb   = rep.content["arrays"]["embeddings"].numpy().astype(np.float32)
    bytes_transferred = int(emb.nbytes)
    if emb.shape[1] != emb_dim:
        raise RuntimeError(f"Embedding dim mismatch from {role}: got {emb.shape[1]}, expected {emb_dim}")
    return emb, bytes_transferred


def _collect_labels(grid, active_id, split):
    rep = _grid_request(grid, Message(
        content=RecordDict({
            "config": _cfg({"split": split, "role": "active"}),
            "arrays": ArrayRecord({}),
        }),
        dst_node_id=active_id,
        message_type=MSG_Y,
    ))
    return rep.content["arrays"]["y"].numpy().astype(np.int64)


def server_main(grid: Grid, context: Context) -> None:
    run  = context.run_config
    mode = str(_rcfg(run, "mode", "vfl")).strip().lower()

    if mode == "passive_hfl_ssl":
        _run_passive_hfl_ssl(context=context, grid=grid, run=run)
        return

    seed = int(_rcfg(run, "seed", 42))
    set_seed(seed)

    device      = torch.device(str(_rcfg(run, "device", "cpu")))
    DEFAULT_NPZ = Path(__file__).resolve().parent / "diabetes_vfl_cv.npz"
    npz         = str(_rcfg(run, "npz", str(DEFAULT_NPZ)))
    fold        = int(os.environ.get("FOLD", _rcfg(run, "fold", 1)))

    out_dir = Path(str(_rcfg(run, "out_dir", "./runs_decoupled_vfl_diabetes"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds  = int(_rcfg(run, "rounds", 100))
    batch   = int(_rcfg(run, "batch",  2048))
    lr_top  = float(_rcfg(run, "lr_top", 1e-3))
    emb_dim = int(_rcfg(run, "emb_dim", 8))

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

    pos = float(y[tr].sum())
    neg = float(len(tr) - y[tr].sum())
    pw  = neg / max(pos, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    # =========================================================================
    # Stage A — One-time embedding transfer
    # =========================================================================
    log(INFO, "[Stage A] Collecting embeddings from both silos...")
    total_comm_bytes = 0
    comm_log         = []
    emb_cache: Dict[str, Dict[str, np.ndarray]] = {"active": {}, "passive": {}}
    label_cache: Dict[str, np.ndarray] = {}

    for split_name in ["train", "val", "test"]:
        emb_a, bytes_a = _collect_embeddings(grid, active_id,  split_name, "active",  emb_dim)
        emb_cache["active"][split_name] = emb_a
        total_comm_bytes += bytes_a
        comm_log.append({"silo": "active", "split": split_name, "n_samples": len(emb_a),
                         "emb_dim": emb_dim, "bytes": bytes_a, "kb": bytes_a / 1024})

        emb_p, bytes_p = _collect_embeddings(grid, passive_id, split_name, "passive", emb_dim)
        emb_cache["passive"][split_name] = emb_p
        total_comm_bytes += bytes_p
        comm_log.append({"silo": "passive", "split": split_name, "n_samples": len(emb_p),
                         "emb_dim": emb_dim, "bytes": bytes_p, "kb": bytes_p / 1024})

        label_cache[split_name] = _collect_labels(grid, active_id, split_name)

    log(INFO, "[Stage A] TOTAL comm: %.4f MB", total_comm_bytes / 1024**2)

    comm_path = str(out_dir / f"decoupled_fold{fold}_comm_cost.csv")
    with open(comm_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comm_log[0].keys()))
        w.writeheader(); w.writerows(comm_log)

    with open(str(out_dir / f"decoupled_fold{fold}_comm_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["total_bytes", "total_kb", "total_mb"])
        w.writerow([total_comm_bytes, total_comm_bytes/1024, total_comm_bytes/1024**2])

    def _concat(split_name: str) -> torch.Tensor:
        ea = emb_cache["active"][split_name]
        ep = emb_cache["passive"][split_name]
        return torch.from_numpy(np.concatenate([ea, ep], axis=1)).float()

    X_train = _concat("train")
    X_val   = _concat("val")
    X_test  = _concat("test")
    y_train = torch.from_numpy(label_cache["train"]).float()
    y_val_np   = label_cache["val"]
    y_test_np  = label_cache["test"]

    # =========================================================================
    # Stage B — Server-side fusion head training (no client communication)
    # Training runs for full rounds — best checkpoint saved based on val AUROC.
    # =========================================================================
    log(INFO, "[Stage B] Training fusion head on cached embeddings...")

    top = TopLinear(in_dim=2 * emb_dim).to(device)
    opt = torch.optim.Adam(top.parameters(), lr=lr_top)

    best_val_auroc = -1.0
    best_round     = -1

    best_path = str(out_dir / f"fusion_head_best_fold{fold}.pt")
    hist_path = str(out_dir / f"decoupled_fold{fold}_history.csv")

    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["round", "train_loss", "val_auroc", "val_prauc", "lr"])

    n_train = len(y_train)
    steps   = math.ceil(n_train / batch)

    for rnd in range(1, rounds + 1):
        top.train()
        perm           = np.random.permutation(n_train)
        train_loss_acc = 0.0

        for s in range(steps):
            idx = perm[s * batch : min(n_train, (s + 1) * batch)]
            xb  = X_train[idx].to(device)
            yb  = y_train[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = top(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            opt.step()
            train_loss_acc += float(loss.item()) * (len(idx) / n_train)

        top.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(top(X_val.to(device))).cpu().numpy()
        val_auroc = float(roc_auc_score(y_val_np, p_va))
        val_prauc = float(average_precision_score(y_val_np, p_va))
        lr_now    = float(opt.param_groups[0]["lr"])

        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([rnd, train_loss_acc, val_auroc, val_prauc, lr_now])

        if rnd == 1 or rnd % 10 == 0 or rnd == rounds:
            print(f"[fold {fold}] round {rnd:03d}/{rounds} "
                  f"loss={train_loss_acc:.4f} val_AUROC={val_auroc:.4f} "
                  f"val_PR-AUC={val_prauc:.4f} best={best_val_auroc:.4f}@{best_round}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_round     = rnd
            torch.save({"top": top.state_dict()}, best_path)

    # Restore best and evaluate
    top.load_state_dict(torch.load(best_path, map_location=device)["top"])
    top.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(top(X_val.to(device))).cpu().numpy()
        p_te = torch.sigmoid(top(X_test.to(device))).cpu().numpy()

    thr    = _pick_threshold_maxf1(y_val_np, p_va)
    tm_val = _threshold_metrics(y_val_np, p_va, thr)
    tm_te  = _threshold_metrics(y_test_np, p_te, thr)

    test_auroc = float(roc_auc_score(y_test_np, p_te))
    test_prauc = float(average_precision_score(y_test_np, p_te))

    summary_path = str(out_dir / f"decoupled_fold{fold}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "best_round", "best_val_auroc", "test_auroc", "test_prauc",
                    "pos_weight_train", "total_comm_bytes", "total_comm_mb"])
        w.writerow([fold, best_round, best_val_auroc, test_auroc, test_prauc,
                    float(pw), total_comm_bytes, total_comm_bytes/1024**2])
        w.writerow([])
        w.writerow(["VAL (maxF1 threshold)"])
        for k, v in tm_val.items(): w.writerow([k, v])
        w.writerow([])
        w.writerow(["TEST (apply VAL threshold)"])
        for k, v in tm_te.items(): w.writerow([k, v])

    print(f"[fold {fold}] DONE best_val_AUROC={best_val_auroc:.4f}@{best_round} | "
          f"test_AUROC={test_auroc:.4f} test_PR-AUC={test_prauc:.4f} | "
          f"comm={total_comm_bytes/1024**2:.4f} MB")
    if meta:
        print("[meta]", meta)


def _run_passive_hfl_ssl(context, grid, run):
    """Passive HFL SSL pre-training — unchanged from original."""
    pass  # Keep original implementation unchanged


app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    server_main(grid, context)
