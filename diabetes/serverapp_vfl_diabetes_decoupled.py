# serverapp_vfl_diabetes_decoupled.py
#
# Server application for the decoupled VFL diabetes experiment.
#
# This server implements Tier 2 of the 10PH-DVFL architecture:
# - Requests frozen embeddings from the active and passive silos once per batch
# - Trains only the lightweight fusion head (TopLinear) on the server side
# - No gradients are returned to any silo at any point
# - Supports two pre-training modes for the passive silo:
#     "vfl": standard decoupled VFL (SSL or supervised active encoder)
#     "passive_hfl_ssl": intra-silo HFL pre-training for the passive silo before VFL fusion
#
# Communication pattern (Tier 2):
#   For each training batch:
#     1. Request active embeddings from active silo (MSG_EMB, view=0)
#     2. Request passive embeddings from passive silo (MSG_EMB, view=1)
#     3. Request labels from active silo (MSG_Y)
#     4. Concatenate embeddings, compute loss, update fusion head locally
#     No gradients are sent back to either silo.

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

# Message type constants must match the handler names registered in the client application.
MSG_EMB = "query.generate_embeddings"
MSG_Y = "query.get_labels"
MSG_HFL_FIT = "query.hfl_fit"


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


def _rcfg(run, key: str, default):
    """
    Read a value from the Flower run configuration.
    Flower converts underscore keys to hyphens internally, so both forms are checked.
    """
    hyphen_key = key.replace("_", "-")
    if key in run:
        return run[key]
    if hyphen_key in run:
        return run[hyphen_key]
    return default


def _sd_to_arrays(sd: Dict[str, torch.Tensor]) -> Dict[str, Array]:
    """Convert a PyTorch state dict to a dict of Flower Arrays for transmission."""
    return {k: Array(v.detach().cpu().numpy().astype(np.float32)) for k, v in sd.items()}


def _arrays_to_sd(arrs: Dict[str, Array], device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert a dict of Flower Arrays back to a PyTorch state dict on the given device."""
    return {k: torch.from_numpy(arr.numpy()).to(device) for k, arr in arrs.items()}


# ---------------------------------------------------------------------------
# Server-side fusion head
# ---------------------------------------------------------------------------

class TopLinear(nn.Module):
    """
    Lightweight fusion head trained on the server in Tier 2.

    Receives concatenated embeddings from the active and passive silos
    and produces a scalar logit for binary classification.
    Input dimensionality must equal emb_dim_active + emb_dim_passive.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(1)


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
    Select the classification threshold that maximises F1 score on the validation set.
    This threshold is then applied to the test set without further tuning.
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


def _threshold_metrics(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
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
    emb_dim: int,
    timeout: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the fusion head over a set of sample indices.

    For each batch:
      1. Requests active embeddings from the active silo node
      2. Requests passive embeddings from the passive silo node
      3. Requests labels from the active silo node
      4. Concatenates embeddings and runs the fusion head forward pass
    Returns arrays of true labels and predicted probabilities.
    """
    top.eval()
    y_all: List[np.ndarray] = []
    p_all: List[np.ndarray] = []

    bs = 4096
    for s in range(0, len(idx), bs):
        bidx = idx[s : s + bs].astype(np.int64)

        rep_x = _grid_request(
            grid,
            Message(
                content=RecordDict(
                    {
                        "config": _cfg({"view": 0, "role": "active"}),
                        "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                    }
                ),
                dst_node_id=active_id,
                message_type=MSG_EMB,
            ),
            timeout=timeout,
        )

        rep_p = _grid_request(
            grid,
            Message(
                content=RecordDict(
                    {
                        "config": _cfg({"view": 1, "role": "passive"}),
                        "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                    }
                ),
                dst_node_id=passive_id,
                message_type=MSG_EMB,
            ),
            timeout=timeout,
        )

        ex = rep_x.content["arrays"]["embedding"].numpy().astype(np.float32)
        ep = rep_p.content["arrays"]["embedding"].numpy().astype(np.float32)
        if ex.shape[1] != emb_dim or ep.shape[1] != emb_dim:
            raise RuntimeError(
                f"Embedding dim mismatch: ex={ex.shape}, ep={ep.shape}, expected emb_dim={emb_dim}"
            )

        rep_y = _grid_request(
            grid,
            Message(
                content=RecordDict(
                    {
                        "config": _cfg({"role": "active"}),
                        "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                    }
                ),
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
    Entry point for the server application.

    Dispatches to the passive HFL pre-training routine if mode is
    'passive_hfl_ssl', otherwise runs the main Tier 2 decoupled VFL
    fusion head training loop.
    """
    run = context.run_config
    mode = str(_rcfg(run, "mode", "vfl")).strip().lower()

    if mode == "passive_hfl_ssl":
        _run_passive_hfl_ssl(context=context, grid=grid, run=run)
        return

    seed = int(_rcfg(run, "seed", 42))
    set_seed(seed)

    device = torch.device(str(_rcfg(run, "device", "cpu")))
    DEFAULT_NPZ = Path(__file__).resolve().parent / "diabetes_vfl_cv.npz"
    npz = str(_rcfg(run, "npz", str(DEFAULT_NPZ)))
    fold = int(os.environ.get("FOLD", _rcfg(run, "fold", 1)))

    out_dir = Path(str(_rcfg(run, "out_dir", "./runs_decoupled_vfl_diabetes"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = str(out_dir)
    print(f"[server] out_dir resolved to: {out_dir}")

    rounds = int(_rcfg(run, "rounds", 100))
    batch = int(_rcfg(run, "batch", 2048))
    lr_top = float(_rcfg(run, "lr_top", 1e-3))
    patience = int(_rcfg(run, "patience", 15))
    emb_dim = int(_rcfg(run, "emb_dim", 8))

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

    # Compute positive class weight from the training split to handle class imbalance.
    pos = float(y[tr].sum())
    neg = float(len(tr) - y[tr].sum())
    pw = neg / max(pos, 1.0)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    # Initialise the server-side fusion head.
    # Input dimension is twice the embedding dimension since active and passive
    # embeddings are concatenated before classification.
    top = TopLinear(in_dim=2 * emb_dim).to(device)
    opt = torch.optim.Adam(top.parameters(), lr=lr_top)

    best_val_auroc = -1.0
    best_round = -1
    no_improve = 0

    # Checkpoint paths for the best fusion head and per-round training history.
    best_path = os.path.join(out_dir, f"fusion_head_best_fold{fold}.pt")
    hist_path = os.path.join(out_dir, f"decoupled_fold{fold}_history.csv")

    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["round", "train_loss", "val_auroc", "val_prauc", "lr"])

    n = len(tr)
    steps = math.ceil(n / batch)

    # Tier 2 training loop.
    # Each round requests frozen embeddings from both silos,
    # concatenates them on the server, and updates only the fusion head.
    # No gradients are returned to any silo at any point.
    for rnd in range(1, rounds + 1):
        top.train()
        perm = np.random.permutation(tr)
        train_loss_acc = 0.0

        for s in range(steps):
            bidx = perm[s * batch : min(n, (s + 1) * batch)].astype(np.int64)

            # Request frozen active-silo embeddings.
            rep_x = _grid_request(
                grid,
                Message(
                    content=RecordDict(
                        {
                            "config": _cfg({"view": 0, "role": "active"}),
                            "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                        }
                    ),
                    dst_node_id=active_id,
                    message_type=MSG_EMB,
                ),
            )

            # Request frozen passive-silo embeddings.
            rep_p = _grid_request(
                grid,
                Message(
                    content=RecordDict(
                        {
                            "config": _cfg({"view": 1, "role": "passive"}),
                            "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                        }
                    ),
                    dst_node_id=passive_id,
                    message_type=MSG_EMB,
                ),
            )

            ex = rep_x.content["arrays"]["embedding"].numpy().astype(np.float32)
            ep = rep_p.content["arrays"]["embedding"].numpy().astype(np.float32)
            if ex.shape[1] != emb_dim or ep.shape[1] != emb_dim:
                raise RuntimeError(
                    f"Embedding dim mismatch in train: ex={ex.shape}, ep={ep.shape}, expected emb_dim={emb_dim}"
                )

            # Request labels from the active silo (passive silo has no label access).
            rep_y = _grid_request(
                grid,
                Message(
                    content=RecordDict(
                        {
                            "config": _cfg({"role": "active"}),
                            "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)}),
                        }
                    ),
                    dst_node_id=active_id,
                    message_type=MSG_Y,
                ),
            )
            yb = rep_y.content["arrays"]["y"].numpy().astype(np.int64)

            # Concatenate active and passive embeddings and update the fusion head.
            z = torch.from_numpy(np.concatenate([ex, ep], axis=1)).to(device)
            yy = torch.from_numpy(yb).float().to(device)

            opt.zero_grad(set_to_none=True)
            logits = top(z)
            loss = criterion(logits, yy)
            loss.backward()
            opt.step()

            frac = len(bidx) / n
            train_loss_acc += float(loss.item()) * frac

        # Evaluate on the validation set using frozen embeddings from both silos.
        y_va, p_va = _eval_probs(grid, top, device, va, active_id, passive_id, emb_dim)
        val_auroc = float(roc_auc_score(y_va, p_va))
        val_prauc = float(average_precision_score(y_va, p_va))
        lr_now = float(opt.param_groups[0]["lr"])

        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([rnd, train_loss_acc, val_auroc, val_prauc, lr_now])

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
            best_round = rnd
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
    y_va, p_va = _eval_probs(grid, top, device, va, active_id, passive_id, emb_dim)
    y_te, p_te = _eval_probs(grid, top, device, te, active_id, passive_id, emb_dim)

    # Select the classification threshold on the validation set using max F1,
    # then apply it to the test set without further tuning.
    thr = _pick_threshold_maxf1(y_va, p_va)
    tm_val = _threshold_metrics(y_va, p_va, thr)
    tm_te = _threshold_metrics(y_te, p_te, thr)

    test_auroc = float(roc_auc_score(y_te, p_te))
    test_prauc = float(average_precision_score(y_te, p_te))

    summary_path = os.path.join(out_dir, f"decoupled_fold{fold}_summary.csv")
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
# Intra-silo HFL pre-training for the passive silo (Tier 1 optional stage)
# ---------------------------------------------------------------------------

def _run_passive_hfl_ssl(context: Context, grid: Grid, run: Dict[str, object]) -> None:
    """
    Run the intra-silo horizontal federated learning pre-training stage
    for the passive silo encoder using a self-supervised objective.

    This implements the optional Tier 1 HFL stage of the 10PH-DVFL architecture,
    where the passive silo trains its encoder across K simulated IID clients
    using FedAvg before the Tier 2 cross-silo fusion begins.

    Each client trains the shared passive encoder locally using two augmented
    tabular views of its partition (cosine similarity loss), then sends updated
    weights back to the server for aggregation. The final aggregated encoder
    checkpoint is saved and used as the passive silo encoder in Tier 2.
    """
    seed = int(_rcfg(run, "seed", 42))
    set_seed(seed)

    npz = str(_rcfg(run, "npz", "diabetes_vfl_cv.npz"))
    fold = int(_rcfg(run, "fold", 1))
    device = torch.device(str(_rcfg(run, "device", "cpu")))

    K = int(_rcfg(run, "K", 10))
    rounds = int(_rcfg(run, "rounds", 20))
    local_epochs = int(_rcfg(run, "local_epochs", 1))
    batch = int(_rcfg(run, "batch", 256))
    lr = float(_rcfg(run, "lr", 1e-3))
    noise_std = float(_rcfg(run, "noise_std", 0.05))
    dropout_p = float(_rcfg(run, "dropout_p", 0.1))

    out_dir = Path(str(_rcfg(run, "out_dir", "./runs_passive_hfl_diabetes")))
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(npz, allow_pickle=True)
    X2 = d["X2"].astype(np.float32)
    folds = list(d["folds"])
    split_obj = folds[fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)

    node_ids = get_node_ids(grid)
    if len(node_ids) < K:
        raise RuntimeError(
            f"Need at least K={K} clients, but grid has {len(node_ids)} nodes. "
            f"Set options.num-supernodes={K} in your simulator config"
        )
    node_ids = node_ids[:K]
    log(INFO, f"[PASSIVE_HFL_SSL] fold={fold} K={K} rounds={rounds} device={device} node_ids={node_ids}")

    from clientapp_vfl_diabetes_decoupled import BottomMLP_Paper
    global_model = BottomMLP_Paper(in_dim=int(X2.shape[1])).to(device)
    global_sd = global_model.state_dict()

    # Partition the passive silo training data into K IID client partitions.
    parts = np.array_split(tr, K)
    client_indices_path = out_dir / f"client_indices_fold{fold}.npz"
    np.savez_compressed(client_indices_path, **{f"c{i}": parts[i].astype(np.int64) for i in range(K)})

    manifest = {
        "fold": fold,
        "K": K,
        "train_size": int(len(tr)),
        "client_sizes": [int(len(p)) for p in parts],
        "node_ids": [int(n) for n in node_ids],
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch": batch,
        "lr": lr,
        "noise_std": noise_std,
        "dropout_p": dropout_p,
    }
    (out_dir / f"partition_manifest_fold{fold}.json").write_text(json.dumps(manifest, indent=2))

    metrics_csv = out_dir / f"hfl_ssl_metrics_fold{fold}.csv"
    with metrics_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "avg_loss"])

        for r in range(1, rounds + 1):
            gw_arrs = ArrayRecord({"global_weights": _sd_to_arrays(global_sd)})
            replies = []
            losses = []
            ns = []

            # Send global encoder weights to each client and collect local updates.
            for rank, nid in enumerate(node_ids):
                msg = Message(
                    message_type=MSG_HFL_FIT,
                    content=RecordDict(
                        {
                            "arrays": gw_arrs,
                            "config": _cfg(
                                {
                                    "role": "passive_hfl",
                                    "client_rank": rank,
                                    "K": K,
                                    "lr": lr,
                                    "local_epochs": local_epochs,
                                    "batch": batch,
                                    "noise_std": noise_std,
                                    "dropout_p": dropout_p,
                                }
                            ),
                        }
                    ),
                    dst_node_id=int(nid),
                )
                rep = _grid_request(grid, msg, timeout=600)
                rep_cfg = rep.content["config"]
                n = int(rep_cfg.get("num_examples", 0))
                loss = float(rep_cfg.get("loss", 0.0))
                ns.append(n)
                losses.append(loss)
                replies.append(rep)

            # Aggregate client updates using FedAvg weighted by local dataset size.
            total = float(sum(ns)) if ns else 1.0
            new_sd = {k: torch.zeros_like(v) for k, v in global_sd.items()}
            for rep, n in zip(replies, ns):
                sd_i = _arrays_to_sd(rep.content["arrays"]["global_weights"], device)
                for k in new_sd.keys():
                    new_sd[k] += (n / total) * sd_i[k]

            global_sd = new_sd
            avg_loss = float(np.mean(losses)) if losses else 0.0
            w.writerow([r, avg_loss])
            if r % max(1, rounds // 5) == 0 or r == 1 or r == rounds:
                log(INFO, f"[PASSIVE_HFL_SSL] round={r}/{rounds} avg_loss={avg_loss:.6f}")

    # Save the final aggregated passive encoder checkpoint for use in Tier 2.
    ckpt_path = out_dir / f"pretrained_passive_bottom_hfl_fold{fold}.pt"
    torch.save({"bottom_state": {k: v.detach().cpu() for k, v in global_sd.items()}}, ckpt_path)
    log(INFO, f"[PASSIVE_HFL_SSL] saved {ckpt_path}")


# ---------------------------------------------------------------------------
# Flower ServerApp entry point
# ---------------------------------------------------------------------------

app = ServerApp()


@app.main()
def main(grid, context) -> None:
    server_main(grid, context)
