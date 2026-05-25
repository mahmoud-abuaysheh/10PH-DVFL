# serverapp_vfl_midas_splitnn_head.py
# Changes vs original:
#   - MIN_ROUNDS guard (=5) — early stopping only activates after round 5
#   - Communication cost tracking per round and total (MB)
#     Forward: 3 clients × batch × EMB_DIM × 4 bytes
#     Backward: 3 clients × batch × EMB_DIM × 4 bytes
#   - total_rounds, comm_cost_per_round_MB, comm_cost_total_MB in summary.csv
#   - train_log.csv — per-round loss, val_auroc, val_prauc
#   - Aggregation: run python serverapp_vfl_midas_splitnn_head.py --aggregate
from __future__ import annotations

import os
import csv
import argparse
from logging import INFO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


ART_DIR    = Path(os.environ.get("ART_DIR", ".")).resolve()
EMB_DIM    = _env_int("EMB_DIM", 256)

HEAD_HIDDEN  = _env_int("HEAD_HIDDEN", 512)
HEAD_DROPOUT = _env_float("HEAD_DROPOUT", 0.2)
HEAD_LR      = _env_float("HEAD_LR", 1e-4)
HEAD_WD      = _env_float("HEAD_WD", 1e-4)

ROUNDS               = _env_int("ROUNDS", 20)
MIN_ROUNDS           = _env_int("MIN_ROUNDS", 5)       # early stopping guard
SEED                 = _env_int("SEED", 42)
DEVICE       = os.environ.get("DEVICE", "cpu")
DEVICE_TORCH = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE           = _env_int("BATCH_SIZE", 2)
EVAL_EVERY           = _env_int("EVAL_EVERY", 1)
EARLY_STOP_PATIENCE  = _env_int("EARLY_STOP_PATIENCE", 7)
MIN_DELTA            = _env_float("MIN_DELTA", 1e-4)
IMAGE_ROOT           = os.environ.get("IMAGE_ROOT", "")

# ── Communication cost helpers ────────────────────────────────────────────────
# Per batch, per direction (forward OR backward): 3 clients × B × EMB_DIM × float32
def _comm_bytes_per_batch(batch_size: int) -> int:
    """Bytes for ONE direction (forward embeddings OR backward gradients)."""
    return 3 * batch_size * EMB_DIM * 4  # float32 = 4 bytes

def _comm_mb_per_round(n_train: int, batch_size: int) -> float:
    """Total MB exchanged in one round (forward + backward per batch)."""
    n_batches   = int(np.ceil(n_train / batch_size))
    bytes_round = n_batches * _comm_bytes_per_batch(batch_size) * 2  # ×2: fwd+bwd
    return bytes_round / (1024 ** 2)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _grid_request(grid: Grid, msg: Message, timeout: int = 600) -> Message:
    reps = grid.send_and_receive([msg], timeout=timeout)
    if not reps or not reps[0].has_content():
        raise RuntimeError("Client reply has no content (client crashed or raised).")
    return reps[0]


def _cfg(**kwargs) -> ConfigRecord:
    return ConfigRecord({k: v for k, v in kwargs.items()})


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    return 1.0 / (1.0 + np.exp(-x))


def _threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray,
                     step: float = 0.01) -> List[Dict[str, float]]:
    y_true = y_true.astype(int).reshape(-1)
    y_prob = y_prob.astype(float).reshape(-1)
    rows: List[Dict[str, float]] = []
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
            youdenJ=float(rec+spec-1),
            tp=float(tp), tn=float(tn), fp=float(fp), fn=float(fn),
        ))
    return rows


def _pick_best(rows: List[Dict[str, float]], key: str) -> Dict[str, float]:
    return sorted(
        rows,
        key=lambda r: (-float(r[key]), -float(r["f1"]),
                       -float(r["acc"]), float(r["threshold"]))
    )[0]


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


class OldMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    set_seed(SEED)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    log(INFO, f"[SERVER] ART_DIR={ART_DIR}")
    log(INFO, f"[SERVER] IMAGE_ROOT={IMAGE_ROOT}")
    log(INFO, f"[SERVER] EMB_DIM={EMB_DIM} DEVICE={DEVICE_TORCH}")
    log(INFO, f"[SERVER] ROUNDS={ROUNDS} MIN_ROUNDS={MIN_ROUNDS} "
              f"BATCH_SIZE={BATCH_SIZE} EVAL_EVERY={EVAL_EVERY}")
    log(INFO, f"[SERVER] HEAD_HIDDEN={HEAD_HIDDEN} HEAD_LR={HEAD_LR} HEAD_WD={HEAD_WD}")

    node_ids_sorted = sorted(list(grid.get_node_ids()))
    log(INFO, f"[SERVER] node_ids={node_ids_sorted}")
    if len(node_ids_sorted) < 3:
        raise RuntimeError(f"Need 3 nodes, got {len(node_ids_sorted)}: {node_ids_sorted}")

    roles: Dict[str, int] = {
        "active": node_ids_sorted[0],
        "p6":     node_ids_sorted[1],
        "p1":     node_ids_sorted[2],
    }
    log(INFO, f"[SERVER] assigned: active={roles['active']} "
              f"p6={roles['p6']} p1={roles['p1']}")

    # ── Init roles ────────────────────────────────────────────────────────────
    for role, nid in roles.items():
        msg = Message(
            content=RecordDict({"config": _cfg(role=role, image_root=IMAGE_ROOT)}),
            message_type="query.init_role",
            dst_node_id=nid,
        )
        _ = _grid_request(grid, msg)

    # ── Labels from active client ─────────────────────────────────────────────
    meta_active = _request_meta(grid, roles["active"])
    y_train = torch.from_numpy(meta_active["y_train"]).float().view(-1, 1)
    y_val   = torch.from_numpy(meta_active["y_val"]).float().view(-1, 1)
    y_test  = torch.from_numpy(meta_active["y_test"]).float().view(-1, 1)
    n_train = len(y_train)
    log(INFO, f"[SERVER] n_train={n_train} n_val={len(y_val)} n_test={len(y_test)}")

    # ── Communication cost per round ──────────────────────────────────────────
    comm_mb_per_round = _comm_mb_per_round(n_train, BATCH_SIZE)
    log(INFO, f"[SERVER] comm_cost_per_round={comm_mb_per_round:.4f} MB")

    # ── Model ─────────────────────────────────────────────────────────────────
    head      = OldMLP(in_dim=EMB_DIM * 3, hidden=HEAD_HIDDEN,
                       dropout=HEAD_DROPOUT).to(DEVICE_TORCH)
    opt       = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = -1.0
    best_round     = 0
    total_rounds   = 0
    bad            = 0
    train_log      = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for r in range(1, ROUNDS + 1):
        total_rounds = r
        head.train()
        perm       = np.random.permutation(n_train)
        total_loss = 0.0
        nb         = 0

        for start in range(0, n_train, BATCH_SIZE):
            idx      = perm[start:start + BATCH_SIZE]
            idx_list = [int(i) for i in idx.tolist()]

            emb = _get_embeddings_batch(
                grid, roles, split="train", idx_list=idx_list).to(DEVICE_TORCH)
            emb.requires_grad_(True)

            opt.zero_grad(set_to_none=True)
            logits = head(emb)
            loss   = criterion(logits, y_train[idx].to(DEVICE_TORCH))
            loss.backward()

            grad_chunks = emb.grad.detach().cpu().split(
                [EMB_DIM, EMB_DIM, EMB_DIM], dim=1)
            opt.step()
            _push_gradients_batch(grid, roles, split="train",
                                  idx_list=idx_list, grad_chunks=grad_chunks)

            total_loss += float(loss.item())
            nb         += 1

        avg_loss = total_loss / max(1, nb)

        if (r % EVAL_EVERY) == 0:
            val_auroc, val_ap, _  = _eval_split(
                grid, roles, head, y_val,  split="val",  return_probs=True)
            test_auroc, test_ap, _ = _eval_split(
                grid, roles, head, y_test, split="test", return_probs=True)

            train_log.append(dict(
                round=r, train_loss=avg_loss,
                val_auroc=val_auroc, val_prauc=val_ap,
                test_auroc=test_auroc, test_prauc=test_ap,
            ))

            log(INFO, f"[SERVER] round={r} loss={avg_loss:.4f} | "
                      f"val AUROC={val_auroc:.4f} AP={val_ap:.4f} | "
                      f"test AUROC={test_auroc:.4f} AP={test_ap:.4f}")

            # Early stopping — only after MIN_ROUNDS
            if val_auroc > best_val_auroc + MIN_DELTA:
                best_val_auroc = val_auroc
                best_round     = r
                bad            = 0
                torch.save(head.state_dict(), ART_DIR / "head_best.pt")
            else:
                if r >= MIN_ROUNDS:   # guard: don't stop before MIN_ROUNDS
                    bad += 1
                    if bad >= EARLY_STOP_PATIENCE:
                        log(INFO, f"[SERVER] Early stopping at round {r} "
                                  f"(best val AUROC={best_val_auroc:.4f})")
                        break
        else:
            train_log.append(dict(
                round=r, train_loss=avg_loss,
                val_auroc=None, val_prauc=None,
                test_auroc=None, test_prauc=None,
            ))
            log(INFO, f"[SERVER] round={r} loss={avg_loss:.4f}")

    _write_csv(ART_DIR / "train_log.csv", train_log)

    # ── Load best head ─────────────────────────────────────────────────────────
    ckpt = ART_DIR / "head_best.pt"
    if ckpt.exists():
        head.load_state_dict(torch.load(ckpt, map_location=DEVICE_TORCH,
                                         weights_only=False))
        log(INFO, f"[SERVER] Loaded best head (best_round={best_round})")

    val_auroc,  val_ap,  val_probs  = _eval_split(
        grid, roles, head, y_val,  split="val",  return_probs=True)
    test_auroc, test_ap, test_probs = _eval_split(
        grid, roles, head, y_test, split="test", return_probs=True)

    y_val_np  = y_val.numpy().reshape(-1)
    y_test_np = y_test.numpy().reshape(-1)

    thr_val = _threshold_sweep(y_val_np,  val_probs,  step=0.01)
    thr_te  = _threshold_sweep(y_test_np, test_probs, step=0.01)
    _write_csv(ART_DIR / "threshold_metrics_val.csv",  thr_val)
    _write_csv(ART_DIR / "threshold_metrics_test.csv", thr_te)

    best_val_you = _pick_best(thr_val, "youdenJ")
    best_val_f1  = _pick_best(thr_val, "f1")

    comm_total_mb = comm_mb_per_round * total_rounds

    summary = {
        "best_round":              float(best_round),
        "total_rounds":            float(total_rounds),
        "val_auroc":               float(val_auroc),
        "val_pr_auc":              float(val_ap),
        "test_auroc":              float(test_auroc),
        "test_pr_auc":             float(test_ap),
        "val_thr_best_youden":     float(best_val_you["threshold"]),
        "val_thr_best_f1":         float(best_val_f1["threshold"]),
        "comm_cost_per_round_MB":  float(comm_mb_per_round),
        "comm_cost_total_MB":      float(comm_total_mb),
    }
    _write_csv(ART_DIR / "summary.csv", [summary])

    log(INFO, f"[SERVER] Final VAL  AUROC={val_auroc:.4f} AP={val_ap:.4f}")
    log(INFO, f"[SERVER] Final TEST AUROC={test_auroc:.4f} AP={test_ap:.4f}")
    log(INFO, f"[SERVER] best_round={best_round} total_rounds={total_rounds}")
    log(INFO, f"[SERVER] comm/round={comm_mb_per_round:.4f} MB  "
              f"comm_total={comm_total_mb:.4f} MB")
    log(INFO, f"[SERVER] Wrote: head_best.pt, train_log.csv, "
              f"threshold_metrics_val/test.csv, summary.csv")


# ── Helper functions ──────────────────────────────────────────────────────────

def _request_meta(grid: Grid, active_node_id: int) -> Dict[str, np.ndarray]:
    msg = Message(content=RecordDict(),
                  message_type="query.meta", dst_node_id=active_node_id)
    rep  = _grid_request(grid, msg, timeout=600)
    arrs = rep.content["arrays"]
    return {
        "y_train": arrs["y_train"].numpy(),
        "y_val":   arrs["y_val"].numpy(),
        "y_test":  arrs["y_test"].numpy(),
    }


def _get_embeddings_batch(grid: Grid, roles: Dict[str, int],
                          split: str, idx_list: List[int]) -> torch.Tensor:
    messages: List[Message] = []
    for role, nid in roles.items():
        pos = {"active": 0, "p6": 1, "p1": 2}[role]
        messages.append(Message(
            content=RecordDict({"config": _cfg(split=split, pos=pos,
                                               indices=idx_list)}),
            message_type="query.get_embeddings",
            dst_node_id=nid,
        ))

    replies = grid.send_and_receive(messages, timeout=1200)
    if len(replies) != len(messages):
        raise RuntimeError(f"Expected {len(messages)} embedding replies, "
                           f"got {len(replies)}")

    bsz = len(idx_list)
    emb = torch.zeros((bsz, EMB_DIM * 3), dtype=torch.float32)
    for rep in replies:
        if not rep.has_content():
            raise RuntimeError("Client reply has no content.")
        arrays = rep.content["arrays"]
        pos    = int(rep.content["config"]["pos"])
        x      = torch.from_numpy(arrays["embedding"].numpy()).float()
        if x.shape[0] != bsz or x.shape[1] != EMB_DIM:
            raise RuntimeError(f"Bad embedding shape {tuple(x.shape)} "
                               f"expected ({bsz},{EMB_DIM})")
        emb[:, pos * EMB_DIM:(pos + 1) * EMB_DIM] = x
    return emb


def _push_gradients_batch(
    grid: Grid, roles: Dict[str, int], split: str,
    idx_list: List[int],
    grad_chunks: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    role_order = ["active", "p6", "p1"]
    messages: List[Message] = []
    for i, role in enumerate(role_order):
        nid  = roles[role]
        grad = grad_chunks[i].contiguous().numpy().astype(np.float32)
        msg  = Message(
            content=RecordDict({
                "gradients": ArrayRecord({"local_gradients": Array(grad)}),
                "config":    _cfg(split=split, indices=idx_list),
            }),
            message_type="train.apply_gradients",
            dst_node_id=nid,
        )
        messages.append(msg)
    grid.push_messages(messages)


@torch.no_grad()
def _eval_split(
    grid: Grid, roles: Dict[str, int], head: nn.Module,
    y: torch.Tensor, split: str, return_probs: bool = False,
) -> Tuple[float, float, np.ndarray]:
    head.eval()
    n         = len(y)
    probs_all = np.zeros((n,), dtype=np.float32)
    for start in range(0, n, BATCH_SIZE):
        idx_list = list(range(start, min(start + BATCH_SIZE, n)))
        emb      = _get_embeddings_batch(grid, roles, split=split,
                                          idx_list=idx_list).to(DEVICE_TORCH)
        logits   = head(emb).detach().cpu().numpy().reshape(-1)
        probs    = _sigmoid_np(logits).astype(np.float32, copy=False)
        probs_all[start:start + len(idx_list)] = probs
    y_np  = y.numpy().reshape(-1)
    auroc = float(roc_auc_score(y_np, probs_all))
    ap    = float(average_precision_score(y_np, probs_all))
    return auroc, ap, probs_all