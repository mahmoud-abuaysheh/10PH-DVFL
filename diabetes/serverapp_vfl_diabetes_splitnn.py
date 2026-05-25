# serverapp_vfl_diabetes_splitnn.py
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

# ✅ MUST match client @app.query(...) names
MSG_EMB  = "query.generate_embeddings"
MSG_Y    = "query.get_labels"
MSG_CKPT = "query.checkpoint_bottom"
MSG_RST  = "query.restore_best_bottom"

# ✅ MUST match client @app.train(...) names
MSG_BWD = "train.backward"


# ---------------- utils ----------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _arr_i64(x: np.ndarray) -> Array:
    return Array(np.asarray(x, dtype=np.int64))


def _arr_f32(x: np.ndarray) -> Array:
    return Array(np.asarray(x, dtype=np.float32))


def _cfg(d: Dict[str, object] | None = None) -> ConfigRecord:
    return ConfigRecord(d or {})


def _grid_request(grid: Grid, msg: Message, timeout: int = 300) -> Message:
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


# ---------------- model ----------------
class TopLinear(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 (active) + 8 (passive) = 16
        self.fc = nn.Linear(16, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(1)


def _checkpoint_clients(grid: Grid, active_id: int, passive_id: int, fold: int, out_dir: str) -> None:
    """Save bottom models on both clients at the current (best) round."""
    for node_id, view in [(active_id, 0), (passive_id, 1)]:
        _grid_request(grid, Message(
            content=RecordDict({"config": ConfigRecord({"view": view, "fold": fold, "out_dir": out_dir})}),
            dst_node_id=node_id,
            message_type=MSG_CKPT,
        ))


def _restore_best_clients(grid: Grid, active_id: int, passive_id: int, fold: int, out_dir: str) -> None:
    """Restore bottom models on both clients from best checkpoint."""
    for node_id, view in [(active_id, 0), (passive_id, 1)]:
        _grid_request(grid, Message(
            content=RecordDict({"config": ConfigRecord({"view": view, "fold": fold, "out_dir": out_dir})}),
            dst_node_id=node_id,
            message_type=MSG_RST,
        ))


def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    y = d["y"].astype(np.int64)
    folds = list(d["folds"])
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return y, folds, meta


def _pick_threshold_maxf1(y_true: np.ndarray, p: np.ndarray) -> float:
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


def server_main(grid: Grid, context: Context) -> None:
    run = context.run_config
    seed = int(run.get("seed", 42))
    set_seed(seed)

    device = torch.device(str(run.get("device", "cpu")))
    DEFAULT_NPZ = Path(__file__).resolve().parent / "diabetes_vfl_cv.npz"
    npz = str(run.get("npz", str(DEFAULT_NPZ)))
    fold = int(os.environ.get("FOLD", run.get("fold", 1)))

    out_dir = str(run.get("out_dir", "./runs_vfl_diabetes_splitnn"))
    os.makedirs(out_dir, exist_ok=True)

    rounds = int(run.get("rounds", 100))
    batch = int(run.get("batch", ))
    lr_top = float(run.get("lr_top", 1e-3))
    patience = int(run.get("patience", 15))

    plateau_patience = int(run.get("plateau_patience", 5))
    plateau_factor = float(run.get("plateau_factor", 0.5))
    min_lr = float(run.get("min_lr", 1e-6))

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

    pos = float(y[tr].sum())
    neg = float(len(tr) - y[tr].sum())
    pw = neg / max(pos, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    top = TopLinear().to(device)
    opt = torch.optim.Adam(top.parameters(), lr=lr_top)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=plateau_factor, patience=plateau_patience, min_lr=min_lr
    )

    best_val_auroc = -1.0
    best_round = -1
    no_improve = 0

    best_path = os.path.join(out_dir, f"splitnn_best_fold{fold}.pt")
    hist_path = os.path.join(out_dir, f"splitnn_fold{fold}_history.csv")

    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["round", "train_loss", "val_auroc", "val_prauc", "lr"])

    n = len(tr)
    steps = math.ceil(n / batch)

    for rnd in range(1, rounds + 1):
        top.train()
        perm = np.random.permutation(tr)
        train_loss_acc = 0.0

        for s in range(steps):
            bidx = perm[s * batch: min(n, (s + 1) * batch)].astype(np.int64)

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

            z = torch.from_numpy(np.concatenate([ex, ep], axis=1)).to(device).requires_grad_(True)
            yy = torch.from_numpy(yb).float().to(device)

            opt.zero_grad(set_to_none=True)
            logits = top(z)
            loss = criterion(logits, yy)
            loss.backward()

            dz = z.grad.detach().cpu().numpy().astype(np.float32)

            # each bottom returns 8-dim embedding -> split 16-dim gradient
            dx = dz[:, :8].copy()
            dp = dz[:, 8:].copy()

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

        y_va, p_va = _eval_probs(grid, top, device, va, active_id, passive_id)
        val_auroc = float(roc_auc_score(y_va, p_va))
        val_prauc = float(average_precision_score(y_va, p_va))
        sch.step(val_auroc)
        lr_now = float(opt.param_groups[0]["lr"])

        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([rnd, train_loss_acc, val_auroc, val_prauc, lr_now])

        if rnd == 1 or rnd % 10 == 0 or rnd == rounds:
            print(
                f"[fold {fold}] round {rnd:03d}/{rounds} train_loss={train_loss_acc:.4f} "
                f"val_AUROC={val_auroc:.4f} val_PR-AUC={val_prauc:.4f} "
                f"best={best_val_auroc:.4f}@{best_round} lr={lr_now:.2e}"
            )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_round = rnd
            torch.save({"top": top.state_dict()}, best_path)
            _checkpoint_clients(grid, active_id, passive_id, fold, out_dir)
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"[fold {fold}] early stop at round {rnd} (no val AUROC improvement for {patience} rounds)")
                break

    top.load_state_dict(torch.load(best_path, map_location=device)["top"])
    _restore_best_clients(grid, active_id, passive_id, fold, out_dir)
    y_va, p_va = _eval_probs(grid, top, device, va, active_id, passive_id)
    y_te, p_te = _eval_probs(grid, top, device, te, active_id, passive_id)

    thr = _pick_threshold_maxf1(y_va, p_va)
    tm_val = _threshold_metrics(y_va, p_va, thr)
    tm_te = _threshold_metrics(y_te, p_te, thr)

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


app = ServerApp()

@app.main()
def main(grid, context) -> None:
    server_main(grid, context)