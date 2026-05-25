# serverapp_vfl_glioma_splitnn.py  — matches DIA SplitNN pattern exactly
#
# Key fixes vs previous glioma server:
#   FIX-B: opt.step() (head) happens AFTER gradients sent to clients and
#           clients have stepped — same order as DIA.
#   FIX-C: uses send_and_receive (barrier) for gradient push, not push_messages.
#           Ensures bottom weights are updated before next forward pass.
#   FIX-D: TopLinear(16→1) matches DIA — single linear layer, not 3-layer MLP.
#           Prevents head overfitting on top of noisy embeddings.

from __future__ import annotations

import copy
import csv
import os
from logging import INFO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TopMLP(nn.Module):
    """Matches decoupled Glioma head exactly: in_dim → 16 → 8 → 1."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, 8),      nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(1)


def _arr_i64(x: np.ndarray) -> Array:
    return Array(np.asarray(x, dtype=np.int64))


def _arr_f32(x: np.ndarray) -> Array:
    return Array(np.asarray(x, dtype=np.float32))


def load_npz(npz_path: str):
    d     = np.load(npz_path, allow_pickle=True)
    y     = d["y"].astype(np.int64)
    folds = list(d["folds"])
    return y, folds


def safe_auroc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else float("nan")
    except Exception:
        return float("nan")


def safe_prauc(y_true, y_score):
    try:
        return float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else float("nan")
    except Exception:
        return float("nan")


def confusion_counts(y_true, y_pred):
    y_true = y_true.astype(np.int64).reshape(-1)
    y_pred = y_pred.astype(np.int64).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tn, fp, fn, tp


def metrics_from_counts(tn, fp, fn, tp):
    eps = 1e-12
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1   = (2 * prec * sens) / max(prec + sens, eps)
    return acc, sens, spec, prec, f1


def find_best_thresholds(y_true, y_prob):
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


def write_threshold_report(csv_path, y_true, y_prob):
    thr_map = find_best_thresholds(y_true, y_prob)
    rows = []
    for rule, thr in thr_map.items():
        y_pred = (y_prob >= thr).astype(np.int64)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        acc, sens, spec, prec, f1 = metrics_from_counts(tn, fp, fn, tp)
        rows.append([rule, thr, acc, sens, spec, prec, f1, tn, fp, fn, tp])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule","thr","acc","sens","spec","prec","f1","tn","fp","fn","tp"])
        w.writerows(rows)


def write_threshold_curve(csv_path, y_true, y_prob, max_points=2000):
    y_true = y_true.astype(np.int64).reshape(-1)
    y_prob = y_prob.astype(np.float64).reshape(-1)
    thr    = np.unique(y_prob)
    if thr.size == 0:
        thr = np.array([0.5])
    if thr.size > max_points:
        thr = thr[np.linspace(0, thr.size - 1, max_points).astype(int)]
    rows = []
    for t in thr:
        y_pred = (t >= t)  # placeholder
        y_pred = (y_prob >= t).astype(np.int64)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        acc, sens, spec, prec, f1 = metrics_from_counts(tn, fp, fn, tp)
        fpr    = fp / max(fp + tn, 1)
        youden = sens - fpr
        rows.append([float(t), acc, sens, spec, prec, f1, youden, tn, fp, fn, tp])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["thr","acc","sens","spec","prec","f1","youden","tn","fp","fn","tp"])
        w.writerows(rows)


def save_test_plots(out_dir, fold, y_true, y_prob):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = y_true.astype(np.int64).reshape(-1)
    y_prob = y_prob.astype(np.float64).reshape(-1)

    if len(np.unique(y_true)) >= 2:
        fpr_c, tpr_c, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
    else:
        fpr_c, tpr_c, auroc = np.array([0., 1.]), np.array([0., 1.]), float("nan")
    plt.figure()
    plt.plot(fpr_c, tpr_c, label=f"ROC (AUROC={auroc:.4f})")
    plt.plot([0, 1], [0, 1], label="chance")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"test_roc_fold{fold}.png", dpi=200); plt.close()

    if len(np.unique(y_true)) >= 2:
        prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
    else:
        prec_c, rec_c, ap = np.array([1., 0.]), np.array([0., 1.]), float("nan")
    plt.figure()
    plt.plot(rec_c, prec_c, label=f"PR (AP={ap:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"test_pr_fold{fold}.png", dpi=200); plt.close()

    curve_csv = out_dir / f"test_threshold_curve_fold{fold}.csv"
    write_threshold_curve(curve_csv, y_true, y_prob)

    import pandas as pd
    df = pd.read_csv(curve_csv)
    for col, fname in [("f1", "f1"), ("youden", "youden")]:
        plt.figure()
        plt.plot(df["thr"], df[col], label=col)
        plt.xlabel("threshold"); plt.ylabel(col); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"test_{fname}_vs_thr_fold{fold}.png", dpi=200); plt.close()
    plt.figure()
    for col in ["acc", "sens", "spec"]:
        plt.plot(df["thr"], df[col], label=col)
    plt.xlabel("threshold"); plt.ylabel("metric"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"test_acc_sens_spec_vs_thr_fold{fold}.png", dpi=200); plt.close()


# ---------------------------------------------------------------------------
# VFL communication helpers
# ---------------------------------------------------------------------------

def _request_embeddings(
    grid: Grid, node_ids: list, batch_idx: np.ndarray, out_dim: int
) -> Tuple[torch.Tensor, List[int]]:
    """Send embedding request to both nodes. Returns concatenated z + byte counts."""
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

    z = torch.cat(emb_list, dim=1).requires_grad_()
    return z, bytes_up


def _push_embedding_grads_barrier(
    grid: Grid, node_ids: list,
    grads_per_pos: List[torch.Tensor],
    batch_idx: np.ndarray,
) -> List[int]:
    """FIX-C: send_and_receive (barrier) — wait for both clients to finish stepping."""
    msgs = []
    bytes_down = []
    for view, nid in enumerate(node_ids):
        grad_np = grads_per_pos[view].detach().cpu().numpy().astype(np.float32)
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
    grid.send_and_receive(msgs)   # FIX-C: barrier, not fire-and-forget
    return bytes_down


def _checkpoint_clients(grid: Grid, node_ids: list) -> None:
    msgs = [
        Message(
            content=RecordDict({"arrays": ArrayRecord({}), "config": ConfigRecord({})}),
            message_type="query.checkpoint_bottom",
            dst_node_id=nid,
        )
        for nid in node_ids
    ]
    grid.send_and_receive(msgs)


def _restore_best_clients(grid: Grid, node_ids: list) -> None:
    msgs = [
        Message(
            content=RecordDict({"arrays": ArrayRecord({}), "config": ConfigRecord({})}),
            message_type="query.restore_best_bottom",
            dst_node_id=nid,
        )
        for nid in node_ids
    ]
    grid.send_and_receive(msgs)


# ---------------------------------------------------------------------------
# ServerApp
# ---------------------------------------------------------------------------
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    DEFAULT_NPZ = Path(__file__).resolve().parent / "glioma_aligned_vfl_hfl_cv.npz"
    rc          = getattr(context, "run_config", {}) or {}

    npz:        str   = str(rc.get("npz", str(DEFAULT_NPZ)))
    fold:       int   = int(os.environ.get("FOLD", rc.get("fold", 1)))
    seed:       int   = int(rc.get("seed", 42))
    device:     str   = str(rc.get("device", "cpu"))
    lr_head:    float = float(rc.get("lr_head", 1e-3))
    lr_bottom:  float = float(rc.get("lr_bottom", 1e-3))
    batch_size: int   = int(rc.get("batch_size", 64))
    epochs:     int   = int(os.environ.get("EPOCHS", rc.get("epochs", 100)))
    out_dim:    int   = int(rc.get("out_feature_dim", 16))  # 16 per silo → 32 concat, matches decoupled
    patience:   int   = int(rc.get("patience", 0))          # 0 = no early stop

    out_dir = Path(rc.get("out_dir",
                          str(Path(__file__).resolve().parent / "runs_splitnn_vfl")))
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
    log(INFO, "Node IDs: %s → view=0(X1),  %s → view=1(X2)", node_ids[0], node_ids[1])

    y_all, folds_data = load_npz(npz)
    split_obj = folds_data[fold - 1]
    split     = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr_idx = split["train"].astype(np.int64)
    va_idx = split["val"].astype(np.int64)
    te_idx = split["test"].astype(np.int64)

    # Head matches decoupled exactly: TopMLP(32 → 16 → 8 → 1)
    head = TopMLP(in_dim=out_dim * 2).to(dev)
    opt  = torch.optim.Adam(head.parameters(), lr=lr_head)

    y_tr       = torch.from_numpy(y_all[tr_idx]).float()
    pos        = float(y_tr.sum().item())
    neg        = float(y_tr.numel() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)]).float().to(dev)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    steps_per_epoch = int(np.ceil(len(tr_idx) / batch_size))
    total_steps     = epochs * steps_per_epoch
    log(INFO, "fold=%d | epochs=%d | steps/epoch=%d | total=%d | out_dim_per_silo=%d",
        fold, epochs, steps_per_epoch, total_steps, out_dim)

    totals = {
        "train": {"up": [0, 0], "down": [0, 0], "total": 0},
        "val":   {"up": [0, 0], "down": [0, 0], "total": 0},
        "test":  {"up": [0, 0], "down": [0, 0], "total": 0},
    }
    best_val:        dict          = {"epoch": 0, "val_loss": float("nan"),
                                      "val_auroc": float("nan"), "val_prauc": float("nan")}
    best_head_state: Optional[dict] = None
    no_improve      = 0
    rng             = np.random.default_rng(seed)
    global_step     = 0
    test_loss = test_auroc = test_prauc = float("nan")

    with open(metrics_csv, "w", newline="") as fmet, \
         open(comm_csv,    "w", newline="") as fcom:

        met_w = csv.writer(fmet)
        com_w = csv.writer(fcom)
        met_w.writerow(["epoch","train_loss","val_loss","val_auroc","val_prauc","lr_head"])
        com_w.writerow(["phase","epoch","step","global_step",
                        "bytes_up_x1","bytes_up_x2","bytes_down_x1","bytes_down_x2","total"])

        # ----------------------------------------------------------------
        # Training loop
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

                # --- forward ---
                z, up_bytes = _request_embeddings(grid, node_ids, b, out_dim)
                z = z.to(dev)

                logits = head(z)
                yb     = torch.from_numpy(y_all[b]).float().to(dev)
                loss   = criterion(logits, yb)

                tr_loss_sum += float(loss.item()) * int(b.size)
                tr_loss_n   += int(b.size)

                # --- backward through head ---
                opt.zero_grad(set_to_none=True)
                loss.backward()

                grads         = z.grad.detach()
                grads_per_pos = list(grads.split([out_dim, out_dim], dim=1))

                # FIX-C: barrier — wait for both clients to step their bottoms
                down_bytes = _push_embedding_grads_barrier(
                    grid, node_ids, grads_per_pos, b
                )

                # FIX-B: head steps AFTER clients have stepped
                opt.step()

                bu0, bu1 = int(up_bytes[0]),   int(up_bytes[1])
                bd0, bd1 = int(down_bytes[0]), int(down_bytes[1])
                total    = bu0 + bu1 + bd0 + bd1
                totals["train"]["up"][0]   += bu0; totals["train"]["up"][1]   += bu1
                totals["train"]["down"][0] += bd0; totals["train"]["down"][1] += bd1
                totals["train"]["total"]   += total
                com_w.writerow(["train", ep, s+1, global_step, bu0, bu1, bd0, bd1, total])

                if global_step % 50 == 0:
                    with torch.no_grad():
                        p   = torch.sigmoid(logits).detach().cpu().numpy()
                        acc = float(((p >= 0.5) == (yb.cpu().numpy() >= 0.5)).mean() * 100)
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
                    totals["val"]["up"][0] += bu0; totals["val"]["up"][1] += bu1
                    totals["val"]["total"] += bu0 + bu1
                    com_w.writerow(["val", ep, s+1, global_step, bu0, bu1, 0, 0, bu0+bu1])

                val_logits = torch.cat(val_logits_all).numpy()
                val_prob   = 1.0 / (1.0 + np.exp(-val_logits))
                val_y      = torch.cat(val_y_all).numpy().astype(np.float32)
                val_loss   = val_loss_sum / max(val_n, 1)
                val_auroc  = safe_auroc(val_y, val_prob)
                val_prauc  = safe_prauc(val_y, val_prob)

            train_loss = tr_loss_sum / max(tr_loss_n, 1)
            met_w.writerow([ep, train_loss, val_loss, val_auroc, val_prauc,
                            float(opt.param_groups[0]["lr"])])
            fmet.flush(); fcom.flush()
            log(INFO, "[VAL] ep=%d loss=%.4f AUROC=%.4f PR-AUC=%.4f",
                ep, val_loss, val_auroc, val_prauc)

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
                log(INFO, "[CKPT] ep=%d AUROC=%.4f — head+bottoms saved", ep, val_auroc)
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    log(INFO, "[EARLY STOP] ep=%d no improvement for %d epochs", ep, patience)
                    break

        # ----------------------------------------------------------------
        # Restore best, then test
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
                totals["test"]["up"][0] += bu0; totals["test"]["up"][1] += bu1
                totals["test"]["total"] += bu0 + bu1
                com_w.writerow(["test", epochs, s+1, global_step, bu0, bu1, 0, 0, bu0+bu1])

            test_logits = torch.cat(test_logits_all).numpy()
            test_prob   = 1.0 / (1.0 + np.exp(-test_logits))
            test_y      = torch.cat(test_y_all).numpy().astype(np.float32)
            test_loss   = test_loss_sum / max(test_n, 1)
            test_auroc  = safe_auroc(test_y, test_prob)
            test_prauc  = safe_prauc(test_y, test_prob)

        log(INFO, "[TEST] loss=%.4f AUROC=%.4f PR-AUC=%.4f", test_loss, test_auroc, test_prauc)
        write_threshold_report(test_thresholds_csv, test_y, test_prob)
        save_test_plots(out_dir, fold, test_y, test_prob)

    # ----------------------------------------------------------------
    # Communication summaries
    # ----------------------------------------------------------------
    train_total = totals["train"]["total"]
    all_total   = train_total + totals["val"]["total"] + totals["test"]["total"]

    with open(comm_summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope","bytes_up_x1","bytes_up_x2","bytes_down_x1","bytes_down_x2","total_bytes"])
        w.writerow(["train_only",
                    totals["train"]["up"][0], totals["train"]["up"][1],
                    totals["train"]["down"][0], totals["train"]["down"][1], train_total])
        w.writerow(["train_val_test",
                    totals["train"]["up"][0]+totals["val"]["up"][0]+totals["test"]["up"][0],
                    totals["train"]["up"][1]+totals["val"]["up"][1]+totals["test"]["up"][1],
                    totals["train"]["down"][0]+totals["val"]["down"][0]+totals["test"]["down"][0],
                    totals["train"]["down"][1]+totals["val"]["down"][1]+totals["test"]["down"][1],
                    all_total])

    with open(experiment_summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold","best_val_epoch","best_val_loss","best_val_auroc","best_val_prauc",
                    "test_loss","test_auroc","test_prauc",
                    "train_only_bytes_up_x1","train_only_bytes_up_x2",
                    "train_only_bytes_down_x1","train_only_bytes_down_x2","train_only_total_bytes",
                    "train_val_test_bytes_up_x1","train_val_test_bytes_up_x2",
                    "train_val_test_bytes_down_x1","train_val_test_bytes_down_x2",
                    "train_val_test_total_bytes"])
        w.writerow([fold,
                    best_val["epoch"], best_val["val_loss"],
                    best_val["val_auroc"], best_val["val_prauc"],
                    test_loss, test_auroc, test_prauc,
                    totals["train"]["up"][0], totals["train"]["up"][1],
                    totals["train"]["down"][0], totals["train"]["down"][1], train_total,
                    totals["train"]["up"][0]+totals["val"]["up"][0]+totals["test"]["up"][0],
                    totals["train"]["up"][1]+totals["val"]["up"][1]+totals["test"]["up"][1],
                    totals["train"]["down"][0]+totals["val"]["down"][0]+totals["test"]["down"][0],
                    totals["train"]["down"][1]+totals["val"]["down"][1]+totals["test"]["down"][1],
                    all_total])

    log(INFO, "Done. Outputs: %s", str(out_dir))