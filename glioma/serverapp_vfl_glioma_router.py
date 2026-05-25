# serverapp_vfl_glioma_router.py  (FIXED + THRESHOLD METRICS)
#
# Additions vs previous version:
#   - parse_threshold_txt(): reads the per-fold threshold .txt saved by the client
#   - write_fold_threshold_csv(): writes test_thresholds_fold{N}.csv with
#     one row per threshold rule (best_f1, best_youden, fixed_0.5)
#   - write_experiment_summary(): aggregates all folds into experiment_summary.csv
#     with mean±std for AUROC, PRAUC, and every threshold metric
#   - mkdir bug fixed: out_dir.mkdir called before out_dir.parent.mkdir

from __future__ import annotations

import csv
import os
import re
from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _arr_i64(x: np.ndarray) -> Array:
    return Array(np.asarray(x, dtype=np.int64))


def _arr_f32(x: np.ndarray) -> Array:
    return Array(np.asarray(x, dtype=np.float32))


def _nid(n) -> int:
    return int(n)


def _cfg_with_active(active_nid: int, extra: dict | None = None) -> ConfigRecord:
    d = {"active_node_id": str(int(active_nid))}
    if extra:
        d.update(extra)
    return ConfigRecord(d)


def load_folds(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    return list(d["folds"])


def _require_arrays(rep: Message, where: str, dst_nid: int, msg_type: str) -> None:
    if not rep.has_content():
        raise RuntimeError(
            f"[server] {where} reply has no content "
            f"(dst_node_id={dst_nid}, msg_type={msg_type})"
        )
    if "arrays" not in rep.content:
        keys = list(rep.content.keys())
        raise RuntimeError(
            f"[server] {where} reply missing 'arrays' "
            f"(dst_node_id={dst_nid}, msg_type={msg_type}, keys={keys})"
        )


# ---------------------------------------------------------------------------
# Client RPC helpers
# ---------------------------------------------------------------------------

def query_role(grid: Grid, nid: int, active_nid: int) -> int:
    msg_type = "query.get_role"
    msg = Message(
        content=RecordDict({"arrays": ArrayRecord({}), "config": _cfg_with_active(active_nid)}),
        message_type=msg_type,
        dst_node_id=_nid(nid),
    )
    rep = grid.send_and_receive([msg])[0]
    _require_arrays(rep, where="query_role", dst_nid=_nid(nid), msg_type=msg_type)
    return int(rep.content["arrays"]["role_code"].numpy()[0])


def broadcast_active_node_id(grid: Grid, node_ids: list[int], active_node_id: int) -> None:
    msgs = [
        Message(
            content=RecordDict({"arrays": ArrayRecord({}), "config": _cfg_with_active(active_node_id)}),
            message_type="query.set_active_node",
            dst_node_id=_nid(nid),
        )
        for nid in node_ids
    ]
    _ = grid.send_and_receive(msgs)


def request_embedding(
    grid: Grid,
    nid: int,
    batch_idx: np.ndarray,
    phase: str,
    active_nid: int,
) -> Tuple[np.ndarray, int]:
    msg_type = "query.generate_embeddings"
    msg = Message(
        content=RecordDict({
            "arrays": ArrayRecord({"batch_idx": _arr_i64(batch_idx)}),
            "config": _cfg_with_active(active_nid, {"phase": phase}),
        }),
        message_type=msg_type,
        dst_node_id=_nid(nid),
    )
    rep = grid.send_and_receive([msg])[0]
    _require_arrays(rep, where="request_embedding", dst_nid=_nid(nid), msg_type=msg_type)
    if "embedding" not in rep.content["arrays"]:
        raise RuntimeError(
            f"[server] request_embedding reply missing 'embedding' "
            f"(dst_node_id={_nid(nid)}, phase={phase}, "
            f"array_keys={list(rep.content['arrays'].keys())})"
        )
    emb = rep.content["arrays"]["embedding"].numpy().astype(np.float32)
    return emb, int(emb.nbytes)


def active_step(
    grid: Grid,
    active_nid: int,
    batch_idx: np.ndarray,
    z_passive: np.ndarray,
    phase: str,
) -> Tuple[float, Optional[np.ndarray]]:
    msg_type = "train.active_step"
    msg = Message(
        content=RecordDict({
            "arrays": ArrayRecord({
                "batch_idx": _arr_i64(batch_idx),
                "z_passive":  _arr_f32(z_passive),
            }),
            "config": _cfg_with_active(active_nid, {"phase": phase}),
        }),
        message_type=msg_type,
        dst_node_id=_nid(active_nid),
    )
    rep = grid.send_and_receive([msg])[0]
    _require_arrays(rep, where="active_step", dst_nid=_nid(active_nid), msg_type=msg_type)
    if "loss" not in rep.content["arrays"]:
        raise RuntimeError(
            f"[server] active_step reply missing 'loss' "
            f"(dst_node_id={_nid(active_nid)}, phase={phase}, "
            f"array_keys={list(rep.content['arrays'].keys())})"
        )
    loss = float(rep.content["arrays"]["loss"].numpy()[0])
    if phase == "train":
        if "grad_z_passive" not in rep.content["arrays"]:
            raise RuntimeError(
                f"[server] active_step(train) reply missing 'grad_z_passive' "
                f"(dst_node_id={_nid(active_nid)}, "
                f"array_keys={list(rep.content['arrays'].keys())})"
            )
        grad = rep.content["arrays"]["grad_z_passive"].numpy().astype(np.float32)
        return loss, grad
    return loss, None


def push_gradients(
    grid: Grid,
    passive_nid: int,
    batch_idx: np.ndarray,
    grad_z: np.ndarray,
    phase: str,
    active_nid: int,
) -> int:
    msg = Message(
        content=RecordDict({
            "arrays": ArrayRecord({
                "batch_idx":       _arr_i64(batch_idx),
                "local_gradients": _arr_f32(grad_z),
            }),
            "config": _cfg_with_active(active_nid, {"phase": phase}),
        }),
        message_type="train.apply_gradients",
        dst_node_id=_nid(passive_nid),
    )
    grid.push_messages([msg])
    return int(grad_z.nbytes)


def finalize_metrics(
    grid: Grid,
    active_nid: int,
    phase: str,
    save_dir: str,
    tag: str,
) -> Tuple[float, float, float]:
    msg_type = "train.finalize_metrics"
    msg = Message(
        content=RecordDict({
            "arrays": ArrayRecord({}),
            "config": _cfg_with_active(
                active_nid,
                {"phase": phase, "save_dir": str(save_dir), "tag": str(tag)},
            ),
        }),
        message_type=msg_type,
        dst_node_id=_nid(active_nid),
    )
    rep = grid.send_and_receive([msg])[0]
    _require_arrays(rep, where="finalize_metrics", dst_nid=_nid(active_nid), msg_type=msg_type)
    needed  = ["loss", "auroc", "prauc"]
    missing = [k for k in needed if k not in rep.content["arrays"]]
    if missing:
        raise RuntimeError(
            f"[server] finalize_metrics reply missing {missing} "
            f"(dst_node_id={_nid(active_nid)}, phase={phase}, "
            f"array_keys={list(rep.content['arrays'].keys())})"
        )
    loss  = float(rep.content["arrays"]["loss"].numpy()[0])
    auroc = float(rep.content["arrays"]["auroc"].numpy()[0])
    prauc = float(rep.content["arrays"]["prauc"].numpy()[0])
    return loss, auroc, prauc


def run_pretraining(
    grid: Grid,
    active_nid: int,
    passive_nid: int,
    sup_pre_epochs: int,
    ssl_pre_epochs: int,
    batch_size: int,
    ssl_noise_std: float,
) -> None:
    msgs = []

    if int(sup_pre_epochs) > 0:
        msgs.append(Message(
            content=RecordDict({
                "arrays": ArrayRecord({}),
                "config": _cfg_with_active(
                    active_nid,
                    {"epochs": int(sup_pre_epochs), "batch_size": int(batch_size)},
                ),
            }),
            message_type="train.pretrain_supervised",
            dst_node_id=_nid(active_nid),
        ))
    elif int(ssl_pre_epochs) > 0:
        msgs.append(Message(
            content=RecordDict({
                "arrays": ArrayRecord({}),
                "config": _cfg_with_active(
                    active_nid,
                    {
                        "epochs":     int(ssl_pre_epochs),
                        "batch_size": int(batch_size),
                        "noise_std":  float(ssl_noise_std),
                    },
                ),
            }),
            message_type="train.pretrain_ssl_active",
            dst_node_id=_nid(active_nid),
        ))

    if int(ssl_pre_epochs) > 0:
        msgs.append(Message(
            content=RecordDict({
                "arrays": ArrayRecord({}),
                "config": _cfg_with_active(
                    active_nid,
                    {
                        "epochs":     int(ssl_pre_epochs),
                        "batch_size": int(batch_size),
                        "noise_std":  float(ssl_noise_std),
                    },
                ),
            }),
            message_type="train.pretrain_ssl",
            dst_node_id=_nid(passive_nid),
        ))

    if msgs:
        _ = grid.send_and_receive(msgs)
    else:
        print("[server] run_pretraining: no pretraining requested (sup=0, ssl=0). Skipping.")


def _checkpoint_all_clients(grid: Grid, node_ids: list[int], active_nid: int) -> None:
    msgs = [
        Message(
            content=RecordDict({
                "arrays": ArrayRecord({}),
                "config": _cfg_with_active(active_nid),
            }),
            message_type="query.checkpoint_bottom",
            dst_node_id=_nid(nid),
        )
        for nid in node_ids
    ]
    _ = grid.send_and_receive(msgs)


def _restore_best_all_clients(grid: Grid, node_ids: list[int], active_nid: int) -> None:
    msgs = [
        Message(
            content=RecordDict({
                "arrays": ArrayRecord({}),
                "config": _cfg_with_active(active_nid),
            }),
            message_type="query.restore_best_bottom",
            dst_node_id=_nid(nid),
        )
        for nid in node_ids
    ]
    _ = grid.send_and_receive(msgs)


def _clean_suffix(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


# ---------------------------------------------------------------------------
# Threshold metrics helpers
# ---------------------------------------------------------------------------

THRESHOLD_METRIC_COLS = [
    "threshold", "acc", "precision", "recall", "specificity",
    "f1", "balanced_acc", "tp", "tn", "fp", "fn",
]
THRESHOLD_RULES = ["best_f1", "best_youden", "fixed_0.5"]


def parse_threshold_txt(txt_path: Path) -> Dict[str, Dict[str, float]]:
    """Parse the threshold .txt file written by the client's finalize_metrics.
    Returns dict: rule -> {metric: value, ...}
    """
    result: Dict[str, Dict[str, float]] = {}
    if not txt_path.exists():
        return result

    current_rule = None
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # section header like [best_f1]
            m = re.match(r"^\[(.+)\]$", line)
            if m:
                current_rule = m.group(1)
                result[current_rule] = {}
                continue
            # key=value
            if "=" in line and current_rule is not None:
                k, v = line.split("=", 1)
                try:
                    result[current_rule][k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return result


def write_fold_threshold_csv(
    csv_path: Path,
    fold: int,
    auroc: float,
    prauc: float,
    thr_data: Dict[str, Dict[str, float]],
) -> None:
    """Write per-fold threshold CSV with one row per threshold rule."""
    header = ["fold", "rule", "auroc", "prauc"] + THRESHOLD_METRIC_COLS
    rows = []
    for rule in THRESHOLD_RULES:
        metrics = thr_data.get(rule, {})
        row = [fold, rule, auroc, prauc] + [
            metrics.get(c, float("nan")) for c in THRESHOLD_METRIC_COLS
        ]
        rows.append(row)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_experiment_summary(
    summary_path: Path,
    fold_records: List[Dict],
) -> None:
    """Aggregate all fold records into a single experiment_summary.csv.

    Each fold_record is:
      {"fold": int, "auroc": float, "prauc": float,
       "best_f1": {metric: val}, "best_youden": {metric: val}, "fixed_0.5": {metric: val}}

    Writes:
      - one row per fold (fold=1..5)
      - a mean row  (fold="mean")
      - a std  row  (fold="std")
    """
    header = ["fold", "auroc", "prauc"]
    for rule in THRESHOLD_RULES:
        for col in THRESHOLD_METRIC_COLS:
            header.append(f"{rule}__{col}")

    rows = []
    for rec in fold_records:
        row = [rec["fold"], rec["auroc"], rec["prauc"]]
        for rule in THRESHOLD_RULES:
            m = rec.get(rule, {})
            for col in THRESHOLD_METRIC_COLS:
                row.append(m.get(col, float("nan")))
        rows.append(row)

    # compute mean and std over numeric columns (skip fold col)
    arr = np.array([[r[i] for i in range(1, len(header))] for r in rows], dtype=np.float64)
    mean_row = ["mean"] + list(np.nanmean(arr, axis=0))
    std_row  = ["std"]  + list(np.nanstd(arr,  axis=0))

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
        w.writerow(mean_row)
        w.writerow(std_row)


# ---------------------------------------------------------------------------
# ServerApp
# ---------------------------------------------------------------------------
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    here        = Path(__file__).resolve().parent
    DEFAULT_NPZ = here / "glioma_aligned_vfl_hfl_cv.npz"
    rc          = getattr(context, "run_config", {}) or {}

    npz:            str   = os.environ.get("NPZ",   str(rc.get("npz",   str(DEFAULT_NPZ))))
    fold:           int   = int(os.environ.get("FOLD",  rc.get("fold",  1)))
    seed:           int   = int(os.environ.get("SEED",  rc.get("seed",  0)))
    batch_size:     int   = int(os.environ.get("BATCH_SIZE", rc.get("batch_size", 64)))
    epochs:         int   = int(os.environ.get("EPOCHS", rc.get("epochs", 100)))

    sup_pre_epochs: int   = int(os.environ.get("SUP_PRE_EPOCHS", rc.get("sup_pre_epochs", 10)))
    ssl_pre_epochs: int   = int(os.environ.get("SSL_PRE_EPOCHS", rc.get("ssl_pre_epochs", 10)))
    ssl_noise_std:  float = float(os.environ.get("SSL_NOISE_STD", rc.get("ssl_noise_std", 0.1)))
    freeze_passive: int   = int(os.environ.get("FREEZE_PASSIVE", "1"))

    base_tag   = f"pre_sup{sup_pre_epochs}_ssl{ssl_pre_epochs}_vfl{epochs}_" + (
        "frozen" if freeze_passive == 1 else "finetune"
    )
    run_suffix = _clean_suffix(os.environ.get("RUN_SUFFIX", ""))
    tag        = base_tag + (f"__{run_suffix}" if run_suffix else "")

    out_root_default = here / "runs_immediate_active"
    out_root = Path(os.environ.get("OUT_DIR", str(rc.get("out_dir", str(out_root_default))))).resolve()
    out_dir  = out_root / tag

    # WSL/Python 3.10 bug: FileExistsError even with exist_ok=True on /mnt/c paths
    import errno as _errno
    for _d in (out_root, out_dir):
        try:
            _d.mkdir(parents=True, exist_ok=True)
        except OSError as _e:
            if _e.errno != _errno.EEXIST:
                raise

    metrics_csv      = out_dir / f"metrics_fold{fold}.csv"
    comm_csv         = out_dir / f"comm_fold{fold}.csv"
    thr_csv          = out_dir / f"test_thresholds_fold{fold}.csv"
    summary_csv      = out_root / f"experiment_summary__{run_suffix or tag}.csv"

    set_seed(seed)

    node_ids   = sorted([int(n) for n in list(grid.get_node_ids())])
    if len(node_ids) != 2:
        raise ValueError(f"Expected 2 clients, got {len(node_ids)}: {node_ids}")
    log(INFO, "Node order (sorted): %s", node_ids)

    active_nid  = int(node_ids[0])
    passive_nid = int(node_ids[1])

    log(INFO, "Broadcasting ACTIVE_NODE_ID=%s to all clients", active_nid)
    broadcast_active_node_id(grid, node_ids, active_node_id=active_nid)

    roles = {nid: query_role(grid, nid, active_nid=active_nid) for nid in node_ids}
    log(INFO, "ACTIVE=%s PASSIVE=%s roles=%s", active_nid, passive_nid, roles)
    log(INFO, "Config npz=%s fold=%d seed=%d bs=%d epochs=%d", npz, fold, seed, batch_size, epochs)
    log(INFO, "Run tag=%s out_dir=%s", tag, str(out_dir))
    log(INFO, "Pretrain: sup=%d ssl=%d | freeze_passive=%d | RUN_SUFFIX=%s",
        sup_pre_epochs, ssl_pre_epochs, freeze_passive, run_suffix)

    folds     = load_folds(npz)
    split_obj = folds[fold - 1]
    split     = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr_idx    = split["train"].astype(np.int64)
    va_idx    = split["val"].astype(np.int64)
    te_idx    = split["test"].astype(np.int64)

    rng             = np.random.default_rng(seed)
    steps_per_epoch = int(np.ceil(len(tr_idx) / batch_size))
    global_step     = 0
    best_val_auroc: float = float("nan")

    with open(metrics_csv, "w", newline="") as fmet, \
         open(comm_csv,    "w", newline="") as fcom:

        met_w = csv.writer(fmet)
        com_w = csv.writer(fcom)
        met_w.writerow(["epoch", "train_loss", "val_loss", "val_auroc", "val_prauc"])
        com_w.writerow(["phase", "epoch", "step_in_epoch", "global_step",
                        "bytes_up_x2", "bytes_down_x2", "total_bytes"])

        # ----------------------------------------------------------------
        # Pretraining barrier
        # ----------------------------------------------------------------
        log(INFO, "Starting pretraining barrier...")
        run_pretraining(
            grid=grid,
            active_nid=active_nid,
            passive_nid=passive_nid,
            sup_pre_epochs=sup_pre_epochs,
            ssl_pre_epochs=ssl_pre_epochs,
            batch_size=batch_size,
            ssl_noise_std=ssl_noise_std,
        )
        log(INFO, "Pretraining complete. Starting downstream VFL training.")

        # ----------------------------------------------------------------
        # Training loop
        # ----------------------------------------------------------------
        for ep in range(1, epochs + 1):
            rng.shuffle(tr_idx)
            tr_loss_sum = 0.0
            tr_n        = 0

            for s in range(steps_per_epoch):
                global_step += 1
                b = tr_idx[s * batch_size : (s + 1) * batch_size]
                if b.size == 0:
                    continue

                z2, up_bytes  = request_embedding(grid, passive_nid, b, phase="train", active_nid=active_nid)
                loss, grad_z2 = active_step(grid, active_nid, b, z2, phase="train")

                tr_loss_sum += float(loss) * int(b.size)
                tr_n        += int(b.size)

                down_bytes = 0
                if freeze_passive == 0 and grad_z2 is not None:
                    down_bytes = push_gradients(grid, passive_nid, b, grad_z2,
                                                phase="train", active_nid=active_nid)

                total_bytes = int(up_bytes) + int(down_bytes)
                com_w.writerow(["train", ep, s + 1, global_step,
                                up_bytes, down_bytes, total_bytes])

            # ----------------------------------------------------------------
            # Validation
            # ----------------------------------------------------------------
            n_val_steps = int(np.ceil(len(va_idx) / batch_size))
            for s in range(n_val_steps):
                b = va_idx[s * batch_size : (s + 1) * batch_size]
                if b.size == 0:
                    continue
                z2, up_bytes = request_embedding(grid, passive_nid, b, phase="val", active_nid=active_nid)
                _loss, _     = active_step(grid, active_nid, b, z2, phase="val")
                com_w.writerow(["val", ep, s + 1, global_step, up_bytes, 0, up_bytes])

            val_loss, val_auroc, val_prauc = finalize_metrics(
                grid, active_nid, phase="val", save_dir=str(out_dir), tag=tag
            )
            train_loss = tr_loss_sum / max(tr_n, 1)

            met_w.writerow([ep, train_loss, val_loss, val_auroc, val_prauc])
            fmet.flush(); fcom.flush()
            log(INFO, "[VAL] ep=%d train_loss=%.4f val_loss=%.4f AUROC=%.4f PR-AUC=%.4f",
                ep, train_loss, val_loss, val_auroc, val_prauc)

            improved = (
                not np.isnan(val_auroc)
                and (np.isnan(best_val_auroc) or val_auroc > best_val_auroc)
            )
            if improved:
                best_val_auroc = val_auroc
                _checkpoint_all_clients(grid, node_ids, active_nid)
                log(INFO, "[CKPT] Saved all client weights at epoch=%d (AUROC=%.4f)",
                    ep, val_auroc)

        # ----------------------------------------------------------------
        # Restore best weights before test
        # ----------------------------------------------------------------
        log(INFO, "[CKPT] Restoring best-val weights in all clients for test evaluation...")
        _restore_best_all_clients(grid, node_ids, active_nid)

        # ----------------------------------------------------------------
        # Test pass
        # ----------------------------------------------------------------
        n_test_steps = int(np.ceil(len(te_idx) / batch_size))
        for s in range(n_test_steps):
            b = te_idx[s * batch_size : (s + 1) * batch_size]
            if b.size == 0:
                continue
            z2, up_bytes = request_embedding(grid, passive_nid, b, phase="test", active_nid=active_nid)
            _loss, _     = active_step(grid, active_nid, b, z2, phase="test")
            com_w.writerow(["test", epochs, s + 1, global_step, up_bytes, 0, up_bytes])

        test_loss, test_auroc, test_prauc = finalize_metrics(
            grid, active_nid, phase="test", save_dir=str(out_dir), tag=tag
        )
        log(INFO, "[TEST] fold=%d loss=%.4f AUROC=%.4f PR-AUC=%.4f",
            fold, test_loss, test_auroc, test_prauc)

    # ----------------------------------------------------------------
    # Threshold CSV for this fold
    # ----------------------------------------------------------------
    thr_txt = out_dir / f"thresholds_{tag}_test_fold{fold}.txt"
    thr_data = parse_threshold_txt(thr_txt)
    write_fold_threshold_csv(thr_csv, fold, test_auroc, test_prauc, thr_data)
    log(INFO, "[THR] Written test_thresholds_fold%d.csv", fold)

    # ----------------------------------------------------------------
    # Aggregate experiment_summary.csv (append/rebuild across folds)
    # ----------------------------------------------------------------
    # Read existing fold records from summary if present, then upsert this fold
    existing: List[Dict] = []
    if summary_csv.exists():
        with open(summary_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fold_val = row.get("fold", "")
                    if fold_val in ("mean", "std"):
                        continue
                    existing.append(row)
                except Exception:
                    pass

    # Build record for this fold
    new_record: Dict = {
        "fold":  fold,
        "auroc": test_auroc,
        "prauc": test_prauc,
    }
    for rule in THRESHOLD_RULES:
        new_record[rule] = thr_data.get(rule, {})

    # Upsert: replace existing record for this fold if present
    updated = [r for r in existing if str(r.get("fold", "")) != str(fold)]

    # Re-parse existing rows back into the record format
    fold_records: List[Dict] = []
    header_cols = ["auroc", "prauc"] + [
        f"{rule}__{col}" for rule in THRESHOLD_RULES for col in THRESHOLD_METRIC_COLS
    ]
    for r in updated:
        rec: Dict = {
            "fold":  r.get("fold", "?"),
            "auroc": float(r.get("auroc", float("nan"))),
            "prauc": float(r.get("prauc", float("nan"))),
        }
        for rule in THRESHOLD_RULES:
            rec[rule] = {}
            for col in THRESHOLD_METRIC_COLS:
                key = f"{rule}__{col}"
                try:
                    rec[rule][col] = float(r.get(key, float("nan")))
                except (ValueError, TypeError):
                    rec[rule][col] = float("nan")
        fold_records.append(rec)

    fold_records.append(new_record)
    # sort by fold number
    fold_records.sort(key=lambda x: int(x["fold"]) if str(x["fold"]).isdigit() else 99)

    write_experiment_summary(summary_csv, fold_records)
    log(INFO, "[SUMMARY] experiment_summary updated: %s (folds so far: %d)",
        summary_csv.name, len(fold_records))