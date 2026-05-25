# serverapp_vfl_diabetes_decoupled.py
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

INFO = 20

# MUST match client query handler names
MSG_EMB = "query.generate_embeddings"
MSG_Y = "query.get_labels"
MSG_HFL_FIT = "query.hfl_fit"


# ---------------- utils ----------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _arr_i64(x: np.ndarray) -> Array:
    return Array(np.asarray(x, dtype=np.int64))


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


def _rcfg(run, key: str, default):
    """Flower converts underscore keys to hyphens in run_config. Check both."""
    hyphen_key = key.replace("_", "-")
    if key in run:
        return run[key]
    if hyphen_key in run:
        return run[hyphen_key]
    return default


def _sd_to_arrays(sd: Dict[str, torch.Tensor]) -> Dict[str, Array]:
    return {k: Array(v.detach().cpu().numpy().astype(np.float32)) for k, v in sd.items()}


def _arrays_to_sd(arrs: Dict[str, Array], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(arr.numpy()).to(device) for k, arr in arrs.items()}


# ---------------- model ----------------
class TopLinear(nn.Module):
    """Top head for decoupled VFL. Input dim must match concat(emb_active, emb_passive)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(1)


class TeacherHead(nn.Module):
    """Teacher head for ACTIVE-only embeddings (emb_dim -> 1)."""

    def __init__(self, in_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(1)


def _load_teacher_head(teacher_ckpt_path: str, device: torch.device, emb_dim: int) -> TeacherHead:
    if not os.path.exists(teacher_ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt_path}")

    state = torch.load(teacher_ckpt_path, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError(f"Teacher ckpt is not a dict: type={type(state)}")

    if "head_state" not in state or not isinstance(state["head_state"], dict):
        raise RuntimeError(f"Teacher ckpt missing 'head_state' dict keys={list(state.keys())}")

    # raw keys {"weight","bias"} -> TeacherHead expects {"fc.weight","fc.bias"}
    raw_sd = state["head_state"]
    prefixed_sd = {f"fc.{k}": v for k, v in raw_sd.items()}

    head = TeacherHead(in_dim=emb_dim).to(device)
    head.load_state_dict(prefixed_sd, strict=True)

    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    print(f"[teacher] loaded teacher head from: {teacher_ckpt_path}")
    return head


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
    emb_dim: int,
    timeout: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
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
            raise RuntimeError(f"Embedding dim mismatch: ex={ex.shape}, ep={ep.shape}, expected emb_dim={emb_dim}")

        rep_y = _grid_request(
            grid,
            Message(
                content=RecordDict(
                    {"config": _cfg({"role": "active"}), "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)})}
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


def server_main(grid: Grid, context: Context) -> None:
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

    # ---- KD settings (ONLY used if mode == "kd") ----
    kd_alpha = float(_rcfg(run, "kd_alpha", 0.5))
    kd_T = float(_rcfg(run, "kd_T", 2.0))
    teacher_ckpt_path = _rcfg(run, "teacher_ckpt_path", None)
    teacher_ckpt_dir = str(_rcfg(run, "teacher_ckpt_dir", "./runs_active_sup_diabetes"))

    teacher_head: Optional[TeacherHead] = None
    if mode == "kd" and kd_alpha > 0.0:
        if teacher_ckpt_path is None:
            teacher_ckpt_path = os.path.join(teacher_ckpt_dir, f"pretrained_active_sup_teacher_fold{fold}.pt")
        else:
            teacher_ckpt_path = str(teacher_ckpt_path)
        teacher_head = _load_teacher_head(teacher_ckpt_path, device=device, emb_dim=emb_dim)
        print(f"[KD] enabled: kd_alpha={kd_alpha} kd_T={kd_T}")
    else:
        kd_alpha = 0.0
        teacher_ckpt_path = ""
        print("[KD] disabled (mode != kd)")

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

    criterion_sup = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    criterion_kd = nn.BCEWithLogitsLoss()  # for soft targets

    top = TopLinear(in_dim=2 * emb_dim).to(device)
    opt = torch.optim.Adam(top.parameters(), lr=lr_top)

    best_val_auroc = -1.0
    best_round = -1
    no_improve = 0

    best_path = os.path.join(out_dir, f"decoupled_best_fold{fold}.pt")
    hist_path = os.path.join(out_dir, f"decoupled_fold{fold}_history.csv")

    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["round", "train_loss", "train_sup_loss", "train_kd_loss", "val_auroc", "val_prauc", "lr"]
        )

    n = len(tr)
    steps = math.ceil(n / batch)

    for rnd in range(1, rounds + 1):
        top.train()
        perm = np.random.permutation(tr)

        train_loss_acc = 0.0
        sup_loss_acc = 0.0
        kd_loss_acc = 0.0

        for s in range(steps):
            bidx = perm[s * batch : min(n, (s + 1) * batch)].astype(np.int64)

            rep_x = _grid_request(
                grid,
                Message(
                    content=RecordDict(
                        {"config": _cfg({"view": 0, "role": "active"}), "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)})}
                    ),
                    dst_node_id=active_id,
                    message_type=MSG_EMB,
                ),
            )

            rep_p = _grid_request(
                grid,
                Message(
                    content=RecordDict(
                        {"config": _cfg({"view": 1, "role": "passive"}), "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)})}
                    ),
                    dst_node_id=passive_id,
                    message_type=MSG_EMB,
                ),
            )

            ex = rep_x.content["arrays"]["embedding"].numpy().astype(np.float32)
            ep = rep_p.content["arrays"]["embedding"].numpy().astype(np.float32)
            if ex.shape[1] != emb_dim or ep.shape[1] != emb_dim:
                raise RuntimeError(f"Embedding dim mismatch in train: ex={ex.shape}, ep={ep.shape}, expected emb_dim={emb_dim}")

            rep_y = _grid_request(
                grid,
                Message(
                    content=RecordDict({"config": _cfg({"role": "active"}), "arrays": ArrayRecord({"batch_idx": _arr_i64(bidx)})}),
                    dst_node_id=active_id,
                    message_type=MSG_Y,
                ),
            )
            yb = rep_y.content["arrays"]["y"].numpy().astype(np.int64)

            z = torch.from_numpy(np.concatenate([ex, ep], axis=1)).to(device)
            yy = torch.from_numpy(yb).float().to(device)

            opt.zero_grad(set_to_none=True)
            logits_s = top(z)

            sup_loss = criterion_sup(logits_s, yy)

            if teacher_head is not None and kd_alpha > 0.0:
                ex_t = torch.from_numpy(ex).to(device)
                with torch.no_grad():
                    logits_t = teacher_head(ex_t)
                    p_t = torch.sigmoid(logits_t / kd_T)
                kd_loss = criterion_kd(logits_s / kd_T, p_t) * (kd_T * kd_T)
                loss = (1.0 - kd_alpha) * sup_loss + kd_alpha * kd_loss
            else:
                kd_loss = torch.tensor(0.0, device=device)
                loss = sup_loss

            loss.backward()
            opt.step()

            frac = len(bidx) / n
            train_loss_acc += float(loss.item()) * frac
            sup_loss_acc += float(sup_loss.item()) * frac
            kd_loss_acc += float(kd_loss.item()) * frac

        y_va, p_va = _eval_probs(grid, top, device, va, active_id, passive_id, emb_dim)
        val_auroc = float(roc_auc_score(y_va, p_va))
        val_prauc = float(average_precision_score(y_va, p_va))
        lr_now = float(opt.param_groups[0]["lr"])

        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([rnd, train_loss_acc, sup_loss_acc, kd_loss_acc, val_auroc, val_prauc, lr_now])

        if rnd == 1 or rnd % 10 == 0 or rnd == rounds:
            print(
                f"[fold {fold}] round {rnd:03d}/{rounds} "
                f"loss={train_loss_acc:.4f} sup={sup_loss_acc:.4f} kd={kd_loss_acc:.4f} "
                f"val_AUROC={val_auroc:.4f} val_PR-AUC={val_prauc:.4f} "
                f"best={best_val_auroc:.4f}@{best_round} lr={lr_now:.2e}"
            )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_round = rnd
            torch.save({"top": top.state_dict()}, best_path)
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"[fold {fold}] early stop at round {rnd} (no val AUROC improvement for {patience} rounds)")
                break

    top.load_state_dict(torch.load(best_path, map_location=device)["top"])
    y_va, p_va = _eval_probs(grid, top, device, va, active_id, passive_id, emb_dim)
    y_te, p_te = _eval_probs(grid, top, device, te, active_id, passive_id, emb_dim)

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
        w.writerow(["KD settings"])
        w.writerow(["kd_alpha", float(kd_alpha)])
        w.writerow(["kd_T", float(kd_T)])
        w.writerow(["teacher_ckpt_path", teacher_ckpt_path])
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


def _run_passive_hfl_ssl(context: Context, grid: Grid, run: Dict[str, object]) -> None:
    # unchanged: your passive HFL SSL pretrain
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

    ckpt_path = out_dir / f"pretrained_passive_bottom_hfl_fold{fold}.pt"
    torch.save({"bottom_state": {k: v.detach().cpu() for k, v in global_sd.items()}}, ckpt_path)
    log(INFO, f"[PASSIVE_HFL_SSL] saved {ckpt_path}")


app = ServerApp()


@app.main()
def main(grid, context) -> None:
    server_main(grid, context)