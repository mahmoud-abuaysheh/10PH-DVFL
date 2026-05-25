# clientapp_vfl_glioma_router_client_both_ssl.py  (FIXED)
#
# Fixes applied:
#   BUG-3 FIXED: pretrain_supervised now uses BCEWithLogitsLoss with
#                pos_weight computed from the train split — matching the
#                pos_weight used by _ensure_vfl_head during downstream VFL.
#                This removes the class-imbalance bias in the active
#                encoder's pretrained initialization.
#
#   BUG-4 FIXED: The active node now accumulates a best-val checkpoint of
#                both (active_bottom, head) together, keyed by the epoch at
#                which the server received the best val AUROC.  The server
#                sends "checkpoint_bottom" / "restore_best_bottom" messages
#                (same protocol as the immediate baseline) so that the
#                active bottom and head are always in sync at test time.
#                NOTE: passive bottom is either frozen (FREEZE_PASSIVE=1,
#                default) or finetuned with gradients.  If finetuned, it is
#                also checkpointed/restored via the same messages.
#
#   EPOCHS:      Default changed from 40 → 100.

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import os
import time
import torch
import torch.nn as nn
from flwr.app import Array, ArrayRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

# ---------------- process-local cache ----------------
_CACHE: Dict[str, Dict] = {}
_ACTIVE_NODE_ID_GLOBAL: str = ""


def _cache_key(context: Context) -> str:
    nid = getattr(context, "node_id", None)
    return str(nid) if nid is not None else str(id(context))


def _get_cache(context: Context) -> Dict:
    key = _cache_key(context)
    if key not in _CACHE:
        _CACHE[key] = {}
    return _CACHE[key]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _current_tag() -> str:
    sup_pre_epochs  = int(os.environ.get("SUP_PRE_EPOCHS", "10"))
    ssl_pre_epochs  = int(os.environ.get("SSL_PRE_EPOCHS", "10"))
    epochs          = int(os.environ.get("EPOCHS", "100"))
    freeze_passive  = int(os.environ.get("FREEZE_PASSIVE", "1"))
    return (
        f"pre_sup{sup_pre_epochs}_ssl{ssl_pre_epochs}_vfl{epochs}_"
        + ("frozen" if freeze_passive == 1 else "finetune")
    )


def _runs_root() -> Path:
    here     = Path(__file__).resolve().parent
    out_root = Path(os.environ.get("OUT_DIR", str(here / "runs_immediate_active")))
    return out_root / _current_tag()


def load_npz(npz_path: str):
    d  = np.load(npz_path, allow_pickle=True)
    X1 = d["X1"].astype(np.float32)
    X2 = d["X2"].astype(np.float32)
    y  = d["y"].astype(np.int64)
    folds = list(d["folds"])
    return X1, X2, y, folds


# ---------------- Models ----------------
class BottomMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(1)


class ReconHead(nn.Module):
    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------- Role helpers ----------------
def _my_node_id(context: Context) -> int:
    nid = getattr(context, "node_id", None)
    if nid is None:
        nid = int(id(context) % (10 ** 9))
    return int(nid)


def _active_node_id_str(cache: Dict) -> str:
    v = cache.get("ACTIVE_NODE_ID", "")
    v = "" if v is None else str(v).strip()
    if v:
        return v
    global _ACTIVE_NODE_ID_GLOBAL
    return str(_ACTIVE_NODE_ID_GLOBAL).strip()


def _is_active(context: Context) -> bool:
    cache    = _get_cache(context)
    my_id    = str(getattr(context, "node_id", "")).strip()
    active_id = _active_node_id_str(cache)
    if not my_id or not active_id:
        return False
    return my_id == active_id


# ---------------- Init once ----------------
def _init_if_needed(context: Context) -> None:
    cache = _get_cache(context)
    if cache.get("ready", False):
        return

    here        = Path(__file__).resolve().parent
    DEFAULT_NPZ = here / "glioma_aligned_vfl_hfl_cv.npz"

    rc      = getattr(context, "run_config", {}) or {}
    npz:    str   = os.environ.get("NPZ",     str(rc.get("npz",     str(DEFAULT_NPZ))))
    fold:   int   = int(os.environ.get("FOLD",  rc.get("fold",  1)))
    seed:   int   = int(os.environ.get("SEED",  rc.get("seed",  0)))
    device: str   = os.environ.get("DEVICE",    str(rc.get("device", "cpu")))

    lr_bottom: float = float(os.environ.get("LR_BOTTOM", rc.get("lr_bottom", 1e-3)))
    lr_head:   float = float(os.environ.get("LR_HEAD",   rc.get("lr_head",   1e-3)))
    dropout:   float = float(os.environ.get("DROPOUT",   rc.get("dropout",   0.0)))
    out_dim:   int   = int(os.environ.get("OUT_FEATURE_DIM", rc.get("out_feature_dim", 16)))

    cache["seed"] = seed
    set_seed(seed)
    dev = torch.device(device)

    X1, X2, y, folds = load_npz(npz)
    split_obj = folds[fold - 1]
    split     = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)
    va = split["val"].astype(np.int64)
    te = split["test"].astype(np.int64)

    mu1 = X1[tr].mean(axis=0, keepdims=True);  sd1 = X1[tr].std(axis=0, keepdims=True) + 1e-8
    mu2 = X2[tr].mean(axis=0, keepdims=True);  sd2 = X2[tr].std(axis=0, keepdims=True) + 1e-8
    X1s = (X1 - mu1) / sd1
    X2s = (X2 - mu2) / sd2

    cache["dev"]        = dev
    cache["fold"]       = fold
    cache["tr"]         = tr
    cache["va"]         = va
    cache["te"]         = te
    cache["out_dim"]    = out_dim
    cache["lr_bottom"]  = lr_bottom
    cache["lr_head"]    = lr_head
    cache["dropout"]    = dropout
    cache["npz_path"]   = npz
    cache["x1_mu"]      = mu1.astype(np.float32)
    cache["x1_sd"]      = sd1.astype(np.float32)
    cache["x2_mu"]      = mu2.astype(np.float32)
    cache["x2_sd"]      = sd2.astype(np.float32)

    cache["X1"] = torch.from_numpy(X1s).float().to(dev)
    cache["X2"] = torch.from_numpy(X2s).float().to(dev)
    cache["y"]  = torch.from_numpy(y.astype(np.float32)).float().to(dev)

    cache["active_bottom"]     = BottomMLP(in_dim=int(X1.shape[1]), out_dim=out_dim, dropout=dropout).to(dev)
    cache["opt_active_bottom"] = torch.optim.Adam(cache["active_bottom"].parameters(), lr=lr_bottom)

    cache["passive_bottom"]     = BottomMLP(in_dim=int(X2.shape[1]), out_dim=out_dim, dropout=dropout).to(dev)
    cache["opt_passive_bottom"] = torch.optim.Adam(cache["passive_bottom"].parameters(), lr=lr_bottom)

    cache["passive_ckpt_loaded"] = False
    cache["active_ckpt_loaded"]  = False

    # VFL head (active only, lazy init on first active_step)
    cache["head"]      = None
    cache["opt_head"]  = None
    cache["criterion"] = None

    # eval/test buffers
    cache["buf"] = {
        "val":  {"prob": [], "y": [], "loss_sum": 0.0, "n": 0},
        "test": {"prob": [], "y": [], "loss_sum": 0.0, "n": 0},
    }

    # Checkpoints are disk-based (see checkpoint_bottom/restore_best_bottom)

    cache["ready"] = True
    print(
        f"[RouterClient init] node={_my_node_id(context)} device={device} fold={fold} "
        f"X1dim={X1.shape[1]} X2dim={X2.shape[1]} out_dim={out_dim}"
    )


# ---------------- Checkpoint loaders ----------------
def _ensure_passive_ckpt_loaded(context: Context) -> None:
    cache = _get_cache(context)
    if cache.get("passive_ckpt_loaded", False):
        return
    if _is_active(context):
        return
    passive_ckpt = os.environ.get("PASSIVE_BOTTOM_CKPT", "").strip()
    if not passive_ckpt:
        cache["passive_ckpt_loaded"] = True
        return
    ckpt_path = Path(passive_ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PASSIVE_BOTTOM_CKPT not found: {ckpt_path}")
    ckpt = _safe_torch_load(ckpt_path)
    if "bottom_state" not in ckpt:
        raise KeyError(f"Checkpoint missing 'bottom_state': {ckpt_path}")
    cache["passive_bottom"].load_state_dict(ckpt["bottom_state"], strict=True)
    cache["passive_ckpt_loaded"] = True
    print(f"[RouterClient][Passive init] loaded PASSIVE_BOTTOM_CKPT={ckpt_path}")


def _ensure_active_ckpt_loaded(context: Context) -> None:
    cache = _get_cache(context)
    if cache.get("active_ckpt_loaded", False):
        return
    if not _is_active(context):
        return
    active_ckpt = os.environ.get("ACTIVE_BOTTOM_CKPT", "").strip()
    if not active_ckpt:
        cache["active_ckpt_loaded"] = True
        return
    ckpt_path = Path(active_ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ACTIVE_BOTTOM_CKPT not found: {ckpt_path}")
    ckpt = _safe_torch_load(ckpt_path)
    if "bottom_state" not in ckpt:
        raise KeyError(f"Checkpoint missing 'bottom_state': {ckpt_path}")
    cache["active_bottom"].load_state_dict(ckpt["bottom_state"], strict=True)
    cache["active_ckpt_loaded"] = True
    print(f"[RouterClient][Active init] loaded ACTIVE_BOTTOM_CKPT={ckpt_path}")


def _ensure_vfl_head(context: Context, z2_dim: int) -> None:
    cache = _get_cache(context)
    if cache["head"] is not None:
        return
    dev:     torch.device = cache["dev"]
    out_dim: int          = int(cache["out_dim"])
    lr_head: float        = float(cache["lr_head"])

    head     = TopMLP(in_dim=out_dim + z2_dim).to(dev)
    opt_head = torch.optim.Adam(head.parameters(), lr=lr_head)

    tr     = cache["tr"]
    y_tr   = cache["y"][torch.from_numpy(tr).long().to(dev)]
    pos    = float(y_tr.sum().item())
    neg    = float((y_tr.numel() - y_tr.sum()).item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(dev)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    cache["head"]      = head
    cache["opt_head"]  = opt_head
    cache["criterion"] = criterion


# ---------------- Threshold helpers ----------------
def _compute_threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict:
    y    = y.astype(int)
    pred = (p >= thr).astype(int)
    tp   = int(((pred == 1) & (y == 1)).sum())
    tn   = int(((pred == 0) & (y == 0)).sum())
    fp   = int(((pred == 1) & (y == 0)).sum())
    fn   = int(((pred == 0) & (y == 1)).sum())
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1   = (2 * prec * rec) / max(prec + rec, 1e-12)
    bal_acc = 0.5 * (rec + spec)
    return {
        "threshold": float(thr),
        "acc": float(acc), "precision": float(prec),
        "recall": float(rec), "specificity": float(spec),
        "f1": float(f1), "balanced_acc": float(bal_acc),
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
    }


def _best_thresholds(y: np.ndarray, p: np.ndarray) -> Dict:
    thrs = np.unique(np.clip(p, 0.0, 1.0))
    thrs = np.unique(np.concatenate([thrs, np.array([0.5], dtype=np.float32)]))
    thrs.sort()
    best_f1     = (-1.0, 0.5)
    best_youden = (-1.0, 0.5)
    for t in thrs:
        m      = _compute_threshold_metrics(y, p, float(t))
        youden = m["recall"] + m["specificity"] - 1.0
        if m["f1"] > best_f1[0]:
            best_f1 = (m["f1"], float(t))
        if youden > best_youden[0]:
            best_youden = (youden, float(t))
    return {
        "best_f1":     _compute_threshold_metrics(y, p, best_f1[1]),
        "best_youden": _compute_threshold_metrics(y, p, best_youden[1]),
        "fixed_0.5":   _compute_threshold_metrics(y, p, 0.5),
    }


# ============================================================================
# ClientApp
# ============================================================================
app = ClientApp()


# ---------------------------------------------------------------------------
# Common routing handlers
# ---------------------------------------------------------------------------
@app.query("set_active_node")
def set_active_node(msg: Message, context: Context) -> Message:
    cache     = _get_cache(context)
    active_nid = str(msg.content["config"].get("active_node_id", "")).strip()
    cache["ACTIVE_NODE_ID"] = active_nid
    global _ACTIVE_NODE_ID_GLOBAL
    _ACTIVE_NODE_ID_GLOBAL = active_nid
    print(f"[RouterClient] received ACTIVE_NODE_ID={active_nid} my node={_my_node_id(context)}")
    return Message(content=RecordDict(), reply_to=msg)


@app.query("get_role")
def get_role(msg: Message, context: Context) -> Message:
    _init_if_needed(context)
    for _ in range(100):
        cache     = _get_cache(context)
        my_id     = str(getattr(context, "node_id", "")).strip()
        active_id = _active_node_id_str(cache)
        if my_id and active_id:
            break
        time.sleep(0.05)
    role = 1 if _is_active(context) else 0
    arrs = ArrayRecord({"role_code": Array(np.array([role], dtype=np.int64))})
    return Message(content=RecordDict({"arrays": arrs}), reply_to=msg)


# ---------------------------------------------------------------------------
# BUG-4 FIX: checkpoint / restore_best (same protocol as immediate baseline)
# Handles both active and passive nodes transparently.
# ---------------------------------------------------------------------------
@app.query("checkpoint_bottom")
def checkpoint_bottom(msg: Message, context: Context) -> Message:
    """DISK-BASED: Save current bottom (and head if active) as best-val checkpoint.
    Disk-based to survive Ray dispatching restore to a different actor than checkpoint.
    """
    _init_if_needed(context)
    cache   = _get_cache(context)
    out_dir = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent / "runs_decoupled_vfl")))
    fold    = int(cache["fold"])
    # Use tag subdir to match server's output structure
    tag_dir = out_dir / _current_tag()
    tag_dir.mkdir(parents=True, exist_ok=True)

    if _is_active(context):
        p = tag_dir / f"ckpt_active_bottom_fold{fold}.pt"
        torch.save(cache["active_bottom"].state_dict(), p)
        if cache["head"] is not None:
            ph = tag_dir / f"ckpt_head_fold{fold}.pt"
            torch.save(cache["head"].state_dict(), ph)
        print(f"[RouterClient][Active pid={os.getpid()}] checkpoint saved -> {p}")
    else:
        p = tag_dir / f"ckpt_passive_bottom_fold{fold}.pt"
        torch.save(cache["passive_bottom"].state_dict(), p)
        print(f"[RouterClient][Passive pid={os.getpid()}] checkpoint saved -> {p}")
    return Message(content=RecordDict(), reply_to=msg)


@app.query("restore_best_bottom")
def restore_best_bottom(msg: Message, context: Context) -> Message:
    """DISK-BASED: Restore best-val bottom (and head if active) before test evaluation.
    Works regardless of which Ray actor receives this message.
    """
    _init_if_needed(context)
    cache   = _get_cache(context)
    dev     = cache["dev"]
    out_dir = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parent / "runs_decoupled_vfl")))
    fold    = int(cache["fold"])
    tag_dir = out_dir / _current_tag()

    if _is_active(context):
        p = tag_dir / f"ckpt_active_bottom_fold{fold}.pt"
        if p.exists():
            cache["active_bottom"].load_state_dict(
                torch.load(p, map_location=dev, weights_only=False)
            )
            print(f"[RouterClient][Active pid={os.getpid()}] active_bottom restored from {p}")
        else:
            print(f"[RouterClient][Active pid={os.getpid()}] WARNING: {p} not found — using final weights!")
        ph = tag_dir / f"ckpt_head_fold{fold}.pt"
        if ph.exists() and cache["head"] is not None:
            cache["head"].load_state_dict(
                torch.load(ph, map_location=dev, weights_only=False)
            )
            print(f"[RouterClient][Active pid={os.getpid()}] head restored from {ph}")
    else:
        p = tag_dir / f"ckpt_passive_bottom_fold{fold}.pt"
        if p.exists():
            cache["passive_bottom"].load_state_dict(
                torch.load(p, map_location=dev, weights_only=False)
            )
            print(f"[RouterClient][Passive pid={os.getpid()}] passive_bottom restored from {p}")
        else:
            print(f"[RouterClient][Passive pid={os.getpid()}] WARNING: {p} not found — using final weights!")
    return Message(content=RecordDict(), reply_to=msg)


# ---------------------------------------------------------------------------
# PASSIVE handlers
# ---------------------------------------------------------------------------
@app.query("generate_embeddings")
def generate_embeddings(msg: Message, context: Context) -> Message:
    _init_if_needed(context)
    _ensure_passive_ckpt_loaded(context)
    if _is_active(context):
        raise RuntimeError("generate_embeddings called on active node (routing error)")
    cache = _get_cache(context)
    X2: torch.Tensor = cache["X2"]
    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    idx_t     = torch.from_numpy(batch_idx).long().to(X2.device)
    cache["passive_bottom"].eval()
    with torch.no_grad():
        emb = (
            cache["passive_bottom"](X2.index_select(0, idx_t))
            .detach().cpu().numpy().astype(np.float32)
        )
    arrs = ArrayRecord({"embedding": Array(emb)})
    return Message(content=RecordDict({"arrays": arrs}), reply_to=msg)


@app.train("apply_gradients")
def apply_gradients(msg: Message, context: Context) -> Message:
    _init_if_needed(context)
    _ensure_passive_ckpt_loaded(context)
    if _is_active(context):
        raise RuntimeError("apply_gradients called on active node (routing error)")
    phase = str(msg.content["config"].get("phase", "train"))
    if phase != "train":
        return Message(content=RecordDict(), reply_to=msg)
    if int(os.environ.get("FREEZE_PASSIVE", "1")) == 1:
        return Message(content=RecordDict(), reply_to=msg)
    cache   = _get_cache(context)
    X2: torch.Tensor = cache["X2"]
    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    grad_np   = msg.content["arrays"]["local_gradients"].numpy().astype(np.float32)
    idx_t   = torch.from_numpy(batch_idx).long().to(X2.device)
    grad_t  = torch.from_numpy(grad_np).float().to(X2.device)
    cache["passive_bottom"].train()
    cache["opt_passive_bottom"].zero_grad(set_to_none=True)
    emb = cache["passive_bottom"](X2.index_select(0, idx_t))
    emb.backward(grad_t)
    cache["opt_passive_bottom"].step()
    return Message(content=RecordDict(), reply_to=msg)


@app.train("pretrain_ssl")
def pretrain_ssl(msg: Message, context: Context) -> Message:
    """SSL pretraining for the PASSIVE node (denoising autoencoder)."""
    _init_if_needed(context)
    _ensure_passive_ckpt_loaded(context)
    if _is_active(context):
        raise RuntimeError("pretrain_ssl called on active node (routing error)")
    cache  = _get_cache(context)
    dev    = cache["dev"]
    epochs     = int(msg.content["config"].get("epochs",     int(os.environ.get("SSL_PRE_EPOCHS",  "10"))))
    batch_size = int(msg.content["config"].get("batch_size", int(os.environ.get("BATCH_SIZE",       "64"))))
    lr         = float(msg.content["config"].get("lr",       float(cache["lr_bottom"])))
    noise_std  = float(msg.content["config"].get("noise_std", float(os.environ.get("SSL_NOISE_STD", "0.1"))))
    if epochs <= 0:
        print("[RouterClient][Passive][SSL] epochs<=0, skipping.")
        return Message(content=RecordDict(), reply_to=msg)
    X2: torch.Tensor = cache["X2"]
    tr  = cache["tr"].copy()
    rng = np.random.default_rng(int(cache["seed"]))
    bottom: nn.Module = cache["passive_bottom"]
    bottom.train()
    recon_head = ReconHead(emb_dim=int(cache["out_dim"]), out_dim=int(X2.shape[1])).to(dev)
    opt = torch.optim.Adam(list(bottom.parameters()) + list(recon_head.parameters()), lr=lr)
    mse = nn.MSELoss()
    for ep in range(1, epochs + 1):
        rng.shuffle(tr)
        loss_sum = 0.0; n = 0
        for s in range(int(np.ceil(len(tr) / batch_size))):
            b = tr[s * batch_size : (s + 1) * batch_size]
            if b.size == 0:
                continue
            bt  = torch.from_numpy(b).long().to(dev)
            xb  = X2.index_select(0, bt)
            xn  = xb + noise_std * torch.randn_like(xb)
            opt.zero_grad(set_to_none=True)
            z   = bottom(xn)
            xr  = recon_head(z)
            loss = mse(xr, xb)
            loss.backward(); opt.step()
            loss_sum += float(loss.item()) * int(b.size); n += int(b.size)
        print(f"[RouterClient][Passive][SSL] epoch={ep}/{epochs} avg_recon_mse={loss_sum/max(n,1):.6f}")
    out_dir = _runs_root(); out_dir.mkdir(parents=True, exist_ok=True)
    fold = int(cache["fold"])
    ckpt = {
        "bottom_state": cache["passive_bottom"].state_dict(),
        "fold": fold, "seed": int(cache["seed"]), "out_dim": int(cache["out_dim"]),
        "npz": str(cache["npz_path"]),
        "x2_mu": cache["x2_mu"], "x2_sd": cache["x2_sd"], "ssl": True,
    }
    save_path = out_dir / f"pretrained_passive_bottom_fold{fold}.pt"
    torch.save(ckpt, save_path)
    print(f"[RouterClient][Passive] ssl pretrain done epochs={epochs} | saved -> {save_path}")
    return Message(content=RecordDict(), reply_to=msg)


# ---------------------------------------------------------------------------
# ACTIVE handlers
# ---------------------------------------------------------------------------
@app.train("pretrain_ssl_active")
def pretrain_ssl_active(msg: Message, context: Context) -> Message:
    """SSL pretraining for the ACTIVE node (denoising autoencoder)."""
    _init_if_needed(context)
    if not _is_active(context):
        raise RuntimeError("pretrain_ssl_active called on passive node (routing error)")
    cache  = _get_cache(context)
    dev    = cache["dev"]
    epochs     = int(msg.content["config"].get("epochs",     int(os.environ.get("SSL_PRE_EPOCHS_ACTIVE", "10"))))
    batch_size = int(msg.content["config"].get("batch_size", int(os.environ.get("BATCH_SIZE",             "64"))))
    lr         = float(msg.content["config"].get("lr",       float(cache["lr_bottom"])))
    noise_std  = float(msg.content["config"].get("noise_std", float(os.environ.get("SSL_NOISE_STD",       "0.1"))))
    if epochs <= 0:
        print("[RouterClient][Active][SSL] epochs<=0, skipping.")
        return Message(content=RecordDict(), reply_to=msg)
    X1: torch.Tensor = cache["X1"]
    tr  = cache["tr"].copy()
    rng = np.random.default_rng(int(cache["seed"]))
    bottom: nn.Module = cache["active_bottom"]
    bottom.train()
    recon_head = ReconHead(emb_dim=int(cache["out_dim"]), out_dim=int(X1.shape[1])).to(dev)
    opt = torch.optim.Adam(list(bottom.parameters()) + list(recon_head.parameters()), lr=lr)
    mse = nn.MSELoss()
    for ep in range(1, epochs + 1):
        rng.shuffle(tr)
        loss_sum = 0.0; n = 0
        for s in range(int(np.ceil(len(tr) / batch_size))):
            b = tr[s * batch_size : (s + 1) * batch_size]
            if b.size == 0:
                continue
            bt  = torch.from_numpy(b).long().to(dev)
            xb  = X1.index_select(0, bt)
            xn  = xb + noise_std * torch.randn_like(xb)
            opt.zero_grad(set_to_none=True)
            z   = bottom(xn)
            xr  = recon_head(z)
            loss = mse(xr, xb)
            loss.backward(); opt.step()
            loss_sum += float(loss.item()) * int(b.size); n += int(b.size)
        print(f"[RouterClient][Active][SSL] epoch={ep}/{epochs} avg_recon_mse={loss_sum/max(n,1):.6f}")
    out_dir = _runs_root(); out_dir.mkdir(parents=True, exist_ok=True)
    fold = int(cache["fold"])
    ckpt = {
        "bottom_state": cache["active_bottom"].state_dict(),
        "fold": fold, "seed": int(cache["seed"]), "out_dim": int(cache["out_dim"]),
        "npz": str(cache["npz_path"]),
        "x1_mu": cache["x1_mu"], "x1_sd": cache["x1_sd"], "ssl": True,
    }
    save_path = out_dir / f"pretrained_active_bottom_ssl_fold{fold}.pt"
    torch.save(ckpt, save_path)
    print(f"[RouterClient][Active] ssl pretrain done epochs={epochs} | saved -> {save_path}")
    return Message(content=RecordDict(), reply_to=msg)


@app.train("pretrain_supervised")
def pretrain_supervised(msg: Message, context: Context) -> Message:
    """Supervised pretraining for the ACTIVE node."""
    _init_if_needed(context)
    if not _is_active(context):
        raise RuntimeError("pretrain_supervised called on passive node (routing error)")
    cache  = _get_cache(context)
    dev    = cache["dev"]
    epochs     = int(msg.content["config"].get("epochs",     int(os.environ.get("SUP_PRE_EPOCHS", "10"))))
    batch_size = int(msg.content["config"].get("batch_size", int(os.environ.get("BATCH_SIZE",      "64"))))
    lr         = float(msg.content["config"].get("lr",       float(cache["lr_bottom"])))

    local_head = TopMLP(in_dim=int(cache["out_dim"])).to(dev)
    opt = torch.optim.Adam(
        list(cache["active_bottom"].parameters()) + list(local_head.parameters()), lr=lr
    )

    # BUG-3 FIX: use pos_weight to match the VFL downstream head and handle
    # class imbalance properly during supervised pretraining.
    tr     = cache["tr"]
    y_tr   = cache["y"][torch.from_numpy(tr).long().to(dev)]
    pos    = float(y_tr.sum().item())
    neg    = float((y_tr.numel() - y_tr.sum()).item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(dev)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)   # <-- FIX

    X1: torch.Tensor = cache["X1"]
    y:  torch.Tensor = cache["y"]
    tr_copy = tr.copy()
    rng = np.random.default_rng(int(cache["seed"]))

    cache["active_bottom"].train()
    local_head.train()
    for ep in range(1, epochs + 1):
        rng.shuffle(tr_copy)
        loss_sum = 0.0; n = 0
        for s in range(int(np.ceil(len(tr_copy) / batch_size))):
            b = tr_copy[s * batch_size : (s + 1) * batch_size]
            if b.size == 0:
                continue
            bt  = torch.from_numpy(b).long().to(dev)
            xb  = X1.index_select(0, bt)
            yb  = y.index_select(0, bt)
            opt.zero_grad(set_to_none=True)
            z1     = cache["active_bottom"](xb)
            logits = local_head(z1)
            loss   = criterion(logits, yb)
            loss.backward(); opt.step()
            loss_sum += float(loss.item()) * int(b.size); n += int(b.size)
        print(f"[RouterClient][Active][Sup] epoch={ep}/{epochs} avg_loss={loss_sum/max(n,1):.6f}")

    out_dir = _runs_root(); out_dir.mkdir(parents=True, exist_ok=True)
    fold = int(cache["fold"])
    ckpt = {
        "bottom_state": cache["active_bottom"].state_dict(),
        "fold": fold, "seed": int(cache["seed"]), "out_dim": int(cache["out_dim"]),
        "npz": str(cache["npz_path"]),
        "x1_mu": cache["x1_mu"], "x1_sd": cache["x1_sd"], "supervised": True,
    }
    save_path = out_dir / f"pretrained_active_bottom_sup_fold{fold}.pt"
    torch.save(ckpt, save_path)
    print(f"[RouterClient][Active] supervised pretrain done epochs={epochs} | saved -> {save_path}")
    return Message(content=RecordDict(), reply_to=msg)


@app.train("active_step")
def active_step(msg: Message, context: Context) -> Message:
    _init_if_needed(context)
    if not _is_active(context):
        raise RuntimeError("active_step called on passive node (routing error)")

    _ensure_active_ckpt_loaded(context)

    cache  = _get_cache(context)
    dev    = cache["dev"]
    phase  = str(msg.content["config"].get("phase", "train"))
    freeze_active_bottom = int(os.environ.get("FREEZE_ACTIVE_BOTTOM", "0")) == 1

    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    z2_np     = msg.content["arrays"]["z_passive"].numpy().astype(np.float32)

    bt = torch.from_numpy(batch_idx).long().to(dev)
    z2 = torch.from_numpy(z2_np).float().to(dev)
    z2.requires_grad_(phase == "train")

    _ensure_vfl_head(context, z2_dim=int(z2.shape[1]))

    X1:        torch.Tensor = cache["X1"]
    y:         torch.Tensor = cache["y"]
    bottom:    nn.Module    = cache["active_bottom"]
    head:      nn.Module    = cache["head"]
    opt_bottom              = cache["opt_active_bottom"]
    opt_head                = cache["opt_head"]
    criterion               = cache["criterion"]

    if phase == "train":
        bottom.train(); head.train()
        opt_head.zero_grad(set_to_none=True)
        if not freeze_active_bottom:
            opt_bottom.zero_grad(set_to_none=True)

        z1     = bottom(X1.index_select(0, bt))
        z      = torch.cat([z1, z2], dim=1)
        logits = head(z)
        yb     = y.index_select(0, bt)
        loss   = criterion(logits, yb)
        loss.backward()
        opt_head.step()
        if not freeze_active_bottom:
            opt_bottom.step()

        grad_z2 = z2.grad.detach().cpu().numpy().astype(np.float32)
        arrs = ArrayRecord({
            "loss":          Array(np.array([float(loss.item())], dtype=np.float32)),
            "grad_z_passive": Array(grad_z2),
        })
        return Message(content=RecordDict({"arrays": arrs}), reply_to=msg)

    # val / test
    bottom.eval(); head.eval()
    with torch.no_grad():
        z1     = bottom(X1.index_select(0, bt))
        z      = torch.cat([z1, z2], dim=1)
        logits = head(z)
        yb     = y.index_select(0, bt)
        loss   = criterion(logits, yb)
        prob   = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        y_np   = yb.detach().cpu().numpy().astype(np.float32)

    buf = cache["buf"][phase]
    buf["prob"].append(prob.reshape(-1))
    buf["y"].append(y_np.reshape(-1))
    buf["loss_sum"] += float(loss.item()) * int(batch_idx.size)
    buf["n"]        += int(batch_idx.size)

    arrs = ArrayRecord({"loss": Array(np.array([float(loss.item())], dtype=np.float32))})
    return Message(content=RecordDict({"arrays": arrs}), reply_to=msg)


@app.train("finalize_metrics")
def finalize_metrics(msg: Message, context: Context) -> Message:
    _init_if_needed(context)
    if not _is_active(context):
        raise RuntimeError("finalize_metrics called on passive node (routing error)")
    cache = _get_cache(context)
    from sklearn.metrics import roc_auc_score, average_precision_score

    phase    = str(msg.content["config"].get("phase",    "val"))
    save_dir = msg.content["config"].get("save_dir", None)
    tag      = str(msg.content["config"].get("tag",      "run"))

    buf = cache["buf"][phase]
    if buf["n"] == 0:
        arrs = ArrayRecord({
            "loss":  Array(np.array([np.nan], dtype=np.float32)),
            "auroc": Array(np.array([np.nan], dtype=np.float32)),
            "prauc": Array(np.array([np.nan], dtype=np.float32)),
        })
        return Message(content=RecordDict({"arrays": arrs}), reply_to=msg)

    y    = np.concatenate(buf["y"],    axis=0).astype(np.float32)
    p    = np.concatenate(buf["prob"], axis=0).astype(np.float32)
    loss = np.array([buf["loss_sum"] / max(buf["n"], 1)], dtype=np.float32)

    try:
        auroc = roc_auc_score(y, p) if len(np.unique(y)) >= 2 else np.nan
    except Exception:
        auroc = np.nan
    try:
        prauc = average_precision_score(y, p) if len(np.unique(y)) >= 2 else np.nan
    except Exception:
        prauc = np.nan

    if save_dir is not None and str(save_dir).strip() != "":
        out = Path(str(save_dir)); out.mkdir(parents=True, exist_ok=True)
        preds_path = out / f"preds_{tag}_{phase}_fold{cache['fold']}.npz"
        np.savez(preds_path, y=y, p=p)
        thr      = _best_thresholds(y.astype(int), p.astype(float))
        thr_path = out / f"thresholds_{tag}_{phase}_fold{cache['fold']}.txt"
        with open(thr_path, "w") as f:
            f.write(f"phase={phase}\n")
            f.write(f"loss={float(loss[0]):.6f}\n")
            f.write(f"auroc={float(auroc):.6f}\n")
            f.write(f"prauc={float(prauc):.6f}\n\n")
            for k, v in thr.items():
                f.write(f"[{k}]\n")
                for kk, vv in v.items():
                    f.write(f"{kk}={vv}\n")
                f.write("\n")
        print(f"[RouterClient][Active] saved {phase} preds  -> {preds_path}")
        print(f"[RouterClient][Active] saved {phase} thresholds -> {thr_path}")

    # reset buffer
    cache["buf"][phase] = {"prob": [], "y": [], "loss_sum": 0.0, "n": 0}

    arrs = ArrayRecord({
        "loss":  Array(loss),
        "auroc": Array(np.array([auroc],  dtype=np.float32)),
        "prauc": Array(np.array([prauc],  dtype=np.float32)),
    })
    return Message(content=RecordDict({"arrays": arrs}), reply_to=msg)