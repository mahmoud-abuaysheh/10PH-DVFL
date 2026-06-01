# clientapp_vfl_glioma_decoupled.py
#
# Flower client application for the decoupled VFL glioma experiment.
#
# This client implements the silo-side behaviour for both the active and passive
# silos in Tier 2 of the 10PH-DVFL architecture for the glioma dataset.
#
# Two-stage protocol:
#   Stage A — send_embeddings handler:
#     Called once per split (train/val/test) by the server. Returns the full
#     embedding matrix for the requested split in a single message. Since
#     encoders are frozen after Tier 1, this one-time transfer is sufficient
#     for all subsequent fusion head training in Stage B.
#   Stage B — no handlers called:
#     The server trains the fusion head entirely on cached embeddings.
#     No messages are sent to any client node during Stage B.
#
# Roles handled by this client:
#   "active"  : Loads the pre-trained active-silo encoder. Responds to
#               send_embeddings with X1 and get_labels with y.
#   "passive" : Loads the pre-trained passive-silo encoder. Responds to
#               send_embeddings with X2.
#
# Key architectural differences from the diabetes decoupled client:
#   - BottomMLP: input -> 32 -> ReLU -> Dropout(0.0) -> 16 -> ReLU
#   - The Dropout layer is included at dropout=0.0 to maintain checkpoint
#     key compatibility with HFL scripts which define BottomMLP with Dropout.

from __future__ import annotations

from pathlib import Path
from typing import Dict

import json
import numpy as np
import os
import torch
import torch.nn as nn

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

_CACHE: Dict[str, Dict] = {}


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def _cache_key(context: Context) -> str:
    """Return a unique string key identifying this client node."""
    nid = getattr(context, "node_id", None)
    return str(nid) if nid is not None else str(id(context))


def _get_cache(context: Context) -> Dict:
    """Return the cache dict for this client node, initialising it if needed."""
    k = _cache_key(context)
    if k not in _CACHE:
        _CACHE[k] = {}
    return _CACHE[k]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class BottomMLP(nn.Module):
    """
    Bottom encoder used by each silo in the glioma decoupled VFL architecture.

    Architecture: input -> 32 -> ReLU -> Dropout(0.0) -> 16 -> ReLU

    The Dropout layer is included at dropout=0.0 to maintain checkpoint key
    compatibility with clientapp_hfl_passive_glioma.py and
    serverapp_hfl_passive_glioma.py which define BottomMLP with Dropout present.
    """
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_npz(npz_path: str):
    """Load the cross-validation dataset from the NPZ file."""
    d     = np.load(npz_path, allow_pickle=True)
    X1    = d["X1"].astype(np.float32)
    X2    = d["X2"].astype(np.float32)
    y     = d["y"].astype(np.int64)
    folds = list(d["folds"])
    meta  = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return X1, X2, y, folds, meta


def _standardize_using_train(X: np.ndarray, tr_idx: np.ndarray) -> np.ndarray:
    """Standardize using training split statistics only."""
    mu = X[tr_idx].mean(axis=0, keepdims=True)
    sd = X[tr_idx].std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _cfg(d: Dict[str, object] | None = None) -> ConfigRecord:
    """Wrap a plain dict as a Flower ConfigRecord."""
    return ConfigRecord(d or {})


def _unwrap_state_dict(state: object) -> Dict[str, torch.Tensor]:
    """Extract the encoder state dict from a checkpoint file."""
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint is not a dict; got type={type(state)}")
    if "bottom_state" in state and isinstance(state["bottom_state"], dict):
        sd = state["bottom_state"]
    elif "model_state" in state and isinstance(state["model_state"], dict):
        sd = state["model_state"]
    elif "state_dict" in state and isinstance(state["state_dict"], dict):
        sd = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        sd = state["model"]
    else:
        sd = state
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _init_if_needed(context: Context, desired_role: str) -> None:
    """
    Initialise the client node for the role requested by the server.

    Loads the dataset, applies fold-specific standardization, and loads
    the appropriate pre-trained encoder checkpoint. Stores the full
    standardized feature matrices for all splits so that Stage A can
    serve the complete embedding matrix for any split in one call.

    Checkpoint priority:
      active  : supervised checkpoint takes priority over SSL checkpoint
      passive : HFL checkpoint takes priority over local SSL checkpoint
    """
    cache = _get_cache(context)
    key   = f"ready::{desired_role}"
    if cache.get(key, False):
        return

    run     = context.run_config
    DEFAULT_NPZ = Path(__file__).resolve().parent / "glioma_aligned_vfl_hfl_cv.npz"
    npz     = str(run.get("npz", str(DEFAULT_NPZ)))
    fold    = int(os.environ.get("FOLD", run.get("fold", 1)))
    seed    = int(run.get("seed", 42))
    device  = str(run.get("device", "cpu"))
    out_dim = int(run.get("out_feature_dim", 16))

    active_dir  = str(run.get("active_ckpt_dir",  "./runs_sup_pretrain"))
    passive_dir = str(run.get("passive_ckpt_dir", "./runs_passive_ssl_glioma"))

    active_ckpt_sup    = os.path.join(active_dir,  f"pretrained_active_bottom_sup_fold{fold}.pt")
    active_ckpt_ssl    = os.path.join(active_dir,  f"pretrained_active_bottom_ssl_fold{fold}.pt")
    passive_ckpt_hfl   = os.path.join(passive_dir, f"pretrained_passive_bottom_hfl_fold{fold}.pt")
    passive_ckpt_local = os.path.join(passive_dir, f"pretrained_passive_bottom_ssl_fold{fold}.pt")

    set_seed(seed)
    dev = torch.device(device)

    X1, X2, y, folds, meta = load_npz(npz)
    split_obj = folds[fold - 1]
    split_    = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split_["train"].astype(np.int64)
    va = split_["val"].astype(np.int64)
    te = split_["test"].astype(np.int64)

    X1s = _standardize_using_train(X1, tr)
    X2s = _standardize_using_train(X2, tr)

    cache["dev"]  = dev
    cache["meta"] = meta
    cache["fold"] = fold
    cache["tr"]   = tr
    cache["va"]   = va
    cache["te"]   = te

    # Store full standardized feature matrices for all splits so Stage A
    # can serve the complete embedding matrix for any split in one call.
    cache["X1"] = torch.from_numpy(X1s).float()
    cache["X2"] = torch.from_numpy(X2s).float()
    cache["y"]  = torch.from_numpy(y).long()

    m0 = BottomMLP(in_dim=int(X1.shape[1]), out_dim=out_dim).to(dev)
    m1 = BottomMLP(in_dim=int(X2.shape[1]), out_dim=out_dim).to(dev)

    if desired_role == "active":
        if os.path.exists(active_ckpt_sup):
            ckpt_path = active_ckpt_sup
        elif os.path.exists(active_ckpt_ssl):
            ckpt_path = active_ckpt_ssl
        else:
            raise FileNotFoundError(
                f"[ACTIVE] No checkpoint found. Tried:\n"
                f"  {active_ckpt_sup}\n  {active_ckpt_ssl}"
            )
        state = torch.load(ckpt_path, map_location=dev)
        sd    = _unwrap_state_dict(state)
        m0.load_state_dict(sd, strict=True)
        cache["model_active"]  = m0
        cache["model_passive"] = m1
        print(f"[Client init][ACTIVE] loaded {ckpt_path}")

    elif desired_role == "passive":
        if os.path.exists(passive_ckpt_hfl):
            ckpt_path = passive_ckpt_hfl
        elif os.path.exists(passive_ckpt_local):
            ckpt_path = passive_ckpt_local
        else:
            raise FileNotFoundError(
                f"[PASSIVE] No checkpoint found. Tried:\n"
                f"  {passive_ckpt_hfl}\n  {passive_ckpt_local}"
            )
        state = torch.load(ckpt_path, map_location=dev)
        sd    = _unwrap_state_dict(state)
        m1.load_state_dict(sd, strict=True)
        cache["model_active"]  = m0
        cache["model_passive"] = m1
        print(f"[Client init][PASSIVE] loaded {ckpt_path}")

    else:
        raise RuntimeError(f"Unknown desired_role={desired_role}")

    # Freeze all encoder parameters after Tier 1 pre-training.
    for m in (cache["model_active"], cache["model_passive"]):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    cache[key] = True
    print(
        f"[Client ready] key={_cache_key(context)} fold={fold} "
        f"desired_role={desired_role} device={device} npz={npz}"
    )


def _get_split_indices(cache: Dict, split: str) -> torch.Tensor:
    """Return the index tensor for the requested split name."""
    if split == "train":
        return torch.from_numpy(cache["tr"]).long()
    elif split == "val":
        return torch.from_numpy(cache["va"]).long()
    elif split == "test":
        return torch.from_numpy(cache["te"]).long()
    else:
        raise ValueError(f"Unknown split: {split}")


# ---------------------------------------------------------------------------
# Flower ClientApp and handlers
# ---------------------------------------------------------------------------

app = ClientApp()


@app.query("send_embeddings")
def send_embeddings(msg: Message, context: Context) -> Message:
    """
    Stage A handler: return all embeddings for the requested split in one message.

    Called once per split by the server during Stage A. Applies the frozen
    encoder to the full split and returns the complete embedding matrix.
    This one-time transfer eliminates repeated embedding requests during
    Stage B, making the total communication cost equal to one forward pass
    per sample — matching the theoretical protocol cost in the paper.
    """
    role  = str(msg.content["config"].get("role",  "")).strip().lower()
    split = str(msg.content["config"].get("split", "train")).strip().lower()

    _init_if_needed(context, role)
    cache = _get_cache(context)
    dev   = cache["dev"]
    idx   = _get_split_indices(cache, split)

    if role == "active":
        X     = cache["X1"].to(dev)
        model = cache["model_active"]
    elif role == "passive":
        X     = cache["X2"].to(dev)
        model = cache["model_passive"]
    else:
        raise RuntimeError(f"send_embeddings: unexpected role={role}")

    with torch.no_grad():
        emb = model(
            X.index_select(0, idx.to(dev))
        ).detach().cpu().numpy().astype(np.float32)

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord({"embeddings": Array(emb)}),
            "config": _cfg({"n_samples": len(emb), "split": split}),
        }),
        reply_to=msg,
    )


@app.query("get_labels")
def get_labels(msg: Message, context: Context) -> Message:
    """
    Stage A handler: return all labels for the requested split.

    Only the active silo responds to label requests. Called once per split
    during Stage A. The passive silo has no access to labels at any point.
    """
    role  = str(msg.content["config"].get("role",  "")).strip().lower()
    split = str(msg.content["config"].get("split", "train")).strip().lower()

    _init_if_needed(context, role)

    if role != "active":
        raise RuntimeError("get_labels called on non-active node (role mismatch).")

    cache = _get_cache(context)
    idx   = _get_split_indices(cache, split)
    y     = cache["y"]

    with torch.no_grad():
        yb = y.index_select(0, idx).numpy().astype(np.int64)

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord({"y": Array(yb)}),
            "config": _cfg({"n_samples": len(yb), "split": split}),
        }),
        reply_to=msg,
    )
