# clientapp_vfl_diabetes_decoupled.py
#
# Flower client application for the decoupled VFL diabetes experiment.
#
# This client implements the silo-side behaviour for both the active and passive
# silos in Tier 2 of the 10PH-DVFL architecture, and optionally supports
# intra-silo HFL pre-training for the passive silo in Tier 1.
#
# Two-stage protocol:
#   Stage A — send_embeddings handler:
#     Called once per split (train/val/test) by the server. Returns the full
#     embedding matrix for the requested split in a single message. Since
#     encoders are frozen after Tier 1, embeddings are identical for every
#     training round. Sending them once eliminates repeated cross-silo
#     communication during Stage B.
#   Stage B — no handlers called:
#     The server trains the fusion head entirely on cached embeddings.
#     No messages are sent to any client node during Stage B.
#
# Roles handled by this client:
#   "active"      : Loads the pre-trained active-silo encoder. Responds to
#                   send_embeddings requests using X1 and get_labels requests
#                   using y for the requested split.
#   "passive"     : Loads the pre-trained passive-silo encoder. Responds to
#                   send_embeddings requests using X2.
#   "passive_hfl" : Initialises a randomly initialised passive encoder and
#                   trains it locally through the hfl_fit handler.

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

# Module-level cache to persist loaded data and models across handler calls.
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
    """Fix all random seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class BottomMLP_Paper(nn.Module):
    """
    Bottom encoder used by each silo in the decoupled VFL architecture.

    Projects silo-specific input features through two linear layers with
    ReLU activations to produce an 8-dimensional embedding vector.
    Architecture: input -> 16 -> ReLU -> 8 -> ReLU
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_npz(npz_path: str):
    """
    Load the pre-computed cross-validation dataset from the NPZ file.
    Returns feature matrices X1 and X2, labels y, fold splits, and metadata.
    """
    d     = np.load(npz_path, allow_pickle=True)
    X1    = d["X1"].astype(np.float32)
    X2    = d["X2"].astype(np.float32)
    y     = d["y"].astype(np.int64)
    folds = list(d["folds"])
    meta  = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return X1, X2, y, folds, meta


def _standardize_using_train(X: np.ndarray, tr_idx: np.ndarray) -> np.ndarray:
    """
    Standardize features using mean and standard deviation computed from
    the training split only, preventing any leakage from validation or test sets.
    """
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
    """
    Extract the encoder state dict from a checkpoint file.
    Supports multiple checkpoint formats and handles DataParallel checkpoints.
    """
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


def _state_dict_to_arrays(
    sd: Dict[str, torch.Tensor]
) -> Dict[str, Array]:
    """Convert a PyTorch state dict to a dict of Flower Arrays for transmission."""
    return {
        k: Array(v.detach().cpu().numpy().astype(np.float32))
        for k, v in sd.items()
    }


def _arrays_to_state_dict(
    arrs: Dict[str, Array], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Convert a dict of Flower Arrays back to a PyTorch state dict."""
    return {
        k: torch.from_numpy(arr.numpy()).to(device)
        for k, arr in arrs.items()
    }


def _two_view_augment(
    x: torch.Tensor, noise_std: float = 0.05, dropout_p: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create two augmented tabular views for self-supervised HFL pre-training.
    Each view is independently augmented with Gaussian noise and feature dropout.
    """
    def aug(t: torch.Tensor) -> torch.Tensor:
        if noise_std > 0:
            t = t + noise_std * torch.randn_like(t)
        if dropout_p > 0:
            mask = (torch.rand_like(t) > dropout_p).float()
            t    = t * mask
        return t
    return aug(x), aug(x)


def _neg_cos_sim(
    z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Negative cosine similarity used as the self-supervised HFL pre-training loss.
    """
    z1 = z1 / (z1.norm(dim=1, keepdim=True) + eps)
    z2 = z2 / (z2.norm(dim=1, keepdim=True) + eps)
    return 1.0 - (z1 * z2).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _init_if_needed(context: Context, desired_role: str) -> None:
    """
    Initialise the client node for the role requested by the server.

    Called at the start of each handler. Is a no-op if the client has already
    been initialised for the requested role. Loads the dataset, applies
    fold-specific standardization, and loads the appropriate encoder checkpoint.

    Checkpoint priority:
      active  : supervised checkpoint takes priority over SSL checkpoint
      passive : HFL checkpoint takes priority over local SSL checkpoint
    """
    cache = _get_cache(context)
    key   = f"ready::{desired_role}"
    if cache.get(key, False):
        return

    run    = context.run_config
    DEFAULT_NPZ = Path(__file__).resolve().parent / "diabetes_vfl_cv.npz"
    npz    = str(run.get("npz", str(DEFAULT_NPZ)))
    fold   = int(os.environ.get("FOLD", run.get("fold", 1)))
    seed   = int(run.get("seed", 42))
    device = str(run.get("device", "cpu"))

    active_dir  = str(run.get("active_ckpt_dir",  "./runs_active_ssl_diabetes"))
    passive_dir = str(run.get("passive_ckpt_dir", "./runs_passive_ssl_diabetes"))

    active_ckpt_sup   = os.path.join(active_dir,  f"pretrained_active_sup_fold{fold}.pt")
    active_ckpt_ssl   = os.path.join(active_dir,  f"pretrained_active_bottom_ssl_fold{fold}.pt")
    passive_ckpt_hfl  = os.path.join(passive_dir, f"pretrained_passive_bottom_hfl_fold{fold}.pt")
    passive_ckpt_local = os.path.join(passive_dir, f"pretrained_passive_bottom_ssl_fold{fold}.pt")

    set_seed(seed)
    dev = torch.device(device)

    X1, X2, y, folds, meta = load_npz(npz)
    split_obj = folds[fold - 1]
    split_    = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split_["train"].astype(np.int64)
    va = split_["val"].astype(np.int64)
    te = split_["test"].astype(np.int64)

    # Standardize using training split statistics only.
    X1s = _standardize_using_train(X1, tr)
    X2s = _standardize_using_train(X2, tr)

    cache["dev"]  = dev
    cache["meta"] = meta
    cache["fold"] = fold
    cache["tr"]   = tr
    cache["va"]   = va
    cache["te"]   = te

    # Store full standardized feature matrices for all splits.
    # Stage A requests embeddings for the full split at once.
    cache["X1"] = torch.from_numpy(X1s).float()
    cache["X2"] = torch.from_numpy(X2s).float()
    cache["y"]  = torch.from_numpy(y).long()

    m0 = BottomMLP_Paper(in_dim=int(X1.shape[1])).to(dev)
    m1 = BottomMLP_Paper(in_dim=int(X2.shape[1])).to(dev)

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

    elif desired_role == "passive_hfl":
        cache["model_active"]  = m0
        cache["model_passive"] = m1
        for p in cache["model_passive"].parameters():
            p.requires_grad_(True)
        cache["model_passive"].train()
        print("[Client init][PASSIVE_HFL] initialized random bottom")
    else:
        raise RuntimeError(f"Unknown desired_role={desired_role}")

    # Freeze encoders after Tier 1 pre-training.
    if desired_role != "passive_hfl":
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

    Called once per split (train/val/test) by the server during Stage A.
    Applies the frozen encoder to the full split and returns the complete
    embedding matrix. This eliminates the need for repeated per-batch
    embedding requests during Stage B fusion head training.

    The communication cost of this one-time transfer equals exactly one
    forward pass per sample — matching the theoretical protocol cost.
    """
    role  = str(msg.content["config"].get("role",  "")).strip().lower()
    split = str(msg.content["config"].get("split", "train")).strip().lower()

    _init_if_needed(context, role)
    cache = _get_cache(context)
    dev   = cache["dev"]

    idx = _get_split_indices(cache, split)

    if role == "active":
        X     = cache["X1"].to(dev)
        model = cache["model_active"]
    elif role == "passive":
        X     = cache["X2"].to(dev)
        model = cache["model_passive"]
    else:
        raise RuntimeError(f"send_embeddings: unexpected role={role}")

    # Generate embeddings for the full split in one forward pass.
    # Encoders are frozen so no gradients are needed.
    with torch.no_grad():
        emb = model(X.index_select(0, idx.to(dev))).detach().cpu().numpy().astype(np.float32)

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

    Called once per split by the server during Stage A to retrieve labels
    from the active silo. The passive silo has no access to labels at any point.
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


@app.query("hfl_fit")
def hfl_fit(msg: Message, context: Context) -> Message:
    """
    Handle one local HFL training round for passive-silo self-supervised
    pre-training (Tier 1 optional stage).

    Receives global encoder weights, trains locally on the assigned partition
    using two-view cosine similarity, and returns updated weights.
    """
    cfg          = msg.content["config"]
    desired_role = str(cfg.get("role", "")).strip().lower()
    if desired_role != "passive_hfl":
        raise RuntimeError(
            f"hfl_fit called with role={desired_role}, expected passive_hfl"
        )

    _init_if_needed(context, desired_role)
    cache = _get_cache(context)
    dev   = cache["dev"]

    client_rank = int(cfg.get("client_rank", 0))
    K           = int(cfg.get("K", 1))
    X2          = cache["X2"].to(dev)
    fold        = int(os.environ.get("FOLD", context.run_config.get("fold", 1)))

    run         = context.run_config
    DEFAULT_NPZ = Path(__file__).resolve().parent / "diabetes_vfl_cv.npz"
    npz         = str(run.get("npz", str(DEFAULT_NPZ)))
    _, _, _, folds, _ = load_npz(npz)
    split_obj   = folds[fold - 1]
    split_      = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr          = split_["train"].astype(np.int64)

    parts = np.array_split(tr, K)
    if client_rank < 0 or client_rank >= len(parts):
        raise RuntimeError(
            f"client_rank={client_rank} out of range for K={K}"
        )
    local_idx = parts[client_rank].astype(np.int64)

    model = cache["model_passive"]
    model.to(dev)

    if "arrays" in msg.content and "global_weights" in msg.content["arrays"]:
        gw = msg.content["arrays"]["global_weights"]
        sd = _arrays_to_state_dict(gw, dev)
        model.load_state_dict(sd, strict=True)

    lr           = float(cfg.get("lr", 1e-3))
    local_epochs = int(cfg.get("local_epochs", 1))
    batch        = int(cfg.get("batch", 256))
    noise_std    = float(cfg.get("noise_std", 0.05))
    dropout_p    = float(cfg.get("dropout_p", 0.1))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n      = len(local_idx)
    model.train()
    losses = []

    for _ in range(local_epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch):
            idx = local_idx[perm[start : start + batch]]
            xb  = X2.index_select(
                0, torch.from_numpy(idx).long().to(dev)
            )
            x1, x2 = _two_view_augment(xb, noise_std=noise_std, dropout_p=dropout_p)
            loss    = _neg_cos_sim(model(x1), model(x2))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

    avg_loss  = float(np.mean(losses)) if losses else 0.0
    out_sd    = model.state_dict()
    out_arrs  = ArrayRecord({
        "global_weights": _state_dict_to_arrays(out_sd)
    })
    out_cfg   = _cfg({
        "num_examples": int(n),
        "loss": avg_loss,
        "client_rank": client_rank,
        "K": K,
    })
    return Message(
        content=RecordDict({"arrays": out_arrs, "config": out_cfg}),
        reply_to=msg,
    )
