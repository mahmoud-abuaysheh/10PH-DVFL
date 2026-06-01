# clientapp_vfl_glioma_decoupled.py
#
# Flower client application for the decoupled VFL glioma experiment.
#
# This client implements the silo-side behaviour for both the active and passive
# silos in Tier 2 of the 10PH-DVFL architecture for the glioma dataset.
#
# Roles handled by this client:
#   "active"  : Loads the pre-trained active-silo encoder. Responds to
#               embedding requests using X1 and label requests using y.
#               The active silo is the only party with access to labels.
#   "passive" : Loads the pre-trained passive-silo encoder (from local SSL
#               or HFL pre-training). Responds to embedding requests using X2.
#               The passive silo has no access to labels at any point.
#
# All encoders are frozen after Tier 1 pre-training. During Tier 2, clients
# only serve pre-computed embeddings. No gradients are received from the server.
#
# Key architectural differences from the diabetes decoupled client:
#   - BottomMLP: input -> 32 -> ReLU -> Dropout(0.0) -> 16 -> ReLU
#     instead of input -> 16 -> ReLU -> 8 -> ReLU
#   - The Dropout layer is included at dropout=0.0 to maintain checkpoint
#     key compatibility with the HFL pre-training scripts which use the same
#     BottomMLP definition with Dropout present.
#
# Checkpoint loading logic:
#   active  : loads pretrained_active_bottom_sup_fold{fold}.pt or
#             pretrained_active_bottom_ssl_fold{fold}.pt depending on
#             which pre-training mode was used
#   passive : loads pretrained_passive_bottom_hfl_fold{fold}.pt if
#             available (HFL pre-training), otherwise falls back to
#             pretrained_passive_bottom_ssl_fold{fold}.pt (local SSL)

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

# Module-level cache to persist loaded data and models across handler calls
# within the same client process, avoiding repeated disk reads.
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

class BottomMLP(nn.Module):
    """
    Bottom encoder used by each silo in the glioma decoupled VFL architecture.

    Projects silo-specific input features through two linear layers with
    ReLU activations to produce a 16-dimensional embedding vector.
    Architecture: input -> 32 -> ReLU -> Dropout -> 16 -> ReLU

    The Dropout layer is included at dropout=0.0 (disabled by default) to
    maintain checkpoint key compatibility with clientapp_hfl_passive_glioma.py
    and serverapp_hfl_passive_glioma.py which define BottomMLP with Dropout.
    If the Dropout layer were absent here, loading HFL-pre-trained checkpoints
    would fail due to key mismatch in the state dict.
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
    """
    Load the pre-computed cross-validation dataset from the NPZ file.
    Returns feature matrices X1 and X2, labels y, fold splits, and metadata.
    X1 contains active-silo features; X2 contains passive-silo features.
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

    Supports multiple checkpoint formats by checking for common key names
    before falling back to treating the entire dict as a state dict.
    Also handles DataParallel checkpoints by stripping 'module.' prefixes.
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
    DEFAULT_NPZ = Path(__file__).resolve().parent / "glioma_aligned_vfl_hfl_cv.npz"
    npz    = str(run.get("npz", str(DEFAULT_NPZ)))
    fold   = int(os.environ.get("FOLD", run.get("fold", 1)))
    seed   = int(run.get("seed", 42))
    device = str(run.get("device", "cpu"))
    out_dim = int(run.get("out_feature_dim", 16))

    # Read checkpoint directories from the Flower run configuration.
    active_dir  = str(run.get("active_ckpt_dir",  "./runs_sup_pretrain"))
    passive_dir = str(run.get("passive_ckpt_dir", "./runs_passive_ssl_glioma"))

    # Active encoder: supervised pre-training takes priority over SSL.
    active_ckpt_sup = os.path.join(
        active_dir, f"pretrained_active_bottom_sup_fold{fold}.pt"
    )
    active_ckpt_ssl = os.path.join(
        active_dir, f"pretrained_active_bottom_ssl_fold{fold}.pt"
    )

    # Passive encoder: HFL pre-training takes priority over local SSL.
    passive_ckpt_hfl = os.path.join(
        passive_dir, f"pretrained_passive_bottom_hfl_fold{fold}.pt"
    )
    passive_ckpt_local = os.path.join(
        passive_dir, f"pretrained_passive_bottom_ssl_fold{fold}.pt"
    )

    set_seed(seed)
    dev = torch.device(device)

    X1, X2, y, folds, meta = load_npz(npz)
    split_obj = folds[fold - 1]
    split     = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr        = split["train"].astype(np.int64)

    # Standardize both feature matrices using training split statistics only.
    X1s = _standardize_using_train(X1, tr)
    X2s = _standardize_using_train(X2, tr)

    cache["dev"]  = dev
    cache["meta"] = meta
    cache["fold"] = fold
    cache["X1"]   = torch.from_numpy(X1s).float().to(dev)
    cache["X2"]   = torch.from_numpy(X2s).float().to(dev)
    cache["y"]    = torch.from_numpy(y).long().to(dev)

    m0 = BottomMLP(in_dim=int(cache["X1"].shape[1]), out_dim=out_dim).to(dev)
    m1 = BottomMLP(in_dim=int(cache["X2"].shape[1]), out_dim=out_dim).to(dev)

    if desired_role == "active":
        # Load active-silo encoder; supervised checkpoint takes priority.
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
        cache["model_for_view0"] = m0
        cache["model_for_view1"] = m1  # Not used by the active role.
        print(f"[Client init][ACTIVE] loaded {ckpt_path}")

    elif desired_role == "passive":
        # Load passive-silo encoder; HFL checkpoint takes priority over local SSL.
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
        cache["model_for_view0"] = m0  # Not used by the passive role.
        cache["model_for_view1"] = m1
        print(f"[Client init][PASSIVE] loaded {ckpt_path}")

    else:
        raise RuntimeError(f"Unknown desired_role={desired_role}")

    # Freeze all encoder parameters after Tier 1 pre-training.
    # Encoders are never updated during Tier 2 fusion head training.
    for m in (cache["model_for_view0"], cache["model_for_view1"]):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    cache[key] = True
    print(
        f"[Client ready] key={_cache_key(context)} fold={fold} "
        f"desired_role={desired_role} device={device} npz={npz}"
    )


# ---------------------------------------------------------------------------
# Flower ClientApp and handlers
# ---------------------------------------------------------------------------

app = ClientApp()


@app.query("generate_embeddings")
def generate_embeddings(msg: Message, context: Context) -> Message:
    """
    Handle an embedding request from the server.

    Selects the correct feature matrix based on the view parameter:
      view=0 uses X1 (active silo features)
      view=1 uses X2 (passive silo features)
    Applies the corresponding frozen encoder and returns the resulting
    embeddings for the requested batch of sample indices.
    No gradients are computed or stored during this operation.
    """
    desired_role = str(msg.content["config"].get("role", "")).strip().lower()
    _init_if_needed(context, desired_role)
    cache = _get_cache(context)

    view      = int(msg.content["config"].get("view", 0))
    X         = cache["X1"] if view == 0 else cache["X2"]
    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    idx_t     = torch.from_numpy(batch_idx).long().to(X.device)
    model     = cache["model_for_view0"] if view == 0 else cache["model_for_view1"]

    with torch.no_grad():
        emb = model(X.index_select(0, idx_t)).detach().cpu().numpy().astype(np.float32)

    return Message(
        content=RecordDict({"arrays": ArrayRecord({"embedding": Array(emb)})}),
        reply_to=msg,
    )


@app.query("get_labels")
def get_labels(msg: Message, context: Context) -> Message:
    """
    Handle a label request from the server.

    Only the active silo is permitted to respond to label requests.
    Raises a RuntimeError if called on a passive node, which would
    indicate a role mismatch in the server logic.
    """
    desired_role = str(msg.content["config"].get("role", "")).strip().lower()
    _init_if_needed(context, desired_role)
    cache = _get_cache(context)

    if desired_role != "active":
        raise RuntimeError("get_labels called on non-active node (role mismatch).")

    y         = cache["y"]
    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    idx_t     = torch.from_numpy(batch_idx).long().to(y.device)

    with torch.no_grad():
        yb = y.index_select(0, idx_t).detach().cpu().numpy().astype(np.int64)

    return Message(
        content=RecordDict({"arrays": ArrayRecord({"y": Array(yb)})}),
        reply_to=msg,
    )
