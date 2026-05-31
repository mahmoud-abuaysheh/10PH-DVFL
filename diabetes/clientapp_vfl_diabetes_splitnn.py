# clientapp_vfl_diabetes_splitnn.py
#
# Flower client application for the SplitNN VFL baseline in the diabetes
# decoupled VFL experiment.
#
# This client implements the silo-side behaviour for the SplitNN baseline
# (Condition 1), where each silo owns one feature view and trains its local
# bottom encoder end-to-end through gradient signals received from the server.
#
# SplitNN client behaviour per training batch:
#   1. Server requests cut-layer activations (generate_embeddings handler)
#   2. Server computes forward pass, loss, and backward pass
#   3. Server sends the gradient slice for this silo (backward handler)
#   4. Client applies the gradient to update its local bottom encoder
#
# Unlike the decoupled architecture, both activations and gradients cross
# the silo boundary every batch, making the encoder coupled to the server-side
# supervised objective throughout training.
#
# Checkpoint management:
#   The server instructs clients to save and restore encoder checkpoints
#   whenever the best validation AUROC is updated (checkpoint_bottom and
#   restore_best_bottom handlers). This ensures that the best encoder state
#   is used for final test evaluation after early stopping.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

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

class BottomMLP_Paper(nn.Module):
    """
    Bottom encoder used by each silo in the SplitNN VFL baseline.

    Projects silo-specific input features through two linear layers with
    ReLU activations to produce an 8-dimensional cut-layer output.
    Architecture: input -> 16 -> ReLU -> 8 -> ReLU

    In SplitNN, this encoder is updated every training batch through
    gradient signals received from the server-side top model.
    Must match the architecture used in the decoupled VFL client for
    fair comparison between conditions.
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
    Returns feature matrices X1 and X2, labels y, and fold splits.
    X1 contains active-silo features; X2 contains passive-silo features.
    """
    d = np.load(npz_path, allow_pickle=True)
    X1 = d["X1"].astype(np.float32)
    X2 = d["X2"].astype(np.float32)
    y  = d["y"].astype(np.int64)
    folds = list(d["folds"])
    return X1, X2, y, folds


def _standardize_using_train(X: np.ndarray, tr_idx: np.ndarray) -> np.ndarray:
    """
    Standardize features using mean and standard deviation computed from
    the training split only, preventing any leakage from validation or test sets.
    """
    mu = X[tr_idx].mean(axis=0, keepdims=True)
    sd = X[tr_idx].std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _init_if_needed(context: Context) -> None:
    """
    Initialise the client node by loading the dataset and preparing the
    feature matrices for both silos.

    This function is called at the start of each handler and is a no-op
    if the client has already been initialised. Both feature matrices X1
    and X2 are loaded and standardized so that the same client node can
    serve either the active or passive silo role depending on the view
    parameter sent by the server.

    Bottom encoder models are created lazily on first use via _get_model_opt
    to avoid creating unnecessary models for unused views.
    """
    cache = _get_cache(context)
    if cache.get("ready", False):
        return

    DEFAULT_NPZ = Path(__file__).resolve().parent / "diabetes_vfl_cv.npz"
    npz       = str(context.run_config.get("npz", str(DEFAULT_NPZ)))
    fold      = int(os.environ.get("FOLD", context.run_config.get("fold", 1)))
    seed      = int(context.run_config.get("seed", 42))
    device    = str(context.run_config.get("device", "cpu"))
    lr_bottom = float(context.run_config.get("lr_bottom", 1e-3))

    set_seed(seed)
    dev = torch.device(device)

    X1, X2, y, folds = load_npz(npz)
    split_obj = folds[fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)

    # Standardize both feature matrices using training split statistics only.
    X1s = _standardize_using_train(X1, tr)
    X2s = _standardize_using_train(X2, tr)

    cache["dev"]       = dev
    cache["lr_bottom"] = lr_bottom
    cache["X1"]        = torch.from_numpy(X1s).float().to(dev)
    cache["X2"]        = torch.from_numpy(X2s).float().to(dev)
    cache["y"]         = torch.from_numpy(y).long().to(dev)
    cache["models"]    = {}  # Bottom models created lazily per view.
    cache["opts"]      = {}
    cache["ready"]     = True

    print(f"[Client init] key={_cache_key(context)} fold={fold} device={device} npz={npz}")


def _get_model_opt(
    context: Context, view: int, in_dim: int
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Return the bottom encoder and optimizer for the given feature view,
    creating them on first use. View 0 corresponds to the active silo (X1)
    and view 1 to the passive silo (X2).
    """
    cache = _get_cache(context)
    dev   = cache["dev"]
    lr    = float(cache["lr_bottom"])

    if view not in cache["models"]:
        m   = BottomMLP_Paper(in_dim=in_dim).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        cache["models"][view] = m
        cache["opts"][view]   = opt

    return cache["models"][view], cache["opts"][view]


# ---------------------------------------------------------------------------
# Flower ClientApp and handlers
# ---------------------------------------------------------------------------

app = ClientApp()


@app.query("generate_embeddings")
def generate_embeddings(msg: Message, context: Context) -> Message:
    """
    Handle a cut-layer activation request from the server.

    Selects the correct feature matrix based on the view parameter:
      view=0 uses X1 (active silo features)
      view=1 uses X2 (passive silo features)
    Applies the bottom encoder in inference mode and returns the resulting
    cut-layer activations for the requested batch of sample indices.

    Note: In SplitNN, the encoder is in training mode during backward
    passes. It is switched to eval mode here for the forward pass only
    to avoid affecting batch normalisation statistics if present.
    """
    _init_if_needed(context)
    cache = _get_cache(context)

    cfg  = msg.content.get("config", None)
    view = int(cfg.get("view", 0) if cfg is not None else 0)
    X    = cache["X1"] if view == 0 else cache["X2"]

    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    idx_t     = torch.from_numpy(batch_idx).long().to(X.device)

    model, _ = _get_model_opt(context, view=view, in_dim=int(X.shape[1]))
    model.eval()
    with torch.no_grad():
        emb = model(X.index_select(0, idx_t)).detach().cpu().numpy().astype(np.float32)

    arrs = ArrayRecord({"embedding": Array(emb)})
    return Message(content=RecordDict({"arrays": arrs}), reply_to=msg)


@app.query("get_labels")
def get_labels(msg: Message, context: Context) -> Message:
    """
    Return labels from the active silo for the requested batch.

    Only the active silo node holds labels. This handler is called only
    on the active silo node; calling it on the passive node would indicate
    a server-side role assignment error.
    """
    _init_if_needed(context)
    cache = _get_cache(context)
    y     = cache["y"]

    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    idx_t     = torch.from_numpy(batch_idx).long().to(y.device)

    with torch.no_grad():
        yb = y.index_select(0, idx_t).detach().cpu().numpy().astype(np.int64)

    arrs = ArrayRecord({"y": Array(yb)})
    return Message(content=RecordDict({"arrays": arrs}), reply_to=msg)


@app.train("apply_gradients")
def apply_gradients(msg: Message, context: Context) -> Message:
    """
    Apply the gradient slice received from the server to update the local
    bottom encoder.

    The server computes the full backward pass and splits the gradient at
    the concatenation point into per-silo slices. This handler receives
    the gradient slice for this silo's feature view and uses it to update
    the local encoder via a standard backward pass through the cut layer.

    This is the core SplitNN update step that couples the local encoder
    to the server-side supervised objective through gradient exchange.
    """
    _init_if_needed(context)
    cache = _get_cache(context)

    cfg  = msg.content.get("config", None)
    view = int(cfg.get("view", 0) if cfg is not None else 0)
    X    = cache["X1"] if view == 0 else cache["X2"]

    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    grad_np   = msg.content["arrays"]["local_gradients"].numpy().astype(np.float32)

    idx_t  = torch.from_numpy(batch_idx).long().to(X.device)
    grad_t = torch.from_numpy(grad_np).float().to(X.device)

    model, opt = _get_model_opt(context, view=view, in_dim=int(X.shape[1]))
    model.train()
    opt.zero_grad(set_to_none=True)

    # Re-run the forward pass with gradient tracking enabled so that
    # the received gradient can be back-propagated through the encoder.
    emb = model(X.index_select(0, idx_t))
    emb.backward(grad_t)
    opt.step()

    return Message(content=RecordDict(), reply_to=msg)


@app.train("backward")
def backward(msg: Message, context: Context) -> Message:
    """
    Alias for apply_gradients using the handler name expected by the server.
    The server sends gradient messages with message_type='train.backward'.
    """
    return apply_gradients(msg, context)


@app.query("checkpoint_bottom")
def checkpoint_bottom(msg: Message, context: Context) -> Message:
    """
    Save the current bottom encoder weights for the selected view.

    Called by the server whenever a new best validation AUROC is achieved,
    so that the best encoder state can be restored after early stopping
    before final test evaluation.
    """
    _init_if_needed(context)
    cache = _get_cache(context)

    cfg     = msg.content.get("config", None)
    view    = int(cfg.get("view",    0) if cfg is not None else 0)
    fold    = int(cfg.get("fold",    1) if cfg is not None else 1)
    out_dir = str(cfg.get("out_dir", ".") if cfg is not None else ".")

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"splitnn_best_bottom_fold{fold}_view{view}.pt")

    if view in cache.get("models", {}):
        torch.save(cache["models"][view].state_dict(), ckpt_path)
        print(f"[Client] Saved bottom view={view} fold={fold} -> {ckpt_path}")

    return Message(
        content=RecordDict({"config": ConfigRecord({
            "status": "saved", "path": ckpt_path
        })}),
        reply_to=msg,
    )


@app.query("restore_best_bottom")
def restore_best_bottom(msg: Message, context: Context) -> Message:
    """
    Restore the bottom encoder weights from the saved best checkpoint.

    Called by the server after early stopping to restore the encoder state
    that achieved the best validation AUROC before final test evaluation.
    """
    _init_if_needed(context)
    cache = _get_cache(context)

    cfg     = msg.content.get("config", None)
    view    = int(cfg.get("view",    0) if cfg is not None else 0)
    fold    = int(cfg.get("fold",    1) if cfg is not None else 1)
    out_dir = str(cfg.get("out_dir", ".") if cfg is not None else ".")

    ckpt_path = os.path.join(out_dir, f"splitnn_best_bottom_fold{fold}_view{view}.pt")

    if os.path.exists(ckpt_path) and view in cache.get("models", {}):
        dev = cache["dev"]
        cache["models"][view].load_state_dict(
            torch.load(ckpt_path, map_location=dev, weights_only=False)
        )
        print(f"[Client] Restored bottom view={view} fold={fold} from {ckpt_path}")

    return Message(
        content=RecordDict({"config": ConfigRecord({
            "status": "restored", "path": ckpt_path
        })}),
        reply_to=msg,
    )
