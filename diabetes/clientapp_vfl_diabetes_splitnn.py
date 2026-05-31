# clientapp_vfl_diabetes_splitnn.py
# Flower client for the SplitNN-based VFL diabetes experiment.
# Each client owns one feature view and updates its bottom model using gradients
# received from the server-side top model.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import os
import torch
import torch.nn as nn

from flwr.app import Array, ArrayRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

_CACHE: Dict[str, Dict] = {}


def _cache_key(context: Context) -> str:
    nid = getattr(context, "node_id", None)
    return str(nid) if nid is not None else str(id(context))


def _get_cache(context: Context) -> Dict:
    k = _cache_key(context)
    if k not in _CACHE:
        _CACHE[k] = {}
    return _CACHE[k]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BottomMLP_Paper(nn.Module):
    """Bottom encoder used by each silo: input features -> 16 -> 8 with ReLU activations."""
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


def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    X1 = d["X1"].astype(np.float32)
    X2 = d["X2"].astype(np.float32)
    y = d["y"].astype(np.int64)
    folds = list(d["folds"])
    return X1, X2, y, folds


def _standardize_using_train(X: np.ndarray, tr_idx: np.ndarray) -> np.ndarray:
    mu = X[tr_idx].mean(axis=0, keepdims=True)
    sd = X[tr_idx].std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


def _init_if_needed(context: Context) -> None:
    cache = _get_cache(context)
    if cache.get("ready", False):
        return

    DEFAULT_NPZ = Path(__file__).resolve().parent / "diabetes_vfl_cv.npz"
    npz = str(context.run_config.get("npz", str(DEFAULT_NPZ)))
    fold = int(os.environ.get("FOLD", context.run_config.get("fold", 1)))

    seed = int(context.run_config.get("seed", 42))
    device = str(context.run_config.get("device", "cpu"))
    lr_bottom = float(context.run_config.get("lr_bottom", 1e-3))

    set_seed(seed)
    dev = torch.device(device)

    X1, X2, y, folds = load_npz(npz)
    split_obj = folds[fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)

    X1s = _standardize_using_train(X1, tr)
    X2s = _standardize_using_train(X2, tr)

    cache["dev"] = dev
    cache["lr_bottom"] = lr_bottom

    cache["X1"] = torch.from_numpy(X1s).float().to(dev)
    cache["X2"] = torch.from_numpy(X2s).float().to(dev)
    cache["y"] = torch.from_numpy(y).long().to(dev)

    cache["models"] = {}  # Bottom models are created separately for each feature view.
    cache["opts"] = {}

    cache["ready"] = True
    print(f"[Client init] key={_cache_key(context)} fold={fold} device={device} npz={npz}")


def _get_model_opt(context: Context, view: int, in_dim: int) -> Tuple[nn.Module, torch.optim.Optimizer]:
    cache = _get_cache(context)
    dev: torch.device = cache["dev"]
    lr = float(cache["lr_bottom"])

    if view not in cache["models"]:
        m = BottomMLP_Paper(in_dim=in_dim).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        cache["models"][view] = m
        cache["opts"][view] = opt
    return cache["models"][view], cache["opts"][view]


app = ClientApp()


@app.query("generate_embeddings")
def generate_embeddings(msg: Message, context: Context) -> Message:
    _init_if_needed(context)
    cache = _get_cache(context)

    cfg = msg.content.get("config", None)
    view = int(cfg.get("view", 0) if cfg is not None else 0)  # View 0 uses X1; view 1 uses X2.
    X = cache["X1"] if view == 0 else cache["X2"]

    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    idx_t = torch.from_numpy(batch_idx).long().to(X.device)

    model, _ = _get_model_opt(context, view=view, in_dim=int(X.shape[1]))
    model.eval()
    with torch.no_grad():
        emb = model(X.index_select(0, idx_t)).detach().cpu().numpy().astype(np.float32)

    arrs = ArrayRecord({"embedding": Array(emb)})
    return Message(content=RecordDict({"arrays": arrs}), reply_to=msg)


@app.query("get_labels")
def get_labels(msg: Message, context: Context) -> Message:
    """Return labels from the active silo for the requested batch."""
    _init_if_needed(context)
    cache = _get_cache(context)
    y = cache["y"]

    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    idx_t = torch.from_numpy(batch_idx).long().to(y.device)

    with torch.no_grad():
        yb = y.index_select(0, idx_t).detach().cpu().numpy().astype(np.int64)

    arrs = ArrayRecord({"y": Array(yb)})
    return Message(content=RecordDict({"arrays": arrs}), reply_to=msg)


@app.train("apply_gradients")
def apply_gradients(msg: Message, context: Context) -> Message:
    _init_if_needed(context)
    cache = _get_cache(context)

    cfg = msg.content.get("config", None)
    view = int(cfg.get("view", 0) if cfg is not None else 0)
    X = cache["X1"] if view == 0 else cache["X2"]

    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    grad_np = msg.content["arrays"]["local_gradients"].numpy().astype(np.float32)

    idx_t = torch.from_numpy(batch_idx).long().to(X.device)
    grad_t = torch.from_numpy(grad_np).float().to(X.device)

    model, opt = _get_model_opt(context, view=view, in_dim=int(X.shape[1]))
    model.train()
    opt.zero_grad(set_to_none=True)

    emb = model(X.index_select(0, idx_t))
    emb.backward(grad_t)
    opt.step()

    return Message(content=RecordDict(), reply_to=msg)


# Expose the same gradient-update step under the name expected by the server.
@app.train("backward")
def backward(msg: Message, context: Context) -> Message:
    return apply_gradients(msg, context)


@app.query("checkpoint_bottom")
def checkpoint_bottom(msg: Message, context: Context) -> Message:
    """Save the current bottom-model weights for the selected view."""
    _init_if_needed(context)
    cache = _get_cache(context)

    cfg = msg.content.get("config", None)
    view = int(cfg.get("view", 0) if cfg is not None else 0)
    fold = int(cfg.get("fold", 1) if cfg is not None else 1)
    out_dir = str(cfg.get("out_dir", ".") if cfg is not None else ".")

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"splitnn_best_bottom_fold{fold}_view{view}.pt")

    if view in cache.get("models", {}):
        torch.save(cache["models"][view].state_dict(), ckpt_path)
        print(f"[Client] Saved bottom view={view} fold={fold} -> {ckpt_path}")

    from flwr.app import ConfigRecord
    return Message(content=RecordDict({"config": ConfigRecord({"status": "saved", "path": ckpt_path})}), reply_to=msg)


@app.query("restore_best_bottom")
def restore_best_bottom(msg: Message, context: Context) -> Message:
    """Restore the bottom-model weights from the saved best checkpoint."""
    _init_if_needed(context)
    cache = _get_cache(context)

    cfg = msg.content.get("config", None)
    view = int(cfg.get("view", 0) if cfg is not None else 0)
    fold = int(cfg.get("fold", 1) if cfg is not None else 1)
    out_dir = str(cfg.get("out_dir", ".") if cfg is not None else ".")

    ckpt_path = os.path.join(out_dir, f"splitnn_best_bottom_fold{fold}_view{view}.pt")

    if os.path.exists(ckpt_path) and view in cache.get("models", {}):
        dev = cache["dev"]
        cache["models"][view].load_state_dict(
            torch.load(ckpt_path, map_location=dev, weights_only=False)
        )
        print(f"[Client] Restored bottom view={view} fold={fold} from {ckpt_path}")

    from flwr.app import ConfigRecord
    return Message(content=RecordDict({"config": ConfigRecord({"status": "restored", "path": ckpt_path})}), reply_to=msg)
