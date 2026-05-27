# clientapp_vfl_diabetes_decoupled.py
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
    """Baseline bottom: in_dim -> 16 -> 8 with ReLU."""
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
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return X1, X2, y, folds, meta


def _standardize_using_train(X: np.ndarray, tr_idx: np.ndarray) -> np.ndarray:
    mu = X[tr_idx].mean(axis=0, keepdims=True)
    sd = X[tr_idx].std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


def _cfg(d: Dict[str, object] | None = None) -> ConfigRecord:
    return ConfigRecord(d or {})


def _unwrap_state_dict(state: object) -> Dict[str, torch.Tensor]:
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



def _state_dict_to_arrays(sd: Dict[str, torch.Tensor]) -> Dict[str, Array]:
    return {k: Array(v.detach().cpu().numpy().astype(np.float32)) for k, v in sd.items()}

def _arrays_to_state_dict(arrs: Dict[str, Array], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(arr.numpy()).to(device) for k, arr in arrs.items()}

def _two_view_augment(x: torch.Tensor, noise_std: float = 0.05, dropout_p: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    # tabular augmentation: gaussian noise + feature dropout
    def aug(t: torch.Tensor) -> torch.Tensor:
        if noise_std > 0:
            t = t + noise_std * torch.randn_like(t)
        if dropout_p > 0:
            mask = (torch.rand_like(t) > dropout_p).float()
            t = t * mask
        return t
    return aug(x), aug(x)

def _neg_cos_sim(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    z1 = z1 / (z1.norm(dim=1, keepdim=True) + eps)
    z2 = z2 / (z2.norm(dim=1, keepdim=True) + eps)
    return 1.0 - (z1 * z2).sum(dim=1).mean()


def _init_if_needed(context: Context, desired_role: str) -> None:
    """
    desired_role is provided by server per-request:
      - "active": must load active bottom into m0 (X1)
      - "passive": must load passive bottom into m1 (X2)
    """
    cache = _get_cache(context)
    key = f"ready::{desired_role}"
    if cache.get(key, False):
        return

    run = context.run_config

    DEFAULT_NPZ = Path(__file__).resolve().parent / "diabetes_vfl_cv.npz"
    npz = str(run.get("npz", str(DEFAULT_NPZ)))
    fold = int(os.environ.get("FOLD", run.get("fold", 1)))

    seed = int(run.get("seed", 42))
    device = str(run.get("device", "cpu"))

    # Use dirs from run-config (this matches what you pass)
    active_dir = str(run.get("active_ckpt_dir", "./runs_active_ssl_diabetes"))
    passive_dir = str(run.get("passive_ckpt_dir", "./runs_passive_ssl_diabetes"))

    active_ckpt = os.path.join(active_dir, f"pretrained_active_bottom_ssl_fold{fold}.pt")
    # passive could be local SSL or HFL, but we standardize filename for both:
    # - local SSL script saves: pretrained_passive_bottom_ssl_fold{fold}.pt
    # - HFL script saves: pretrained_passive_bottom_hfl_fold{fold}.pt
    passive_ckpt_local = os.path.join(passive_dir, f"pretrained_passive_bottom_ssl_fold{fold}.pt")
    passive_ckpt_hfl = os.path.join(passive_dir, f"pretrained_passive_bottom_hfl_fold{fold}.pt")

    set_seed(seed)
    dev = torch.device(device)

    X1, X2, y, folds, meta = load_npz(npz)
    split_obj = folds[fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)

    X1s = _standardize_using_train(X1, tr)
    X2s = _standardize_using_train(X2, tr)

    cache["dev"] = dev
    cache["meta"] = meta
    cache["fold"] = fold
    cache["X1"] = torch.from_numpy(X1s).float().to(dev)
    cache["X2"] = torch.from_numpy(X2s).float().to(dev)
    cache["y"] = torch.from_numpy(y).long().to(dev)

    m0 = BottomMLP_Paper(in_dim=int(cache["X1"].shape[1])).to(dev)
    m1 = BottomMLP_Paper(in_dim=int(cache["X2"].shape[1])).to(dev)

    if desired_role == "active":
        if not os.path.exists(active_ckpt):
            raise FileNotFoundError(f"[ACTIVE] checkpoint not found: {active_ckpt}")
        state = torch.load(active_ckpt, map_location=dev)
        sd = _unwrap_state_dict(state)
        m0.load_state_dict(sd, strict=True)
        cache["model_for_view0"] = m0
        cache["model_for_view1"] = m1  # unused
        print(f"[Client init][ACTIVE] loaded {active_ckpt}")

    elif desired_role == "passive":
        ckpt_path = passive_ckpt_hfl if os.path.exists(passive_ckpt_hfl) else passive_ckpt_local
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"[PASSIVE] checkpoint not found. Tried:\n  {passive_ckpt_hfl}\n  {passive_ckpt_local}"
            )
        state = torch.load(ckpt_path, map_location=dev)
        sd = _unwrap_state_dict(state)
        m1.load_state_dict(sd, strict=True)
        cache["model_for_view0"] = m0  # unused
        cache["model_for_view1"] = m1
        print(f"[Client init][PASSIVE] loaded {ckpt_path}")

    elif desired_role == "passive_hfl":
        # start from random init; training happens via hfl_fit query
        cache["model_for_view0"] = m0  # unused
        cache["model_for_view1"] = m1  # trainable
        # keep requires_grad True for view1 model (passive)
        for p in cache["model_for_view1"].parameters():
            p.requires_grad_(True)
        cache["model_for_view1"].train()
        print("[Client init][PASSIVE_HFL] initialized random bottom (will be trained via HFL)")
    else:
        raise RuntimeError(f"Unknown desired_role={desired_role}")

    if desired_role != "passive_hfl":
        for m in (cache["model_for_view0"], cache["model_for_view1"]):
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()

    cache[key] = True
    print(f"[Client ready] key={_cache_key(context)} fold={fold} desired_role={desired_role} device={device} npz={npz}")


app = ClientApp()


@app.query("generate_embeddings")
def generate_embeddings(msg: Message, context: Context) -> Message:
    desired_role = str(msg.content["config"].get("role", "")).strip().lower()
    _init_if_needed(context, desired_role)
    cache = _get_cache(context)

    view = int(msg.content["config"].get("view", 0))  # 0->X1(active), 1->X2(passive)
    X = cache["X1"] if view == 0 else cache["X2"]

    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    idx_t = torch.from_numpy(batch_idx).long().to(X.device)

    model = cache["model_for_view0"] if view == 0 else cache["model_for_view1"]
    with torch.no_grad():
        emb = model(X.index_select(0, idx_t)).detach().cpu().numpy().astype(np.float32)

    arrs = ArrayRecord({"embedding": Array(emb)})
    return Message(content=RecordDict({"arrays": arrs}), reply_to=msg)


@app.query("get_labels")
def get_labels(msg: Message, context: Context) -> Message:
    desired_role = str(msg.content["config"].get("role", "")).strip().lower()
    _init_if_needed(context, desired_role)
    cache = _get_cache(context)

    if desired_role != "active":
        raise RuntimeError("get_labels called on non-active node (role mismatch).")

    y = cache["y"]
    batch_idx = msg.content["arrays"]["batch_idx"].numpy().astype(np.int64)
    idx_t = torch.from_numpy(batch_idx).long().to(y.device)

    with torch.no_grad():
        yb = y.index_select(0, idx_t).detach().cpu().numpy().astype(np.int64)

    arrs = ArrayRecord({"y": Array(yb)})
    return Message(content=RecordDict({"arrays": arrs}), reply_to=msg)



@app.query("hfl_fit")
def hfl_fit(msg: Message, context: Context) -> Message:
    """Passive-only HFL SSL pretraining (FedAvg) for K clients.

    Server sends:
      - config: role="passive_hfl", client_rank, K, lr, local_epochs, batch, noise_std, dropout_p
      - arrays: global_weights (state_dict tensors as arrays)
    Client returns:
      - arrays: updated_weights
      - config: num_examples, loss
    """
    cfg = msg.content["config"]
    desired_role = str(cfg.get("role", "")).strip().lower()
    if desired_role != "passive_hfl":
        raise RuntimeError(f"hfl_fit called with role={desired_role}, expected passive_hfl")

    _init_if_needed(context, desired_role)
    cache = _get_cache(context)
    dev: torch.device = cache["dev"]

    # identify this client's local partition
    client_rank = int(cfg.get("client_rank", 0))
    K = int(cfg.get("K", 1))
    X2: torch.Tensor = cache["X2"]
    fold = int(os.environ.get("FOLD", context.run_config.get("fold", 1)))

    # recover train indices from folds (already loaded during init)
    # we stored standardized full X2; now we need fold->train indices again
    # easiest: reload from cache meta? we can recompute by reloading npz once
    run = context.run_config
    DEFAULT_NPZ = Path(__file__).resolve().parent / "diabetes_vfl_cv.npz"
    npz = str(run.get("npz", str(DEFAULT_NPZ)))
    _, _, _, folds, _ = load_npz(npz)
    split_obj = folds[fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)

    parts = np.array_split(tr, K)
    if client_rank < 0 or client_rank >= len(parts):
        raise RuntimeError(f"client_rank={client_rank} out of range for K={K}")
    local_idx = parts[client_rank].astype(np.int64)

    model: nn.Module = cache["model_for_view1"]
    model.to(dev)

    # load global weights if provided
    if "arrays" in msg.content and "global_weights" in msg.content["arrays"]:
        gw = msg.content["arrays"]["global_weights"]
        sd = _arrays_to_state_dict(gw, dev)
        model.load_state_dict(sd, strict=True)

    lr = float(cfg.get("lr", 1e-3))
    local_epochs = int(cfg.get("local_epochs", 1))
    batch = int(cfg.get("batch", 256))
    noise_std = float(cfg.get("noise_std", 0.05))
    dropout_p = float(cfg.get("dropout_p", 0.1))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # train: cosine agreement between two augmented views (no labels)
    n = len(local_idx)
    model.train()
    losses = []
    # shuffle each epoch
    for _ in range(local_epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch):
            idx = local_idx[perm[start:start+batch]]
            xb = X2.index_select(0, torch.from_numpy(idx).long().to(dev))
            x1, x2 = _two_view_augment(xb, noise_std=noise_std, dropout_p=dropout_p)
            z1 = model(x1)
            z2 = model(x2)
            loss = _neg_cos_sim(z1, z2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

    avg_loss = float(np.mean(losses)) if losses else 0.0

    # return updated weights
    out_sd = model.state_dict()
    out_arrs = ArrayRecord({"global_weights": _state_dict_to_arrays(out_sd)})
    out_cfg = _cfg({"num_examples": int(n), "loss": avg_loss, "client_rank": client_rank, "K": K})
    return Message(content=RecordDict({"arrays": out_arrs, "config": out_cfg}), reply_to=msg)
