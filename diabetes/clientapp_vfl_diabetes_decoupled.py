# clientapp_vfl_diabetes_decoupled.py
#
# Flower client application for the decoupled VFL diabetes experiment.
#
# This client implements the silo-side behaviour for both the active and passive
# silos in Tier 2 of the 10PH-DVFL architecture, and optionally supports
# intra-silo HFL pre-training for the passive silo in Tier 1.
#
# Roles handled by this client:
#   "active"      : Loads the pre-trained active-silo encoder. Responds to
#                   embedding requests using X1 and label requests using y.
#                   The active silo is the only party with access to labels.
#   "passive"     : Loads the pre-trained passive-silo encoder (from local SSL
#                   or HFL pre-training). Responds to embedding requests using X2.
#                   The passive silo has no access to labels at any point.
#   "passive_hfl" : Initialises a randomly initialised passive encoder and
#                   trains it locally through the hfl_fit handler using a
#                   self-supervised two-view augmentation objective.
#
# All encoders are frozen after Tier 1 pre-training. During Tier 2, clients
# only serve pre-computed embeddings; no gradients are received from the server.

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


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_npz(npz_path: str):
    """
    Load the pre-computed cross-validation dataset from the NPZ file.
    Returns feature matrices X1 and X2, labels y, fold splits, and metadata.
    X1 contains active-silo features; X2 contains passive-silo features.
    """
    d = np.load(npz_path, allow_pickle=True)
    X1 = d["X1"].astype(np.float32)
    X2 = d["X2"].astype(np.float32)
    y = d["y"].astype(np.int64)
    folds = list(d["folds"])
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return X1, X2, y, folds, meta


def _standardize_using_train(X: np.ndarray, tr_idx: np.ndarray) -> np.ndarray:
    """
    Standardize features using mean and standard deviation computed from
    the training split only, preventing any leakage from validation or test sets.
    """
    mu = X[tr_idx].mean(axis=0, keepdims=True)
    sd = X[tr_idx].std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


def _cfg(d: Dict[str, object] | None = None) -> ConfigRecord:
    """Wrap a plain dict as a Flower ConfigRecord."""
    return ConfigRecord(d or {})


def _unwrap_state_dict(state: object) -> Dict[str, torch.Tensor]:
    """
    Extract the encoder state dict from a checkpoint file.

    Supports multiple checkpoint formats by checking for common key names
    ('bottom_state', 'model_state', 'state_dict', 'model') before falling
    back to treating the entire dict as a state dict. Also handles DataParallel
    checkpoints by stripping 'module.' prefixes from parameter names.
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


def _state_dict_to_arrays(sd: Dict[str, torch.Tensor]) -> Dict[str, Array]:
    """Convert a PyTorch state dict to a dict of Flower Arrays for transmission."""
    return {k: Array(v.detach().cpu().numpy().astype(np.float32)) for k, v in sd.items()}


def _arrays_to_state_dict(arrs: Dict[str, Array], device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert a dict of Flower Arrays back to a PyTorch state dict on the given device."""
    return {k: torch.from_numpy(arr.numpy()).to(device) for k, arr in arrs.items()}


def _two_view_augment(
    x: torch.Tensor, noise_std: float = 0.05, dropout_p: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create two augmented tabular views of the same input batch for
    self-supervised contrastive pre-training.

    Each view is independently augmented using:
      - Gaussian noise addition (controlled by noise_std)
      - Random feature dropout masking (controlled by dropout_p)
    The encoder is trained to produce similar representations for both views
    of the same sample, encouraging it to learn robust features.
    """
    def aug(t: torch.Tensor) -> torch.Tensor:
        if noise_std > 0:
            t = t + noise_std * torch.randn_like(t)
        if dropout_p > 0:
            mask = (torch.rand_like(t) > dropout_p).float()
            t = t * mask
        return t
    return aug(x), aug(x)


def _neg_cos_sim(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the negative cosine similarity between two embedding tensors.
    Used as the self-supervised loss during HFL passive encoder pre-training.
    Minimising this loss encourages the encoder to produce similar embeddings
    for different augmented views of the same sample.
    """
    z1 = z1 / (z1.norm(dim=1, keepdim=True) + eps)
    z2 = z2 / (z2.norm(dim=1, keepdim=True) + eps)
    return 1.0 - (z1 * z2).sum(dim=1).mean()


def _init_if_needed(context: Context, desired_role: str) -> None:
    """
    Initialise the client node for the role requested by the server.

    This function is called at the start of each handler and is a no-op
    if the client has already been initialised for the requested role.
    Initialisation loads the dataset, applies fold-specific standardization,
    and loads the appropriate pre-trained encoder checkpoint.

    Checkpoint loading logic:
      active      : loads pretrained_active_sup_fold{fold}.pt or
                    pretrained_active_bottom_ssl_fold{fold}.pt depending
                    on which pre-training mode was used
      passive     : loads pretrained_passive_bottom_hfl_fold{fold}.pt if
                    available (HFL pre-training), otherwise falls back to
                    pretrained_passive_bottom_ssl_fold{fold}.pt (local SSL)
      passive_hfl : initialises a randomly initialised encoder for HFL training
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

    # Read checkpoint directories from the Flower run configuration.
    active_dir = str(run.get("active_ckpt_dir", "./runs_active_ssl_diabetes"))
    passive_dir = str(run.get("passive_ckpt_dir", "./runs_passive_ssl_diabetes"))

    # Active encoder checkpoint: supervised pre-training takes priority over SSL.
    active_ckpt_sup = os.path.join(active_dir, f"pretrained_active_sup_fold{fold}.pt")
    active_ckpt_ssl = os.path.join(active_dir, f"pretrained_active_bottom_ssl_fold{fold}.pt")

    # Passive encoder checkpoint: HFL pre-training takes priority over local SSL.
    passive_ckpt_hfl = os.path.join(passive_dir, f"pretrained_passive_bottom_hfl_fold{fold}.pt")
    passive_ckpt_local = os.path.join(passive_dir, f"pretrained_passive_bottom_ssl_fold{fold}.pt")

    set_seed(seed)
    dev = torch.device(device)

    X1, X2, y, folds, meta = load_npz(npz)
    split_obj = folds[fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)

    # Standardize features using training split statistics only.
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
        # Load the active-silo encoder. Supervised pre-training checkpoint
        # takes priority; falls back to SSL checkpoint if not found.
        if os.path.exists(active_ckpt_sup):
            ckpt_path = active_ckpt_sup
        elif os.path.exists(active_ckpt_ssl):
            ckpt_path = active_ckpt_ssl
        else:
            raise FileNotFoundError(
                f"[ACTIVE] No checkpoint found. Tried:\n  {active_ckpt_sup}\n  {active_ckpt_ssl}"
            )
        state = torch.load(ckpt_path, map_location=dev)
        sd = _unwrap_state_dict(state)
        m0.load_state_dict(sd, strict=True)
        cache["model_for_view0"] = m0
        cache["model_for_view1"] = m1  # Not used by the active role.
        print(f"[Client init][ACTIVE] loaded {ckpt_path}")

    elif desired_role == "passive":
        # Load the passive-silo encoder. HFL checkpoint takes priority
        # over local SSL checkpoint if both are present.
        if os.path.exists(passive_ckpt_hfl):
            ckpt_path = passive_ckpt_hfl
        elif os.path.exists(passive_ckpt_local):
            ckpt_path = passive_ckpt_local
        else:
            raise FileNotFoundError(
                f"[PASSIVE] No checkpoint found. Tried:\n  {passive_ckpt_hfl}\n  {passive_ckpt_local}"
            )
        state = torch.load(ckpt_path, map_location=dev)
        sd = _unwrap_state_dict(state)
        m1.load_state_dict(sd, strict=True)
        cache["model_for_view0"] = m0  # Not used by the passive role.
        cache["model_for_view1"] = m1
        print(f"[Client init][PASSIVE] loaded {ckpt_path}")

    elif desired_role == "passive_hfl":
        # Initialise a randomly initialised passive encoder for HFL pre-training.
        # Weights will be replaced by the global model sent by the server
        # at the start of each HFL round.
        cache["model_for_view0"] = m0  # Not used by the passive HFL role.
        cache["model_for_view1"] = m1  # Trainable passive encoder.
        for p in cache["model_for_view1"].parameters():
            p.requires_grad_(True)
        cache["model_for_view1"].train()
        print("[Client init][PASSIVE_HFL] initialized random bottom (will be trained via HFL)")
    else:
        raise RuntimeError(f"Unknown desired_role={desired_role}")

    # Freeze all encoder parameters after Tier 1 pre-training.
    # Encoders are never updated during Tier 2 fusion head training.
    if desired_role != "passive_hfl":
        for m in (cache["model_for_view0"], cache["model_for_view1"]):
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()

    cache[key] = True
    print(
        f"[Client ready] key={_cache_key(context)} fold={fold} "
        f"desired_role={desired_role} device={device} npz={npz}"
    )


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

    view = int(msg.content["config"].get("view", 0))
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
    """
    Handle a label request from the server.

    Only the active silo is permitted to respond to label requests.
    Raises a RuntimeError if this handler is called on a passive node,
    which would indicate a role mismatch in the server logic.
    """
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
    """
    Handle one local HFL training round for passive-silo self-supervised pre-training.

    This handler implements the client-side step of the intra-silo HFL stage
    (Tier 1, optional) in the 10PH-DVFL architecture. It is called once per
    HFL round by the server during passive encoder pre-training.

    Each call:
      1. Loads the current global passive encoder weights sent by the server
      2. Trains the encoder locally on the client's assigned data partition
         using a self-supervised two-view cosine similarity objective
      3. Returns the locally updated weights, the number of local examples,
         and the average training loss to the server for FedAvg aggregation

    The passive encoder is trained without access to any labels. The
    self-supervised objective encourages the encoder to produce consistent
    representations for different augmented views of the same tabular sample.
    """
    cfg = msg.content["config"]
    desired_role = str(cfg.get("role", "")).strip().lower()
    if desired_role != "passive_hfl":
        raise RuntimeError(f"hfl_fit called with role={desired_role}, expected passive_hfl")

    _init_if_needed(context, desired_role)
    cache = _get_cache(context)
    dev: torch.device = cache["dev"]

    # Determine this client's local partition using its rank within the K clients.
    client_rank = int(cfg.get("client_rank", 0))
    K = int(cfg.get("K", 1))
    X2: torch.Tensor = cache["X2"]
    fold = int(os.environ.get("FOLD", context.run_config.get("fold", 1)))

    # Reload fold metadata to reconstruct the training partition for this client.
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

    # Replace local encoder weights with the current global weights from the server.
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

    # Local self-supervised training using two augmented views of each sample.
    n = len(local_idx)
    model.train()
    losses = []
    for _ in range(local_epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch):
            idx = local_idx[perm[start:start + batch]]
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

    # Return updated encoder weights to the server for FedAvg aggregation.
    out_sd = model.state_dict()
    out_arrs = ArrayRecord({"global_weights": _state_dict_to_arrays(out_sd)})
    out_cfg = _cfg({"num_examples": int(n), "loss": avg_loss, "client_rank": client_rank, "K": K})
    return Message(content=RecordDict({"arrays": out_arrs, "config": out_cfg}), reply_to=msg)
