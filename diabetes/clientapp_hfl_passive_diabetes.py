# clientapp_hfl_passive_diabetes.py
#
# Flower client application for intra-silo HFL pre-training of the passive
# silo encoder in the diabetes decoupled VFL experiment.
#
# This client implements the HFL client-side behaviour for the optional Tier 1
# intra-silo horizontal federated learning stage of the 10PH-DVFL architecture.
# Each client holds an IID partition of the passive-silo features (X2) and
# trains the shared passive encoder locally using a Denoising Autoencoder (DAE)
# self-supervised objective before returning updated weights to the server.
#
# The passive silo has no access to labels at any point. All pre-training is
# self-supervised using reconstruction of corrupted input features.
#
# Handlers registered by this client:
#   query.get_metadata   : Returns local partition size and feature dimensionality.
#                          Called once by the server before FedAvg begins.
#   query.get_stats      : Returns fold-specific standardization statistics (mu, sd).
#                          Called once by the server after FedAvg to include in the
#                          saved checkpoint.
#   query.get_embeddings : Applies the converged encoder to the client's slice of
#                          the aligned cohort and returns embeddings. Called once
#                          after FedAvg for Step 5 embedding collection.
#   train.local_train    : Receives global encoder weights, trains locally for
#                          local_epochs using DAE objective, returns updated weights
#                          and average reconstruction MSE for FedAvg aggregation.

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

# Module-level cache to persist loaded data and models across handler calls
# within the same client process, avoiding repeated disk reads.
_CACHE: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def _cache_key(ctx: Context) -> str:
    """Return a unique string key identifying this client node."""
    nid = getattr(ctx, "node_id", None)
    return str(nid) if nid is not None else str(id(ctx))


def _get_cache(ctx: Context) -> Dict[str, Any]:
    """Return the cache dict for this client node, initialising it if needed."""
    k = _cache_key(ctx)
    if k not in _CACHE:
        _CACHE[k] = {}
    return _CACHE[k]


# ---------------------------------------------------------------------------
# Model definitions (must match serverapp_hfl_passive_diabetes.py)
# ---------------------------------------------------------------------------

class BottomMLP_Paper(nn.Module):
    """
    Bottom encoder used by the passive silo in the decoupled VFL architecture.

    Projects passive-silo input features through two linear layers with
    ReLU activations to produce an 8-dimensional embedding vector.
    Architecture: input -> 16 -> ReLU -> 8 -> ReLU

    Must match the architecture defined in serverapp_hfl_passive_diabetes.py
    and clientapp_vfl_diabetes_decoupled.py.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, 8),      nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReconHead(nn.Module):
    """
    Reconstruction head used during DAE self-supervised pre-training.

    Decodes the 8-dimensional encoder output back to the original input
    dimensionality. This head is used only during HFL pre-training and is
    discarded after training. It is never used in Tier 2.
    Architecture: 8 -> 16 -> ReLU -> input_dim
    """
    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 16), nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Array conversion utility
# ---------------------------------------------------------------------------

def _to_f32_tensor(arr: Any, device: torch.device) -> torch.Tensor:
    """
    Convert a Flower Array to a float32 PyTorch tensor on the given device.
    Handles multiple array formats returned by different Flower versions.
    """
    if hasattr(arr, "numpy"):
        x = arr.numpy()
    elif hasattr(arr, "data"):
        x = np.asarray(arr.data)
    else:
        x = np.asarray(arr)
    return torch.from_numpy(x.astype(np.float32)).to(device)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _init_if_needed(ctx: Context) -> None:
    """
    Initialise the client node by loading the dataset and computing the
    fold-specific IID partition assigned to this client.

    This function is called at the start of each handler and is a no-op
    if the client has already been initialised.

    The client's partition is determined by its node_idx (assigned by the
    server via the get_metadata handler) and the number of HFL clients K.
    Training data is shuffled before partitioning using the global seed to
    ensure IID distribution across clients.

    Standardization statistics are computed from the full passive-silo
    training split and applied globally to ensure consistency between
    pre-training and Tier 2 embedding generation.
    """
    cache = _get_cache(ctx)
    if cache.get("ready", False):
        return

    npz_path = os.environ.get("NPZ_PATH", "diabetes_vfl_cv.npz")
    fold     = int(os.environ.get("FOLD", 1))
    seed     = int(os.environ.get("SEED", 42))
    K        = int(os.environ.get("K", 10))
    device   = os.environ.get("DEVICE", "cpu")
    node_id  = getattr(ctx, "node_id", 0)
    node_idx = int(cache.get("node_idx", 0))

    d = np.load(npz_path, allow_pickle=True)
    X2    = d["X2"].astype(np.float32)
    folds = list(d["folds"])
    split_obj = folds[fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)

    # Standardize using training split statistics only to prevent leakage.
    mu  = X2[tr].mean(axis=0).astype(np.float32)
    sd  = (X2[tr].std(axis=0) + 1e-8).astype(np.float32)
    X2s = (X2 - mu) / sd

    # Partition the training data into K IID client partitions using the
    # global seed to ensure reproducibility across runs.
    rng  = np.random.default_rng(seed)
    tr_s = tr.copy()
    rng.shuffle(tr_s)
    parts    = np.array_split(tr_s, K)
    my_idx   = parts[node_idx % K].astype(np.int64)

    dev = torch.device(device)
    cache["X2"]      = torch.from_numpy(X2s).float().to(dev)
    cache["my_idx"]  = my_idx
    cache["in_dim"]  = int(X2.shape[1])
    cache["n_train"] = len(my_idx)
    cache["mu"]      = mu
    cache["sd"]      = sd
    cache["dev"]     = dev
    cache["ready"]   = True

    print(
        f"[Client] node_id={node_id} node_idx={node_idx} fold={fold} "
        f"partition={len(my_idx)} in_dim={cache['in_dim']}"
    )


# ---------------------------------------------------------------------------
# Flower ClientApp and handlers
# ---------------------------------------------------------------------------

app = ClientApp()


@app.query("get_metadata")
def get_metadata(msg: Message, ctx: Context) -> Message:
    """
    Return local partition size and feature dimensionality to the server.

    Called once by the server before FedAvg begins to determine FedAvg
    weights and verify feature space consistency across all clients.
    The server-assigned node_idx is stored before initialisation so that
    the correct partition is selected during _init_if_needed.
    """
    cache = _get_cache(ctx)
    cfg = msg.content.get("config", None)
    if cfg is not None:
        # Save the server-assigned client index before initialisation so
        # that _init_if_needed selects the correct partition for this client.
        cache["node_idx"] = int(cfg.get("node_idx", 0))
    _init_if_needed(ctx)
    return Message(
        content=RecordDict({"config": ConfigRecord({
            "n_train": cache["n_train"],
            "in_dim":  cache["in_dim"],
        })}),
        reply_to=msg,
    )


@app.query("get_stats")
def get_stats(msg: Message, ctx: Context) -> Message:
    """
    Return fold-specific standardization statistics to the server.

    Called once by the server after FedAvg training completes so that
    the mean and standard deviation computed from the training split can
    be included in the saved encoder checkpoint for reproducibility.
    """
    _init_if_needed(ctx)
    cache = _get_cache(ctx)
    return Message(
        content=RecordDict({"arrays": ArrayRecord({
            "mu": Array(cache["mu"].astype(np.float32)),
            "sd": Array(cache["sd"].astype(np.float32)),
        })}),
        reply_to=msg,
    )


@app.query("get_embeddings")
def get_embeddings(msg: Message, ctx: Context) -> Message:
    """
    Apply the converged encoder to the client's slice of the aligned cohort
    and return the resulting embeddings to the server.

    Called once after FedAvg training completes as part of Step 5 in the
    server script. The server provides the final converged encoder weights
    and the indices of the aligned cohort samples assigned to this client.

    Raw passive-silo data never leaves this client node. Only embeddings
    for the requested aligned samples are returned to the server.
    """
    _init_if_needed(ctx)
    cache  = _get_cache(ctx)
    dev    = cache["dev"]
    in_dim = cache["in_dim"]

    arrs         = msg.content["arrays"]
    bottom_state = {
        k[len("bottom_"):]: _to_f32_tensor(arrs[k], dev)
        for k in arrs if k.startswith("bottom_")
    }

    # Load the converged encoder weights sent by the server.
    bottom = BottomMLP_Paper(in_dim=in_dim).to(dev)
    bottom.load_state_dict(bottom_state, strict=True)
    bottom.eval()

    # Use the server-provided aligned indices if present; otherwise use all samples.
    if "aligned_idx" in arrs:
        aligned_idx = np.asarray(arrs["aligned_idx"].numpy()).astype(np.int32).astype(np.int64)
    else:
        aligned_idx = np.arange(cache["X2"].shape[0], dtype=np.int64)

    X2    = cache["X2"]
    idx_t = torch.from_numpy(aligned_idx).long().to(dev)

    with torch.no_grad():
        xb   = X2.index_select(0, idx_t)
        embs = bottom(xb).cpu().numpy().astype(np.float32)

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord({"embeddings": Array(embs)}),
            "config": ConfigRecord({
                "n_aligned": len(aligned_idx),
                "emb_dim":   embs.shape[1],
            }),
        }),
        reply_to=msg,
    )


@app.train("local_train")
def local_train(msg: Message, ctx: Context) -> Message:
    """
    Run one local HFL training step using the DAE self-supervised objective.

    Called once per FedAvg round by the server. Each call:
      1. Loads the current global encoder and reconstruction head weights
      2. Trains locally on the client's assigned IID partition for local_epochs
         using DAE: corrupted input -> encode -> reconstruct -> MSE loss
      3. Returns locally updated weights and average reconstruction MSE

    The reconstruction head is trained alongside the encoder but is not
    used after pre-training. No labels are used at any point, consistent
    with the passive silo's absence of label access in the VFL setting.
    """
    _init_if_needed(ctx)
    cache  = _get_cache(ctx)
    dev    = cache["dev"]
    in_dim = cache["in_dim"]

    cfg          = msg.content["config"]
    local_epochs = int(cfg.get("local_epochs", 1))
    batch_size   = int(cfg.get("batch_size", 256))
    noise_std    = float(cfg.get("noise_std", 0.1))
    lr           = float(cfg.get("lr", 1e-3))
    seed_local   = int(cfg.get("seed", 42))

    # Load the global encoder and reconstruction head weights from the server.
    arrs         = msg.content["arrays"]
    bottom_state = {
        k[len("bottom_"):]: _to_f32_tensor(arrs[k], dev)
        for k in arrs if k.startswith("bottom_")
    }
    recon_state = {
        k[len("recon_"):]: _to_f32_tensor(arrs[k], dev)
        for k in arrs if k.startswith("recon_")
    }

    bottom     = BottomMLP_Paper(in_dim=in_dim).to(dev)
    recon_head = ReconHead(emb_dim=8, out_dim=in_dim).to(dev)
    bottom.load_state_dict(bottom_state, strict=True)
    recon_head.load_state_dict(recon_state, strict=True)
    bottom.train()
    recon_head.train()

    opt    = torch.optim.Adam(
        list(bottom.parameters()) + list(recon_head.parameters()), lr=lr
    )
    mse_fn = nn.MSELoss()
    X2     = cache["X2"]
    idx    = cache["my_idx"]
    rng    = np.random.default_rng(seed_local)

    # Local DAE training loop.
    # Each epoch shuffles the local partition and trains on all batches.
    # Gaussian noise is added to the input; the encoder is trained to
    # reconstruct the original clean features from the corrupted version.
    n, loss_sum = 0, 0.0
    for _ in range(local_epochs):
        idx_s = idx.copy()
        rng.shuffle(idx_s)
        for s in range(0, len(idx_s), batch_size):
            b = idx_s[s:s + batch_size]
            if len(b) == 0:
                continue
            bt   = torch.from_numpy(b).long().to(dev)
            xb   = X2.index_select(0, bt)
            xn   = xb + noise_std * torch.randn_like(xb)  # DAE corruption step.
            opt.zero_grad(set_to_none=True)
            loss = mse_fn(recon_head(bottom(xn)), xb)     # MSE against clean input.
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(b)
            n += len(b)

    recon_mse = float(loss_sum / max(n, 1))

    # Return locally updated encoder and reconstruction head weights to the server.
    upd_b = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in bottom.state_dict().items()}
    upd_r = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in recon_head.state_dict().items()}

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord({
                **{f"bottom_{k}": Array(v) for k, v in upd_b.items()},
                **{f"recon_{k}":  Array(v) for k, v in upd_r.items()},
            }),
            "config": ConfigRecord({"recon_mse": recon_mse}),
        }),
        reply_to=msg,
    )
