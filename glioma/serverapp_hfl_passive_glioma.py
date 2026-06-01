# serverapp_hfl_passive_glioma.py
#
# Flower 1.26.1 server application for intra-silo HFL pre-training of the
# passive silo encoder in the glioma decoupled VFL experiment.
#
# This script implements the optional Tier 1 intra-silo horizontal federated
# learning stage of the 10PH-DVFL architecture for the glioma dataset. The
# passive silo holds no labels at any point; this stage uses a self-supervised
# Denoising Autoencoder (DAE) objective to pre-train the passive encoder across
# K simulated IID clients using FedAvg before Tier 2 cross-silo fusion begins.
#
# Architecture differences from the diabetes HFL server:
#   BottomMLP:  input -> 32 -> ReLU -> Dropout(0.0) -> 16 -> ReLU
#               (diabetes: input -> 16 -> ReLU -> 8 -> ReLU)
#   ReconHead:  16 -> 32 -> ReLU -> input_dim
#               (diabetes: 8 -> 16 -> ReLU -> input_dim)
#   out_dim = 16 (diabetes: 8)
#   run tag uses "bottom32_out16" (diabetes: "bottom16_out8")
#
# Step 5 — Aligned cohort embedding collection:
#   After FedAvg training, each HFL client applies the converged encoder to
#   its local slice of the aligned cohort and returns embeddings. Raw data
#   never leaves any HFL client node. The server assembles per-node embeddings
#   into a single silo-level embedding matrix and saves it alongside the
#   checkpoint.
#
#   Partition indices for Step 5 are read directly from the NPZ file under
#   hfl_partitions["iid"][f"K{K}"]["silo2"], guaranteeing consistency with
#   the IID partitions used during FedAvg training. This differs from the
#   diabetes HFL server which reads partition indices from a separate JSON file.

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

INFO = 20


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility across numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env_int(k: str, d: int) -> int:
    """Read an integer from environment variables with a default fallback."""
    return int(os.environ.get(k, str(d)))


def _env_float(k: str, d: float) -> float:
    """Read a float from environment variables with a default fallback."""
    return float(os.environ.get(k, str(d)))


# ---------------------------------------------------------------------------
# Model definitions (must match clientapp_hfl_passive_glioma.py and
# clientapp_vfl_glioma_decoupled.py exactly)
# ---------------------------------------------------------------------------

class BottomMLP(nn.Module):
    """
    Bottom encoder used by the passive silo in the glioma decoupled VFL architecture.

    Projects passive-silo input features through two linear layers with
    ReLU activations to produce a 16-dimensional embedding vector.
    Architecture: input -> 32 -> ReLU -> Dropout(0.0) -> 16 -> ReLU

    The Dropout layer is included at dropout=0.0 (disabled) to maintain
    checkpoint key compatibility with the VFL client script which loads
    these checkpoints for Tier 2 fusion.
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


class ReconHead(nn.Module):
    """
    Reconstruction head used during DAE self-supervised pre-training.

    Decodes the 16-dimensional encoder output back to the original input
    dimensionality. Used only during HFL pre-training and discarded after.
    Architecture: 16 -> 32 -> ReLU -> input_dim
    """
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Flower grid utilities
# ---------------------------------------------------------------------------

def _grid_request(grid: Grid, msg: Message, timeout: int = 300) -> Message:
    """
    Send a single message to a client node and return the reply.
    Raises RuntimeError if the reply is empty or has no content.
    """
    reps = grid.send_and_receive([msg], timeout=timeout)
    if not reps or not reps[0].has_content():
        raise RuntimeError("Empty reply from client")
    return reps[0]


def _state_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Compute the total byte size of a PyTorch state dict for communication tracking."""
    return sum(v.cpu().numpy().nbytes for v in state_dict.values())


def _fedavg(
    states: List[Dict[str, torch.Tensor]],
    weights: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate client model updates using FedAvg weighted by partition size.
    Weights are normalised to sum to 1 before aggregation.
    """
    w   = weights / max(weights.sum(), 1e-12)
    avg = {}
    for key in states[0].keys():
        avg[key] = sum(sd[key] * float(wi) for sd, wi in zip(states, w))
    return avg


def _write_csv(path: Path, rows: List[Dict]) -> None:
    """Write a list of dicts to a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


# ---------------------------------------------------------------------------
# Flower ServerApp entry point
# ---------------------------------------------------------------------------

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main server function for intra-silo HFL passive encoder pre-training.

    Orchestrates the full HFL pre-training pipeline:
      Step 1: Query each client for partition size and input dimensionality
      Step 2: Initialise the global model and prepare for broadcasting
      Step 3: Run FedAvg rounds with DAE self-supervised objective
      Step 4: Save the final aggregated encoder checkpoint
      Step 5: Collect aligned-cohort embeddings from all HFL clients
    """
    rc = getattr(context, "run_config", {}) or {}

    fold         = int(os.environ.get("FOLD",         rc.get("fold",         1)))
    seed         = int(os.environ.get("SEED",         rc.get("seed",         42)))
    rounds       = int(os.environ.get("ROUNDS",       rc.get("rounds",       100)))
    local_epochs = int(os.environ.get("LOCAL_EPOCHS", rc.get("local_epochs", 1)))
    batch_size   = int(os.environ.get("BATCH_SIZE",   rc.get("batch_size",   64)))
    noise_std    = float(os.environ.get("NOISE_STD",  rc.get("noise_std",    0.1)))
    lr           = float(os.environ.get("LR",         rc.get("lr",           1e-3)))
    set_seed(seed)

    out_dir = Path(os.environ.get(
        "OUT_DIR", rc.get("out_dir", "runs_passive_hfl_glioma_flower")
    ))
    K       = len(list(grid.get_node_ids()))
    tag     = f"hfl_ssl_fedavg_K{K}_R{rounds}_E{local_epochs}_bottom32_out16"
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    log(INFO,
        "[HFL-SETUP] fold=%d K=%d rounds=%d local_epochs=%d batch=%d noise=%.2f",
        fold, K, rounds, local_epochs, batch_size, noise_std)

    node_ids = sorted(list(grid.get_node_ids()))
    log(INFO, "[HFL-SETUP] Node IDs: %s", node_ids)

    # ---------------------------------------------------------------------------
    # Step 1: Query each client for partition size and input dimensionality
    # ---------------------------------------------------------------------------

    log(INFO, "[HFL] Step 1: querying client metadata...")
    sizes  = []
    in_dim = None

    for node_idx, nid in enumerate(node_ids):
        rep = _grid_request(grid, Message(
            content=RecordDict({"config": ConfigRecord({
                "fold": fold, "seed": seed, "node_idx": node_idx,
            })}),
            dst_node_id=nid,
            message_type="query.get_metadata",
        ))
        cfg = rep.content["config"]
        n   = int(cfg.get("n_train", 0))
        d   = int(cfg.get("in_dim",  0))
        sizes.append(n)
        if in_dim is None:
            in_dim = d
        log(INFO, "[HFL] node=%d node_idx=%d n_train=%d in_dim=%d",
            nid, node_idx, n, d)

    sizes = np.array(sizes, dtype=np.float64)
    log(INFO, "[HFL] Partition sizes: %s  total=%d", sizes.tolist(), int(sizes.sum()))

    npz_path = os.environ.get("NPZ_PATH", "glioma_aligned_vfl_hfl_cv.npz")

    with open(run_dir / f"partition_manifest_fold{fold}.json", "w") as f:
        json.dump({
            "fold": fold, "K": K, "rounds": rounds,
            "local_epochs": local_epochs, "batch_size": batch_size,
            "noise_std": noise_std, "seed": seed,
            "bottom_arch": "in->32->16",
            "recon_arch":  "16->32->in",
            "sizes": [int(s) for s in sizes.tolist()],
        }, f, indent=2)

    # ---------------------------------------------------------------------------
    # Step 2: Initialise global model
    # ---------------------------------------------------------------------------

    torch.manual_seed(seed)
    global_bottom = BottomMLP(in_dim=in_dim, out_dim=16)
    global_recon  = ReconHead(out_dim=in_dim)

    bottom_state = {k: v.detach().cpu() for k, v in global_bottom.state_dict().items()}
    recon_state  = {k: v.detach().cpu() for k, v in global_recon.state_dict().items()}

    model_bytes = _state_bytes(bottom_state) + _state_bytes(recon_state)
    log(INFO, "[HFL] Model size: %.2f KB (bottom + recon head)", model_bytes / 1024)

    # ---------------------------------------------------------------------------
    # Step 3: FedAvg rounds
    # ---------------------------------------------------------------------------

    total_comm_bytes = 0

    with open(run_dir / f"hfl_ssl_metrics_fold{fold}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "round", "mean_recon_mse", "weighted_recon_mse",
            "min_client_mse", "max_client_mse",
            "round_comm_bytes", "round_comm_kb",
            "cumulative_comm_bytes", "cumulative_comm_mb",
        ])

        for r in range(1, rounds + 1):
            client_bottom_states = []
            client_recon_states  = []
            client_mses          = []
            round_comm = 0

            for node_idx, nid in enumerate(node_ids):
                # Download cost: server sends global model weights to this client.
                round_comm += model_bytes

                rep = _grid_request(grid, Message(
                    content=RecordDict({
                        "arrays": ArrayRecord({
                            **{f"bottom_{k}": Array(v.numpy().astype(np.float32))
                               for k, v in bottom_state.items()},
                            **{f"recon_{k}": Array(v.numpy().astype(np.float32))
                               for k, v in recon_state.items()},
                        }),
                        "config": ConfigRecord({
                            "round":        r,
                            "local_epochs": local_epochs,
                            "batch_size":   batch_size,
                            "noise_std":    noise_std,
                            "lr":           lr,
                            # Per-client per-round seed ensures diverse augmentation.
                            "seed": seed + 10000 * r + 97 * node_ids.index(nid),
                        }),
                    }),
                    dst_node_id=nid,
                    message_type="train.local_train",
                ))

                # Upload cost: server receives locally updated weights.
                arrs = rep.content["arrays"]
                recv_bottom = {
                    k[len("bottom_"):]: torch.from_numpy(
                        np.asarray(arrs[k].numpy()).astype(np.float32)
                    )
                    for k in arrs if k.startswith("bottom_")
                }
                recv_recon = {
                    k[len("recon_"):]: torch.from_numpy(
                        np.asarray(arrs[k].numpy()).astype(np.float32)
                    )
                    for k in arrs if k.startswith("recon_")
                }
                mse_val    = float(rep.content["config"].get("recon_mse", 0.0))
                recv_bytes = _state_bytes(recv_bottom) + _state_bytes(recv_recon)
                round_comm += recv_bytes

                client_bottom_states.append(recv_bottom)
                client_recon_states.append(recv_recon)
                client_mses.append(mse_val)

            # Aggregate client updates using FedAvg weighted by partition size.
            bottom_state = _fedavg(client_bottom_states, sizes)
            recon_state  = _fedavg(client_recon_states,  sizes)

            client_mses  = np.array(client_mses)
            mean_mse     = float(client_mses.mean())
            weighted_mse = float((client_mses * (sizes / sizes.sum())).sum())
            min_mse      = float(client_mses.min())
            max_mse      = float(client_mses.max())
            total_comm_bytes += round_comm

            log(INFO,
                "[HFL][fold=%d] round=%d/%d mean_mse=%.6f weighted_mse=%.6f "
                "round_comm=%.2f KB cumulative=%.4f MB",
                fold, r, rounds, mean_mse, weighted_mse,
                round_comm / 1024, total_comm_bytes / 1024**2)

            w.writerow([
                r, mean_mse, weighted_mse, min_mse, max_mse,
                round_comm, round_comm / 1024,
                total_comm_bytes, total_comm_bytes / 1024**2,
            ])
            f.flush()

    # ---------------------------------------------------------------------------
    # Step 4: Save the final aggregated encoder checkpoint
    # ---------------------------------------------------------------------------

    # Retrieve fold-specific standardization statistics from the first client.
    rep = _grid_request(grid, Message(
        content=RecordDict({"config": ConfigRecord({"request": "stats"})}),
        dst_node_id=node_ids[0],
        message_type="query.get_stats",
    ))
    mu = np.asarray(rep.content["arrays"]["mu"].numpy()).astype(np.float32)
    sd = np.asarray(rep.content["arrays"]["sd"].numpy()).astype(np.float32)

    ckpt = {
        "bottom_state": bottom_state,
        "recon_state":  recon_state,
        "fold":         fold,
        "seed":         seed,
        "out_dim":      16,
        "npz":          npz_path,
        "x2_mu":        mu,
        "x2_sd":        sd,
        "ssl":          True,
        "hfl": {
            "K": K, "rounds": rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "noise_std": noise_std,
        },
        "total_comm_bytes": total_comm_bytes,
        "total_comm_mb":    total_comm_bytes / 1024**2,
    }

    ckpt_path = run_dir / f"pretrained_passive_bottom_hfl_fold{fold}.pt"
    torch.save(ckpt, ckpt_path)

    _write_csv(run_dir / f"comm_summary_fold{fold}.csv", [{
        "fold":              fold,
        "K":                 K,
        "rounds":            rounds,
        "model_size_kb":     model_bytes / 1024,
        "total_comm_bytes":  total_comm_bytes,
        "total_comm_kb":     total_comm_bytes / 1024,
        "total_comm_mb":     total_comm_bytes / 1024**2,
        "comm_per_round_kb": (total_comm_bytes / rounds) / 1024,
    }])

    log(INFO, "[HFL-DONE] fold=%d total_comm=%.4f MB checkpoint=%s",
        fold, total_comm_bytes / 1024**2, ckpt_path)

    # ---------------------------------------------------------------------------
    # Step 5: Collect aligned-cohort embeddings from HFL clients
    # ---------------------------------------------------------------------------
    # After FedAvg training, each HFL client applies the converged encoder to
    # its local slice of the aligned cohort and returns the resulting embeddings.
    # Raw passive-silo data never leaves any HFL client node at any point.
    #
    # Partition indices are read directly from the NPZ file under
    # hfl_partitions["iid"][f"K{K}"]["silo2"], guaranteeing consistency with
    # the IID partitions used during FedAvg training. This is the glioma-specific
    # approach; the diabetes HFL server reads partition indices from a separate
    # JSON file instead.

    log(INFO, "[HFL] Step 5: collecting aligned-cohort embeddings from fog nodes...")

    d_npz   = np.load(npz_path, allow_pickle=True)
    folds_  = list(d_npz["folds"])
    split_o = folds_[fold - 1]
    split_  = split_o.item() if hasattr(split_o, "item") else split_o
    hp      = split_["hfl_partitions"]
    hp      = hp.item() if hasattr(hp, "item") else hp
    k_key   = f"K{K}"
    if k_key not in hp["iid"]:
        raise ValueError(
            f"hfl_partitions has no iid/{k_key}. "
            f"Available keys: {list(hp['iid'].keys())}"
        )
    # silo2 = passive silo (X2): list of K arrays, one per fog node.
    al_parts = hp["iid"][k_key]["silo2"]

    node_embeddings:  Dict[int, np.ndarray] = {}
    node_aligned_idx: Dict[int, np.ndarray] = {}

    for node_idx, nid in enumerate(node_ids):
        local_aligned = np.asarray(al_parts[node_idx]).astype(np.int64)
        node_aligned_idx[node_idx] = local_aligned

        rep = _grid_request(grid, Message(
            content=RecordDict({
                "arrays": ArrayRecord({
                    **{f"bottom_{k}": Array(
                        v.numpy().astype(np.float32)
                        if isinstance(v, torch.Tensor)
                        else np.asarray(v).astype(np.float32))
                       for k, v in bottom_state.items()},
                    # Send indices as int32 to avoid float32 precision loss.
                    "aligned_idx": Array(local_aligned.astype(np.int32)),
                }),
                "config": ConfigRecord({"fold": fold}),
            }),
            dst_node_id=nid,
            message_type="query.get_embeddings",
        ))

        embs = np.asarray(
            rep.content["arrays"]["embeddings"].numpy()
        ).astype(np.float32)
        node_embeddings[node_idx] = embs
        log(INFO, "[HFL] node_idx=%d n_aligned=%d emb_shape=%s",
            node_idx, len(local_aligned), embs.shape)

    # Reassemble the full silo-level embedding matrix in original sample index order.
    all_idx  = np.concatenate([node_aligned_idx[i] for i in range(K)])
    all_embs = np.concatenate([node_embeddings[i]  for i in range(K)], axis=0)
    sort_ord = np.argsort(all_idx)
    aligned_embeddings = all_embs[sort_ord]
    aligned_indices    = all_idx[sort_ord]

    emb_path = run_dir / f"passive_aligned_embeddings_hfl_fold{fold}.npz"
    np.savez(
        emb_path,
        embeddings=aligned_embeddings,
        aligned_indices=aligned_indices,
        x2_mu=mu, x2_sd=sd,
        fold=np.array(fold),
    )

    log(INFO, "[HFL] Aligned embeddings saved: shape=%s  path=%s",
        aligned_embeddings.shape, emb_path)
