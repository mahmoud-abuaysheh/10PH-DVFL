# serverapp_hfl_passive_diabetes.py
#
# Flower 1.26.1 server application for intra-silo HFL pre-training of the
# passive silo encoder in the diabetes decoupled VFL experiment.
#
# This script implements the optional Tier 1 intra-silo horizontal federated
# learning stage of the 10PH-DVFL architecture. The passive silo holds no
# labels at any point; this stage uses a self-supervised Denoising Autoencoder
# (DAE) objective to pre-train the passive encoder across K simulated IID
# clients using FedAvg before Tier 2 cross-silo fusion begins.
#
# Algorithm: FedAvg with DAE self-supervised objective.
#   - Each client holds an IID partition of the passive features (X2).
#   - Each round: server broadcasts global encoder weights to all clients;
#     each client trains locally for local_epochs using DAE; clients return
#     updated weights; server aggregates using FedAvg weighted by partition size.
#   - After convergence, the aggregated encoder checkpoint is saved for use
#     as the passive silo encoder in Tier 2.
#
# Communication cost is measured as actual bytes sent and received per round,
# accumulated across all rounds and clients.
#
# Step 5 — Aligned cohort embedding collection:
#   After FedAvg training, each HFL client applies the converged encoder to
#   its local slice of the aligned cohort and returns embeddings. Raw data
#   never leaves any HFL client node. The server assembles per-node embeddings
#   into a single silo-level embedding matrix and saves it alongside the
#   checkpoint. This step demonstrates the correct architectural logic for
#   a real distributed deployment where passive silo data resides at the HFL
#   clients rather than at a single central node.
#
#   Note: The current Tier 2 simulation loads the HFL checkpoint directly and
#   runs the encoder forward pass centrally (simulation simplification). In a
#   real deployment, the pre-generated embedding file produced here should be
#   used directly in the Tier 2 VFL client.

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
# Model definitions (must match clientapp_hfl_passive_diabetes.py)
# ---------------------------------------------------------------------------

class BottomMLP_Paper(nn.Module):
    """
    Bottom encoder used by the passive silo in the decoupled VFL architecture.

    Projects passive-silo input features through two linear layers with
    ReLU activations to produce an 8-dimensional embedding vector.
    Architecture: input -> 16 -> ReLU -> 8 -> ReLU

    Must match the architecture defined in clientapp_hfl_passive_diabetes.py
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
    dimensionality. This head is local to the pre-training stage and is
    discarded after HFL training completes. It is never used in Tier 2.
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


def _state_to_arrays(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Array]:
    """Convert a PyTorch state dict to a dict of Flower Arrays for transmission."""
    return {k: Array(v.cpu().numpy()) for k, v in state_dict.items()}


def _arrays_to_state(arrays: Dict[str, Array]) -> Dict[str, torch.Tensor]:
    """Convert a dict of Flower Arrays back to a PyTorch state dict."""
    return {k: torch.from_numpy(np.asarray(v).astype(np.float32)) for k, v in arrays.items()}


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
    w = weights / max(weights.sum(), 1e-12)
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
      Step 2: Initialise the global model and broadcast to all clients
      Step 3: Run FedAvg rounds with DAE self-supervised objective
      Step 4: Save the final aggregated encoder checkpoint
      Step 5: Collect aligned-cohort embeddings from all HFL clients
    """

    # Read hyperparameters from environment variables.
    # Environment variables are used to allow per-fold job submission
    # without modifying the pyproject.toml configuration.
    fold         = _env_int("FOLD", 1)
    seed         = _env_int("SEED", 42)
    rounds       = _env_int("ROUNDS", 100)
    local_epochs = _env_int("LOCAL_EPOCHS", 1)
    batch_size   = _env_int("BATCH_SIZE", 256)
    noise_std    = _env_float("NOISE_STD", 0.1)
    set_seed(seed)

    out_dir = Path(os.environ.get("OUT_DIR", "runs_passive_hfl_diabetes_flower"))
    K = len(list(grid.get_node_ids()))
    tag = f"hfl_ssl_fedavg_K{K}_R{rounds}_E{local_epochs}_bottom16_out8"
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    log(INFO,
        "[HFL-SETUP] fold=%d K=%d rounds=%d local_epochs=%d batch=%d noise=%.2f",
        fold, K, rounds, local_epochs, batch_size, noise_std)

    node_ids = sorted(list(grid.get_node_ids()))
    log(INFO, "[HFL-SETUP] Node IDs: %s", node_ids)

    # ---------------------------------------------------------------------------
    # Step 1: Query each client for its partition size and input dimensionality
    # ---------------------------------------------------------------------------
    # The server needs to know each client's local partition size to compute
    # FedAvg weights and to verify that all clients share the same feature space.

    log(INFO, "[HFL] Step 1: querying client metadata...")
    sizes = []
    in_dim = None

    for node_idx, nid in enumerate(node_ids):
        rep = _grid_request(grid, Message(
            content=RecordDict({"config": ConfigRecord({
                "fold": fold, "seed": seed, "node_idx": node_idx
            })}),
            dst_node_id=nid,
            message_type="query.get_metadata",
        ))
        cfg = rep.content["config"]
        n = int(cfg.get("n_train", 0))
        d = int(cfg.get("in_dim", 0))
        sizes.append(n)
        if in_dim is None:
            in_dim = d
        log(INFO, "[HFL] node=%d node_idx=%d n_train=%d in_dim=%d", nid, node_idx, n, d)

    sizes = np.array(sizes, dtype=np.float64)
    log(INFO, "[HFL] Partition sizes: %s  total=%d", sizes.tolist(), int(sizes.sum()))

    # Save partition manifest for reproducibility documentation.
    with open(run_dir / f"partition_manifest_fold{fold}.json", "w") as f:
        json.dump({
            "fold": fold, "K": K, "rounds": rounds,
            "local_epochs": local_epochs, "batch_size": batch_size,
            "noise_std": noise_std, "seed": seed,
            "bottom_arch": "in->16->8",
            "sizes": [int(s) for s in sizes.tolist()],
        }, f, indent=2)

    # ---------------------------------------------------------------------------
    # Step 2: Initialise global model and prepare for broadcasting
    # ---------------------------------------------------------------------------
    # The global model is initialised on the server with a fixed seed for
    # reproducibility. Initial weights are broadcast to all clients at the
    # start of the first FedAvg round.

    torch.manual_seed(seed)
    global_bottom = BottomMLP_Paper(in_dim=in_dim)
    global_recon  = ReconHead(emb_dim=8, out_dim=in_dim)

    bottom_state = {k: v.detach().cpu() for k, v in global_bottom.state_dict().items()}
    recon_state  = {k: v.detach().cpu() for k, v in global_recon.state_dict().items()}

    model_bytes = _state_bytes(bottom_state) + _state_bytes(recon_state)
    log(INFO, "[HFL] Model size: %.2f KB (bottom + recon head)", model_bytes / 1024)

    # ---------------------------------------------------------------------------
    # Step 3: FedAvg rounds
    # ---------------------------------------------------------------------------
    # Each round broadcasts the current global weights to all K clients,
    # collects locally updated weights after DAE training, and aggregates
    # using FedAvg weighted by each client's partition size.
    # Communication cost is tracked in both directions (download and upload).

    total_comm_bytes = 0
    round_logs = []

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
                send_bytes = model_bytes
                round_comm += send_bytes

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
                            # Per-client per-round seed ensures diverse augmentation
                            # across clients within each round.
                            "seed": seed + 10000 * r + 97 * node_ids.index(nid),
                        }),
                    }),
                    dst_node_id=nid,
                    message_type="train.local_train",
                ))

                # Upload cost: server receives locally updated weights from this client.
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
                mse_val = float(rep.content["config"].get("recon_mse", 0.0))

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

            round_logs.append({
                "round": r,
                "mean_recon_mse": mean_mse,
                "round_comm_kb": round_comm / 1024,
                "cumulative_comm_mb": total_comm_bytes / 1024**2,
            })

    # ---------------------------------------------------------------------------
    # Step 4: Save the final aggregated encoder checkpoint
    # ---------------------------------------------------------------------------
    # The converged passive encoder checkpoint is saved for use in Tier 2.
    # The reconstruction head is also saved for completeness but is not used
    # in downstream stages. Standardization statistics are retrieved from the
    # first client node for inclusion in the checkpoint.

    npz_path = os.environ.get("NPZ_PATH", "diabetes_vfl_cv.npz")

    # Retrieve fold-specific standardization parameters from the first client.
    # These are needed to ensure consistent feature scaling between pre-training
    # and Tier 2 embedding generation.
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
        "out_dim":      8,
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

    # Save communication cost summary for reporting in the paper.
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
    # The server assembles per-node embeddings into a single silo-level matrix
    # sorted by original sample index and saves it alongside the checkpoint.
    #
    # This step demonstrates the correct architectural logic for a real
    # distributed deployment. In the current simulation, the Tier 2 VFL client
    # loads the checkpoint directly and runs the encoder centrally as a
    # simplification. In a real deployment, this pre-generated embedding file
    # should replace the checkpoint loading step in the Tier 2 VFL client.

    log(INFO, "[HFL] Step 5: collecting aligned-cohort embeddings from fog nodes...")

    # Load pre-computed per-node partitions from the JSON file generated
    # by make_iid_partitions.py. This ensures consistency with the partitions
    # used during FedAvg training.
    partition_json = os.environ.get(
        "PARTITION_JSON",
        f"partitions_K{K}_train.json"
    )
    with open(partition_json) as pf:
        part_data = json.load(pf)

    fold_entry = part_data["folds"][fold - 1]
    client_indices = fold_entry["client_indices"]  # List of K lists of sample indices.

    if len(client_indices) != K:
        raise ValueError(
            f"Partition file has {len(client_indices)} nodes but K={K}. "
            f"Check PARTITION_JSON env var points to the correct file."
        )

    node_embeddings:  Dict[int, np.ndarray] = {}
    node_aligned_idx: Dict[int, np.ndarray] = {}

    for node_idx, nid in enumerate(node_ids):
        local_aligned = np.asarray(client_indices[node_idx], dtype=np.int64)
        node_aligned_idx[node_idx] = local_aligned

        rep = _grid_request(grid, Message(
            content=RecordDict({
                "arrays": ArrayRecord({
                    **{f"bottom_{k}": Array(
                        v.numpy().astype(np.float32)
                        if isinstance(v, torch.Tensor)
                        else np.asarray(v).astype(np.float32))
                       for k, v in bottom_state.items()},
                    # Send indices as int32 to avoid float32 precision loss
                    # on large sample index values.
                    "aligned_idx": Array(local_aligned.astype(np.int32)),
                }),
                "config": ConfigRecord({"fold": fold}),
            }),
            dst_node_id=nid,
            message_type="query.get_embeddings",
        ))

        embs = np.asarray(rep.content["arrays"]["embeddings"].numpy()).astype(np.float32)
        node_embeddings[node_idx] = embs
        log(INFO, "[HFL] node_idx=%d n_aligned=%d emb_shape=%s",
            node_idx, len(local_aligned), embs.shape)

    # Reassemble the full silo-level embedding matrix in original sample index order.
    # Embeddings are concatenated across nodes and then sorted by the original
    # X2 row indices to ensure consistent alignment with the active silo.
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
        x2_mu=mu,
        x2_sd=sd,
        fold=np.array(fold),
    )

    log(INFO, "[HFL] Aligned embeddings saved: shape=%s  path=%s",
        aligned_embeddings.shape, emb_path)
