# serverapp_hfl_passive_diabetes.py
# Flower 1.26.1 — True federated HFL for passive silo pretraining.
# Replaces run_passive_hfl_pretrain_sim_diabetes.py with real Flower federation.
# Communication cost is measured as actual bytes sent/received per round.
#
# Algorithm: FedAvg with denoising autoencoder (DAE) SSL objective.
# Each client holds an IID partition of the passive features (X2).
# Server aggregates bottom + recon_head weights each round.
#
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


# ── Env helpers ───────────────────────────────────────────────────────────────
def _env_int(k, d):   return int(os.environ.get(k, str(d)))
def _env_float(k, d): return float(os.environ.get(k, str(d)))

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Models (must match client) ─────────────────────────────────────────────────
class BottomMLP_Paper(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, 8),      nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class ReconHead(nn.Module):
    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 16), nn.ReLU(),
            nn.Linear(16, out_dim),
        )
    def forward(self, z): return self.net(z)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _grid_request(grid, msg, timeout=300):
    reps = grid.send_and_receive([msg], timeout=timeout)
    if not reps or not reps[0].has_content():
        raise RuntimeError("Empty reply from client")
    return reps[0]


def _state_to_arrays(state_dict) -> Dict[str, Array]:
    return {k: Array(v.cpu().numpy()) for k, v in state_dict.items()}


def _arrays_to_state(arrays) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.asarray(v).astype(np.float32)) for k, v in arrays.items()}


def _state_bytes(state_dict) -> int:
    return sum(v.cpu().numpy().nbytes for v in state_dict.values())


def _fedavg(states: List[Dict[str, torch.Tensor]],
            weights: np.ndarray) -> Dict[str, torch.Tensor]:
    w = weights / max(weights.sum(), 1e-12)
    avg = {}
    for key in states[0].keys():
        avg[key] = sum(sd[key] * float(wi) for sd, wi in zip(states, w))
    return avg


def _write_csv(path, rows):
    if not rows: return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


# ── App ───────────────────────────────────────────────────────────────────────
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # ── Hyperparams ───────────────────────────────────────────────────────────
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

    log(INFO, "[HFL-SETUP] fold=%d K=%d rounds=%d local_epochs=%d batch=%d noise=%.2f",
        fold, K, rounds, local_epochs, batch_size, noise_std)

    node_ids = sorted(list(grid.get_node_ids()))
    log(INFO, "[HFL-SETUP] Node IDs: %s", node_ids)

    # ── Step 1: Ask each client for its partition size and in_dim ─────────────
    log(INFO, "[HFL] Querying client metadata...")
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

    # Save partition manifest
    with open(run_dir / f"partition_manifest_fold{fold}.json", "w") as f:
        json.dump({
            "fold": fold, "K": K, "rounds": rounds,
            "local_epochs": local_epochs, "batch_size": batch_size,
            "noise_std": noise_std, "seed": seed,
            "bottom_arch": "in->16->8",
            "sizes": [int(s) for s in sizes.tolist()],
        }, f, indent=2)

    # ── Step 2: Init global model and broadcast to all clients ────────────────
    torch.manual_seed(seed)
    global_bottom = BottomMLP_Paper(in_dim=in_dim)
    global_recon  = ReconHead(emb_dim=8, out_dim=in_dim)

    bottom_state = {k: v.detach().cpu() for k, v in global_bottom.state_dict().items()}
    recon_state  = {k: v.detach().cpu() for k, v in global_recon.state_dict().items()}

    model_bytes = _state_bytes(bottom_state) + _state_bytes(recon_state)
    log(INFO, "[HFL] Model size: %.2f KB (bottom+recon)", model_bytes / 1024)

    # ── Step 3: FedAvg rounds ─────────────────────────────────────────────────
    total_comm_bytes = 0
    round_logs = []

    with open(run_dir / f"hfl_ssl_metrics_fold{fold}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "mean_recon_mse", "weighted_recon_mse",
                    "min_client_mse", "max_client_mse",
                    "round_comm_bytes", "round_comm_kb",
                    "cumulative_comm_bytes", "cumulative_comm_mb"])

        for r in range(1, rounds + 1):
            client_bottom_states = []
            client_recon_states  = []
            client_mses          = []
            round_comm = 0

            for node_idx, nid in enumerate(node_ids):
                # Send global weights → client (download cost)
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
                            "seed":         seed + 10000 * r + 97 * node_ids.index(nid),
                        }),
                    }),
                    dst_node_id=nid,
                    message_type="train.local_train",
                ))

                # Receive updated weights ← client (upload cost)
                arrs = rep.content["arrays"]
                recv_bottom = {k[len("bottom_"):]: torch.from_numpy(np.asarray(arrs[k].numpy()).astype(np.float32))
                               for k in arrs if k.startswith("bottom_")}
                recv_recon  = {k[len("recon_"):]:  torch.from_numpy(np.asarray(arrs[k].numpy()).astype(np.float32))
                               for k in arrs if k.startswith("recon_")}
                mse_val = float(rep.content["config"].get("recon_mse", 0.0))

                recv_bytes = _state_bytes(recv_bottom) + _state_bytes(recv_recon)
                round_comm += recv_bytes

                client_bottom_states.append(recv_bottom)
                client_recon_states.append(recv_recon)
                client_mses.append(mse_val)

            # FedAvg aggregation
            bottom_state = _fedavg(client_bottom_states, sizes)
            recon_state  = _fedavg(client_recon_states,  sizes)

            client_mses = np.array(client_mses)
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

            w.writerow([r, mean_mse, weighted_mse, min_mse, max_mse,
                        round_comm, round_comm / 1024,
                        total_comm_bytes, total_comm_bytes / 1024**2])
            f.flush()

            round_logs.append({
                "round": r, "mean_recon_mse": mean_mse,
                "round_comm_kb": round_comm / 1024,
                "cumulative_comm_mb": total_comm_bytes / 1024**2,
            })

    # ── Step 4: Save final checkpoint ─────────────────────────────────────────
    npz_path = os.environ.get("NPZ_PATH", "diabetes_vfl_cv.npz")

    # Get standardization params from first client
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

    # Save comm summary
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
    # ── Step 5: Option A — collect embeddings from fog nodes ──────────────────
    # Each fog node applies the converged encoder to its local slice of the
    # aligned cohort and returns embeddings. Raw data never leaves the fog node.
    # The server concatenates per-node embeddings into a single silo-level
    # embedding matrix and saves it alongside the checkpoint.
    #
    # Per-node index partitions are read from the pre-computed JSON partition
    # file (partitions_K{K}_train.json), guaranteeing consistency with the
    # partitions used during FedAvg training.
    log(INFO, "[HFL] Step 5: collecting aligned-cohort embeddings from fog nodes...")

    # Load pre-computed per-node partitions from JSON
    partition_json = os.environ.get(
        "PARTITION_JSON",
        f"partitions_K{K}_train.json"
    )
    import json as _json
    with open(partition_json) as _f:
        part_data = _json.load(_f)

    # folds is a list of dicts; fold index is fold-1
    fold_entry = part_data["folds"][fold - 1]
    client_indices = fold_entry["client_indices"]  # list of K lists

    if len(client_indices) != K:
        raise ValueError(
            f"Partition file has {len(client_indices)} nodes but K={K}. "
            f"Check PARTITION_JSON env var points to the correct file."
        )

    node_embeddings  = {}   # node_idx -> np.ndarray (n_local_aligned, emb_dim)
    node_aligned_idx = {}   # node_idx -> aligned indices in original X2 space

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

    # Reconstruct full aligned embedding matrix in original index order
    all_idx  = np.concatenate([node_aligned_idx[i] for i in range(K)])
    all_embs = np.concatenate([node_embeddings[i]  for i in range(K)], axis=0)
    sort_ord = np.argsort(all_idx)
    aligned_embeddings = all_embs[sort_ord]   # shape: (n_aligned, emb_dim)
    aligned_indices    = all_idx[sort_ord]    # original X2 row indices

    emb_path = run_dir / f"passive_aligned_embeddings_hfl_fold{fold}.npz"
    np.savez(emb_path,
             embeddings=aligned_embeddings,
             aligned_indices=aligned_indices,
             x2_mu=mu, x2_sd=sd,
             fold=np.array(fold))

    log(INFO, "[HFL] Aligned embeddings saved: shape=%s  path=%s",
        aligned_embeddings.shape, emb_path)
