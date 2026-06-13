# serverapp_vfl_midas_decoupled_byol.py
# Flower 1.26.1 compatible — uses ServerApp() + @app.main()
#
# SS-VFL-I: One-time embedding transfer then pure server-side supervised training.
#
# Protocol:
#   Stage A — ONE-TIME embedding transfer:
#       Server requests ALL embeddings (train+val+test) from each client ONCE.
#       Server caches them. Communication cost measured here.
#       Clients are never contacted again after Stage A.
#   Stage B — Supervised training on cached embeddings (no client communication):
#       ROUNDS=20, EARLY_STOP_PATIENCE=7, MIN_DELTA=1e-4,
#       BATCH_SIZE=64, HEAD_LR=1e-4, HEAD_WD=1e-4, HEAD_HIDDEN=512.
#
# Environment variables (set before running flwr run .):
#   FOLD_NUM              — fold index (1-5)
#   ART_DIR               — directory with features_{modality}_fold{N}.npz
#   FOLD_NPZ_DIR          — directory with active_dscope_fold{N}.npz (for labels)
#   OUT_DIR               — output directory
#   SEED                  — random seed (default: 42)
#   ROUNDS                — max training rounds (default: 20)
#   EARLY_STOP_PATIENCE   — patience for early stopping (default: 7)
#   BATCH_SIZE            — batch size (default: 64)
#   HEAD_LR               — head learning rate (default: 1e-4)
#   HEAD_WD               — head weight decay (default: 1e-4)
#   HEAD_HIDDEN           — head hidden dim (default: 512)
#   EMB_DIM               — embedding dim per silo (default: 256)
from __future__ import annotations

import csv, os
from logging import INFO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp


# ── Env helpers ───────────────────────────────────────────────────────────────
def _env_int(k, d):   return int(os.environ.get(k, str(d)))
def _env_float(k, d): return float(os.environ.get(k, str(d)))

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _grid_request(grid, msg, timeout=600):
    reps = grid.send_and_receive([msg], timeout=timeout)
    if not reps or not reps[0].has_content():
        raise RuntimeError("Client reply empty / no content")
    return reps[0]

def _to_np(x):
    if isinstance(x, np.ndarray): return x
    if hasattr(x, "numpy"):       return x.numpy()
    if hasattr(x, "data"):        return np.asarray(x.data)
    return np.asarray(x)

def _sigmoid_np(x):
    x = x.astype(np.float64, copy=False)
    return 1.0 / (1.0 + np.exp(-x))

def _write_csv(path, rows):
    if not rows: return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def _threshold_sweep(y_true, y_prob, step=0.01):
    y_true = y_true.astype(int).reshape(-1)
    y_prob = y_prob.astype(float).reshape(-1)
    rows = []
    for thr in np.arange(0.0, 1.0001, step):
        yp   = (y_prob >= thr).astype(int)
        tp   = int(((y_true==1)&(yp==1)).sum())
        tn   = int(((y_true==0)&(yp==0)).sum())
        fp   = int(((y_true==0)&(yp==1)).sum())
        fn   = int(((y_true==1)&(yp==0)).sum())
        acc  = (tp+tn)/max(1,tp+tn+fp+fn)
        prec = tp/max(1,tp+fp)
        rec  = tp/max(1,tp+fn)
        spec = tn/max(1,tn+fp)
        f1   = 2*prec*rec/max(1e-12,prec+rec)
        rows.append(dict(threshold=float(thr), acc=float(acc), f1=float(f1),
                         precision=float(prec), recall=float(rec),
                         specificity=float(spec), youdenJ=float(rec+spec-1),
                         tp=tp, tn=tn, fp=fp, fn=fn))
    return rows

def _pick_best(rows, key):
    return sorted(rows, key=lambda r: (-r[key], -r["f1"], -r["acc"], r["threshold"]))[0]


# ── Model — identical to standard VFL OldMLP ─────────────────────────────────
class OldMLP(nn.Module):
    def __init__(self, in_dim, hidden=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(x)


# ── RPC helpers ───────────────────────────────────────────────────────────────
def rpc_set_modality(grid, dst, modality):
    msg = Message(
        message_type="query.set_modality",
        content=RecordDict({"config": ConfigRecord({"modality": modality})}),
        dst_node_id=dst,
    )
    _grid_request(grid, msg)


def rpc_send_embeddings(grid, dst, phase):
    """Stage A: pull ALL embeddings for one split from one client."""
    msg = Message(
        message_type="query.send_embeddings",
        content=RecordDict({
            "arrays": ArrayRecord({}),
            "config": ConfigRecord({"phase": phase}),
        }),
        dst_node_id=dst,
    )
    rep  = _grid_request(grid, msg, timeout=300)
    emb  = _to_np(rep.content["arrays"]["embeddings"]).astype(np.float32)
    lbl  = _to_np(rep.content["arrays"]["labels"]).astype(np.float32)
    comm = int(emb.nbytes + lbl.nbytes)
    return emb, lbl, comm


# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _eval_split(head, X, y_np, device, batch_size):
    head.eval()
    probs_all = []
    for i in range(0, len(y_np), batch_size):
        xb     = X[i:i+batch_size].to(device)
        logits = head(xb).detach().cpu().numpy().reshape(-1)
        probs_all.append(_sigmoid_np(logits).astype(np.float32))
    probs = np.concatenate(probs_all)
    auroc = float(roc_auc_score(y_np, probs))
    ap    = float(average_precision_score(y_np, probs))
    return auroc, ap, probs


# ── App ───────────────────────────────────────────────────────────────────────
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Hyperparams
    seed         = _env_int("SEED", 42)
    fold         = _env_int("FOLD_NUM", 1)
    emb_dim      = _env_int("EMB_DIM", 256)
    rounds       = _env_int("ROUNDS", 20)
    patience     = _env_int("EARLY_STOP_PATIENCE", 7)
    min_delta    = _env_float("MIN_DELTA", 1e-4)
    batch_size   = _env_int("BATCH_SIZE", 64)
    head_hidden  = _env_int("HEAD_HIDDEN", 512)
    head_dropout = _env_float("HEAD_DROPOUT", 0.2)
    head_lr      = _env_float("HEAD_LR", 1e-4)
    head_wd      = _env_float("HEAD_WD", 1e-4)
    eval_every   = _env_int("EVAL_EVERY", 1)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(os.environ.get("OUT_DIR", "runs_midas_dvfl_byol"))
    run_dir = out_dir / f"fold{fold}_byol_decoupled"
    run_dir.mkdir(parents=True, exist_ok=True)

    log(INFO, "[SETUP] fold=%d seed=%d device=%s rounds=%d patience=%d batch=%d lr=%s",
        fold, seed, device, rounds, patience, batch_size, head_lr)

    # ── Node assignment ───────────────────────────────────────────────────────
    node_ids = sorted(list(grid.get_node_ids()))
    log(INFO, "[SETUP] Node IDs: %s", node_ids)
    if len(node_ids) != 3:
        raise ValueError(f"Expected 3 clients, got {len(node_ids)}")

    env_active = os.environ.get("ACTIVE_DST", "").strip()
    env_p6     = os.environ.get("P6_DST", "").strip()
    env_p1     = os.environ.get("P1_DST", "").strip()

    if env_active and env_p6 and env_p1:
        active_dst, p6_dst, p1_dst = int(env_active), int(env_p6), int(env_p1)
    else:
        active_dst, p6_dst, p1_dst = node_ids[0], node_ids[1], node_ids[2]

    log(INFO, "[SETUP] active=%d(dscope) p6=%d(6in) p1=%d(1ft)",
        active_dst, p6_dst, p1_dst)

    rpc_set_modality(grid, active_dst, "dscope")
    rpc_set_modality(grid, p6_dst,    "6in")
    rpc_set_modality(grid, p1_dst,    "1ft")

    # =========================================================================
    # STAGE A — One-time embedding transfer
    # =========================================================================
    log(INFO, "=== Stage A: One-time embedding transfer ===")

    total_comm_bytes = 0
    comm_log = []
    cache  = {"dscope": {}, "6in": {}, "1ft": {}}

    for phase in ["train", "val", "test"]:
        for dst, mod in [(active_dst,"dscope"), (p6_dst,"6in"), (p1_dst,"1ft")]:
            emb, lbl, comm = rpc_send_embeddings(grid, dst, phase)
            cache[mod][phase] = emb
            total_comm_bytes += comm
            comm_log.append({"modality":mod, "phase":phase,
                             "n_samples":len(emb), "emb_dim":emb.shape[1],
                             "bytes":comm, "kb":comm/1024})
            log(INFO, "[StageA] %s/%s shape=%s comm=%.2f KB",
                mod, phase, emb.shape, comm/1024)

    log(INFO, "[StageA] TOTAL comm: %.4f MB", total_comm_bytes/1024**2)
    _write_csv(run_dir/"stageA_comm_cost.csv", comm_log)
    _write_csv(run_dir/"stageA_comm_summary.csv", [{
        "total_bytes": total_comm_bytes,
        "total_kb":    total_comm_bytes/1024,
        "total_mb":    total_comm_bytes/1024**2,
    }])

    # Load labels from fold_npz on server side
    fold_npz_dir = Path(os.environ.get("FOLD_NPZ_DIR", "fold_npz"))
    active_npz   = fold_npz_dir / f"active_dscope_fold{fold}.npz"
    log(INFO, "[StageA] Loading labels from %s", active_npz)
    ld = np.load(active_npz, allow_pickle=True)
    labels = {
        "train": ld["y_train"].astype(np.int64),
        "val":   ld["y_val"].astype(np.int64),
        "test":  ld["y_test"].astype(np.int64),
    }
    log(INFO, "[StageA] Labels: train=%d val=%d test=%d",
        len(labels["train"]), len(labels["val"]), len(labels["test"]))

    def _concat(phase):
        return torch.from_numpy(np.concatenate(
            [cache["dscope"][phase], cache["6in"][phase], cache["1ft"][phase]], axis=1
        )).float()

    X_train = _concat("train")
    X_val   = _concat("val")
    X_test  = _concat("test")
    y_train = labels["train"]
    y_val   = labels["val"]
    y_test  = labels["test"]

    log(INFO, "[StageA] train=%s val=%s test=%s", X_train.shape, X_val.shape, X_test.shape)

    # =========================================================================
    # STAGE B — Supervised training on cached embeddings (no client comm)
    # =========================================================================
    log(INFO, "=== Stage B: Supervised training (no client comm) ===")

    head      = OldMLP(in_dim=emb_dim*3, hidden=head_hidden, dropout=head_dropout).to(device)
    opt       = torch.optim.AdamW(head.parameters(), lr=head_lr, weight_decay=head_wd)
    criterion = nn.BCEWithLogitsLoss()
    y_train_t = torch.from_numpy(y_train).float().view(-1, 1)
    n_train   = len(y_train)

    best_val_auroc = -1.0
    best_round     = 0
    bad            = 0
    best_path      = run_dir / "head_best.pt"

    with (run_dir/"stageB_train_log.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "train_loss", "val_auroc", "val_prauc"])

        for r in range(1, rounds + 1):
            head.train()
            perm      = np.random.permutation(n_train)
            tr_losses = []

            for start in range(0, n_train, batch_size):
                idx  = perm[start:start+batch_size]
                xb   = X_train[idx].to(device)
                yb   = y_train_t[idx].to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(head(xb), yb)
                loss.backward()
                opt.step()
                tr_losses.append(float(loss.item()))

            avg_loss = float(np.mean(tr_losses))

            if (r % eval_every) == 0:
                val_auroc, val_ap, _ = _eval_split(head, X_val, y_val, device, batch_size)
                log(INFO, "[Round %02d/%d] loss=%.4f val_auroc=%.4f val_prauc=%.4f",
                    r, rounds, avg_loss, val_auroc, val_ap)
                w.writerow([r, avg_loss, val_auroc, val_ap]); f.flush()

                if val_auroc > best_val_auroc + min_delta:
                    best_val_auroc = val_auroc
                    best_round     = r
                    bad            = 0
                    torch.save(head.state_dict(), best_path)
                    log(INFO, "[Round %02d] New best val_auroc=%.4f saved.", r, best_val_auroc)
                else:
                    bad += 1
                    if bad >= patience:
                        log(INFO, "[EarlyStopping] patience=%d at round %d.", patience, r)
                        break
            else:
                log(INFO, "[Round %02d/%d] loss=%.4f", r, rounds, avg_loss)
                w.writerow([r, avg_loss, "", ""]); f.flush()

    # ── Final evaluation ──────────────────────────────────────────────────────
    if best_path.exists():
        head.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
        log(INFO, "[FINAL] Loaded best (round=%d val_auroc=%.4f)", best_round, best_val_auroc)

    val_auroc,  val_ap,  val_probs  = _eval_split(head, X_val,  y_val,  device, batch_size)
    test_auroc, test_ap, test_probs = _eval_split(head, X_test, y_test, device, batch_size)

    thr_val  = _threshold_sweep(y_val,  val_probs)
    thr_test = _threshold_sweep(y_test, test_probs)
    _write_csv(run_dir/"threshold_metrics_val.csv",  thr_val)
    _write_csv(run_dir/"threshold_metrics_test.csv", thr_test)

    best_you = _pick_best(thr_val, "youdenJ")
    best_f1  = _pick_best(thr_val, "f1")

    summary = {
        "fold":                fold,
        "best_round":          best_round,
        "val_auroc":           val_auroc,
        "val_pr_auc":          val_ap,
        "test_auroc":          test_auroc,
        "test_pr_auc":         test_ap,
        "val_thr_best_youden": best_you["threshold"],
        "val_thr_best_f1":     best_f1["threshold"],
        "total_comm_bytes":    total_comm_bytes,
        "total_comm_mb":       total_comm_bytes/1024**2,
    }
    _write_csv(run_dir/"summary.csv", [summary])

    log(INFO, "[RESULT] fold=%d test_auroc=%.4f test_prauc=%.4f comm=%.4f MB",
        fold, test_auroc, test_ap, total_comm_bytes/1024**2)
