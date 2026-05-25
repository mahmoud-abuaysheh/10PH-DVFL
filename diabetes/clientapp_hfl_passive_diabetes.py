# clientapp_hfl_passive_diabetes.py
# Flower 1.26.1 — Passive silo HFL client.
# Fix 1: node_idx passed by server in config (not env var)
# Fix 2: explicit float32 casting when unpacking Array objects
from __future__ import annotations
import os
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

_CACHE: Dict[str, Dict[str, Any]] = {}

def _key(ctx): 
    nid = getattr(ctx, "node_id", None)
    return str(nid) if nid is not None else str(id(ctx))

def _st(ctx):
    k = _key(ctx)
    if k not in _CACHE: _CACHE[k] = {}
    return _CACHE[k]

class BottomMLP_Paper(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU())
    def forward(self, x): return self.net(x)

class ReconHead(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim,16),nn.ReLU(),nn.Linear(16,out_dim))
    def forward(self, z): return self.net(z)

def _to_f32_tensor(arr, device):
    if hasattr(arr, "numpy"):    x = arr.numpy()
    elif hasattr(arr, "data"):   x = np.asarray(arr.data)
    else:                        x = np.asarray(arr)
    return torch.from_numpy(x.astype(np.float32)).to(device)

def _init_if_needed(ctx):
    st = _st(ctx)
    if st.get("ready", False): return
    npz_path = os.environ.get("NPZ_PATH", "diabetes_vfl_cv.npz")
    fold     = int(os.environ.get("FOLD", 1))
    seed     = int(os.environ.get("SEED", 42))
    K        = int(os.environ.get("K", 10))
    device   = os.environ.get("DEVICE", "cpu")
    node_id  = getattr(ctx, "node_id", 0)
    node_idx = int(st.get("node_idx", 0))  # set by server via get_metadata

    d = np.load(npz_path, allow_pickle=True)
    X2 = d["X2"].astype(np.float32)
    folds = list(d["folds"])
    split_obj = folds[fold - 1]
    split = split_obj.item() if hasattr(split_obj, "item") else split_obj
    tr = split["train"].astype(np.int64)

    mu  = X2[tr].mean(axis=0).astype(np.float32)
    sd  = (X2[tr].std(axis=0) + 1e-8).astype(np.float32)
    X2s = (X2 - mu) / sd

    rng = np.random.default_rng(seed)
    tr_s = tr.copy(); rng.shuffle(tr_s)
    parts  = np.array_split(tr_s, K)
    my_idx = parts[node_idx % K].astype(np.int64)

    dev = torch.device(device)
    st["X2"]      = torch.from_numpy(X2s).float().to(dev)
    st["my_idx"]  = my_idx
    st["in_dim"]  = int(X2.shape[1])
    st["n_train"] = len(my_idx)
    st["mu"]      = mu
    st["sd"]      = sd
    st["dev"]     = dev
    st["ready"]   = True
    print(f"[Client] node_id={node_id} node_idx={node_idx} fold={fold} partition={len(my_idx)} in_dim={st['in_dim']}")

app = ClientApp()

@app.query("get_metadata")
def get_metadata(msg: Message, ctx: Context) -> Message:
    st = _st(ctx)
    cfg = msg.content.get("config", None)
    if cfg is not None:
        st["node_idx"] = int(cfg.get("node_idx", 0))  # MUST set before _init_if_needed
    _init_if_needed(ctx)
    return Message(content=RecordDict({"config": ConfigRecord({
        "n_train": st["n_train"], "in_dim": st["in_dim"],
    })}), reply_to=msg)

@app.query("get_stats")
def get_stats(msg: Message, ctx: Context) -> Message:
    _init_if_needed(ctx)
    st = _st(ctx)
    return Message(content=RecordDict({"arrays": ArrayRecord({
        "mu": Array(st["mu"].astype(np.float32)),
        "sd": Array(st["sd"].astype(np.float32)),
    })}), reply_to=msg)

@app.train("local_train")
def local_train(msg: Message, ctx: Context) -> Message:
    _init_if_needed(ctx)
    st     = _st(ctx)
    dev    = st["dev"]
    in_dim = st["in_dim"]

    cfg          = msg.content["config"]
    local_epochs = int(cfg.get("local_epochs", 1))
    batch_size   = int(cfg.get("batch_size", 256))
    noise_std    = float(cfg.get("noise_std", 0.1))
    lr           = float(os.environ.get("LR", "1e-3"))
    seed_local   = int(cfg.get("seed", 42))

    arrs         = msg.content["arrays"]
    bottom_state = {k[len("bottom_"):]: _to_f32_tensor(arrs[k], dev)
                    for k in arrs if k.startswith("bottom_")}
    recon_state  = {k[len("recon_"):]:  _to_f32_tensor(arrs[k], dev)
                    for k in arrs if k.startswith("recon_")}

    bottom     = BottomMLP_Paper(in_dim=in_dim).to(dev)
    recon_head = ReconHead(emb_dim=8, out_dim=in_dim).to(dev)
    bottom.load_state_dict(bottom_state, strict=True)
    recon_head.load_state_dict(recon_state, strict=True)
    bottom.train(); recon_head.train()

    opt    = torch.optim.Adam(list(bottom.parameters())+list(recon_head.parameters()), lr=lr)
    mse_fn = nn.MSELoss()
    X2     = st["X2"]
    idx    = st["my_idx"]
    rng    = np.random.default_rng(seed_local)

    n, loss_sum = 0, 0.0
    for _ in range(local_epochs):
        idx_s = idx.copy(); rng.shuffle(idx_s)
        for s in range(0, len(idx_s), batch_size):
            b = idx_s[s:s+batch_size]
            if len(b) == 0: continue
            bt   = torch.from_numpy(b).long().to(dev)
            xb   = X2.index_select(0, bt)
            xn   = xb + noise_std * torch.randn_like(xb)
            opt.zero_grad(set_to_none=True)
            loss = mse_fn(recon_head(bottom(xn)), xb)
            loss.backward(); opt.step()
            loss_sum += float(loss.item()) * len(b); n += len(b)

    recon_mse = float(loss_sum / max(n, 1))
    upd_b = {k: v.detach().cpu().numpy().astype(np.float32) for k,v in bottom.state_dict().items()}
    upd_r = {k: v.detach().cpu().numpy().astype(np.float32) for k,v in recon_head.state_dict().items()}

    return Message(content=RecordDict({
        "arrays": ArrayRecord({
            **{f"bottom_{k}": Array(v) for k,v in upd_b.items()},
            **{f"recon_{k}":  Array(v) for k,v in upd_r.items()},
        }),
        "config": ConfigRecord({"recon_mse": recon_mse}),
    }), reply_to=msg)