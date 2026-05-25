# clientapp_vfl_midas_decoupled_byol.py
#
# SS-VFL-I faithful client implementation.
# Stage A (one-time): Client sends ALL pre-extracted BYOL+PCA embeddings
#                     (train + val + test) to server cache in one shot.
# Stage B onward:     Client is idle — server trains purely on cached embeddings.
#
# Communication cost: only the one-time embedding transfer is counted.
#
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

# ── per-node state cache ──────────────────────────────────────────────────────
_CACHE: Dict[str, Dict[str, Any]] = {}

def _node_key(ctx: Context) -> str:
    nid = getattr(ctx, "node_id", None)
    return str(nid) if nid is not None else str(id(ctx))

def _st(ctx: Context) -> Dict[str, Any]:
    k = _node_key(ctx)
    if k not in _CACHE:
        _CACHE[k] = {}
    return _CACHE[k]

# ── NPZ name mapping ──────────────────────────────────────────────────────────
NPZ_PREFIX = {
    "dscope": "features_dscope",
    "6in":    "features_6in",
    "1ft":    "features_1ft",
}

def _init_if_needed(ctx: Context) -> None:
    st = _st(ctx)
    if st.get("ready", False):
        return

    mod = st.get("MODALITY", "")
    if mod not in {"dscope", "6in", "1ft"}:
        return

    rc  = getattr(ctx, "run_config", {}) or {}
    art_dir = Path(os.environ.get("ART_DIR", str(rc.get("art_dir", "")))).resolve()
    fold    = int(os.environ.get("FOLD_NUM", rc.get("fold_num", "1")))

    npz_path = art_dir / f"{NPZ_PREFIX[mod]}_fold{fold}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Feature NPZ not found: {npz_path}\n"
            f"Run extract_features_byol_midas.py first."
        )

    d = np.load(npz_path, allow_pickle=True)
    st["X_train"] = d["X_train"].astype(np.float32)   # (423, 256)
    st["X_val"]   = d["X_val"].astype(np.float32)     # (105, 256)
    st["X_test"]  = d["X_test"].astype(np.float32)    # (132, 256)

    # Active silo also holds labels
    if mod == "dscope":
        active_npz = art_dir / f"active_dscope_fold{fold}.npz"
        if active_npz.exists():
            ld = np.load(active_npz, allow_pickle=True)
            st["y_train"] = ld["y_train"].astype(np.float32)
            st["y_val"]   = ld["y_val"].astype(np.float32)
            st["y_test"]  = ld["y_test"].astype(np.float32)

    st["modality"] = mod
    st["fold"]     = fold
    st["ready"]    = True


def _cfg(ctx: Context, st: Dict[str, Any], extra: Dict | None = None) -> ConfigRecord:
    nid = getattr(ctx, "node_id", None)
    d: Dict[str, Any] = {
        "node_id":  str(nid if nid is not None else -1),
        "modality": str(st.get("MODALITY", "")),
        "fold":     str(st.get("fold", "")),
    }
    if extra:
        d.update(extra)
    return ConfigRecord(d)


# ── Client App ────────────────────────────────────────────────────────────────
app = ClientApp()


@app.query("set_modality")
def set_modality(msg: Message, ctx: Context) -> Message:
    st = _st(ctx)
    if not st.get("MODALITY_LOCKED", False):
        try:
            mod = str(msg.content["config"].get("modality", "")).strip().lower()
            if mod:
                st["MODALITY"] = mod
                st["MODALITY_LOCKED"] = True
        except Exception:
            pass
    return Message(content=RecordDict({"config": _cfg(ctx, st)}), reply_to=msg)


@app.query("send_embeddings")
def send_embeddings(msg: Message, ctx: Context) -> Message:
    """
    SS-VFL-I Stage A: Send ALL embeddings for a given split to the server.
    This is called ONCE per split (train / val / test).
    Communication cost is measured at server side by counting bytes received.
    """
    st = _st(ctx)
    _init_if_needed(ctx)

    if not st.get("ready", False):
        empty = np.zeros((0, 256), dtype=np.float32)
        arrs  = ArrayRecord({"embeddings": Array(empty), "labels": Array(np.zeros(0, dtype=np.float32))})
        return Message(content=RecordDict({"arrays": arrs, "config": _cfg(ctx, st, {"error": "not_ready"})}), reply_to=msg)

    phase = str(msg.content["config"].get("phase", "train"))

    if phase == "train":
        X = st["X_train"]
        y = st.get("y_train", np.zeros(0, dtype=np.float32))
    elif phase == "val":
        X = st["X_val"]
        y = st.get("y_val", np.zeros(0, dtype=np.float32))
    else:  # test
        X = st["X_test"]
        y = st.get("y_test", np.zeros(0, dtype=np.float32))

    arrs = ArrayRecord({
        "embeddings": Array(X),
        "labels":     Array(y),
    })
    return Message(
        content=RecordDict({"arrays": arrs, "config": _cfg(ctx, st, {"phase": phase, "n_samples": str(len(X))})}),
        reply_to=msg,
    )


@app.query("ping")
def ping(msg: Message, ctx: Context) -> Message:
    """Health check — server calls this to verify connectivity."""
    st = _st(ctx)
    return Message(content=RecordDict({"config": _cfg(ctx, st, {"status": "ok"})}), reply_to=msg)