# clientapp_vfl_midas_splitnn_proj256.py
# Changes vs original:
#   - Fold-specific NPZ: FOLD_NPZ_DIR/active_dscope_fold{N}.npz
#   - FOLD_NUM env var selects which fold
#   - Fast image lookup (build once, avoid os.listdir per image)
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import log

INFO = 20


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


ART_DIR      = Path(os.environ.get("ART_DIR", ".")).resolve()
FOLD_NPZ_DIR = Path(os.environ.get("FOLD_NPZ_DIR", str(ART_DIR))).resolve()
FOLD_NUM     = _env_int("FOLD_NUM", 1)
EMB_DIM      = _env_int("EMB_DIM", 256)
DEVICE       = os.environ.get("DEVICE", "cpu")
DEVICE_T     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIENT_LR    = _env_float("CLIENT_LR", 1e-4)
CLIENT_WD    = _env_float("CLIENT_WD", 1e-4)
FREEZE_BB    = _env_int("FREEZE_BACKBONE", 0)
SEED         = _env_int("SEED", 42)
PROJ_HID     = _env_int("PROJ_HIDDEN", 512)
PROJ_DROP    = _env_float("PROJ_DROPOUT", 0.2)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Build lookup once per image_root to avoid os.listdir per image
_image_lookup: Dict[str, str] = {}

def _build_lookup(image_root: str) -> None:
    global _image_lookup
    if _image_lookup:
        return
    for fname in os.listdir(image_root):
        _image_lookup[fname.lower()] = os.path.join(image_root, fname)

def resolve_image_path(image_root: str, filename: str) -> str:
    _build_lookup(image_root)
    filename = str(filename)
    base, _ = os.path.splitext(filename)
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        for candidate in [base + ext, base + "_cropped" + ext]:
            found = _image_lookup.get(candidate.lower())
            if found:
                return found
    raise FileNotFoundError(f"Image not found in {image_root}: {filename}")


class ProjectionMLP(nn.Module):
    """ResNet2048 -> PROJ_HID -> EMB_DIM with LayerNorm/GELU/Dropout."""
    def __init__(self, in_dim: int = 2048, hidden: int = 512,
                 out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_actor_states: Dict[int, "RoleState"] = {}


class RoleState:
    def __init__(self) -> None:
        self.role:       Optional[str]                   = None
        self.image_root: Optional[str]                   = None
        self.paths:      Dict[str, np.ndarray]           = {}
        self.y:          Dict[str, np.ndarray]           = {}
        self.backbone:   Optional[nn.Module]             = None
        self.proj:       Optional[nn.Module]             = None
        self.optim:      Optional[torch.optim.Optimizer] = None
        self.tf_train    = None
        self.tf_eval     = None

    def load(self, role: str, image_root: str) -> None:
        self.role       = role
        self.image_root = image_root
        set_seed(SEED)

        # ── Fold-specific NPZ ─────────────────────────────────────────────────
        npz_map = {
            "active": FOLD_NPZ_DIR / f"active_dscope_fold{FOLD_NUM}.npz",
            "p6":     FOLD_NPZ_DIR / f"passive_6in_fold{FOLD_NUM}.npz",
            "p1":     FOLD_NPZ_DIR / f"passive_1ft_fold{FOLD_NUM}.npz",
        }
        npz_path = npz_map.get(role)
        if npz_path is None:
            raise RuntimeError(f"Unknown role: {role}")
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing NPZ for role={role}: {npz_path}")

        d = np.load(npz_path, allow_pickle=True)
        for split in ["train", "val", "test"]:
            self.paths[split] = d[f"paths_{split}"].astype(str)

        if role == "active":
            for split in ["train", "val", "test"]:
                self.y[split] = d[f"y_{split}"].astype(np.float32).reshape(-1)

        # ── Transforms ────────────────────────────────────────────────────────
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        self.tf_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.tf_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # ── ResNet50 + ProjectionMLP ───────────────────────────────────────────
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        backbone.to(DEVICE_T)
        if FREEZE_BB == 1:
            for p in backbone.parameters():
                p.requires_grad = False

        proj   = ProjectionMLP(2048, PROJ_HID, EMB_DIM, PROJ_DROP).to(DEVICE_T)
        params = [p for p in list(backbone.parameters()) + list(proj.parameters())
                  if p.requires_grad]
        optim  = torch.optim.AdamW(params, lr=CLIENT_LR, weight_decay=CLIENT_WD)

        self.backbone = backbone
        self.proj     = proj
        self.optim    = optim

        log(INFO, f"[CLIENT {role}] NPZ={npz_path.name} fold={FOLD_NUM} | "
                  f"FREEZE={FREEZE_BB} | PROJ=2048->{PROJ_HID}->{EMB_DIM} | "
                  f"device={DEVICE_T} | n_train={len(self.paths['train'])}")

    def _load_batch(self, split: str, indices: List[int],
                    training: bool) -> torch.Tensor:
        tf   = self.tf_train if training else self.tf_eval
        imgs = []
        for i in indices:
            path = resolve_image_path(self.image_root, self.paths[split][int(i)])
            imgs.append(tf(Image.open(path).convert("RGB")))
        return torch.stack(imgs).to(DEVICE_T)

    def forward_embeddings(self, split: str, indices: List[int]) -> torch.Tensor:
        training = (split == "train")
        x = self._load_batch(split, indices, training)
        self.backbone.train(training and FREEZE_BB == 0)
        self.proj.train(training)
        with torch.set_grad_enabled(training):
            emb = self.proj(self.backbone(x))
        return emb

    def apply_gradients(self, split: str, indices: List[int],
                        grad: np.ndarray) -> None:
        x   = self._load_batch(split, indices, training=True)
        self.backbone.train(FREEZE_BB == 0)
        self.proj.train(True)
        emb = self.proj(self.backbone(x))
        g   = torch.from_numpy(grad.astype(np.float32)).to(DEVICE_T)
        if g.shape != emb.shape:
            raise RuntimeError(
                f"Grad shape {tuple(g.shape)} != emb shape {tuple(emb.shape)}")
        self.optim.zero_grad(set_to_none=True)
        emb.backward(g)
        self.optim.step()


def _get_state(context: Context) -> RoleState:
    nid = context.node_id
    if nid in _actor_states:
        return _actor_states[nid]
    state  = RoleState()
    cfg_st = context.state.config_records
    if "role_cfg" in cfg_st:
        saved = cfg_st["role_cfg"]
        state.load(str(saved["role"]), str(saved["image_root"]))
    _actor_states[nid] = state
    return state


def _cfg_get(rec: ConfigRecord, key: str, default=None):
    try:
        return rec[key]
    except Exception:
        return default


app = ClientApp()


@app.query("init_role")
def init_role(message: Message, context: Context) -> Message:
    cfg        = message.content["config"]
    role       = str(_cfg_get(cfg, "role"))
    image_root = str(_cfg_get(cfg, "image_root", ""))

    if role not in {"active", "p6", "p1"}:
        raise RuntimeError(f"Bad role: {role}")
    if not image_root or not os.path.exists(image_root):
        raise RuntimeError(f"IMAGE_ROOT invalid or missing: {image_root}")

    context.state.config_records["role_cfg"] = ConfigRecord({
        "role": role, "image_root": image_root,
    })
    state = RoleState()
    state.load(role, image_root)
    _actor_states[context.node_id] = state

    return Message(
        content=RecordDict({"config": ConfigRecord({"ok": True})}),
        reply_to=message,
    )


@app.query("meta")
def query_meta(message: Message, context: Context) -> Message:
    state = _get_state(context)
    if state.role != "active":
        raise RuntimeError(f"Only active serves labels (role={state.role})")
    arrs = ArrayRecord({
        "y_train": Array(state.y["train"]),
        "y_val":   Array(state.y["val"]),
        "y_test":  Array(state.y["test"]),
    })
    return Message(content=RecordDict({"arrays": arrs}), reply_to=message)


@app.query("get_embeddings")
def query_get_embeddings(message: Message, context: Context) -> Message:
    state   = _get_state(context)
    cfg     = message.content["config"]
    split   = str(cfg["split"])
    pos     = int(cfg["pos"])
    indices = list(cfg["indices"])
    emb     = (state.forward_embeddings(split, indices)
                   .detach().cpu().numpy().astype(np.float32))
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord({"embedding": Array(emb)}),
            "config": ConfigRecord({"pos": pos}),
        }),
        reply_to=message,
    )


@app.train("apply_gradients")
def train_apply_gradients(message: Message, context: Context) -> Message:
    state   = _get_state(context)
    cfg     = message.content["config"]
    split   = str(cfg["split"])
    indices = list(cfg["indices"])
    g       = message.content["gradients"]["local_gradients"].numpy()
    state.apply_gradients(split, indices, g)
    return Message(
        content=RecordDict({"config": ConfigRecord({"ok": True})}),
        reply_to=message,
    )