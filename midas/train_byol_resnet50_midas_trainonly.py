#!/usr/bin/env python3
"""
train_byol_resnet50_midas_trainonly.py

Corrected MIDAS BYOL pretraining script.

Purpose:
- Train BYOL encoders for MIDAS using TRAINING images only.
- Validation and test images are excluded from BYOL pretraining.
- This avoids validation-set contamination and keeps validation independent
  for checkpoint/model selection in downstream experiments.

Outputs:
- byol_{modality}_fold{fold}.pt  — final epoch encoder used for feature extraction

Note: Following standard BYOL practice, the final epoch checkpoint is used
for downstream feature extraction. The BYOL loss is governed by EMA dynamics
and is not a reliable standalone checkpoint selection criterion.
"""

from __future__ import annotations

import os
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Image lookup
# ---------------------------------------------------------------------------

def build_image_lookup(image_root: str) -> Dict[str, str]:
    """
    Scan image_root once and create a lookup:
    lowercase filename -> absolute path.
    This avoids repeated os.listdir calls on WSL/NTFS.
    """
    lookup: Dict[str, str] = {}

    for fname in os.listdir(image_root):
        lookup[fname.lower()] = os.path.join(image_root, fname)

    print(f"[DATA] Lookup built: {len(lookup)} files indexed from {image_root}")
    return lookup


def resolve_from_lookup(lookup: Dict[str, str], filename: str) -> str:
    """
    Resolve filename using the pre-built lookup.
    Supports .jpg, .jpeg, and *_cropped variants.
    """
    base, _ = os.path.splitext(str(filename))

    for ext in [".jpg", ".jpeg"]:
        for candidate in [base + ext, base + "_cropped" + ext]:
            hit = lookup.get(candidate.lower())
            if hit:
                return hit

    raise FileNotFoundError(f"Image not found in lookup: {filename}")


# ---------------------------------------------------------------------------
# Fold file naming
# ---------------------------------------------------------------------------

NPZ_NAME = {
    "dscope": "active_dscope",
    "6in": "passive_6in",
    "1ft": "passive_1ft",
}


def load_pretrain_paths(fold_npz_dir: str, modality: str, fold: int) -> List[str]:
    """
    Load fold-specific image paths for BYOL pretraining.

    Corrected behavior:
    - Use TRAINING paths only.
    - Exclude validation paths from BYOL.
    - Exclude test paths from BYOL.

    This ensures:
    - BYOL representation learning uses only the training partition.
    - Validation remains independent for downstream checkpoint/model selection.
    - Test remains fully unseen.
    """
    npz_path = Path(fold_npz_dir) / f"{NPZ_NAME[modality]}_fold{fold}.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Fold NPZ not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)

    paths_train = d["paths_train"].astype(str).tolist()
    paths_val = d["paths_val"].astype(str).tolist()
    paths_test = d["paths_test"].astype(str).tolist()

    # IMPORTANT FIX:
    # Use only training paths for BYOL pretraining.
    all_paths = sorted(set(paths_train))

    print(f"[DATA] fold={fold} modality={modality}")
    print(f"[DATA] train={len(paths_train)} used for BYOL")
    print(f"[DATA] val={len(paths_val)} EXCLUDED from BYOL")
    print(f"[DATA] test={len(paths_test)} EXCLUDED from BYOL")
    print(f"[DATA] unique SSL pretraining paths={len(all_paths)}")

    if len(all_paths) == 0:
        raise RuntimeError(f"No training paths found for fold={fold}, modality={modality}")

    return all_paths


# ---------------------------------------------------------------------------
# BYOL augmentations
# ---------------------------------------------------------------------------

def make_byol_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(0.1, 0.1, 0.1, 0.0)],
            p=0.8,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(23, sigma=(0.1, 2.0))],
            p=0.5,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class BYOLDataset(Dataset):
    def __init__(self, paths: List[str], image_root: str, image_size: int = 224):
        self.transform = make_byol_transform(image_size)

        lookup = build_image_lookup(image_root)
        self.abs_paths = [resolve_from_lookup(lookup, p) for p in paths]

        print(f"[DATA] All {len(self.abs_paths)} training-only paths resolved. Ready.")

    def __len__(self) -> int:
        return len(self.abs_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.abs_paths[idx]).convert("RGB")
        return self.transform(img), self.transform(img)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_encoder() -> nn.Module:
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    return nn.Sequential(*list(base.children())[:-1])


class OnlineNetwork(nn.Module):
    def __init__(
        self,
        proj_hidden: int = 4096,
        proj_out: int = 256,
        pred_hidden: int = 4096,
    ):
        super().__init__()
        self.encoder = build_encoder()
        self.projector = MLP(2048, proj_hidden, proj_out)
        self.predictor = MLP(proj_out, pred_hidden, proj_out)

    def forward(self, x: torch.Tensor):
        h = torch.flatten(self.encoder(x), 1)
        z = self.projector(h)
        p = self.predictor(z)
        return p, z


class TargetNetwork(nn.Module):
    def __init__(
        self,
        proj_hidden: int = 4096,
        proj_out: int = 256,
    ):
        super().__init__()
        self.encoder = build_encoder()
        self.projector = MLP(2048, proj_hidden, proj_out)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.flatten(self.encoder(x), 1)
        return self.projector(h)


# ---------------------------------------------------------------------------
# BYOL objective and EMA update
# ---------------------------------------------------------------------------

def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()


@torch.no_grad()
def ema_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)


def cosine_tau(
    step: int,
    total_steps: int,
    tau_base: float = 0.996,
    tau_max: float = 1.0,
) -> float:
    return tau_max - (tau_max - tau_base) * (
        math.cos(math.pi * step / total_steps) + 1
    ) / 2.0


def save_encoder(online: OnlineNetwork, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(online.encoder.state_dict(), path)
    print(f"[BYOL] Encoder saved -> {path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 70}")
    print("[BYOL TRAIN-ONLY VERSION]")
    print(f"[BYOL] modality={args.modality} | fold={args.fold} | device={device}")
    print(f"[BYOL] epochs={args.epochs} | batch={args.batch_size} | lr={args.lr}")
    print(f"[BYOL] validation and test images are excluded from BYOL pretraining")
    print(f"[BYOL] final epoch checkpoint will be used for feature extraction")
    print(f"{'=' * 70}")

    paths = load_pretrain_paths(args.fold_npz_dir, args.modality, args.fold)

    dataset = BYOLDataset(
        paths=paths,
        image_root=args.image_root,
        image_size=args.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )

    if len(loader) == 0:
        raise RuntimeError(
            f"DataLoader is empty. Reduce --batch_size below {args.batch_size}."
        )

    print(f"[BYOL] SSL training images={len(dataset)} | steps/epoch={len(loader)}")

    online = OnlineNetwork(
        proj_hidden=args.proj_hidden,
        proj_out=args.proj_out,
        pred_hidden=args.pred_hidden,
    ).to(device)

    target = TargetNetwork(
        proj_hidden=args.proj_hidden,
        proj_out=args.proj_out,
    ).to(device)

    target.encoder.load_state_dict(online.encoder.state_dict())
    target.projector.load_state_dict(online.projector.state_dict())

    for param in target.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        online.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(loader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.01,
    )

    use_amp = bool(args.amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(args.out_dir)
    ckpt_final = out_dir / f"byol_{args.modality}_fold{args.fold}.pt"

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        online.train()
        epoch_loss = 0.0

        for v1, v2 in loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                p1, _ = online(v1)
                p2, _ = online(v2)

                zt1 = target(v1)
                zt2 = target(v2)

                loss = 0.5 * byol_loss(p1, zt2) + 0.5 * byol_loss(p2, zt1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            tau = cosine_tau(
                step=global_step,
                total_steps=total_steps,
                tau_base=args.tau_base,
                tau_max=args.tau_max,
            )

            ema_update(online, target, tau)

            epoch_loss += float(loss.item())
            global_step += 1

        avg_loss = epoch_loss / len(loader)
        lr_now = scheduler.get_last_lr()[0]

        tau_now = cosine_tau(
            step=global_step,
            total_steps=total_steps,
            tau_base=args.tau_base,
            tau_max=args.tau_max,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | "
            f"lr={lr_now:.2e} | "
            f"tau={tau_now:.4f}"
        )

    # Save final epoch encoder — used for downstream feature extraction
    save_encoder(online, ckpt_final)

    print(f"\n[BYOL] Done: modality={args.modality} fold={args.fold}")
    print(f"[BYOL] Final checkpoint -> {ckpt_final}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold_npz_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument(
        "--modality",
        required=True,
        choices=["dscope", "6in", "1ft"],
    )
    parser.add_argument(
        "--fold",
        required=True,
        type=int,
        choices=[1, 2, 3, 4, 5],
    )
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--proj_hidden", type=int, default=4096)
    parser.add_argument("--proj_out", type=int, default=256)
    parser.add_argument("--pred_hidden", type=int, default=4096)

    parser.add_argument("--tau_base", type=float, default=0.996)
    parser.add_argument("--tau_max", type=float, default=1.0)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
