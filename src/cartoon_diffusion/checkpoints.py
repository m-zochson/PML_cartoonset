"""Checkpoint helpers shared by training, sampling, and evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def load_checkpoint(path: str | os.PathLike[str], map_location=None):
    return torch.load(path, map_location=map_location)


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    step: int,
    model,
    ema,
    opt=None,
    attrs,
    attribute_dims,
    image_size: int,
    timesteps: int,
    precision: str | None = None,
    grad_clip: float | None = None,
    lr_schedule: str | None = None,
    min_lr: float | None = None,
    scaler=None,
) -> None:
    """Save a training checkpoint with one-level .bak rotation."""
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    payload = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.shadow.state_dict(),
        "attribute_dims": list(attribute_dims),
        "attrs": list(attrs),
        "image_size": image_size,
        "timesteps": timesteps,
    }
    if precision is not None:
        payload["precision"] = precision
    if grad_clip is not None:
        payload["grad_clip"] = grad_clip
    if lr_schedule is not None:
        payload["lr_schedule"] = lr_schedule
    if min_lr is not None:
        payload["min_lr"] = min_lr
    if opt is not None:
        payload["opt"] = opt.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def checkpoint_stem(path: str | os.PathLike[str]) -> str:
    return Path(path).stem
