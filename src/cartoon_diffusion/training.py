"""Training loop for the conditional Cartoon Set DDPM."""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime

import torch

from cartoon_diffusion.checkpoints import load_checkpoint, save_checkpoint
from cartoon_diffusion.data import CartoonSetDataset, build_dataloader
from cartoon_diffusion.diffusion import GaussianDiffusion
from cartoon_diffusion.model import UNet


@dataclass
class TrainConfig:
    root: str
    attrs: list[str]
    image_size: int = 32
    batch: int = 128
    steps: int = 40000
    lr: float = 2e-4
    lr_schedule: str = "constant"
    min_lr: float = 0.0
    precision: str = "fp32"
    grad_clip: float | None = None
    timesteps: int = 1000
    p_uncond: float = 0.1
    limit: int | None = None
    workers: int = 0
    ckpt: str | None = None
    resume: bool = False
    save_every: int = 5000
    cache: bool = False
    rebuild_cache: bool = False


class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def infinite(loader):
    while True:
        yield from loader


def default_root(dataset_variant: str) -> str:
    return "data/cartoonset100k" if dataset_variant == "100k" else "data/cartoonset10k"


def default_classifier_ckpt(dataset_variant: str) -> str:
    return "classifier100k.pt" if dataset_variant == "100k" else "classifier.pt"


def normalize_precision(precision: str) -> str:
    aliases = {
        "32": "fp32",
        "float32": "fp32",
        "16": "fp16",
        "float16": "fp16",
        "amp": "fp16",
        "amp_fp16": "fp16",
        "mixed16": "fp16",
        "bfloat16": "bf16",
        "amp_bf16": "bf16",
    }
    value = aliases.get(str(precision).lower(), str(precision).lower())
    if value not in {"fp32", "fp16", "bf16"}:
        raise ValueError("--precision must be one of: fp32, fp16, bf16")
    return value


def amp_dtype_for(precision: str, device: str):
    precision = normalize_precision(precision)
    if precision == "fp32":
        return precision, False, None
    if device != "cuda":
        raise ValueError(f"precision={precision!r} requires CUDA; use precision=fp32 on CPU")
    if precision == "bf16":
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise ValueError("precision='bf16' was requested, but this CUDA device does not support bf16")
        return precision, True, torch.bfloat16
    return precision, True, torch.float16


def validate_grad_clip(grad_clip: float | None) -> float | None:
    if grad_clip is None:
        return None
    value = float(grad_clip)
    if value <= 0:
        raise ValueError("--grad_clip must be positive, or omitted to disable clipping")
    return value


def normalize_lr_schedule(lr_schedule: str) -> str:
    value = str(lr_schedule).lower()
    if value not in {"constant", "cosine"}:
        raise ValueError("--lr_schedule must be one of: constant, cosine")
    return value


def validate_min_lr(lr: float, min_lr: float) -> float:
    value = float(min_lr)
    if value < 0:
        raise ValueError("--min_lr must be non-negative")
    if value > float(lr):
        raise ValueError("--min_lr must be <= --lr")
    return value


def learning_rate_for_step(
    *,
    base_lr: float,
    min_lr: float,
    schedule: str,
    step: int,
    total_steps: int,
) -> float:
    schedule = normalize_lr_schedule(schedule)
    if schedule == "constant":
        return float(base_lr)
    if total_steps <= 1:
        return float(min_lr)
    progress = min(max((step - 1) / (total_steps - 1), 0.0), 1.0)
    factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (base_lr - min_lr) * factor)


def set_optimizer_lr(opt, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def train(config: TrainConfig) -> None:
    if config.ckpt is None:
        if config.resume:
            raise ValueError("--resume needs an explicit --ckpt")
        config.ckpt = datetime.now().strftime("run_%Y%m%d_%H%M%S.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision, amp_enabled, amp_dtype = amp_dtype_for(config.precision, device)
    grad_clip = validate_grad_clip(config.grad_clip)
    lr_schedule = normalize_lr_schedule(config.lr_schedule)
    min_lr = validate_min_lr(config.lr, config.min_lr)
    clip_msg = "off" if grad_clip is None else f"{grad_clip:g}"
    print(
        f"device: {device}   precision: {precision}   grad_clip: {clip_msg}   "
        f"lr_schedule: {lr_schedule}"
    )

    ds = CartoonSetDataset(
        config.root,
        image_size=config.image_size,
        cond_attributes=tuple(config.attrs),
        limit=config.limit,
        cache=config.cache,
        rebuild_cache=config.rebuild_cache,
    )
    print(f"images: {len(ds)}  attrs: {config.attrs}  dims: {ds.attribute_dims}")
    loader = build_dataloader(ds, batch_size=config.batch, num_workers=config.workers)

    model = UNet(ds.attribute_dims, image_size=config.image_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M")

    diff = GaussianDiffusion(timesteps=config.timesteps)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    ema = EMA(model)

    start_step = 0
    if config.resume:
        if not os.path.exists(config.ckpt):
            raise FileNotFoundError(f"--resume set but checkpoint {config.ckpt!r} not found")
        ck = load_checkpoint(config.ckpt, map_location=device)
        if ck.get("attribute_dims") != ds.attribute_dims:
            raise ValueError("attribute dims in checkpoint != dataset; refusing to resume")
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        else:
            print("WARNING: checkpoint has no optimizer state; Adam restarts cold.")
        if precision == "fp16" and "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_step = int(ck["step"])
        print(f"resumed from {config.ckpt} @ step {start_step}")
        if start_step >= config.steps:
            print(f"already at/after target {config.steps}; raise --steps to train further.")
            return

    print(f"training steps {start_step + 1} -> {config.steps}")
    model.train()
    data = infinite(loader)
    t_window = time.time()
    window_start = start_step
    for step in range(start_step + 1, config.steps + 1):
        step_lr = learning_rate_for_step(
            base_lr=config.lr,
            min_lr=min_lr,
            schedule=lr_schedule,
            step=step,
            total_steps=config.steps,
        )
        set_optimizer_lr(opt, step_lr)

        x0, labels = next(data)
        x0, labels = x0.to(device), labels.to(device)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
            loss = diff.p_losses(model, x0, labels, p_uncond=config.p_uncond)

        opt.zero_grad(set_to_none=True)
        if precision == "fp16":
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        ema.update(model)

        if step % 100 == 0 or step == start_step + 1:
            now = time.time()
            done = step - window_start
            sps = done / max(now - t_window, 1e-9)
            eta_s = (config.steps - step) / max(sps, 1e-9)
            eta = time.strftime("%H:%M:%S", time.gmtime(eta_s))
            print(
                f"step {step:>6}/{config.steps}  loss {loss.item():.4f}  "
                f"lr {step_lr:.2e}  {sps:5.2f} step/s  ETA {eta}"
            )
            t_window, window_start = now, step

        if step % config.save_every == 0 or step == config.steps:
            save_checkpoint(
                config.ckpt,
                step=step,
                model=model,
                ema=ema,
                opt=opt,
                attrs=config.attrs,
                attribute_dims=ds.attribute_dims,
                image_size=config.image_size,
                timesteps=config.timesteps,
                precision=precision,
                grad_clip=grad_clip,
                lr_schedule=lr_schedule,
                min_lr=min_lr,
                scaler=scaler if precision == "fp16" else None,
            )
            bak = " (prev -> .bak)" if os.path.exists(config.ckpt + ".bak") else ""
            print(f"saved {config.ckpt} @ step {step}{bak}")

    print("done.")
