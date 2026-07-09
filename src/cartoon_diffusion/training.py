"""Training loop for the conditional Cartoon Set DDPM."""

from __future__ import annotations

import copy
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


def train(config: TrainConfig) -> None:
    if config.ckpt is None:
        if config.resume:
            raise ValueError("--resume needs an explicit --ckpt")
        config.ckpt = datetime.now().strftime("run_%Y%m%d_%H%M%S.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

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
        x0, labels = next(data)
        x0, labels = x0.to(device), labels.to(device)

        loss = diff.p_losses(model, x0, labels, p_uncond=config.p_uncond)
        opt.zero_grad()
        loss.backward()
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
                f"{sps:5.2f} step/s  ETA {eta}"
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
            )
            bak = " (prev -> .bak)" if os.path.exists(config.ckpt + ".bak") else ""
            print(f"saved {config.ckpt} @ step {step}{bak}")

    print("done.")
