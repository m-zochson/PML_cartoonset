"""
Training loop for the conditional DDPM. Device-agnostic: runs on CPU (laptop,
for debugging with --limit) and on CUDA (desktop, full run) with no changes.

Debug on a laptop:
    python train.py --root data/cartoonset10k --limit 64 --steps 20 --batch 16

Full run on the desktop:
    python train.py --root data/cartoonset10k --steps 40000 --batch 128
"""

import argparse
import copy
import os
import time
from datetime import datetime

import torch

from dataset import CartoonSetDataset, build_dataloader
from diffusion import GaussianDiffusion
from model import UNet


class EMA:
    """Exponential moving average of model weights (improves sample quality)."""

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cartoonset10k")
    ap.add_argument("--attrs", nargs="+",
                    default=["eye_color", "hair_color", "face_color"])
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--p_uncond", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only N images (for CPU debugging)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path. If omitted, an auto-named "
                         "'run_YYYYmmdd_HHMMSS.pt' is used so runs never "
                         "collide with each other by accident.")
    ap.add_argument("--save_every", type=int, default=5000)
    args = ap.parse_args()

    if args.ckpt is None:
        args.ckpt = datetime.now().strftime("run_%Y%m%d_%H%M%S.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    ds = CartoonSetDataset(
        args.root, image_size=args.image_size,
        cond_attributes=tuple(args.attrs), limit=args.limit,
    )
    print(f"images: {len(ds)}  attrs: {args.attrs}  dims: {ds.attribute_dims}")
    loader = build_dataloader(ds, batch_size=args.batch, num_workers=args.workers)

    model = UNet(ds.attribute_dims, image_size=args.image_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M")

    diff = GaussianDiffusion(timesteps=args.timesteps)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model)

    model.train()
    data = infinite(loader)
    t_start = None
    for step in range(1, args.steps + 1):
        if step == 1:
            t_start = time.time()  # escluso il tempo di preload del dataset

        x0, labels = next(data)
        x0, labels = x0.to(device), labels.to(device)

        loss = diff.p_losses(model, x0, labels, p_uncond=args.p_uncond)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - t_start
            sps = step / elapsed if elapsed > 0 else 0.0
            eta_min = (args.steps - step) / sps / 60 if sps > 0 else float("inf")
            print(f"step {step:>6}/{args.steps}  loss {loss.item():.4f}  "
                  f"{sps:.2f} step/s  ETA {eta_min:.1f} min")

        if step % args.save_every == 0 or step == args.steps:
            # Safety: rotate any existing checkpoint to .bak before overwriting,
            # so an accidental short/test run never silently destroys a long one.
            if os.path.exists(args.ckpt):
                os.replace(args.ckpt, args.ckpt + ".bak")
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.shadow.state_dict(),
                    "attribute_dims": ds.attribute_dims,
                    "attrs": args.attrs,
                    "image_size": args.image_size,
                    "timesteps": args.timesteps,
                },
                args.ckpt,
            )
            print(f"saved {args.ckpt} @ step {step}"
                  + (f"  (previous backed up to {args.ckpt}.bak)"
                     if os.path.exists(args.ckpt + ".bak") else ""))


if __name__ == "__main__":
    main()
