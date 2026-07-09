"""
Training loop for the conditional DDPM on the 100k Cartoon Set.

Same model/diffusion as train.py; two additions aimed at the long run:

  * Uses dataset100k (recursive glob over the ten sub-folders + disk cache),
    so the CPU never becomes the bottleneck.

  * RESUMABLE. A 4-hour run that gets interrupted (session/PC) can be
    continued from the last checkpoint with `--resume`, restoring the model,
    the EMA shadow, the Adam optimizer state AND the step counter. Without
    restoring the optimizer you'd get an Adam "warm-up" transient every time;
    without the step counter you'd retrain from scratch.

First launch (100k, ~40k steps ~ 2h on a GTX 1080):
    python train100k.py --root data/cartoonset100k --steps 40000 --batch 128 \
        --ckpt run100k.pt

Continue it later (same --ckpt, add --resume; --steps is the FINAL target):
    python train100k.py --root data/cartoonset100k --steps 40000 --batch 128 \
        --ckpt run100k.pt --resume

Push further in a second sitting (raise the target):
    python train100k.py ... --ckpt run100k.pt --resume --steps 70000

Debug on a laptop (CPU):
    python train100k.py --root data/cartoonset10k --limit 64 --steps 20 \
        --batch 16 --ckpt test.pt
"""

import argparse
import copy
import os
import time
from datetime import datetime

import torch

from dataset100k import CartoonSetDataset, build_dataloader
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


def save_checkpoint(path, step, model, ema, opt, args, ds):
    """Atomic-ish save with .bak rotation of any existing checkpoint."""
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.shadow.state_dict(),
            "opt": opt.state_dict(),          # <-- enables clean resume
            "attribute_dims": ds.attribute_dims,
            "attrs": args.attrs,
            "image_size": args.image_size,
            "timesteps": args.timesteps,
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cartoonset100k")
    ap.add_argument("--attrs", nargs="+",
                    default=["eye_color", "hair_color", "face_color"])
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=40000,
                    help="FINAL target step count (with --resume, training "
                         "continues until this many total steps).")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--p_uncond", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only N images (CPU debugging; disables cache)")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 is fastest with the fully-preloaded dataset")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path (also the resume source). If omitted, "
                         "an auto-named 'run_YYYYmmdd_HHMMSS.pt' is used.")
    ap.add_argument("--resume", action="store_true",
                    help="continue training from --ckpt (model+EMA+optimizer+step)")
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--rebuild_cache", action="store_true",
                    help="ignore any existing dataset cache and rebuild it")
    args = ap.parse_args()

    if args.ckpt is None:
        if args.resume:
            ap.error("--resume needs an explicit --ckpt to resume from.")
        args.ckpt = datetime.now().strftime("run_%Y%m%d_%H%M%S.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    ds = CartoonSetDataset(
        args.root, image_size=args.image_size,
        cond_attributes=tuple(args.attrs), limit=args.limit,
        rebuild_cache=args.rebuild_cache,
    )
    print(f"images: {len(ds)}  attrs: {args.attrs}  dims: {ds.attribute_dims}")
    loader = build_dataloader(ds, batch_size=args.batch, num_workers=args.workers)

    model = UNet(ds.attribute_dims, image_size=args.image_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M")

    diff = GaussianDiffusion(timesteps=args.timesteps)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model)

    # ---- resume ----------------------------------------------------------
    start_step = 0
    if args.resume:
        if not os.path.exists(args.ckpt):
            ap.error(f"--resume set but checkpoint {args.ckpt!r} not found.")
        ck = torch.load(args.ckpt, map_location=device)
        if ck.get("attribute_dims") != ds.attribute_dims:
            ap.error("attribute dims in checkpoint != dataset; refusing to resume.")
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        else:
            print("WARNING: checkpoint has no optimizer state; Adam restarts "
                  "cold (small transient, harmless).")
        start_step = ck["step"]
        print(f"resumed from {args.ckpt} @ step {start_step}")
        if start_step >= args.steps:
            print(f"already at/after target {args.steps}; nothing to do. "
                  f"Raise --steps to train further.")
            return

    print(f"training steps {start_step + 1} -> {args.steps}")

    model.train()
    data = infinite(loader)
    t_window = time.time()
    window_start = start_step
    for step in range(start_step + 1, args.steps + 1):
        x0, labels = next(data)
        x0, labels = x0.to(device), labels.to(device)

        loss = diff.p_losses(model, x0, labels, p_uncond=args.p_uncond)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)

        if step % 100 == 0 or step == start_step + 1:
            now = time.time()
            done = step - window_start
            sps = done / max(now - t_window, 1e-9)
            eta_s = (args.steps - step) / max(sps, 1e-9)
            eta = time.strftime("%H:%M:%S", time.gmtime(eta_s))
            print(f"step {step:>6}/{args.steps}  loss {loss.item():.4f}  "
                  f"{sps:5.2f} step/s  ETA {eta}")
            t_window, window_start = now, step

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.ckpt, step, model, ema, opt, args, ds)
            bak = " (prev -> .bak)" if os.path.exists(args.ckpt + ".bak") else ""
            print(f"saved {args.ckpt} @ step {step}{bak}")

    print("done.")


if __name__ == "__main__":
    main()
