"""
Training loop for the conditional DDPM. Device-agnostic: runs on CPU (laptop,
for debugging with --limit) and on CUDA (desktop, full run) with no changes.

Debug on a laptop:
    python train.py --root data/cartoonset10k --limit 64 --steps 20 --batch 16

Full run on the desktop:
    python train.py --root data/cartoonset10k --steps 40000 --batch 128

Full run with training statistics dumped to CSV (for the loss-curve plot):
    python train.py --root data/cartoonset10k --steps 40000 --batch 128 \
        --ckpt run_full_40k_v2.pt --loss_log results/run_full_40k_loss.csv

CSV columns:
    step          global optimisation step
    loss          mean training loss over the last --log_every steps
    loss_std      std of the loss inside that window (DDPM losses are noisy
                  because t is sampled uniformly, so this is large by nature)
    loss_ema      exponential moving average (decay 0.99) of the per-step loss;
                  this is the column to plot for a readable curve
    grad_norm     mean global L2 gradient norm over the window
    lr            current learning rate
    step_per_sec  throughput since step 1
    elapsed_sec   wall-clock seconds since step 1
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


@torch.no_grad()
def grad_global_norm(model):
    """L2 norm of the full gradient vector. Read-only: does not clip or rescale,
    so training dynamics are identical with and without logging enabled."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def open_loss_log(path):
    """Append-mode, line-buffered CSV. Writes the header only on a fresh file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", buffering=1)
    if fresh:
        f.write("step,loss,loss_std,loss_ema,grad_norm,lr,"
                "step_per_sec,elapsed_sec\n")
    return f


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
    ap.add_argument("--log_every", type=int, default=100,
                    help="window size (in steps) over which the logged "
                         "statistics are averaged")
    ap.add_argument("--loss_log", default=None,
                    help="CSV path where training statistics are appended. "
                         "Written line-buffered, so an interrupted run still "
                         "leaves a usable file.")
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

    loss_f = open_loss_log(args.loss_log) if args.loss_log else None

    model.train()
    data = infinite(loader)
    t_start = None
    window_loss, window_gnorm = [], []
    loss_ema = None          # heavily smoothed curve, nice for plotting
    try:
        for step in range(1, args.steps + 1):
            if step == 1:
                t_start = time.time()  # escluso il tempo di preload del dataset

            x0, labels = next(data)
            x0, labels = x0.to(device), labels.to(device)

            loss = diff.p_losses(model, x0, labels, p_uncond=args.p_uncond)
            opt.zero_grad()
            loss.backward()
            gnorm = grad_global_norm(model)
            opt.step()
            ema.update(model)

            l = loss.item()
            window_loss.append(l)
            window_gnorm.append(gnorm)
            loss_ema = l if loss_ema is None else 0.99 * loss_ema + 0.01 * l

            if step % args.log_every == 0 or step == 1:
                n = len(window_loss)
                mean = sum(window_loss) / n
                var = sum((v - mean) ** 2 for v in window_loss) / n
                std = var ** 0.5
                gmean = sum(window_gnorm) / n
                window_loss, window_gnorm = [], []

                elapsed = time.time() - t_start
                sps = step / elapsed if elapsed > 0 else 0.0
                eta_min = (args.steps - step) / sps / 60 if sps > 0 \
                    else float("inf")
                lr = opt.param_groups[0]["lr"]

                print(f"step {step:>6}/{args.steps}  loss {mean:.4f}"
                      f" (±{std:.4f})  |g| {gmean:.3f}  "
                      f"{sps:.2f} step/s  ETA {eta_min:.1f} min")

                if loss_f:
                    loss_f.write(
                        f"{step},{mean:.6f},{std:.6f},{loss_ema:.6f},"
                        f"{gmean:.6f},{lr:.8f},{sps:.4f},{elapsed:.2f}\n"
                    )

            if step % args.save_every == 0 or step == args.steps:
                # Safety: rotate any existing checkpoint to .bak before
                # overwriting, so an accidental short/test run never silently
                # destroys a long one.
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
    finally:
        if loss_f:
            loss_f.close()


if __name__ == "__main__":
    main()
