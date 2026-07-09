"""CLI for training the optional Cartoon Set 32x32 -> 96x96 upscaler."""

from __future__ import annotations

import argparse

from cartoon_diffusion.config import (
    add_config_argument,
    apply_upscaler_train_run_dir,
    parse_with_config,
)
from cartoon_diffusion.upscaling import UpscalerConfig, train_upscaler


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add_config_argument(ap)
    ap.add_argument("--root", default="data/cartoonset10k")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--input_size", type=int, default=32)
    ap.add_argument("--target_size", type=int, default=96)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--residual_blocks", type=int, default=4)
    ap.add_argument("--refinement_blocks", type=int, default=2)
    ap.add_argument("--head_kernel", type=int, default=5)
    ap.add_argument("--residual_kernel", type=int, default=3)
    ap.add_argument("--up_kernel", type=int, default=3)
    ap.add_argument("--refinement_kernel", type=int, default=3)
    ap.add_argument("--tail_kernel", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument(
        "--lr_schedule",
        choices=["constant", "cosine"],
        default="constant",
        help="learning-rate schedule",
    )
    ap.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="minimum LR for cosine schedule",
    )
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument(
        "--l2_weight",
        type=float,
        default=1.0,
        help="weight for the MSE term in the L1 + l2_weight*MSE loss",
    )
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save_every", type=int, default=1000)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = parse_with_config(build_parser(), argv)
    apply_upscaler_train_run_dir(args)
    config = UpscalerConfig(
        root=args.root,
        input_size=args.input_size,
        target_size=args.target_size,
        channels=args.channels,
        residual_blocks=args.residual_blocks,
        refinement_blocks=args.refinement_blocks,
        head_kernel=args.head_kernel,
        residual_kernel=args.residual_kernel,
        up_kernel=args.up_kernel,
        refinement_kernel=args.refinement_kernel,
        tail_kernel=args.tail_kernel,
        batch=args.batch,
        steps=args.steps,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        l2_weight=args.l2_weight,
        precision=args.precision,
        grad_clip=args.grad_clip,
        limit=args.limit,
        workers=args.workers,
        ckpt=args.ckpt,
        resume=args.resume,
        save_every=args.save_every,
    )
    train_upscaler(config)


if __name__ == "__main__":
    main()
