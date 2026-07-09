"""CLI for training the conditional DDPM."""

from __future__ import annotations

import argparse

from cartoon_diffusion.config import (
    add_config_argument,
    apply_train_run_dir,
    parse_with_config,
)
from cartoon_diffusion.training import TrainConfig, default_root, train


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add_config_argument(ap)
    ap.add_argument("--dataset_variant", choices=["10k", "100k"], default="10k")
    ap.add_argument("--root", default=None)
    ap.add_argument(
        "--attrs",
        nargs="+",
        default=["eye_color", "hair_color", "face_color"],
    )
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument(
        "--steps",
        type=int,
        default=40000,
        help="final target step count; with --resume, continues until this total",
    )
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--p_uncond", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--cache", action="store_true", help="enable dataset cache")
    ap.add_argument("--no_cache", action="store_true", help="disable dataset cache")
    ap.add_argument("--rebuild_cache", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = parse_with_config(build_parser(), argv)
    root = args.root or default_root(args.dataset_variant)
    args.root = root
    default_cache = args.dataset_variant == "100k"
    cache = args.cache or (default_cache and not args.no_cache)
    workers = args.workers
    if workers is None:
        workers = 0 if args.dataset_variant == "100k" else 4
    args.cache = cache
    args.workers = workers
    apply_train_run_dir(args)
    config = TrainConfig(
        root=root,
        attrs=args.attrs,
        image_size=args.image_size,
        batch=args.batch,
        steps=args.steps,
        lr=args.lr,
        timesteps=args.timesteps,
        p_uncond=args.p_uncond,
        limit=args.limit,
        workers=workers,
        ckpt=args.ckpt,
        resume=args.resume,
        save_every=args.save_every,
        cache=cache,
        rebuild_cache=args.rebuild_cache,
    )
    train(config)


if __name__ == "__main__":
    main()
