"""CLI for fidelity, diversity, and variance evaluation."""

from __future__ import annotations

import argparse

from cartoon_diffusion.config import (
    add_config_argument,
    apply_eval_run_dir,
    parse_with_config,
)
from cartoon_diffusion.evaluation import evaluate_from_args
from cartoon_diffusion.training import default_classifier_ckpt, default_root


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add_config_argument(ap)
    ap.add_argument("--dataset_variant", choices=["10k", "100k"], default="10k")
    ap.add_argument("--test", choices=["fidelity", "diversity", "variance"], default="fidelity")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--clf_ckpt", default=None)
    ap.add_argument("--weights", type=float, nargs="+", default=[0, 1, 3, 5, 8, 10, 15])
    ap.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    ap.add_argument("--ddim_steps", "--steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--n_samples", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n_conditions", type=int, default=16)
    ap.add_argument("--n_per_condition", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clf_epochs", type=int, default=8)
    ap.add_argument("--clf_batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--rebuild_cache", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = parse_with_config(build_parser(), argv)
    args.root = args.root or default_root(args.dataset_variant)
    apply_eval_run_dir(args)
    if args.ckpt is None:
        args.ckpt = "run100k.pt" if args.dataset_variant == "100k" else "ckpt.pt"
    args.clf_ckpt = args.clf_ckpt or default_classifier_ckpt(args.dataset_variant)
    evaluate_from_args(args)


if __name__ == "__main__":
    main()
