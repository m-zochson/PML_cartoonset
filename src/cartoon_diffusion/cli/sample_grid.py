"""CLI for annotated qualitative sample grids."""

from __future__ import annotations

import argparse

from cartoon_diffusion.config import (
    add_config_argument,
    apply_sample_run_dir,
    parse_with_config,
)
from cartoon_diffusion.sampling import sample_grids


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add_config_argument(ap)
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--vary", default="all")
    ap.add_argument("--weights", type=float, nargs="+", default=[0, 1, 3, 5])
    ap.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    ap.add_argument("--ddim_steps", "--steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--fixed", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell", type=int, default=112)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out_dir", default="outputs/grids")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = parse_with_config(build_parser(), argv)
    apply_sample_run_dir(args)
    sample_grids(args)


if __name__ == "__main__":
    main()
