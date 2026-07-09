"""Config-file and run-directory helpers for CLI entrypoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="YAML config file")


def parse_with_config(parser: argparse.ArgumentParser, argv: list[str] | None = None):
    """Parse args with YAML defaults, while explicit CLI flags win."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(args_list)
    if known.config:
        parser.set_defaults(**load_config(known.config))
    return parser.parse_args(args_list)


def load_config(path: str | os.PathLike[str]) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path!r} must contain a YAML mapping")
    return data


def save_config(path: str | os.PathLike[str], values: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(_plain(values), f, sort_keys=True)


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def apply_train_run_dir(args) -> None:
    if not getattr(args, "run_dir", None):
        return
    root = Path(args.run_dir)
    if args.ckpt is None:
        args.ckpt = str(root / "checkpoints" / "latest.pt")
    root.mkdir(parents=True, exist_ok=True)
    save_config(root / "config.yaml", vars(args))


def apply_eval_run_dir(args) -> None:
    if not getattr(args, "run_dir", None):
        return
    root = Path(args.run_dir)
    if args.ckpt is None:
        args.ckpt = str(root / "checkpoints" / "latest.pt")
    if args.clf_ckpt is None:
        args.clf_ckpt = str(root / "classifiers" / "classifier.pt")
    if args.results_dir == "results":
        args.results_dir = str(root / "results")
    root.mkdir(parents=True, exist_ok=True)
    save_config(root / "eval_config.yaml", vars(args))


def apply_sample_run_dir(args) -> None:
    if not getattr(args, "run_dir", None):
        return
    root = Path(args.run_dir)
    if args.ckpt == "ckpt.pt":
        args.ckpt = str(root / "checkpoints" / "latest.pt")
    if args.out is None and args.out_dir == "outputs/grids":
        args.out_dir = str(root / "grids")
    root.mkdir(parents=True, exist_ok=True)
    save_config(root / "sample_config.yaml", vars(args))
