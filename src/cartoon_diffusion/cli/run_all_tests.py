"""Run fidelity, diversity, and variance evaluations in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def run(cmd):
    print("\n>>> " + " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] after {dt:.0f}s: {' '.join(cmd)}")
        sys.exit(result.returncode)
    print(f"[ok] {dt:.0f}s")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_variant", choices=["10k", "100k"], default="10k")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--weights", type=float, nargs="+", default=[0, 1, 3, 5, 8, 10, 15])
    ap.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    ap.add_argument("--n_samples_fidelity", type=int, default=256)
    ap.add_argument("--n_samples_variance", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n_conditions", type=int, default=16)
    ap.add_argument("--n_per_condition", type=int, default=8)
    ap.add_argument("--python", default=sys.executable)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    weights = [str(w) for w in args.weights]
    common = [
        "-m",
        "cartoon_diffusion.cli.evaluate",
        "--dataset_variant",
        args.dataset_variant,
        "--ckpt",
        args.ckpt,
        "--results_dir",
        args.results_dir,
        "--sampler",
        args.sampler,
    ]
    if args.root:
        common.extend(["--root", args.root])
    print(f"=== dataset={args.dataset_variant} ckpt={args.ckpt} results_dir={args.results_dir} ===")
    run([args.python, *common, "--test", "fidelity", "--weights", *weights, "--n_samples", str(args.n_samples_fidelity)])
    run([args.python, *common, "--test", "diversity", "--weights", *weights, "--n_conditions", str(args.n_conditions), "--n_per_condition", str(args.n_per_condition)])
    run([args.python, *common, "--test", "variance", "--weights", *weights, "--n_samples", str(args.n_samples_variance), "--repeats", str(args.repeats)])


if __name__ == "__main__":
    main()
