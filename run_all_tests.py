#!/usr/bin/env python3
"""
Runs fidelity, diversity, and variance in sequence -- one evaluate.py (or
evaluate100k.py) subprocess per test, each still a single --test invocation.
This is a convenience wrapper only: it does not add a "--test all" mode to
evaluate.py itself, so the one-test-per-run guarantee there is untouched.

Cross-platform (Windows / laptop CPU debug, desktop GPU run) since it uses
subprocess instead of shell syntax.

Usage:
  python run_all_tests.py --ckpt training_finale.pt
  python run_all_tests.py --script evaluate100k.py --root data/cartoonset100k --ckpt run100k.pt
  python run_all_tests.py --ckpt training_finale.pt --weights 0 1 3 5 --n_samples_fidelity 128
"""

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
        print(f"\n[FAILED] after {dt:.0f}s (exit code {result.returncode}): "
              f"{' '.join(cmd)}")
        sys.exit(result.returncode)
    print(f"[ok] {dt:.0f}s")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="evaluate.py",
                    help="evaluate.py (10k) or evaluate100k.py (100k)")
    ap.add_argument("--root", default="data/cartoonset10k")
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0, 1, 3, 5, 8, 10, 15])
    ap.add_argument("--n_samples_fidelity", type=int, default=256)
    ap.add_argument("--n_samples_variance", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n_conditions", type=int, default=16)
    ap.add_argument("--n_per_condition", type=int, default=8)
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter to use for the subprocess calls "
                         "(default: the one running this script)")
    args = ap.parse_args()

    weights_str = [str(w) for w in args.weights]
    common = ["--root", args.root, "--ckpt", args.ckpt,
             "--results_dir", args.results_dir]

    print(f"=== script={args.script} root={args.root} ckpt={args.ckpt} "
          f"results_dir={args.results_dir} ===")
    print(f"=== weights: {weights_str} ===")
    t_total = time.time()

    print("\n[1/3] fidelity -----------------------------------------------------")
    run([args.python, args.script, "--test", "fidelity", *common,
        "--weights", *weights_str,
        "--n_samples", str(args.n_samples_fidelity)])

    print("\n[2/3] diversity ----------------------------------------------------")
    run([args.python, args.script, "--test", "diversity", *common,
        "--weights", *weights_str,
        "--n_conditions", str(args.n_conditions),
        "--n_per_condition", str(args.n_per_condition)])

    print("\n[3/3] variance -----------------------------------------------------")
    run([args.python, args.script, "--test", "variance", *common,
        "--weights", *weights_str,
        "--n_samples", str(args.n_samples_variance),
        "--repeats", str(args.repeats)])

    dt = time.time() - t_total
    print(f"\n=== all three tests done in {int(dt//60)} min {int(dt%60)}s ===")
    print(f"results in: {args.results_dir}/")


if __name__ == "__main__":
    main()
