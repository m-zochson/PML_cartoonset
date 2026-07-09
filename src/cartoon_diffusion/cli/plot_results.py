"""CLI for plotting fidelity result JSON files."""

from __future__ import annotations

import argparse

from cartoon_diffusion.plotting import plot_results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--files", nargs="+", default=None)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no_comparison", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    plot_results(
        results_dir=args.results_dir,
        files=args.files,
        dpi=args.dpi,
        no_comparison=args.no_comparison,
    )


if __name__ == "__main__":
    main()
