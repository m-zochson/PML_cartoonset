"""
Plot attribute fidelity vs guidance weight from the JSON files written by
evaluate.py / evaluate100k.py (see save_results there).

For every results/<tag>_fidelity.json it produces:
  results/<tag>_fidelity.png / .pdf
    one thin muted line per attribute + a thick orange "mean" curve, with the
    best (highest-mean) guidance weight marked.

If two or more runs are found (e.g. the 10k and 100k checkpoints), it also
writes a comparison figure overlaying just the mean curves:
  results/comparison_fidelity.png / .pdf

PNG (raster, --dpi) is for slides; PDF (vector) drops cleanly into LaTeX/beamer.

    # plot everything in results/
    python plot_results.py

    # only specific runs
    python plot_results.py --files results/ckpt_fidelity.json results/run100k_fidelity.json

Requires matplotlib (pip install matplotlib).
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")            # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

ACCENT = "#E8730C"               # orange, matches the report/study-guide styling


def load_run(path):
    """Return (tag, meta, rows) with rows sorted by guidance weight."""
    with open(path) as f:
        obj = json.load(f)
    meta = obj["meta"]
    rows = sorted(obj["results"], key=lambda r: r["weight"])
    base = os.path.basename(path)
    tag = base[:-len("_fidelity.json")] if base.endswith("_fidelity.json") \
        else os.path.splitext(base)[0]
    return tag, meta, rows


def _save(fig, out_base, dpi):
    png, pdf = out_base + ".png", out_base + ".pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_run(tag, meta, rows, results_dir, dpi):
    attrs = meta["attrs"]
    weights = [r["weight"] for r in rows]
    means = [r["mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = plt.get_cmap("tab10")

    # per-attribute curves (muted, thin) so the mean stands out
    for i, a in enumerate(attrs):
        ys = [r["per_attribute"][a] for r in rows]
        ax.plot(weights, ys, color=cmap(i % 10), lw=1.4, alpha=0.55,
                marker="o", ms=3, label=a)

    # mean curve (thick, accent)
    ax.plot(weights, means, color=ACCENT, lw=3.0, marker="o", ms=6,
            label="mean", zorder=5)

    # mark the best (highest-mean) guidance weight
    bi = max(range(len(means)), key=lambda k: means[k])
    ax.axvline(weights[bi], color=ACCENT, ls="--", lw=1.0, alpha=0.5, zorder=1)
    ax.annotate(f"best w = {weights[bi]:g}\nmean = {means[bi]:.3f}",
                xy=(weights[bi], means[bi]),
                xytext=(6, -28), textcoords="offset points",
                fontsize=9, color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=ACCENT, alpha=0.9))

    variant = meta.get("dataset_variant", "?")
    n = meta.get("n_samples", "?")
    steps = meta.get("steps", "?")
    ax.set_title(f"Attribute fidelity vs guidance weight  "
                 f"[{tag}, {variant}, n={n}, {steps} DDIM steps]",
                 fontsize=10)
    ax.set_xlabel("guidance weight  w")
    ax.set_ylabel("attribute fidelity (classifier accuracy)")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(weights)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.9)

    png, pdf = _save(fig, os.path.join(results_dir, f"{tag}_fidelity"), dpi)
    print(f"  {tag}: best w={weights[bi]:g} (mean {means[bi]:.3f})  ->  {png}")
    return weights[bi], means[bi]


def plot_comparison(runs, results_dir, dpi):
    """Overlay the mean curves of every run in a single figure."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, (tag, meta, rows) in enumerate(runs):
        weights = [r["weight"] for r in rows]
        means = [r["mean"] for r in rows]
        variant = meta.get("dataset_variant", "?")
        ax.plot(weights, means, color=cmap(i % 10), lw=2.5, marker="o", ms=5,
                label=f"{tag} ({variant})")

    ax.set_title("Mean attribute fidelity vs guidance weight  —  run comparison",
                 fontsize=10)
    ax.set_xlabel("guidance weight  w")
    ax.set_ylabel("mean attribute fidelity")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.9)

    png, pdf = _save(fig, os.path.join(results_dir, "comparison_fidelity"), dpi)
    print(f"  comparison ({len(runs)} runs)  ->  {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results",
                    help="folder to read *_fidelity.json from and write plots to")
    ap.add_argument("--files", nargs="+", default=None,
                    help="explicit JSON files (default: glob *_fidelity.json)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no_comparison", action="store_true",
                    help="skip the multi-run mean-curve comparison figure")
    args = ap.parse_args()

    paths = args.files or sorted(
        glob.glob(os.path.join(args.results_dir, "*_fidelity.json")))
    if not paths:
        raise SystemExit(
            f"no *_fidelity.json found in {args.results_dir!r}. "
            "Run evaluate.py / evaluate100k.py first.")

    print(f"plotting {len(paths)} run(s):")
    runs = []
    for p in paths:
        tag, meta, rows = load_run(p)
        plot_run(tag, meta, rows, args.results_dir, args.dpi)
        runs.append((tag, meta, rows))

    if len(runs) >= 2 and not args.no_comparison:
        plot_comparison(runs, args.results_dir, args.dpi)

    print("done.")


if __name__ == "__main__":
    main()
