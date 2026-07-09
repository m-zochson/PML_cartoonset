"""
Plot the results written by evaluate.py / evaluate100k.py.

It scans a results folder for
    results/<tag>_fidelity.json      (per-attribute + mean classifier accuracy)
    results/<tag>_diversity.json     (intra-condition pairwise RMS distance)
    results/<tag>_variance.json      (mean +/- std fidelity over repeated seeds)
and produces, for each file found:

    results/<tag>_fidelity.png  / .pdf   thin muted per-attribute curves +
                                         thick orange mean curve, best w marked
    results/<tag>_diversity.png / .pdf   diversity vs w
    results/<tag>_variance.png  / .pdf   mean fidelity with +/-1 std band and
                                         the individual seed runs as dots

If two or more runs share a test type (e.g. the 10k and the 100k checkpoints),
it also writes overlay figures:

    results/comparison_fidelity.{png,pdf}    mean fidelity curves
    results/comparison_diversity.{png,pdf}   diversity curves
    results/comparison_variance.{png,pdf}    mean fidelity + std bands

PNG (raster, --dpi) is for slides; PDF (vector) drops cleanly into LaTeX/beamer.

    # plot everything in results/
    python plot_results.py

    # only specific runs / tests
    python plot_results.py --files results/run_full_40k_fidelity.json
    python plot_results.py --tests fidelity variance

Requires matplotlib (pip install matplotlib).

Note: the sampler is DDPM (full reverse chain); DDIM was abandoned because
classifier-free guidance amplification destroys the samples for w >= 2.5.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")            # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

ACCENT = "#E8730C"               # orange, matches the report/study-guide styling
TESTS = ("fidelity", "diversity", "variance")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_run(path):
    """Return (tag, test, meta, rows) with rows sorted by guidance weight."""
    with open(path) as f:
        obj = json.load(f)
    meta = obj["meta"]
    rows = sorted(obj["results"], key=lambda r: r["weight"])

    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    test = meta.get("test")
    if test is None:                                  # fall back on the filename
        test = next((t for t in TESTS if stem.endswith("_" + t)), "fidelity")
    tag = stem[: -len("_" + test)] if stem.endswith("_" + test) else stem
    return tag, test, meta, rows


def attr_value(row, attrs, a):
    """evaluate.py writes the per-attribute accuracies flat in the row; older
    versions nested them under 'per_attribute'. Support both."""
    if a in row:
        return row[a]
    return row["per_attribute"][a]


def subtitle(meta):
    bits = [f"ckpt={meta.get('ckpt', '?')}",
            f"data={meta.get('dataset_variant', '?')}",
            f"sampler={meta.get('sampler', '?')}"]
    for k, label in (("n_samples", "n"), ("repeats", "repeats"),
                     ("n_conditions", "conds"), ("n_per_condition", "per cond")):
        if k in meta and meta[k] is not None:
            bits.append(f"{label}={meta[k]}")
    return ", ".join(bits)


def _save(fig, out_base, dpi):
    png, pdf = out_base + ".png", out_base + ".pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _finish(ax, weights, xlabel="guidance weight  w"):
    ax.set_xlabel(xlabel)
    ax.set_xticks(weights)
    ax.grid(True, alpha=0.3)


# --------------------------------------------------------------------------- #
# single-run figures
# --------------------------------------------------------------------------- #
def plot_fidelity(tag, meta, rows, results_dir, dpi):
    attrs = meta["attrs"]
    weights = [r["weight"] for r in rows]
    means = [r["mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = plt.get_cmap("tab10")

    for i, a in enumerate(attrs):
        ys = [attr_value(r, attrs, a) for r in rows]
        ax.plot(weights, ys, color=cmap(i % 10), lw=1.4, alpha=0.55,
                marker="o", ms=3, label=a)

    ax.plot(weights, means, color=ACCENT, lw=3.0, marker="o", ms=6,
            label="mean", zorder=5)

    bi = max(range(len(means)), key=lambda k: means[k])
    ax.axvline(weights[bi], color=ACCENT, ls="--", lw=1.0, alpha=0.5, zorder=1)
    ax.annotate(f"best w = {weights[bi]:g}\nmean = {means[bi]:.3f}",
                xy=(weights[bi], means[bi]),
                xytext=(6, -28), textcoords="offset points",
                fontsize=9, color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=ACCENT, alpha=0.9))

    ax.set_title(f"Attribute fidelity vs guidance weight  [{tag}]\n{subtitle(meta)}",
                 fontsize=10)
    ax.set_ylabel("attribute fidelity (classifier accuracy)")
    ax.set_ylim(0.0, 1.02)
    _finish(ax, weights)
    ax.legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.9)

    png, _ = _save(fig, os.path.join(results_dir, f"{tag}_fidelity"), dpi)
    print(f"  {tag} fidelity : best w={weights[bi]:g} (mean {means[bi]:.3f})  ->  {png}")


def plot_diversity(tag, meta, rows, results_dir, dpi):
    weights = [r["weight"] for r in rows]
    div = [r["diversity"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(weights, div, color=ACCENT, lw=2.5, marker="o", ms=6)

    lo, hi = min(div), max(div)
    pad = 0.15 * (hi - lo) if hi > lo else 0.05 * max(hi, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_title(f"Intra-condition diversity vs guidance weight  [{tag}]\n{subtitle(meta)}",
                 fontsize=10)
    ax.set_ylabel("diversity (pairwise RMS pixel distance)")
    _finish(ax, weights)

    png, _ = _save(fig, os.path.join(results_dir, f"{tag}_diversity"), dpi)
    print(f"  {tag} diversity: range [{lo:.4f}, {hi:.4f}]  ->  {png}")


def plot_variance(tag, meta, rows, results_dir, dpi):
    weights = [r["weight"] for r in rows]
    means = [r["mean_fidelity"] for r in rows]
    stds = [r["std_fidelity"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    lo = [m - s for m, s in zip(means, stds)]
    hi = [m + s for m, s in zip(means, stds)]
    ax.fill_between(weights, lo, hi, color=ACCENT, alpha=0.18,
                    label=r"$\pm 1\,\sigma$")
    ax.plot(weights, means, color=ACCENT, lw=2.5, marker="o", ms=6,
            label="mean fidelity", zorder=5)

    # individual seeds, if evaluate.py stored them
    seen = False
    for r in rows:
        for v in r.get("runs", []):
            ax.plot(r["weight"], v, marker=".", ms=6, color="#444444",
                    alpha=0.7, ls="none", zorder=4,
                    label=None if seen else "individual seeds")
            seen = True

    bi = max(range(len(means)), key=lambda k: means[k])
    ax.axvline(weights[bi], color=ACCENT, ls="--", lw=1.0, alpha=0.5, zorder=1)

    ax.set_title(f"Fidelity stability across seeds  [{tag}]\n{subtitle(meta)}",
                 fontsize=10)
    ax.set_ylabel("mean attribute fidelity")
    _finish(ax, weights)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    png, _ = _save(fig, os.path.join(results_dir, f"{tag}_variance"), dpi)
    print(f"  {tag} variance : best w={weights[bi]:g} "
          f"({means[bi]:.3f} +/- {stds[bi]:.3f})  ->  {png}")


PLOTTERS = {"fidelity": plot_fidelity,
            "diversity": plot_diversity,
            "variance": plot_variance}


# --------------------------------------------------------------------------- #
# comparison figures
# --------------------------------------------------------------------------- #
def plot_comparison(test, runs, results_dir, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = plt.get_cmap("tab10")

    for i, (tag, meta, rows) in enumerate(runs):
        c = cmap(i % 10)
        weights = [r["weight"] for r in rows]
        label = f"{tag} ({meta.get('ckpt', '?')})"
        if test == "fidelity":
            ax.plot(weights, [r["mean"] for r in rows], color=c, lw=2.5,
                    marker="o", ms=5, label=label)
        elif test == "diversity":
            ax.plot(weights, [r["diversity"] for r in rows], color=c, lw=2.5,
                    marker="o", ms=5, label=label)
        else:  # variance
            m = [r["mean_fidelity"] for r in rows]
            s = [r["std_fidelity"] for r in rows]
            ax.fill_between(weights, [a - b for a, b in zip(m, s)],
                            [a + b for a, b in zip(m, s)], color=c, alpha=0.15)
            ax.plot(weights, m, color=c, lw=2.5, marker="o", ms=5, label=label)

    ylabel = {"fidelity": "mean attribute fidelity",
              "diversity": "diversity (pairwise RMS pixel distance)",
              "variance": r"mean attribute fidelity ($\pm 1\,\sigma$)"}[test]
    ax.set_title(f"{test.capitalize()} vs guidance weight  —  run comparison",
                 fontsize=10)
    ax.set_ylabel(ylabel)
    if test != "diversity":
        ax.set_ylim(0.0, 1.02)
    _finish(ax, sorted({r["weight"] for _, _, rows in runs for r in rows}))
    ax.legend(fontsize=9, framealpha=0.9)

    png, _ = _save(fig, os.path.join(results_dir, f"comparison_{test}"), dpi)
    print(f"  comparison {test} ({len(runs)} runs)  ->  {png}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results",
                    help="folder to read the *.json from and write plots to")
    ap.add_argument("--files", nargs="+", default=None,
                    help="explicit JSON files (default: glob *_{fidelity,"
                         "diversity,variance}.json)")
    ap.add_argument("--tests", nargs="+", default=list(TESTS), choices=TESTS,
                    help="which tests to plot (default: all)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no_comparison", action="store_true",
                    help="skip the multi-run overlay figures")
    args = ap.parse_args()

    if args.files:
        paths = args.files
    else:
        paths = []
        for t in args.tests:
            paths += sorted(glob.glob(os.path.join(args.results_dir, f"*_{t}.json")))
    if not paths:
        raise SystemExit(
            f"no result JSON found in {args.results_dir!r}. "
            "Run evaluate.py / evaluate100k.py first.")

    print(f"plotting {len(paths)} file(s):")
    by_test = {t: [] for t in TESTS}
    for p in paths:
        tag, test, meta, rows = load_run(p)
        if test not in args.tests:
            continue
        PLOTTERS[test](tag, meta, rows, args.results_dir, args.dpi)
        by_test[test].append((tag, meta, rows))

    if not args.no_comparison:
        for test in args.tests:
            if len(by_test[test]) >= 2:
                plot_comparison(test, by_test[test], args.results_dir, args.dpi)

    print("done.")


if __name__ == "__main__":
    main()
