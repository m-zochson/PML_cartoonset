"""Plot result JSON files produced by the evaluation CLI."""

from __future__ import annotations

import glob
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cartoonset")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cartoon_diffusion.results import load_result_file

ACCENT = "#E8730C"


def load_run(path):
    meta, normalized = load_result_file(path)
    base = os.path.basename(path)
    tag = base[: -len("_fidelity.json")] if base.endswith("_fidelity.json") else os.path.splitext(base)[0]
    return tag, meta, normalized


def _save(fig, out_base, dpi):
    png, pdf = out_base + ".png", out_base + ".pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _sampler_label(meta):
    sampler = meta.get("sampler", "?")
    if sampler == "ddim":
        return f"DDIM {meta.get('ddim_steps', meta.get('steps', '?'))} steps"
    if sampler == "ddpm":
        return f"DDPM {meta.get('timesteps', meta.get('steps', '?'))} steps"
    return str(sampler)


def plot_run(tag, meta, rows, results_dir, dpi):
    attrs = meta["attrs"]
    weights = [r["weight"] for r in rows]
    means = [r["mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, attr in enumerate(attrs):
        ax.plot(
            weights,
            [r[attr] for r in rows],
            color=cmap(i % 10),
            lw=1.4,
            alpha=0.55,
            marker="o",
            ms=3,
            label=attr,
        )
    ax.plot(weights, means, color=ACCENT, lw=3.0, marker="o", ms=6, label="mean", zorder=5)
    best_i = max(range(len(means)), key=lambda idx: means[idx])
    ax.axvline(weights[best_i], color=ACCENT, ls="--", lw=1.0, alpha=0.5, zorder=1)
    ax.annotate(
        f"best w = {weights[best_i]:g}\nmean = {means[best_i]:.3f}",
        xy=(weights[best_i], means[best_i]),
        xytext=(6, -28),
        textcoords="offset points",
        fontsize=9,
        color=ACCENT,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT, alpha=0.9),
    )
    variant = meta.get("dataset_variant", "?")
    n = meta.get("n_samples", "?")
    ax.set_title(
        f"Attribute fidelity vs guidance weight [{tag}, {variant}, n={n}, {_sampler_label(meta)}]",
        fontsize=10,
    )
    ax.set_xlabel("guidance weight  w")
    ax.set_ylabel("attribute fidelity (classifier accuracy)")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(weights)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.9)
    png, _ = _save(fig, os.path.join(results_dir, f"{tag}_fidelity"), dpi)
    print(f"  {tag}: best w={weights[best_i]:g} (mean {means[best_i]:.3f}) -> {png}")


def plot_comparison(runs, results_dir, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, (tag, meta, rows) in enumerate(runs):
        weights = [r["weight"] for r in rows]
        means = [r["mean"] for r in rows]
        variant = meta.get("dataset_variant", "?")
        ax.plot(weights, means, color=cmap(i % 10), lw=2.5, marker="o", ms=5, label=f"{tag} ({variant})")
    ax.set_title("Mean attribute fidelity vs guidance weight - run comparison", fontsize=10)
    ax.set_xlabel("guidance weight  w")
    ax.set_ylabel("mean attribute fidelity")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.9)
    png, _ = _save(fig, os.path.join(results_dir, "comparison_fidelity"), dpi)
    print(f"  comparison ({len(runs)} runs) -> {png}")


def plot_results(results_dir="results", files=None, dpi=150, no_comparison=False):
    paths = files or sorted(glob.glob(os.path.join(results_dir, "*_fidelity.json")))
    if not paths:
        raise SystemExit(f"no *_fidelity.json found in {results_dir!r}")
    print(f"plotting {len(paths)} run(s):")
    runs = []
    for path in paths:
        tag, meta, rows = load_run(path)
        plot_run(tag, meta, rows, results_dir, dpi)
        runs.append((tag, meta, rows))
    if len(runs) >= 2 and not no_comparison:
        plot_comparison(runs, results_dir, dpi)
    print("done.")
