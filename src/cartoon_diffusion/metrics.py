"""Evaluation metric implementations."""

from __future__ import annotations

import time

import torch

from cartoon_diffusion.generation import generate, sample_random_labels
from cartoon_diffusion.results import DiversityRow, FidelityRow, VarianceRow


@torch.no_grad()
def attribute_fidelity(clf, imgs, labels, attribute_dims, batch=256):
    correct = torch.zeros(len(attribute_dims))
    for start in range(0, imgs.size(0), batch):
        logits = clf(imgs[start : start + batch])
        for i in range(len(attribute_dims)):
            correct[i] += (
                logits[i].argmax(1) == labels[start : start + batch, i]
            ).sum().cpu()
    return (correct / imgs.size(0)).tolist()


def _pairwise_rms(x):
    k = x.size(0)
    if k < 2:
        return 0.0
    flat = x.reshape(k, -1)
    total, count = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            total += torch.sqrt(((flat[i] - flat[j]) ** 2).mean()).item()
            count += 1
    return total / count


def run_fidelity(
    model,
    diff,
    clf,
    dims,
    attrs,
    weights,
    n_samples,
    image_size,
    device,
    *,
    seed=0,
    sampler="ddpm",
    ddim_steps=50,
    eta=0.0,
    verbose=True,
):
    torch.manual_seed(seed)
    if verbose:
        print("  w     " + "  ".join(f"{a:>12}" for a in attrs) + "     mean")
    rows = []
    for w in weights:
        t0 = time.time()
        imgs, labels = sample_random_labels(
            model,
            diff,
            dims,
            n_samples,
            image_size,
            w,
            device,
            sampler=sampler,
            ddim_steps=ddim_steps,
            eta=eta,
        )
        acc = attribute_fidelity(clf, imgs, labels, dims)
        mean = sum(acc) / len(acc)
        rows.append(
            FidelityRow(
                weight=w,
                per_attribute={attr: acc[i] for i, attr in enumerate(attrs)},
                mean=mean,
            )
        )
        if verbose:
            cells = "  ".join(f"{a:12.3f}" for a in acc)
            print(f"  {w:<4}  {cells}   {mean:6.3f}   ({time.time()-t0:.0f}s)")
    return rows


def run_diversity(
    model,
    diff,
    dims,
    weights,
    n_conditions,
    n_per_condition,
    image_size,
    device,
    *,
    seed=0,
    sampler="ddpm",
    ddim_steps=50,
    eta=0.0,
    verbose=True,
):
    g = torch.Generator(device=device).manual_seed(seed)
    conditions = torch.stack(
        [torch.randint(0, k, (n_conditions,), device=device, generator=g) for k in dims],
        dim=1,
    )
    labels = conditions.repeat_interleave(n_per_condition, dim=0)
    if verbose:
        print(f"  {n_conditions} conditions x {n_per_condition} samples each")
        print("  w      diversity (mean pairwise pixel RMS)")
    rows = []
    for w in weights:
        t0 = time.time()
        torch.manual_seed(10_000 + seed)
        imgs = generate(
            model,
            diff,
            labels,
            image_size,
            w,
            sampler=sampler,
            batch=n_conditions * n_per_condition,
            ddim_steps=ddim_steps,
            eta=eta,
        )
        imgs = imgs.reshape(n_conditions, n_per_condition, *imgs.shape[1:])
        div = sum(_pairwise_rms(imgs[c]) for c in range(n_conditions)) / n_conditions
        rows.append(DiversityRow(weight=w, diversity=div))
        if verbose:
            print(
                f"  {w:<5}  {div:8.4f}                        "
                f"({time.time()-t0:.0f}s)"
            )
    return rows


def run_variance(
    model,
    diff,
    clf,
    dims,
    attrs,
    weights,
    n_samples,
    image_size,
    device,
    repeats,
    *,
    base_seed=0,
    sampler="ddpm",
    ddim_steps=50,
    eta=0.0,
):
    per_w = {w: [] for w in weights}
    for r in range(repeats):
        print(f"  repeat {r + 1}/{repeats} (seed {base_seed + r}):")
        rows = run_fidelity(
            model,
            diff,
            clf,
            dims,
            attrs,
            weights,
            n_samples,
            image_size,
            device,
            seed=base_seed + r,
            sampler=sampler,
            ddim_steps=ddim_steps,
            eta=eta,
        )
        for row in rows:
            per_w[row.weight].append(row.mean)
    out = []
    print("\n  w       mean_fidelity   std")
    for w in weights:
        vals = per_w[w]
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        out.append(VarianceRow(weight=w, mean_fidelity=mean, std_fidelity=sd, runs=vals))
        print(f"  {w:<5}   {mean:6.3f}          {sd:6.4f}")
    return out
