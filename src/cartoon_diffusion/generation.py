"""Batched generation helpers for DDPM/DDIM evaluation."""

from __future__ import annotations

import torch


@torch.no_grad()
def generate(
    model,
    diff,
    labels,
    image_size,
    guidance_weight,
    *,
    sampler="ddpm",
    batch=128,
    ddim_steps=50,
    eta=0.0,
):
    outs = []
    for start in range(0, labels.size(0), batch):
        lab = labels[start : start + batch]
        if sampler == "ddpm":
            x = diff.ddpm_sample(
                model, lab, image_size=image_size, guidance_weight=guidance_weight
            )
        elif sampler == "ddim":
            x = diff.ddim_sample(
                model,
                lab,
                image_size=image_size,
                guidance_weight=guidance_weight,
                steps=ddim_steps,
                eta=eta,
            )
        else:
            raise ValueError(f"unknown sampler {sampler!r}")
        outs.append(x)
    return torch.cat(outs)


@torch.no_grad()
def sample_random_labels(
    model,
    diff,
    attribute_dims,
    n,
    image_size,
    guidance_weight,
    device,
    *,
    sampler="ddpm",
    batch=128,
    ddim_steps=50,
    eta=0.0,
):
    labels = torch.stack(
        [torch.randint(0, k, (n,), device=device) for k in attribute_dims],
        dim=1,
    )
    imgs = generate(
        model,
        diff,
        labels,
        image_size,
        guidance_weight,
        sampler=sampler,
        batch=batch,
        ddim_steps=ddim_steps,
        eta=eta,
    )
    return imgs, labels
