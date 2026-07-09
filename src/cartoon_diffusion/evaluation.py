"""High-level evaluation orchestration for conditional Cartoon Set generation."""

from __future__ import annotations

import os
import time
from datetime import datetime

import torch

from cartoon_diffusion.checkpoints import checkpoint_stem, load_checkpoint
from cartoon_diffusion.classifier import AttributeClassifier, train_classifier
from cartoon_diffusion.data import CartoonSetDataset
from cartoon_diffusion.diffusion import GaussianDiffusion
from cartoon_diffusion.generation import generate, sample_random_labels
from cartoon_diffusion.metrics import (
    attribute_fidelity,
    run_diversity,
    run_fidelity,
    run_variance,
)
from cartoon_diffusion.model import UNet
from cartoon_diffusion.results import save_results


def load_generator(ckpt_path, device):
    ck = load_checkpoint(ckpt_path, map_location=device)
    attrs, dims = ck["attrs"], ck["attribute_dims"]
    image_size, timesteps = ck["image_size"], ck["timesteps"]
    model = UNet(dims, image_size=image_size).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return ck, model, GaussianDiffusion(timesteps=timesteps), attrs, dims


def evaluate_from_args(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}   test: {args.test}   sampler: {args.sampler}")

    ck, model, diff, attrs, dims = load_generator(args.ckpt, device)
    image_size, timesteps = ck["image_size"], ck["timesteps"]
    print(
        f"generator: attrs {attrs}  dims {dims}  image_size {image_size}  "
        f"timesteps {timesteps}"
    )

    clf = None
    if args.test in ("fidelity", "variance"):
        ds = CartoonSetDataset(
            args.root,
            image_size=image_size,
            cond_attributes=tuple(attrs),
            cache=args.dataset_variant == "100k",
            rebuild_cache=args.rebuild_cache,
        )
        if ds.attribute_dims != dims:
            raise ValueError("dataset/ckpt attribute mismatch")
        if os.path.exists(args.clf_ckpt):
            clf = AttributeClassifier(dims).to(device)
            clf.load_state_dict(torch.load(args.clf_ckpt, map_location=device))
            clf.eval()
            print(f"loaded classifier from {args.clf_ckpt}")
        else:
            print("training attribute classifier on real data...")
            os.makedirs(os.path.dirname(args.clf_ckpt) or ".", exist_ok=True)
            clf = train_classifier(
                ds,
                dims,
                device,
                epochs=args.clf_epochs,
                batch=args.clf_batch,
                workers=args.workers,
            )
            torch.save(clf.state_dict(), args.clf_ckpt)
            print(f"saved classifier to {args.clf_ckpt}")

    print(f"\nrunning '{args.test}' over weights {args.weights}\n")
    t_start = time.time()
    if args.test == "fidelity":
        rows = run_fidelity(
            model,
            diff,
            clf,
            dims,
            attrs,
            args.weights,
            args.n_samples,
            image_size,
            device,
            seed=args.seed,
            sampler=args.sampler,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
        )
        extra_meta = {"n_samples": args.n_samples}
    elif args.test == "diversity":
        rows = run_diversity(
            model,
            diff,
            dims,
            args.weights,
            args.n_conditions,
            args.n_per_condition,
            image_size,
            device,
            seed=args.seed,
            sampler=args.sampler,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
        )
        extra_meta = {
            "n_conditions": args.n_conditions,
            "n_per_condition": args.n_per_condition,
        }
    else:
        rows = run_variance(
            model,
            diff,
            clf,
            dims,
            attrs,
            args.weights,
            args.n_samples,
            image_size,
            device,
            args.repeats,
            base_seed=args.seed,
            sampler=args.sampler,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
        )
        extra_meta = {"n_samples": args.n_samples, "repeats": args.repeats}

    print(f"\ntotal time: {time.time() - t_start:.0f}s")
    tag = args.tag or checkpoint_stem(args.ckpt)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "test": args.test,
        "dataset_variant": args.dataset_variant,
        "root": args.root,
        "ckpt": args.ckpt,
        "clf_ckpt": args.clf_ckpt if args.test != "diversity" else None,
        "attrs": list(attrs),
        "attribute_dims": list(dims),
        "image_size": image_size,
        "timesteps": timesteps,
        "sampler": args.sampler,
        "ddim_steps": args.ddim_steps if args.sampler == "ddim" else None,
        "eta": args.eta if args.sampler == "ddim" else None,
        "seed": args.seed,
        "weights": list(args.weights),
        "run_dir": args.run_dir,
        "config": args.config,
    }
    meta.update(extra_meta)
    json_path, csv_path = save_results(args.results_dir, tag, args.test, meta, rows)
    print(f"saved results to:\n  {json_path}\n  {csv_path}")


__all__ = [
    "AttributeClassifier",
    "attribute_fidelity",
    "evaluate_from_args",
    "generate",
    "load_generator",
    "run_diversity",
    "run_fidelity",
    "run_variance",
    "sample_random_labels",
    "save_results",
    "train_classifier",
]
