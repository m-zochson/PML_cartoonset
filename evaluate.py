"""
Evaluation for the conditional DDPM (Cartoon Set 10k).

SAMPLING IS DDPM-ONLY. DDIM was removed from the evaluation path: under
classifier-free guidance its accelerated (few-step) trajectory produces a
spurious quality/fidelity collapse that does NOT reflect the trained model
(see report.md, Section 5.1). All metrics here use full ancestral DDPM
sampling over `timesteps` steps.

One test is run per invocation via --test:

  fidelity   Attribute fidelity vs guidance weight w. A multi-head classifier
             trained on the REAL images predicts each attribute of generated
             images; fidelity = fraction matching the requested attribute.
             Sweeping w (including high w) shows the CFG trade-off.

  diversity  Intra-condition diversity vs w. For each of several FIXED
             conditions, generate multiple samples (identical RNG across all
             w so only w changes) and measure mean pairwise pixel RMS distance.
             Detects mode collapse: high fidelity with low diversity means the
             model "cheats" by producing near-identical images.

  variance   Repeat the fidelity sweep with different seeds and report
             mean +/- std of mean fidelity per w, to tell whether small
             differences between weights are real or just sampling noise.

Examples:
  python evaluate.py --test fidelity  --root data/cartoonset10k --ckpt run.pt \
      --weights 0 1 3 5 8 10 15 --n_samples 256
  python evaluate.py --test diversity --root data/cartoonset10k --ckpt run.pt \
      --weights 0 1 3 5 8 10 15 --n_conditions 16 --n_per_condition 8
  python evaluate.py --test variance  --root data/cartoonset10k --ckpt run.pt \
      --weights 0 3 5 --n_samples 128 --repeats 3

Results are written to --results_dir (default: results/) as:
  <tag>_<test>.json   full metadata + per-weight rows
  <tag>_<test>.csv    flat table ready to plot
where <tag> defaults to the checkpoint stem, so tests/runs never collide.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CartoonSetDataset, build_dataloader
from diffusion import GaussianDiffusion
from model import UNet


class AttributeClassifier(nn.Module):
    """Shared conv backbone + one linear head per attribute."""

    def __init__(self, attribute_dims, in_ch=3, base=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, stride=2, padding=1),      # 32 -> 16
            nn.BatchNorm2d(base), nn.ReLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),   # 16 -> 8
            nn.BatchNorm2d(base * 2), nn.ReLU(),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(base * 4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(base * 4, k) for k in attribute_dims]
        )

    def forward(self, x):
        feat = self.backbone(x)
        return [head(feat) for head in self.heads]


def train_classifier(ds, attribute_dims, device, epochs=8, lr=1e-3, batch=128):
    clf = AttributeClassifier(attribute_dims).to(device)
    loader = build_dataloader(ds, batch_size=batch, num_workers=4)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    clf.train()
    for ep in range(1, epochs + 1):
        tot, correct, seen = 0.0, torch.zeros(len(attribute_dims)), 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = clf(x)
            loss = sum(F.cross_entropy(logits[i], y[:, i])
                       for i in range(len(attribute_dims)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            for i in range(len(attribute_dims)):
                correct[i] += (logits[i].argmax(1) == y[:, i]).sum().cpu()
            seen += x.size(0)
        acc = (correct / seen).tolist()
        print(f"  clf epoch {ep}/{epochs}  loss {tot/seen:.3f}  "
              f"train acc {[round(a,3) for a in acc]}")
    clf.eval()
    return clf


# --------------------------------------------------------------------------- #
#  Generation helper (DDPM only)                                              #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ddpm_generate(model, diff, labels, image_size, guidance_weight, batch=128):
    """Generate one image per row of `labels` with full DDPM ancestral sampling.

    Chunks large label tensors into batches for memory; concatenates the result
    back in the original row order.
    """
    outs = []
    for s in range(0, labels.size(0), batch):
        lab = labels[s:s + batch]
        x = diff.ddpm_sample(model, lab, image_size=image_size,
                             guidance_weight=guidance_weight)
        outs.append(x)
    return torch.cat(outs)


@torch.no_grad()
def sample_random_labels(model, diff, attribute_dims, n, image_size,
                         guidance_weight, device, batch=128):
    """Generate n images with random attribute vectors (DDPM). -> imgs, labels."""
    labels = torch.stack(
        [torch.randint(0, k, (n,), device=device) for k in attribute_dims],
        dim=1,
    )
    imgs = ddpm_generate(model, diff, labels, image_size, guidance_weight, batch)
    return imgs, labels


@torch.no_grad()
def attribute_fidelity(clf, imgs, labels, attribute_dims, batch=256):
    """Per-attribute accuracy: classifier prediction vs requested value."""
    correct = torch.zeros(len(attribute_dims))
    for s in range(0, imgs.size(0), batch):
        logits = clf(imgs[s:s + batch])
        for i in range(len(attribute_dims)):
            correct[i] += (logits[i].argmax(1) == labels[s:s + batch, i]).sum().cpu()
    return (correct / imgs.size(0)).tolist()


def _pairwise_rms(x):
    """Mean over all unordered pairs of per-element RMS distance.

    x: (k, C, H, W) in [-1, 1]. Returns a scalar; larger => more diverse.
    """
    k = x.size(0)
    if k < 2:
        return 0.0
    flat = x.reshape(k, -1)
    tot, cnt = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            tot += torch.sqrt(((flat[i] - flat[j]) ** 2).mean()).item()
            cnt += 1
    return tot / cnt


# --------------------------------------------------------------------------- #
#  Tests                                                                       #
# --------------------------------------------------------------------------- #
def run_fidelity(model, diff, clf, dims, attrs, weights, n_samples,
                 image_size, device, seed=0, verbose=True):
    torch.manual_seed(seed)
    if verbose:
        header = "  w     " + "  ".join(f"{a:>12}" for a in attrs) + "     mean"
        print(header)
    rows = []
    for w in weights:
        t0 = time.time()
        imgs, labels = sample_random_labels(
            model, diff, dims, n_samples, image_size, w, device)
        acc = attribute_fidelity(clf, imgs, labels, dims)
        mean = sum(acc) / len(acc)
        row = {"weight": w, "mean": mean}
        row.update({a: acc[i] for i, a in enumerate(attrs)})
        rows.append(row)
        if verbose:
            cells = "  ".join(f"{a:12.3f}" for a in acc)
            print(f"  {w:<4}  {cells}   {mean:6.3f}   ({time.time()-t0:.0f}s)")
    return rows


def run_diversity(model, diff, dims, weights, n_conditions, n_per_condition,
                  image_size, device, seed=0, verbose=True):
    # Fixed set of conditions, reused across every weight (deterministic).
    g = torch.Generator(device=device).manual_seed(seed)
    conditions = torch.stack(
        [torch.randint(0, k, (n_conditions,), device=device, generator=g)
         for k in dims], dim=1)                         # (C, A)
    labels = conditions.repeat_interleave(n_per_condition, dim=0)  # (C*k, A)

    if verbose:
        print(f"  {n_conditions} conditions x {n_per_condition} samples each")
        print("  w      diversity (mean pairwise pixel RMS)")
    rows = []
    for w in weights:
        t0 = time.time()
        # Identical RNG for every w (same x_T and same ancestral noise), so the
        # ONLY thing that changes between weights is the guidance strength.
        torch.manual_seed(10_000 + seed)
        imgs = ddpm_generate(model, diff, labels, image_size, w,
                             batch=n_conditions * n_per_condition)
        imgs = imgs.reshape(n_conditions, n_per_condition, *imgs.shape[1:])
        per_cond = [_pairwise_rms(imgs[c]) for c in range(n_conditions)]
        div = sum(per_cond) / len(per_cond)
        rows.append({"weight": w, "diversity": div})
        if verbose:
            print(f"  {w:<5}  {div:8.4f}                        "
                  f"({time.time()-t0:.0f}s)")
    return rows


def run_variance(model, diff, clf, dims, attrs, weights, n_samples,
                 image_size, device, repeats, base_seed=0):
    per_w = {w: [] for w in weights}
    for r in range(repeats):
        print(f"  repeat {r+1}/{repeats} (seed {base_seed + r}):")
        rows = run_fidelity(model, diff, clf, dims, attrs, weights, n_samples,
                            image_size, device, seed=base_seed + r, verbose=True)
        for row in rows:
            per_w[row["weight"]].append(row["mean"])
    out = []
    print("\n  w       mean_fidelity   std")
    for w in weights:
        vals = per_w[w]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        out.append({"weight": w, "mean_fidelity": m, "std_fidelity": sd,
                    "runs": vals})
        print(f"  {w:<5}   {m:6.3f}          {sd:6.4f}")
    return out


# --------------------------------------------------------------------------- #
#  Persistence                                                                 #
# --------------------------------------------------------------------------- #
def save_results(results_dir, tag, test, meta, rows, fieldnames):
    """Write results/<tag>_<test>.{json,csv}. Returns the two paths."""
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f"{tag}_{test}.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "results": rows}, f, indent=2)
    csv_path = os.path.join(results_dir, f"{tag}_{test}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


DATASET_VARIANT = "10k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["fidelity", "diversity", "variance"],
                    default="fidelity",
                    help="which test to run (one per invocation)")
    ap.add_argument("--root", default="data/cartoonset10k")
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--clf_ckpt", default="classifier.pt")
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0, 1, 3, 5, 8, 10, 15])
    # fidelity / variance
    ap.add_argument("--n_samples", type=int, default=256,
                    help="images per weight (fidelity, variance)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="number of seeds to average over (variance)")
    # diversity
    ap.add_argument("--n_conditions", type=int, default=16,
                    help="distinct conditions held fixed across w (diversity)")
    ap.add_argument("--n_per_condition", type=int, default=8,
                    help="samples generated per condition (diversity)")
    # misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clf_epochs", type=int, default=8)
    ap.add_argument("--clf_batch", type=int, default=128)
    ap.add_argument("--results_dir", default="results",
                    help="folder for the JSON/CSV result files")
    ap.add_argument("--tag", default=None,
                    help="basename for result files (default: checkpoint stem)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}   test: {args.test}   sampler: DDPM (full chain)")

    ck = torch.load(args.ckpt, map_location=device)
    attrs, dims = ck["attrs"], ck["attribute_dims"]
    image_size = ck["image_size"]
    timesteps = ck["timesteps"]
    print(f"generator: attrs {attrs}  dims {dims}  image_size {image_size}  "
          f"timesteps {timesteps}")

    # generator with EMA weights
    model = UNet(dims, image_size=image_size).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    diff = GaussianDiffusion(timesteps=timesteps)

    # classifier only needed for fidelity / variance
    clf = None
    if args.test in ("fidelity", "variance"):
        ds = CartoonSetDataset(args.root, image_size=image_size,
                               cond_attributes=tuple(attrs))
        assert ds.attribute_dims == dims, "dataset/ckpt attribute mismatch"
        if os.path.exists(args.clf_ckpt):
            clf = AttributeClassifier(dims).to(device)
            clf.load_state_dict(torch.load(args.clf_ckpt, map_location=device))
            clf.eval()
            print(f"loaded classifier from {args.clf_ckpt}")
        else:
            print("training attribute classifier on real data...")
            clf = train_classifier(ds, dims, device, epochs=args.clf_epochs,
                                   batch=args.clf_batch)
            torch.save(clf.state_dict(), args.clf_ckpt)
            print(f"saved classifier to {args.clf_ckpt}")

    print(f"\nrunning '{args.test}' over weights {args.weights} "
          f"(DDPM, {timesteps} steps/sample)\n")
    t_start = time.time()

    if args.test == "fidelity":
        rows = run_fidelity(model, diff, clf, dims, attrs, args.weights,
                            args.n_samples, image_size, device, seed=args.seed)
        fieldnames = ["weight"] + list(attrs) + ["mean"]
        extra_meta = {"n_samples": args.n_samples}
    elif args.test == "diversity":
        rows = run_diversity(model, diff, dims, args.weights, args.n_conditions,
                            args.n_per_condition, image_size, device,
                            seed=args.seed)
        fieldnames = ["weight", "diversity"]
        extra_meta = {"n_conditions": args.n_conditions,
                      "n_per_condition": args.n_per_condition}
    else:  # variance
        rows = run_variance(model, diff, clf, dims, attrs, args.weights,
                            args.n_samples, image_size, device,
                            repeats=args.repeats, base_seed=args.seed)
        fieldnames = ["weight", "mean_fidelity", "std_fidelity"]
        extra_meta = {"n_samples": args.n_samples, "repeats": args.repeats}

    print(f"\ntotal time: {time.time()-t_start:.0f}s")

    tag = args.tag or os.path.splitext(os.path.basename(args.ckpt))[0]
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "test": args.test,
        "dataset_variant": DATASET_VARIANT,
        "root": args.root,
        "ckpt": args.ckpt,
        "clf_ckpt": args.clf_ckpt if args.test != "diversity" else None,
        "attrs": list(attrs),
        "attribute_dims": list(dims),
        "image_size": image_size,
        "timesteps": timesteps,
        "sampler": "ddpm",
        "seed": args.seed,
        "weights": list(args.weights),
    }
    meta.update(extra_meta)
    json_path, csv_path = save_results(args.results_dir, tag, args.test,
                                       meta, rows, fieldnames)
    print(f"saved results to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
