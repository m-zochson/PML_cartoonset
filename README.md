# Conditional DDPM on Cartoon Set (Classifier-Free Guidance)

Conditional Denoising Diffusion Probabilistic Model that generates Google
Cartoon Set faces controlled by three categorical attributes
(`eye_color`, `hair_color`, `face_color`), using **classifier-free guidance
(CFG)**. The novelty with respect to the course material (which covers only
*unconditional* DDPMs) is attribute conditioning + CFG: a tunable guidance
weight `w` exposed at sampling time.

**Headline finding:** the guidance-weight sweep gives opposite conclusions
depending on the sampler. Under DDIM, fidelity collapses for `w ≳ 3`; under
full ancestral DDPM sampling (same checkpoint, same classifier), fidelity
rises monotonically with `w` and plateaus around `w ≈ 3`. The cause is DDIM's
explicit `x0`-reconstruction step, which amplifies the guided noise by a large
coefficient at high `t` (~157× at `t=999` for this schedule) and injects
artifacts. **DDIM was abandoned; every reported result uses DDPM sampling.**
See [`report.md`](report.md), Section 5.1, for the full derivation and
comparison tables.

## Requirements

```
pip install torch torchvision pillow numpy matplotlib
```

Use a CUDA-enabled torch build for GPU training (developed against
`torch==2.6.0+cu124`). All scripts are device-agnostic (`cuda` if available,
else `cpu`) — the same code runs on a CPU-only laptop (development/debugging)
and on a CUDA desktop (full training; developed on a GTX 1080, 8 GB).

## Data

Download the Cartoon Set from https://google.github.io/cartoonset/, extract
it, and point `--root` at the folder of paired files:

```
data/cartoonset10k/
    csXXXX.png      # RGBA, transparent background
    csXXXX.csv      # 18 rows: "attr_name", "value_index", "cardinality"
    ...
```

Images are resized to 32x32 and the transparent background is alpha-composited
onto white. Conditioning defaults to three low-cardinality colour attributes:
`eye_color` (5), `hair_color` (10), `face_color` (11) — cardinalities are
parsed from the data, never hard-coded. Change the set with `--attrs`.

Two dataset variants are supported:

- **10k** (`dataset.py`) — flat folder, loaded straight into RAM.
- **100k** (`dataset100k.py`) — drop-in replacement; handles the nested
  `0/ .. 9/` sub-folder layout of the 100k download via a recursive glob, and
  builds an on-disk `uint8` tensor cache (written atomically, temp file +
  `os.replace`) so decoding/compositing/resizing only happens once.

## Project structure

| file                     | role                                                          |
|--------------------------|----------------------------------------------------------------|
| `dataset.py`             | Cartoon Set (10k) dataset: alpha compositing, attribute parsing, dataloader |
| `dataset100k.py`         | Same, for the 100k set: recursive glob + atomic disk cache      |
| `model.py`               | Conditional U-Net `eps_theta(x_t, t, c)` (~4.45M params): sinusoidal time embedding + one embedding table per attribute (with a null row for CFG), summed and injected into every residual block, self-attention at the 8x8 resolution |
| `diffusion.py`           | Gaussian diffusion: linear beta schedule, `q_sample`, training loss with CFG condition-dropout, guided **DDPM** and **DDIM** samplers |
| `train.py`               | Training loop (10k) with EMA and collision-proof checkpointing |
| `train100k.py`           | Training loop for the 100k set (same conventions, `--resume`, `--rebuild_cache`) |
| `sample.py`              | Qualitative attribute x guidance-weight image grids (DDPM only — no step-count knob) |
| `evaluate.py`            | Attribute classifier + conditioning-fidelity metric vs `w` (10k) |
| `evaluate100k.py`        | Same, for the 100k checkpoint/classifier                        |
| `plot_results.py`        | Turns the JSON/CSV results into publication-ready PNG/PDF plots, incl. a 10k-vs-100k comparison figure |
| `generate_interactive.ipynb` | Notebook UI (`ipywidgets`) to pick attributes and sample interactively from a trained checkpoint; also recovers a human-readable colour legend for each attribute value directly from the training data |
| `report.md` / `report.tex` | Full write-up: theory (forward/backward diffusion, DDIM, CFG derivation), implementation, and the DDIM-vs-DDPM evaluation story |
| `README.md`              | this file |

## Quick test on a laptop (CPU)

Verifies the full chain runs; **not** meant to produce good images. Finishes in
about a minute with these tiny settings. Use a throwaway `--ckpt` name for test
runs so they never collide with a real training checkpoint.

```
python dataset.py data/cartoonset10k
python train.py    --root data/cartoonset10k --limit 64 --steps 20 --batch 16 --workers 0 --save_every 20 --ckpt test_ckpt.pt
python sample.py   --ckpt test_ckpt.pt --vary hair_color --weights 0 3 --out grid_test.png
python evaluate.py --root data/cartoonset10k --ckpt test_ckpt.pt --weights 0 3 --n_samples 16 --clf_epochs 1 --clf_batch 16 --sampler ddpm
```

Noisy images, flat loss, and near-random accuracy are expected here. On
Windows, always pass `--workers 0`.

## Full run on the desktop (GPU)

```
# 1. train on the 10k set (checkpoints to run_full_40k.pt)
python train.py --root data/cartoonset10k --steps 40000 --batch 128 --ckpt run_full_40k.pt

# 2. qualitative grids: rows = attribute values, cols = guidance weights
#    (one annotated PNG per attribute: grid_eye_color.png, grid_hair_color.png, grid_face_color.png)
python sample.py --ckpt run_full_40k.pt --vary all --weights 0 1 3 5 8 10 15

# 3. quantitative fidelity sweep — always pass --sampler ddpm for reported numbers
python evaluate.py --root data/cartoonset10k --ckpt run_full_40k.pt \
    --weights 0 1 3 5 8 10 15 --n_samples 512 --sampler ddpm --tag run_full_40k

# 4. plot everything found under results/
python plot_results.py
```

The 100k comparison run follows the same pattern with `dataset100k.py` /
`train100k.py` / `evaluate100k.py` (`--root data/cartoonset100k`).

**Important — DDIM is not used for any reported result.** `evaluate.py` /
`evaluate100k.py` still expose `--sampler {ddim,ddpm}` so the artifact in
`report.md` §5.1 can be reproduced on demand, but the default reported
sampler is always `ddpm`. `sample.py` was simplified to DDPM-only ancestral
sampling (it runs the full reverse chain — there is no `--steps` flag).

## Evaluation: three tests, per checkpoint tag

Results are written to `results/<tag>_<test>.{json,csv}`, where `<tag>`
defaults to the checkpoint name (e.g. `run_full_40k`, `run100k`):

- **`fidelity`** — for each guidance weight `w`, generate images from random
  attribute vectors and measure the fraction whose attributes are classified
  correctly by an independent attribute classifier (trained only on real
  images, never shared weights with the generator). This is the central
  result; see `report.md` §5 for the DDIM-vs-DDPM comparison and §5.2 for the
  DDPM table.
- **`diversity`** — average pairwise pixel-distance between images generated
  under the *same* condition, to check CFG isn't trading fidelity for mode
  collapse.
- **`variance`** — the fidelity sweep repeated over multiple seeds (mean +
  std), to distinguish a real effect from run-to-run noise.

**Headline numbers (DDPM, `w` sweep `0 1 3 5 8 10 15`):**

| dataset | best `w` (mean fidelity) | mean fidelity @ best `w` | diversity trend |
|---------|---------------------------|---------------------------|------------------|
| 10k, 40k steps | 3 | ~0.86 | rises through `w=8-10`, mild dip by `w=15` |
| 100k, 40k steps | 3 (close to `w=1,5`) | ~0.82 | rises through `w=5`, mild dip by `w=15` |

Both dataset variants plateau around `w=3` and degrade mildly at very high
guidance (`w=15`) rather than collapsing outright — the DDIM-only collapse
described above never appears under DDPM. See `plot_results.py` output
(`results/comparison_fidelity.pdf`, etc.) and `report.md` for the full
per-attribute breakdown.

## Checkpoint naming and safety

- **`--ckpt` controls the output file.** Always give test/debug runs their own
  name (e.g. `--ckpt test_ckpt.pt`) so they can never overwrite a real training
  checkpoint. If `--ckpt` is omitted, `train.py` / `train100k.py` auto-generate
  a timestamped name (`run_YYYYmmdd_HHMMSS.pt`) instead of defaulting to a
  single shared filename, precisely to avoid accidental collisions between
  runs.
- **One level of backup rotation** is still built in as a safety net: if the
  target checkpoint file already exists, it is renamed to `<name>.bak` right
  before the new one is written. This only protects against a single
  accidental overwrite — a second overwrite in a row will also overwrite the
  `.bak`. Don't rely on it instead of using distinct names.
- **Avoid syncing the working folder with OneDrive/Dropbox/etc.** while
  training: cloud sync rewriting a large `.pt` file mid-write is a common cause
  of "corrupted zip archive" errors when loading a checkpoint. Keep the
  project in a plain local folder, or pause sync during training.
- Keep separate `--clf_ckpt` names for the 10k and 100k attribute classifiers
  (e.g. `classifier.pt` vs. `classifier100k.pt`) — they're trained on
  different data and shouldn't be mixed across evaluation runs.

## Notes

- **EMA weights** (`ckpt["ema"]`, decay `0.999`) are used for all
  sampling/evaluation; they give cleaner samples than the raw weights.
- **DDPM vs. DDIM**: training always uses `T=1000`. DDIM (`ddim_sample` in
  `diffusion.py`) remains in the codebase purely as a fast preview during
  development and to reproduce the sampler-artifact comparison — it is
  **not** used for any figure or number in `report.md`. Full ancestral
  `ddpm_sample` (1000 steps, no step-count knob) is used everywhere else.
- **Guidance weight convention**: `epsilon_hat = (1+w)*eps_theta(c) -
  w*eps_theta(null)`, so `w=0` is the *pure conditional* model, not
  unconditional. Fidelity rises with `w`, plateaus around `w≈3`, and degrades
  only mildly even out to `w=15` under DDPM.
- **FID** is intentionally omitted (needs InceptionV3 weights); attribute
  fidelity + diversity are used instead as the quantitative measures.
- The interactive notebook (`generate_interactive.ipynb`) also recovers a
  human-readable colour legend (index -> hex/colour name) directly from the
  training images, since Cartoon Set publishes only palette indices.
