# Conditional DDPM on Cartoon Set

This project trains a class-conditional Denoising Diffusion Probabilistic Model
on Google's Cartoon Set. Generation is controlled by categorical attributes
(`eye_color`, `hair_color`, `face_color`) and uses classifier-free guidance
(CFG), so the guidance weight `w` can be swept at sampling time.

## Environment

The project is managed with `uv` and targets Python 3.12:

```bash
uv sync
```

The default PyTorch source in `pyproject.toml` is the CUDA 12.6 wheel index
(`https://download.pytorch.org/whl/cu126`). If the training machine needs a
newer CUDA wheel, change only the `pytorch-cu126` index/source entries to the
appropriate PyTorch index such as `cu128`, then rerun `uv sync`.

Verify the environment with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run pytest
```

## Data

Download Cartoon Set from https://google.github.io/cartoonset/ and extract it
under `data/`:

```text
data/cartoonset10k/
    csXXXX.png
    csXXXX.csv

data/cartoonset100k/
    0/csXXXX.png
    0/csXXXX.csv
    ...
```

The unified dataset loader recursively discovers image/CSV pairs, so both the
flat 10k layout and nested 100k layout work. Images are resized to 32x32,
alpha-composited onto white, and normalized to `[-1, 1]`.

Inspect a dataset:

```bash
uv run python dataset.py data/cartoonset10k
uv run python dataset100k.py data/cartoonset100k --cache
```

## Project Structure

```text
src/cartoon_diffusion/
  data.py          unified 10k/100k dataset loader with optional cache
  model.py         conditional U-Net
  diffusion.py     DDPM/DDIM process and classifier-free guidance
  training.py      EMA, resume, checkpointing, train loop
  classifier.py    attribute classifier used for fidelity evaluation
  generation.py    batched DDPM/DDIM generation helpers
  metrics.py       fidelity, diversity, variance metrics
  results.py       typed result rows and schema-compatible save/load
  evaluation.py    high-level evaluation orchestration
  sampling.py      annotated guidance-weight grids
  plotting.py      result plotting with legacy/new schema support
  cli/             command-line entrypoints
configs/           reusable YAML experiment configs
docs/              report and presentation-facing material
scripts/           canonical thin wrappers
notebooks/         interactive sampler notebook
tests/             smoke/unit tests
results/           tracked experiment artifacts
```

Root-level files such as `train.py`, `evaluate.py`, and `sample.py` are kept as
thin compatibility wrappers for the original commands.

## Quick CPU Smoke Test

This verifies the full pipeline on a tiny subset. The images and metrics will
not be meaningful.

```bash
uv run python train.py --config configs/debug.yaml
uv run python sample.py --run_dir runs/debug --vary hair_color --weights 0 3 --sampler ddim --ddim_steps 10
uv run python evaluate.py --run_dir runs/debug --root data/cartoonset10k --weights 0 3 --n_samples 16 --sampler ddim --ddim_steps 10 --clf_epochs 1 --clf_batch 16
```

## Full Runs

The preferred workflow is config + run directory. Each run directory collects
the training config (`config.yaml`), command snapshots (`eval_config.yaml` and
`sample_config.yaml`), checkpoint, classifier, metric files, and generated
grids.

Train on 10k:

```bash
uv run python scripts/train.py --config configs/train_10k_40k.yaml
```

Training precision is controlled by `precision:` in the YAML config or by
`--precision` on the command line. Valid values are `fp32`, `fp16`, and `bf16`;
the full training configs default to `bf16`. For example:

```bash
uv run python scripts/train.py --config configs/train_10k_40k.yaml --precision bf16
uv run python scripts/train.py --config configs/train_10k_40k.yaml --precision fp16
```

Use `fp16` instead if the CUDA device does not support bf16.

The learning-rate schedule is controlled by:

```yaml
lr_schedule: constant   # or cosine
min_lr: 0.0             # cosine floor
```

For a cosine decay run:

```bash
uv run python scripts/train.py --config configs/train_10k_40k.yaml --lr_schedule cosine --min_lr 1e-5
```

The full training configs also enable conservative global gradient clipping:

```yaml
grad_clip: 1.0
```

Set it to an empty value or override with `--grad_clip` to change it.

Train or resume on 100k:

```bash
uv run python scripts/train.py --config configs/train_100k.yaml
uv run python scripts/train.py --config configs/train_100k.yaml --resume --steps 70000
```

Sample qualitative grids. DDPM is the default reported sampler; DDIM is kept as
a fast preview/artifact reproduction path.

```bash
uv run python scripts/sample.py --run_dir runs/run_full_40k --vary hair_color --weights 0 1 3 5
uv run python scripts/sample.py --run_dir runs/run_full_40k --vary hair_color --weights 0 1 3 5 --sampler ddim --ddim_steps 50
```

Optional upscaling. The diffusion model still generates `32x32` images; this
separate supervised CNN learns to enlarge real Cartoon Set images from `32x32`
to `96x96` and can be used as presentation-only post-processing for generated
PNGs. The default architecture uses a `5x5` head convolution, four `3x3`
residual blocks, PixelShuffle `x3`, two high-resolution `3x3` refinement
blocks, and a final `5x5` RGB projection. Its reconstruction objective is
`L1 + l2_weight * MSE`.

```bash
uv run python scripts/train_upscaler.py --config configs/train_upscaler_96.yaml
uv run python scripts/train_upscaler.py --config configs/train_upscaler_96.yaml --lr_schedule cosine --min_lr 1e-5
uv run python scripts/upscale.py --ckpt runs/upscaler_96/checkpoints/latest.pt --input outputs/grids/grid_hair_color.png
uv run python scripts/upscale.py --ckpt runs/upscaler_96/checkpoints/latest.pt --input_dir outputs/grids
```

```yaml
l2_weight: 1.0
head_kernel: 5
residual_kernel: 3
refinement_blocks: 2
tail_kernel: 5
```

Run metrics:

```bash
uv run python scripts/evaluate.py --config configs/eval_10k.yaml --test fidelity
uv run python scripts/run_all_tests.py --dataset_variant 100k --root data/cartoonset100k --ckpt runs/run100k/checkpoints/latest.pt --results_dir runs/run100k/results
uv run python scripts/plot_results.py
```

Explicit CLI flags still work and override config values, so one-off changes do
not require editing the YAML files.

## Notes

- New checkpoints include raw weights, EMA weights, optimizer state, attributes,
  image size, and diffusion timesteps. Old checkpoints without optimizer state
  still load for sampling/evaluation and can resume with a warning.
- New result JSON files use a flat schema (`weight`, attributes, `mean`) plus
  `schema_version=2`; plotting also accepts the legacy `per_attribute` schema.
- `docs/report.md` contains the project report.
- `data/`, `checkpoints/`, `outputs/`, and `runs/` are local artifacts and
  ignored by git.
