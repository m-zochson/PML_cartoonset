# Conditional DDPM on Cartoon Set (Classifier-Free Guidance)

Conditional denoising diffusion model that generates Google Cartoon Set faces
controlled by categorical attributes, using **classifier-free guidance (CFG)**.
The novelty over the course material (which covers unconditional DDPMs) is the
attribute conditioning + CFG: a tunable guidance weight `w` at sampling time.

## Requirements

```
pip install torch torchvision pillow numpy
```

Use a CUDA-enabled torch build for GPU training. All scripts are
device-agnostic (`cuda` if available, else `cpu`) — the same code runs on the
laptop (CPU, debugging) and on the desktop (GTX 1080, full training).

## Data

Download the Cartoon Set from https://google.github.io/cartoonset/ (start with
the **10k** set), extract it, and point `--root` at the folder of paired files:

```
data/cartoonset10k/
    csXXXX.png      # RGBA, transparent background
    csXXXX.csv      # 18 rows: "attr_name", "value_index", "cardinality"
    ...
```

Images are resized to 32x32 and the transparent background is composited on
white. Conditioning defaults to three low-cardinality colour attributes:
`eye_color` (5), `hair_color` (10), `face_color` (11). Change with `--attrs`.

## Project structure

| file          | role                                                          |
|---------------|---------------------------------------------------------------|
| `dataset.py`  | Cartoon Set dataset: alpha compositing, attribute parsing, dataloader |
| `model.py`    | Conditional U-Net `eps_theta(x_t, t, c)` with per-attribute embeddings + null token for CFG |
| `diffusion.py`| Gaussian diffusion: schedule, training loss (with CFG dropout), DDPM + DDIM sampling |
| `train.py`    | Training loop with EMA and checkpointing                      |
| `sample.py`   | Qualitative attribute x guidance-weight image grids           |
| `evaluate.py` | Attribute classifier + conditioning-fidelity metric vs `w`    |

## Quick test on a laptop (CPU)

Verifies the full chain runs; **not** meant to produce good images. Finishes in
about a minute with these tiny settings. Use a throwaway `--ckpt` name for test
runs so they never collide with a real training checkpoint.

```
python dataset.py data/cartoonset10k
python train.py    --root data/cartoonset10k --limit 64 --steps 20 --batch 16 --workers 0 --save_every 20 --ckpt test_ckpt.pt
python sample.py   --ckpt test_ckpt.pt --vary hair_color --weights 0 3 --steps 10 --out grid_test.png
python evaluate.py --root data/cartoonset10k --ckpt test_ckpt.pt --weights 0 3 --n_samples 16 --steps 10 --clf_epochs 1 --clf_batch 16
```

Noisy images, flat loss, and near-random accuracy are expected here.

## Full run on the desktop (GPU)

```
# 1. train (checkpoints to training_finale.pt)
python train.py --root data/cartoonset10k --steps 40000 --batch 128 --ckpt training_finale.pt

# 2. qualitative grid: rows = hair_color values, cols = guidance weights
python sample.py --ckpt training_finale.pt --vary hair_color --weights 0 1 3 5 --steps 50 --out grid_hair.png

# 3. quantitative: attribute fidelity vs guidance weight
python evaluate.py --root data/cartoonset10k --ckpt training_finale.pt --weights 0 1 3 5 --n_samples 512 --steps 50
```

## Checkpoint naming and safety

- **`--ckpt` controls the output file.** Always give test/debug runs their own
  name (e.g. `--ckpt test_ckpt.pt`) so they can never overwrite a real training
  checkpoint. If `--ckpt` is omitted, `train.py` auto-generates a timestamped
  name (`run_YYYYmmdd_HHMMSS.pt`) instead of defaulting to a single shared
  filename, precisely to avoid accidental collisions between runs.
- **One level of backup rotation** is still built in as a safety net: if the
  target checkpoint file already exists, it is renamed to `<name>.bak` right
  before the new one is written. This only protects against a single
  accidental overwrite — a second overwrite in a row will also overwrite the
  `.bak`. Don't rely on it instead of using distinct names.
- **Avoid syncing the working folder with OneDrive/Dropbox/etc.** while
  training: cloud sync rewriting a large `.pt` file mid-write is a common cause
  of "corrupted zip archive" errors when loading a checkpoint. Keep the project
  in a plain local folder, or pause sync during training.

## Notes

- **EMA weights** (`ckpt["ema"]`) are used for all sampling/evaluation; they
  give cleaner samples than the raw weights.
- **DDPM vs DDIM**: training uses `T=1000`. For sampling, `sample.py` and
  `evaluate.py` use DDIM (`--steps 50`) — far faster, same trained model. Full
  `ddpm_sample` (1000 steps) is available in `diffusion.py` for best quality.
- **Guidance weight**: `w=0` is unconditional-mixed; higher `w` sharpens
  attribute adherence at some cost to diversity. The fidelity table should rise
  with `w` then plateau — that is the CFG story to report.
- **Resuming / more data**: for the 100k set pass `preload=False` in the
  dataset to avoid loading everything into RAM.
- **FID** is intentionally omitted (needs InceptionV3 weights). Add it on the
  desktop with `torchmetrics.image.fid.FrechetInceptionDistance` or `pytorch-fid`.
