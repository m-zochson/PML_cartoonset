# Conditional Denoising Diffusion with Classifier-Free Guidance on the Cartoon Set

**Probabilistic Machine Learning — Project Report**
Matteo — University of Trieste

---

## Abstract

This project implements a *class-conditional* Denoising Diffusion Probabilistic Model (DDPM) that generates cartoon avatars from Google's Cartoon Set, with generation controlled by a small set of categorical attributes. The methodological element that is *novel with respect to the course* (which covers unconditional DDPMs) is the combination of attribute conditioning and *classifier-free guidance* (CFG), which exposes a tunable guidance weight $w$ at sampling time. The report first develops the theory (forward/backward diffusion, the noise-prediction objective, DDIM sampling, and the derivation of classifier-free guidance) and then documents the full implementation — data pipeline, conditional U-Net, diffusion process, training loop, and evaluation — together with the evaluation methodology (attribute-fidelity vs. guidance weight).

---

## 1. Overview and motivation

A DDPM learns to generate data by reversing a fixed noising process. The course presents the *unconditional* case: the model learns $p(\mathbf{x}_0)$ and produces samples with no control over their content. For a labelled dataset such as the Cartoon Set, this leaves the most interesting question untouched: can we ask the model for a face with a *specific* hair colour, eye colour or skin tone?

The project answers this by (i) conditioning the denoising network on the image attributes, $\epsilon_\theta(\mathbf{x}_t, t, c)$, and (ii) training it so that a single network represents both the conditional and the unconditional model, which at sampling time can be linearly combined to *amplify* the influence of the condition. This is classifier-free guidance. The unconditional DDPM is the course baseline; the conditioning mechanism and CFG are the contribution.

---

## 2. Theoretical background

### 2.1 Forward diffusion process

Let $\mathbf{x}_0 \sim p(\mathbf{x}_0)$ be a data sample. The forward process is a Markov chain that gradually adds Gaussian noise over $T$ steps, with a fixed variance schedule $\beta_1, \dots, \beta_T \in (0,1)$:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\big(\mathbf{x}_t;\, \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I}\big).$$

Define $\alpha_t \coloneqq 1-\beta_t$ and $\bar\alpha_t \coloneqq \prod_{i=1}^{t}\alpha_i$. A key property is that the marginal at *any* step has a closed form. Using the reparameterization

$$\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}$$

recursively and merging the independent Gaussian noises,

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t-1} \\
&= \sqrt{\alpha_t\alpha_{t-1}}\,\mathbf{x}_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\,\hat\epsilon_{t-2}
= \dots = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\hat\epsilon,
\end{aligned}
$$

so that

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\big(\mathbf{x}_t;\, \sqrt{\bar\alpha_t}\,\mathbf{x}_0,\, (1-\bar\alpha_t)\mathbf{I}\big)$$

*(1)*

As $t \to T$ with $\bar\alpha_T \to 0$, $q(\mathbf{x}_T \mid \mathbf{x}_0) \to \mathcal{N}(\mathbf{0}, \mathbf{I})$: the data is destroyed into white noise, which is the tractable prior we can sample from directly.

> **Why this matters.** Equation (1) lets us jump to *any* noise level in one shot, without simulating the whole chain. This is what makes training cheap: for a random $t$ we draw $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and form $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ in a single operation. In the code this is exactly `q_sample`.

### 2.2 Backward process and the denoising parametrization

Generation requires the reverse chain. The true reverse kernel

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$

is intractable, so we fit a parametric Gaussian

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol\mu_\theta(\mathbf{x}_t, t), \Sigma_\theta).$$

Crucially, the posterior conditioned on $\mathbf{x}_0$ is tractable and Gaussian:

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\!\big(\mathbf{x}_{t-1};\, \tilde{\boldsymbol\mu}_t(\mathbf{x}_t, \mathbf{x}_0),\, \tilde\sigma_t \mathbf{I}\big), \qquad \tilde\sigma_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.$$

Rather than predicting $\mathbf{x}_0$ directly, DDPMs predict the *noise*. Inverting $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ and substituting, the posterior mean can be written purely in terms of the noise $\epsilon$:

$$\tilde{\boldsymbol\mu}_t(\mathbf{x}_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Big)$$

*(2)*

We therefore train a network $\epsilon_\theta(\mathbf{x}_t, t)$ to predict the noise that was added. The full variational bound simplifies (Ho et al., 2020) to the unweighted denoising objective:

$$L(\theta) = \mathbb{E}_{t,\mathbf{x}_0,\epsilon}\Big[\big\|\,\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\, t\big)\big\|^2\Big]$$

*(3)*

Here $t$ is drawn uniformly over $\{1, \dots, T\}$, $\mathbf{x}_0$ from the dataset, and $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Predicting a target that is marginally standard-normal is empirically easier to optimize than predicting $\mathbf{x}_0$.

### 2.3 Ancestral (DDPM) sampling

Given a trained $\epsilon_\theta$, we start from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and iterate the reverse step for $t = T, \dots, 1$ using (2):

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(\mathbf{x}_t, t)\Big) + \sqrt{\beta_t}\,\mathbf{z}, \qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

*(4)*

with $\mathbf{z} = \mathbf{0}$ at the last step. The implementation uses $\sigma_t^2 = \beta_t$ (one of the two variances proposed by Ho et al.), which is simple and works well in practice.

### 2.4 DDIM: fast deterministic sampling

Ancestral sampling needs all $T$ (here $1000$) sequential steps. DDIM (Song et al., 2021) defines a non-Markovian process with the *same* training objective, allowing sampling on a short sub-sequence of steps. With the noise prediction, one first estimates the clean image

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\bar\alpha_t}},$$

and then moves to the previous retained step $s < t$:

$$\mathbf{x}_s = \sqrt{\bar\alpha_s}\,\hat{\mathbf{x}}_0 + \sqrt{1-\bar\alpha_s-\sigma^2}\,\epsilon_\theta(\mathbf{x}_t, t) + \sigma\,\mathbf{z}, \qquad \sigma = \eta\sqrt{\tfrac{1-\bar\alpha_s}{1-\bar\alpha_t}\big(1-\tfrac{\bar\alpha_t}{\bar\alpha_s}\big)}.$$

With $\eta = 0$ the process is deterministic and typically needs only $50$–$100$ steps for good quality. **This project initially used DDIM for the evaluation grids and metrics (for speed), but abandoned it after the guidance-weight sweep (Section 5) showed it produces a spurious quality collapse under classifier-free guidance that does not reflect the underlying conditional model — see the discussion in Section 5.1. All quantitative and qualitative results reported here use full ancestral DDPM sampling instead.**

### 2.5 Conditioning and classifier-free guidance

**Conditional model.** We make the network depend on a condition $c$ (the attributes): $\epsilon_\theta(\mathbf{x}_t, t, c)$. Everything above carries over unchanged; $c$ is simply an extra input.

**Score view.** Noise prediction is equivalent to score estimation:

$$\epsilon_\theta(\mathbf{x}_t, t) \approx -\sqrt{1-\bar\alpha_t}\,\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t)$$

For the conditional case, Bayes' rule gives

$$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t \mid c) = \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t}\log p(c \mid \mathbf{x}_t).$$

*Classifier guidance* (Dhariwal & Nichol, 2021) adds the gradient of an external classifier $p(c \mid \mathbf{x}_t)$, scaled by a weight, to sharpen conditioning — at the cost of training a separate noisy-image classifier.

**Classifier-free guidance.** Ho & Salimans (2022) avoid the external classifier. Rearranging the identity above, $\nabla\log p(c \mid \mathbf{x}_t) = \nabla\log p(\mathbf{x}_t \mid c) - \nabla\log p(\mathbf{x}_t)$, and amplifying this term by a guidance weight $w$ gives a modified score

$$\tilde\nabla = \nabla\log p(\mathbf{x}_t \mid c) + w\big(\nabla\log p(\mathbf{x}_t \mid c) - \nabla\log p(\mathbf{x}_t)\big).$$

Translating back to noise predictions, the *guided noise* used at sampling time is

$$\hat\epsilon = (1+w)\,\epsilon_\theta(\mathbf{x}_t, t, c) - w\,\epsilon_\theta(\mathbf{x}_t, t, \varnothing)$$

*(5)*

where $\varnothing$ is a special *null* condition standing for "unconditional". Both the conditional term and the unconditional term below come from the *same* network:

$$\epsilon_\theta(\cdot, c) \qquad \text{and} \qquad \epsilon_\theta(\cdot, \varnothing).$$

To make this possible, during training the condition is replaced by $\varnothing$ with a small probability $p_\text{uncond}$ (condition dropout), so the single network learns both models jointly.

> **Reading the guidance weight $w$.** $w=0$ recovers the plain conditional model. Increasing $w$ pushes samples toward regions where the requested attribute is more strongly expressed, improving *attribute fidelity* but eventually reducing diversity and introducing artifacts. The sweet spot is found empirically — and measuring it is exactly what the evaluation script does.

---

## 3. Dataset and preprocessing

The Cartoon Set (Google) provides 2D cartoon avatars, each described by 18 categorical attributes (e.g. `hair_color`, `eye_color`, `face_color`, `glasses`). Each image `csXXXX.png` is stored as RGBA (with a transparent background) alongside a `csXXXX.csv` file containing, one per line, `"attr_name", value_index, cardinality`. The project uses the 10k subset at $32\times32$ resolution as the starting point.

Preprocessing steps (all in `dataset.py`):

1. **Alpha compositing.** The transparent background is composited onto a solid white background; otherwise the network would waste capacity modelling meaningless noise around the figure.
2. **Resize** to $32\times32$ (bilinear). The flat, low-texture nature of cartoon art makes low resolution effective and cheap to train.
3. **Normalization** to $[-1,1]$ via $\mathbf{x} \mapsto \mathbf{x}/127.5 - 1$, matching the standard-normal noise scale of the diffusion process.
4. **Attribute parsing.** A configurable subset of attributes is used for conditioning; their cardinalities are *derived from the data* (the CSV third column) and validated for consistency, so nothing is hard-coded. The default conditioning set is the three unambiguous colour attributes `eye_color` (5), `hair_color` (10), `face_color` (11).

---

## 4. Implementation

The codebase is six modules, all device-agnostic (`cuda` if available else `cpu`) so identical code runs on a laptop CPU for debugging and on the desktop GPU for the real training run.

### 4.1 `dataset.py` — data pipeline

Implements `CartoonSetDataset`, which scans the folder for `.png` files, pairs each with its `.csv`, parses labels once, and stacks the selected attribute indices into an $(N, n_\text{attr})$ tensor. Two design points worth noting:

- **Preloading.** At $32\times32$, the 10k images occupy $\approx 123$ MB, so all images are decoded once into a single tensor at init, making training free of disk I/O. (`preload=False` for the 100k set.)
- **`attribute_dims`** exposes the list of cardinalities $[5, 10, 11]$, consumed by the model to size its embedding tables.

A `limit` argument restricts the number of images, enabling fast CPU debugging (`--limit 64`).

### 4.2 `model.py` — conditional U-Net

The network $\epsilon_\theta(\mathbf{x}_t, t, c)$ is a compact U-Net ($\approx 4.4$ M parameters at `base`=64) with the following ingredients.

- **Time embedding.** A sinusoidal embedding of $t$ passed through an MLP, giving a vector that is injected into every residual block (FiLM-style additive conditioning).
- **Attribute conditioning.** *One embedding table per attribute*, each of size $K_i + 1$: the extra row (index $K_i$) is the *null* token used for CFG. The per-attribute embeddings are summed and added to the time embedding. The whole condition is thus a single vector added alongside time information.
- **Residual blocks** with GroupNorm + SiLU, and a **self-attention** block at the $8\times8$ resolution to capture longer-range structure.
- **Skip connections.** A symmetric encoder/decoder with (down + 1) residual blocks per level on the way up, guaranteeing that every encoder skip is consumed at the matching resolution.

The helper `null_labels(B)` returns the all-null condition $[K_0, K_1, \dots]$, used to form the unconditional prediction in Eq. (5).

### 4.3 `diffusion.py` — the diffusion process

Implements `GaussianDiffusion` with a linear $\beta$ schedule ($\beta_1 = 10^{-4}$ to $\beta_T = 0.02$, $T = 1000$) and precomputed $\alpha_t, \bar\alpha_t, \sqrt{\bar\alpha_t}, \sqrt{1-\bar\alpha_t}$:

- `q_sample` — the closed-form forward step, Eq. (1).
- `p_losses` — the training loss, Eq. (3), *including CFG condition dropout*: with probability $p_\text{uncond} = 0.1$ the whole label vector is replaced by the null condition, so the network learns the unconditional model too.
- `ddpm_sample` — full $T$-step ancestral sampling, Eq. (4), with guidance. **This is the sampler used for all reported results (see Section 5.1).**
- `ddim_sample` — fast deterministic sampling (default 50 steps), with guidance. Kept in the codebase for reference and rapid iteration during development, but **not used for reported results** after the artifact described in Section 5.1 was identified.
- `_guided_eps` — combines conditional and unconditional predictions via Eq. (5); $w=0$ short-circuits to a single forward.

### 4.4 `train.py` — training loop

A standard loop (Adam, $\text{lr} = 2\times10^{-4}$) with two additions relevant to diffusion:

- **EMA** (exponential moving average, decay $0.999$) of the model weights. Diffusion samples are noticeably cleaner from EMA weights, so a shadow copy is maintained and saved alongside the raw weights.
- **Checkpointing.** Every `--save_every` steps the full state (raw + EMA weights, attribute dims, image size, timesteps) is saved. If `--ckpt` is omitted, a timestamped filename is generated so separate runs never collide; a one-level `.bak` rotation guards against a single accidental overwrite.

### 4.5 `sample.py` — qualitative grids

Loads a checkpoint (EMA weights) and produces an image grid whose rows are the values of a chosen attribute and whose columns are guidance weights, with the other attributes held fixed. The *same starting noise* is reused per row across columns, so only $w$ changes and the effect of guidance is visually isolated — the figure for the presentation. **Grids used in this report are generated with full DDPM sampling (Section 5.1); the script also supports DDIM for quick low-cost previews during development, but those are not used for any reported figure.**

### 4.6 `evaluate.py` — quantitative evaluation

Two components:

- **Attribute classifier.** A small multi-head CNN (shared conv backbone, one linear head per attribute) trained on the *real* images. This is the measuring instrument, independent of the generator, and cached to disk so it is trained only once.
- **Attribute fidelity vs. $w$.** For each guidance weight, the generator produces images from random attribute vectors; the classifier then predicts each attribute, and fidelity is the fraction of generated images whose attribute matches the one requested. Sweeping $w$ produces the central quantitative result.

Both `evaluate.py` and `evaluate100k.py` expose a `--sampler {ddim,ddpm}` flag. **`ddpm` is the setting used for every reported number**; `ddim` is retained only to reproduce the sampler-artifact comparison in Section 5.1.

---

## 5. Evaluation methodology

The core claim to validate is that classifier-free guidance improves how faithfully generated images honour the requested attributes. The experiment:

1. Train the generator (`train.py`) and the attribute classifier (`evaluate.py`).
2. For $w \in \{0, 1, 3, 5\}$, generate a fixed number of images from random attribute vectors and classify them.
3. Report per-attribute and mean fidelity as a function of $w$.

This is complemented by the qualitative grids from `sample.py`. FID is intentionally left out (it requires InceptionV3 weights).

### 5.1 Sampler choice materially changes the conclusion: DDIM was abandoned

The fidelity sweep was first run with DDIM (50 steps) for speed, giving the following result on the 10k-image, 40k-step checkpoint:

| $w$ | eye_color | hair_color | face_color | mean |
|-----|-----------|------------|------------|------|
| 0   | 0.654     | 0.930      | 0.693      | 0.759 |
| 1   | 0.730     | 0.791      | 0.635      | 0.719 |
| 3   | 0.490     | 0.623      | 0.260      | 0.458 |
| 5   | 0.441     | 0.650      | 0.191      | 0.428 |

Read at face value, this says guidance *hurts* fidelity beyond $w\approx1$ — the opposite of what CFG is supposed to do, and inconsistent with the qualitative grids at low $w$, which look clean and on-attribute. Raising the DDIM step count to 250 did not fix it (checked interactively in `generate_interactive.ipynb`).

Repeating the identical sweep with full ancestral DDPM sampling (same checkpoint, same classifier) gives a very different picture:

| $w$ | eye_color | hair_color | face_color | mean |
|-----|-----------|------------|------------|------|
| 0   | 0.672     | 0.938      | 0.750      | 0.786 |
| 1   | 0.703     | 0.992      | 0.781      | 0.826 |
| 3   | 0.813     | 1.000      | 0.758      | 0.857 |
| 5   | 0.758     | 1.000      | 0.773      | 0.844 |

With DDPM, mean fidelity *rises* with $w$ and plateaus around 0.84–0.86 — the textbook CFG pattern — and stays there at $w=5$ instead of collapsing. `face_color` is the attribute most affected by the discrepancy (0.69→0.19 under DDIM vs. a stable 0.75–0.78 under DDPM), consistent with it being a spatially global, easily-saturated colour attribute.

**Diagnosis.** The two samplers use the *same* trained $\epsilon_\theta$ and the *same* guided noise $\hat\epsilon$ (Eq. 5) — the discrepancy is purely a property of *how* that noise is used to step through the chain. DDIM reconstructs $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$ explicitly at every step; at high $t$ this coefficient is large (e.g. $\approx 157$ at $t=999$ for this schedule), so the extrapolation error introduced by the guided (and therefore off-manifold) $\hat\epsilon$ is amplified and clipped, injecting high-frequency artifacts that compound over the sub-sampled trajectory. DDPM's ancestral update never forms $\hat{\mathbf{x}}_0$ explicitly and takes 1000 small steps instead of 50–250 larger ones, so the same guided noise is applied far more gradually and does not blow up the same way.

**Decision.** DDIM is abandoned as the sampler for all reported results in this project. `ddim_sample` remains in `diffusion.py` purely for documentation and as a fast (but qualitatively unreliable under guidance) preview tool during development; every number and figure in this report is produced with `ddpm_sample`. Both `evaluate.py` and `evaluate100k.py` expose `--sampler {ddim,ddpm}` precisely to make this choice explicit and reproducible.

### 5.2 Result table (DDPM, 10k, 40k steps)

| $w$ | eye_color | hair_color | face_color | mean |
|-----|-----------|------------|------------|------|
| 0   | 0.672     | 0.938      | 0.750      | 0.786 |
| 1   | 0.703     | 0.992      | 0.781      | 0.826 |
| 3   | 0.813     | 1.000      | 0.758      | 0.857 |
| 5   | 0.758     | 1.000      | 0.773      | 0.844 |

`hair_color` saturates at $w\geq3$ (ceiling effect: the classifier is essentially always right once guidance is on), while `eye_color` and `face_color` continue to benefit from moderate guidance and plateau rather than degrade, at least up to $w=5$. Whether fidelity eventually degrades at higher $w$ (as diversity collapses) under DDPM is an open question addressed in Section 6.

---

## 6. Current status and next steps

All six modules are written and validated end-to-end, and the full 40k-step training run on the 10k-image dataset is complete on the desktop GPU (GTX 1080, 8 GB). The guidance-weight sweep has been run to completion with both samplers, revealing the DDIM sampling artifact documented in Section 5.1; all further evaluation uses DDPM.

Remaining and follow-up work:

- **Higher-$w$ sweep under DDPM** (e.g. $w \in \{8, 10, 15\}$) to check whether fidelity eventually degrades once guidance is pushed far enough, or whether the plateau observed at $w=3$–$5$ (Section 5.2) simply continues — this distinguishes "CFG has a genuine ceiling on this dataset" from "the ceiling in the DDIM evaluation was entirely a sampler artifact."
- **Perceptual/diversity check.** Fidelity alone cannot detect mode collapse: a model could satisfy the requested attribute by generating near-identical images. A simple pairwise diversity metric (e.g. average LPIPS or pixel-distance between same-condition samples) across $w$ would confirm whether the DDPM plateau also preserves within-condition variety, or whether high $w$ trades diversity for fidelity even without the DDIM artifact.
- **100k-dataset run** (`train100k.py`, `evaluate100k.py`) as a comparison baseline against the 10k result, now that the sampler choice is fixed to DDPM.
- **Project presentation** covering the full pipeline and, in particular, the DDIM-vs-DDPM finding as a methodological result in its own right.
