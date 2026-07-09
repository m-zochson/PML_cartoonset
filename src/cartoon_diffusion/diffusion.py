"""
Gaussian diffusion process (DDPM) with classifier-free guidance.

    diff = GaussianDiffusion(timesteps=1000)
    loss = diff.p_losses(model, x0, labels, p_uncond=0.1)   # training
    imgs = diff.ddim_sample(model, labels, guidance_weight=3.0, steps=50)
"""

import torch
import torch.nn.functional as F


def _gather(a, t, shape):
    """Gather 1-D schedule values at timesteps t and reshape to broadcast."""
    out = a.to(t.device).gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_ab = torch.sqrt(alpha_bars)
        self.sqrt_1m_ab = torch.sqrt(1.0 - alpha_bars)

    # ---- forward process ------------------------------------------------
    def q_sample(self, x0, t, noise):
        return _gather(self.sqrt_ab, t, x0.shape) * x0 + \
               _gather(self.sqrt_1m_ab, t, x0.shape) * noise

    def p_losses(self, model, x0, labels, p_uncond=0.1):
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.timesteps, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # classifier-free guidance: drop the whole condition to null w.p. p_uncond
        labels = labels.clone()
        drop = torch.rand(B, device=device) < p_uncond
        if drop.any():
            null = model.null_labels(B, device)
            labels[drop] = null[drop]

        pred = model(x_t, t, labels)
        return F.mse_loss(pred, noise)

    # ---- guided noise prediction ---------------------------------------
    def _guided_eps(self, model, x_t, t, labels, guidance_weight):
        if guidance_weight == 0:
            return model(x_t, t, labels)
        B = x_t.shape[0]
        null = model.null_labels(B, x_t.device)
        # one batched forward for cond + uncond
        eps_c = model(x_t, t, labels)
        eps_u = model(x_t, t, null)
        return (1 + guidance_weight) * eps_c - guidance_weight * eps_u

    # ---- DDPM ancestral sampling ---------------------------------------
    @torch.no_grad()
    def ddpm_sample(self, model, labels, image_size=32, guidance_weight=3.0):
        model.eval()
        device = next(model.parameters()).device
        B = labels.shape[0]
        x = torch.randn(B, 3, image_size, image_size, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps = self._guided_eps(model, x, t, labels, guidance_weight)
            beta = self.betas[i]
            ab = self.alpha_bars[i]
            alpha = self.alphas[i]
            mean = (x - beta / torch.sqrt(1 - ab) * eps) / torch.sqrt(alpha)
            if i > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)

    # ---- DDIM (fast, deterministic) ------------------------------------
    @torch.no_grad()
    def ddim_sample(self, model, labels, image_size=32, guidance_weight=3.0,
                    steps=50, eta=0.0):
        model.eval()
        device = next(model.parameters()).device
        B = labels.shape[0]
        x = torch.randn(B, 3, image_size, image_size, device=device)

        seq = torch.linspace(0, self.timesteps - 1, steps, dtype=torch.long)
        seq = list(seq.tolist())
        for j in reversed(range(len(seq))):
            i = seq[j]
            i_prev = seq[j - 1] if j > 0 else -1
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps = self._guided_eps(model, x, t, labels, guidance_weight)

            ab = self.alpha_bars[i]
            ab_prev = self.alpha_bars[i_prev] if i_prev >= 0 else torch.tensor(1.0)
            x0 = ((x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)).clamp(-1, 1)

            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev))
            dir_xt = torch.sqrt(1 - ab_prev - sigma ** 2) * eps
            x = torch.sqrt(ab_prev) * x0 + dir_xt
            if eta > 0 and i_prev >= 0:
                x = x + sigma * torch.randn_like(x)
        return x.clamp(-1, 1)
