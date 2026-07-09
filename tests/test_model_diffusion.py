import torch

from cartoon_diffusion.diffusion import GaussianDiffusion
from cartoon_diffusion.model import UNet


def _small_model():
    return UNet(
        [2, 3],
        base=8,
        ch_mult=(1,),
        num_res_blocks=1,
        emb_dim=16,
        attn_at=(),
        image_size=8,
        groups=4,
    )


def test_unet_forward_and_null_labels():
    model = _small_model()
    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0, 1])
    labels = torch.tensor([[0, 1], [1, 2]])

    out = model(x, t, labels)
    assert out.shape == x.shape
    assert model.null_labels(2, x.device).tolist() == [[2, 3], [2, 3]]


def test_diffusion_sampling_smoke():
    model = _small_model()
    diff = GaussianDiffusion(timesteps=4)
    x0 = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0, 3])
    noise = torch.randn_like(x0)
    labels = torch.tensor([[0, 1], [1, 2]])

    assert diff.q_sample(x0, t, noise).shape == x0.shape
    assert diff.p_losses(model, x0, labels).ndim == 0
    assert diff.ddpm_sample(model, labels, image_size=8, guidance_weight=0).shape == x0.shape
    assert diff.ddim_sample(model, labels, image_size=8, guidance_weight=0, steps=2).shape == x0.shape
