import torch
import pytest

from cartoon_diffusion.diffusion import GaussianDiffusion
from cartoon_diffusion.model import UNet
from cartoon_diffusion.training import (
    amp_dtype_for,
    learning_rate_for_step,
    normalize_precision,
    validate_grad_clip,
)


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


def test_training_precision_validation():
    assert normalize_precision("fp32") == "fp32"
    assert normalize_precision("float16") == "fp16"
    assert normalize_precision("bfloat16") == "bf16"
    assert amp_dtype_for("fp32", "cpu") == ("fp32", False, None)
    with pytest.raises(ValueError):
        amp_dtype_for("fp16", "cpu")


def test_grad_clip_validation():
    assert validate_grad_clip(None) is None
    assert validate_grad_clip(1) == 1.0
    with pytest.raises(ValueError):
        validate_grad_clip(0)


def test_learning_rate_schedules():
    assert learning_rate_for_step(
        base_lr=2e-4,
        min_lr=0.0,
        schedule="constant",
        step=10,
        total_steps=100,
    ) == 2e-4
    assert learning_rate_for_step(
        base_lr=2e-4,
        min_lr=1e-5,
        schedule="cosine",
        step=1,
        total_steps=100,
    ) == 2e-4
    assert learning_rate_for_step(
        base_lr=2e-4,
        min_lr=1e-5,
        schedule="cosine",
        step=100,
        total_steps=100,
    ) == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        learning_rate_for_step(
            base_lr=2e-4,
            min_lr=0.0,
            schedule="linear",
            step=1,
            total_steps=100,
        )
