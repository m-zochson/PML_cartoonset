from pathlib import Path

import torch
import pytest
from PIL import Image

from cartoon_diffusion.upscaling import (
    PixelShuffleUpscaler,
    UpscalePairsDataset,
    UpscalerConfig,
    load_upscaler_checkpoint,
    model_from_checkpoint,
    train_upscaler,
    validate_block_count,
    validate_kernel_size,
    validate_l2_weight,
)


def _write_cartoon(root: Path, name: str):
    img = Image.new("RGBA", (12, 12), (20, 80, 140, 180))
    img.save(root / f"{name}.png")
    (root / f"{name}.csv").write_text('"eye_color",1,5\n')


def test_upscaler_forward_and_parameter_count():
    model = PixelShuffleUpscaler(scale_factor=3, channels=64, residual_blocks=4)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters())

    assert out.shape == (2, 3, 96, 96)
    assert 780_000 <= n_params <= 790_000


def test_upscale_pairs_dataset_builds_32_to_96_pairs(tmp_path):
    _write_cartoon(tmp_path, "cs0000")
    _write_cartoon(tmp_path, "cs0001")

    ds = UpscalePairsDataset(tmp_path)
    low, target = ds[0]

    assert len(ds) == 2
    assert low.shape == (3, 32, 32)
    assert target.shape == (3, 96, 96)
    assert low.min() >= -1 and low.max() <= 1
    assert target.min() >= -1 and target.max() <= 1


def test_l2_weight_validation():
    assert validate_l2_weight(0) == 0.0
    assert validate_l2_weight(1) == 1.0
    with pytest.raises(ValueError):
        validate_l2_weight(-0.1)


def test_architecture_validation():
    assert validate_kernel_size(5) == 5
    assert validate_block_count(0, "refinement_blocks") == 0
    with pytest.raises(ValueError):
        validate_kernel_size(2)
    with pytest.raises(ValueError):
        validate_block_count(-1, "refinement_blocks")


def test_upscaler_tiny_training_and_checkpoint_load(tmp_path):
    for i in range(4):
        _write_cartoon(tmp_path, f"cs{i:04d}")
    ckpt = tmp_path / "upscaler.pt"
    config = UpscalerConfig(
        root=str(tmp_path),
        channels=8,
        residual_blocks=1,
        refinement_blocks=1,
        head_kernel=5,
        tail_kernel=5,
        batch=2,
        steps=1,
        lr_schedule="cosine",
        min_lr=1e-5,
        l2_weight=0.5,
        precision="fp32",
        workers=0,
        ckpt=str(ckpt),
        save_every=1,
    )

    train_upscaler(config)
    loaded = load_upscaler_checkpoint(ckpt, map_location="cpu")
    model = model_from_checkpoint(loaded)
    model.load_state_dict(loaded["model"])

    assert loaded["step"] == 1
    assert loaded["lr_schedule"] == "cosine"
    assert loaded["min_lr"] == 1e-5
    assert loaded["l2_weight"] == 0.5
    assert loaded["refinement_blocks"] == 1
    assert loaded["head_kernel"] == 5
    assert loaded["tail_kernel"] == 5
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 3, 96, 96)
