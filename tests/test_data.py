from pathlib import Path

import torch
from PIL import Image

from cartoon_diffusion.data import CartoonSetDataset


def _write_cartoon(root: Path, name: str, values=(1, 2, 3)):
    img = Image.new("RGBA", (4, 4), (10, 20, 30, 128))
    img.save(root / f"{name}.png")
    rows = [
        f'"eye_color",{values[0]},5\n',
        f'"hair_color",{values[1]},10\n',
        f'"face_color",{values[2]},11\n',
    ]
    (root / f"{name}.csv").write_text("".join(rows))


def test_cartoon_dataset_parses_labels_and_images(tmp_path):
    _write_cartoon(tmp_path, "cs0000")
    _write_cartoon(tmp_path, "cs0001", values=(2, 3, 4))

    ds = CartoonSetDataset(tmp_path, image_size=8, cache=False, verbose=False)

    assert len(ds) == 2
    assert ds.attribute_dims == [5, 10, 11]
    img, label = ds[0]
    assert img.shape == (3, 8, 8)
    assert img.min() >= -1 and img.max() <= 1
    assert label.tolist() == [1, 2, 3]


def test_cartoon_dataset_cache_and_limit(tmp_path):
    _write_cartoon(tmp_path, "cs0000")
    _write_cartoon(tmp_path, "cs0001", values=(2, 3, 4))

    ds = CartoonSetDataset(tmp_path, image_size=8, cache=True, rebuild_cache=True, verbose=False)
    cache_files = list((tmp_path / ".cache").glob("*.pt"))
    assert len(ds) == 2
    assert cache_files
    assert ds.images.dtype == torch.uint8

    limited = CartoonSetDataset(tmp_path, image_size=8, limit=1, cache=True, verbose=False)
    assert len(limited) == 1
