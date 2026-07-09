"""
Cartoon Set data pipeline for a conditional DDPM.

Expects the extracted Cartoon Set (10k or 100k) as a flat folder of pairs:
    csXXXX.png   (RGBA, transparent background)
    csXXXX.csv   (18 rows: "attr_name", "value_index", "cardinality")

Usage:
    from dataset import CartoonSetDataset, build_dataloader, denormalize

    ds = CartoonSetDataset("data/cartoonset10k",
                           image_size=32,
                           cond_attributes=("eye_color", "hair_color", "face_color"))
    loader = build_dataloader(ds, batch_size=128, num_workers=4)
    imgs, labels = next(iter(loader))       # imgs: (B,3,32,32) in [-1,1]
                                            # labels: (B, n_attr) long
"""

import csv
import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _parse_label_csv(path):
    """Return {attr_name: (value_index, cardinality)} for one image's csv."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            name = row[0].strip().strip('"').strip()
            value = int(row[1].strip().strip('"'))
            card = int(row[2].strip().strip('"'))
            out[name] = (value, card)
    return out


class CartoonSetDataset(Dataset):
    def __init__(
        self,
        root,
        image_size=32,
        cond_attributes=("eye_color", "hair_color", "face_color"),
        bg_color=(255, 255, 255),
        preload=True,
        limit=None,
    ):
        self.root = root
        self.image_size = image_size
        self.cond_attributes = tuple(cond_attributes)
        self.bg_color = bg_color

        pngs = sorted(glob(os.path.join(root, "*.png")))
        if limit is not None:
            pngs = pngs[:limit]
        if not pngs:
            raise FileNotFoundError(f"No .png files found in {root!r}")

        # Pair each png with its csv and parse labels once.
        self.paths, labels_raw = [], []
        for p in pngs:
            c = p[:-4] + ".csv"
            if not os.path.exists(c):
                continue
            self.paths.append(p)
            labels_raw.append(_parse_label_csv(c))
        if not self.paths:
            raise FileNotFoundError(f"Found .png but no matching .csv in {root!r}")

        # Cardinalities derived from the data (third csv column), validated
        # for consistency, so nothing needs to be hard-coded.
        self.cardinalities = {}
        for name in self.cond_attributes:
            cards = {lr[name][1] for lr in labels_raw if name in lr}
            if not cards:
                raise KeyError(
                    f"Attribute {name!r} not found. Available: "
                    f"{sorted(labels_raw[0].keys())}"
                )
            if len(cards) > 1:
                raise ValueError(f"Inconsistent cardinality for {name!r}: {cards}")
            self.cardinalities[name] = cards.pop()

        # Stack the selected attribute indices into a (N, n_attr) long tensor.
        self.labels = torch.tensor(
            [[lr[a][0] for a in self.cond_attributes] for lr in labels_raw],
            dtype=torch.long,
        )

        self.images = self._preload_images() if preload else None

    # -- image loading -----------------------------------------------------
    def _load_image(self, path):
        """RGBA png -> RGB tensor in [-1, 1], alpha composited on bg_color."""
        img = Image.open(path).convert("RGBA")
        bg = Image.new("RGBA", img.size, self.bg_color + (255,))
        img = Image.alpha_composite(bg, img).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float()
        t = t.view(self.image_size, self.image_size, 3).permute(2, 0, 1)
        return t / 127.5 - 1.0  # [0,255] -> [-1,1]

    def _preload_images(self):
        # 10k * 3 * 32 * 32 * 4 bytes ~= 123 MB, trivially fits in RAM and
        # makes training IO-free.
        return torch.stack([self._load_image(p) for p in self.paths])

    # -- Dataset API -------------------------------------------------------
    @property
    def attribute_dims(self):
        """List of cardinalities in cond_attributes order (for the model's
        embedding tables)."""
        return [self.cardinalities[a] for a in self.cond_attributes]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.images[i] if self.images is not None else self._load_image(self.paths[i])
        return img, self.labels[i]


def build_dataloader(dataset, batch_size=128, num_workers=4, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def denormalize(x):
    """[-1,1] tensor -> [0,1] for visualisation / saving."""
    return (x.clamp(-1, 1) + 1) / 2


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/cartoonset10k"
    ds = CartoonSetDataset(root, image_size=32)
    print(f"dataset size      : {len(ds)}")
    print(f"cond attributes   : {ds.cond_attributes}")
    print(f"attribute dims    : {ds.attribute_dims}")
    img, lab = ds[0]
    print(f"image tensor      : {tuple(img.shape)}  "
          f"range [{img.min():.2f}, {img.max():.2f}]")
    print(f"label vector      : {lab.tolist()}")
    loader = build_dataloader(ds, batch_size=8, num_workers=0)
    xb, yb = next(iter(loader))
    print(f"batch images      : {tuple(xb.shape)}")
    print(f"batch labels      : {tuple(yb.shape)}")
