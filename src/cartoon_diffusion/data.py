"""Cartoon Set data pipeline used by both the 10k and 100k experiments."""

from __future__ import annotations

import csv
import hashlib
import os
from glob import glob
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CACHE_VERSION = 1
DEFAULT_ATTRS = ("eye_color", "hair_color", "face_color")


def _parse_label_csv(path: str | os.PathLike[str]) -> dict[str, tuple[int, int]]:
    """Return {attr_name: (value_index, cardinality)} for one image CSV."""
    out: dict[str, tuple[int, int]] = {}
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
    """Dataset for flat 10k and nested 100k Google Cartoon Set folders.

    Images are RGBA PNGs paired with CSV files. They are alpha-composited onto a
    solid background, resized, and returned as float tensors in [-1, 1].
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        image_size: int = 32,
        cond_attributes: tuple[str, ...] = DEFAULT_ATTRS,
        bg_color: tuple[int, int, int] = (255, 255, 255),
        preload: bool = True,
        limit: int | None = None,
        cache: bool = False,
        cache_dir: str | os.PathLike[str] | None = None,
        rebuild_cache: bool = False,
        recursive: bool = True,
        verbose: bool = True,
    ):
        self.root = str(root)
        self.image_size = image_size
        self.cond_attributes = tuple(cond_attributes)
        self.bg_color = tuple(bg_color)
        self.verbose = verbose
        self.cache_dir = str(cache_dir or Path(self.root) / ".cache")

        pattern = "**/*.png" if recursive else "*.png"
        pngs = sorted(glob(str(Path(self.root) / pattern), recursive=recursive))
        if limit is not None:
            pngs = pngs[:limit]
        if not pngs:
            raise FileNotFoundError(f"No .png files found under {self.root!r}.")

        use_cache = cache and limit is None
        images = labels_all = None
        all_attr_names = all_cards = None
        self._rel_paths: list[str] = []

        if use_cache and not rebuild_cache:
            loaded = self._try_load_cache(len(pngs))
            if loaded is not None:
                images, labels_all, all_attr_names, all_cards, self._rel_paths = loaded

        if images is None:
            paths, labels_raw = self._discover_pairs(pngs)
            all_attr_names = list(labels_raw[0].keys())
            all_cards = [labels_raw[0][name][1] for name in all_attr_names]
            labels_all = torch.tensor(
                [[lr[name][0] for name in all_attr_names] for lr in labels_raw],
                dtype=torch.int16,
            )
            self._rel_paths = [os.path.relpath(p, self.root) for p in paths]
            if preload or use_cache:
                if self.verbose:
                    print(
                        f"[dataset] preprocessing {len(paths)} images "
                        f"@ {image_size}px"
                    )
                images = self._preload_images_uint8(paths)
            if use_cache and images is not None:
                self._save_cache(images, labels_all, all_attr_names, all_cards)

        name_to_col = {name: i for i, name in enumerate(all_attr_names)}
        cols = []
        self.cardinalities: dict[str, int] = {}
        for name in self.cond_attributes:
            if name not in name_to_col:
                raise KeyError(
                    f"Attribute {name!r} not found. Available: {all_attr_names}"
                )
            col = name_to_col[name]
            cols.append(col)
            self.cardinalities[name] = int(all_cards[col])

        self.labels = labels_all[:, cols].long()
        self.images = images
        self._all_attr_names = all_attr_names
        self.paths = [str(Path(self.root) / rp) for rp in self._rel_paths]

    def _discover_pairs(self, pngs: list[str]):
        paths, labels_raw = [], []
        for png in pngs:
            csv_path = png[:-4] + ".csv"
            if not os.path.exists(csv_path):
                continue
            paths.append(png)
            labels_raw.append(_parse_label_csv(csv_path))
        if not paths:
            raise FileNotFoundError(f"Found .png but no matching .csv under {self.root!r}.")
        return paths, labels_raw

    def _cache_path(self) -> str:
        key = "|".join(
            [
                os.path.abspath(self.root),
                str(self.image_size),
                str(self.bg_color),
                str(CACHE_VERSION),
            ]
        )
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return str(Path(self.cache_dir) / f"cartoon_{self.image_size}px_{h}.pt")

    def _try_load_cache(self, n_expected: int):
        path = self._cache_path()
        if not os.path.exists(path):
            return None
        try:
            blob = torch.load(path, map_location="cpu")
            if blob.get("version") != CACHE_VERSION:
                return None
            images = blob["images"]
            if images.shape[0] != n_expected:
                if self.verbose:
                    print(
                        f"[dataset] cache count {images.shape[0]} != "
                        f"{n_expected}; rebuilding."
                    )
                return None
            if self.verbose:
                print(f"[dataset] loaded cache {path} ({images.shape[0]} images)")
            return (
                images,
                blob["labels_all"],
                blob["all_attr_names"],
                blob["all_cards"],
                blob["rel_paths"],
            )
        except Exception as exc:  # noqa: BLE001 - corrupt cache should rebuild
            if self.verbose:
                print(f"[dataset] cache unreadable ({exc}); rebuilding.")
            return None

    def _save_cache(self, images, labels_all, all_attr_names, all_cards):
        path = self._cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".tmp.{os.getpid()}"
        torch.save(
            {
                "version": CACHE_VERSION,
                "image_size": self.image_size,
                "bg_color": self.bg_color,
                "images": images,
                "labels_all": labels_all,
                "all_attr_names": all_attr_names,
                "all_cards": all_cards,
                "rel_paths": self._rel_paths,
            },
            tmp,
        )
        os.replace(tmp, path)
        if self.verbose:
            mb = images.numel() / 1e6
            print(f"[dataset] wrote cache {path} (~{mb:.0f} MB)")

    def _load_image_uint8(self, path: str | os.PathLike[str]) -> torch.Tensor:
        """RGBA png -> uint8 RGB tensor (3,H,W), alpha composited on bg_color."""
        img = Image.open(path).convert("RGBA")
        bg = Image.new("RGBA", img.size, self.bg_color + (255,))
        img = Image.alpha_composite(bg, img).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        return t.view(self.image_size, self.image_size, 3).permute(2, 0, 1)

    def _preload_images_uint8(self, paths: list[str]) -> torch.Tensor:
        out = torch.empty(
            (len(paths), 3, self.image_size, self.image_size), dtype=torch.uint8
        )
        report_every = max(1, len(paths) // 20)
        for i, path in enumerate(paths):
            out[i] = self._load_image_uint8(path)
            if self.verbose and (i + 1) % report_every == 0 and len(paths) >= 100:
                print(f"    {i + 1}/{len(paths)}")
        return out

    @staticmethod
    def _to_float(u8: torch.Tensor) -> torch.Tensor:
        return u8.float() / 127.5 - 1.0

    @property
    def attribute_dims(self) -> list[int]:
        return [self.cardinalities[a] for a in self.cond_attributes]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int):
        if self.images is not None:
            img = self._to_float(self.images[i])
        else:
            img = self._to_float(self._load_image_uint8(self.paths[i]))
        return img, self.labels[i]


def build_dataloader(dataset, batch_size=128, num_workers=0, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] tensor -> [0,1] for visualisation / saving."""
    return (x.clamp(-1, 1) + 1) / 2
