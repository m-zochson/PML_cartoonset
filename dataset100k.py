"""
Cartoon Set data pipeline for the 100k set (also works on the 10k set).

Differences vs dataset.py, for the larger set:

  1. RECURSIVE glob. The 100k download extracts into ten sub-folders
     (`0/ .. 9/`), each with 10k paired files, instead of one flat folder.
     We glob `root/**/*.png` so both layouts (flat 10k, nested 100k) work.

  2. DISK CACHE of the preprocessed tensor. Decoding + compositing + resizing
     100k 500x500 PNGs takes minutes and, if done lazily per batch, makes the
     4-core CPU the bottleneck instead of the GPU. So on the first run we
     preprocess once and dump a compact uint8 tensor (+ all labels) to a
     `.pt` cache; every later run (a resumed training, evaluate100k, ...)
     reloads it in a couple of seconds. Images are stored as uint8
     (~307 MB for 100k @ 32px, a quarter of the float size) and converted to
     [-1,1] float only when a sample is requested.

     The cache is written atomically (temp file + os.replace) so a run that
     is killed mid-write never leaves a corrupt cache behind.

Usage is a drop-in replacement for dataset.py:

    from dataset100k import CartoonSetDataset, build_dataloader, denormalize
    ds = CartoonSetDataset("data/cartoonset100k", image_size=32)
"""

import csv
import hashlib
import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CACHE_VERSION = 1  # bump if the preprocessing (compositing/resize/layout) changes


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
        cache=True,
        cache_dir=None,
        rebuild_cache=False,
        verbose=True,
    ):
        self.root = root
        self.image_size = image_size
        self.cond_attributes = tuple(cond_attributes)
        self.bg_color = tuple(bg_color)
        self.verbose = verbose

        # ---- discover files (recursive: flat 10k AND nested 100k) ---------
        pngs = sorted(glob(os.path.join(root, "**", "*.png"), recursive=True))
        if limit is not None:
            pngs = pngs[:limit]
        if not pngs:
            raise FileNotFoundError(
                f"No .png files found under {root!r} (searched recursively)."
            )

        # A cache is only meaningful for the full set; --limit is for debugging
        # and would otherwise write a truncated cache under the same key.
        self.cache_dir = cache_dir or os.path.join(root, ".cache")
        use_cache = cache and (limit is None)

        images = labels_all = None
        all_attr_names = all_cards = None

        if use_cache and not rebuild_cache:
            loaded = self._try_load_cache(len(pngs))
            if loaded is not None:
                (images, labels_all, all_attr_names, all_cards,
                 self._rel_paths) = loaded

        if images is None:
            # ---- parse every csv once (all attributes, not just cond ones) -
            paths, labels_raw = [], []
            for p in pngs:
                c = p[:-4] + ".csv"
                if not os.path.exists(c):
                    continue
                paths.append(p)
                labels_raw.append(_parse_label_csv(c))
            if not paths:
                raise FileNotFoundError(
                    f"Found .png but no matching .csv under {root!r}."
                )

            # Stable ordered list of ALL attribute names + their cardinalities,
            # taken from the first csv (schema is identical across the set).
            all_attr_names = list(labels_raw[0].keys())
            all_cards = [labels_raw[0][n][1] for n in all_attr_names]
            # (N, n_all) int16 value matrix, cond-attribute-independent.
            labels_all = torch.tensor(
                [[lr[n][0] for n in all_attr_names] for lr in labels_raw],
                dtype=torch.int16,
            )

            if preload or use_cache:
                if self.verbose:
                    print(f"[dataset100k] preprocessing {len(paths)} images "
                          f"@ {image_size}px (first run, this is the slow part)...")
                images = self._preload_images_uint8(paths)
            self._rel_paths = [os.path.relpath(p, root) for p in paths]

            if use_cache and images is not None:
                self._save_cache(images, labels_all, all_attr_names, all_cards,
                                 self._rel_paths)

        # ---- select the conditioning attributes ---------------------------
        name_to_col = {n: i for i, n in enumerate(all_attr_names)}
        self.cardinalities = {}
        cols = []
        for name in self.cond_attributes:
            if name not in name_to_col:
                raise KeyError(
                    f"Attribute {name!r} not found. Available: {all_attr_names}"
                )
            cols.append(name_to_col[name])
            self.cardinalities[name] = all_cards[name_to_col[name]]

        self.labels = labels_all[:, cols].long()
        self.images = images  # uint8 (N,3,H,W) or None (lazy)
        self._all_attr_names = all_attr_names

        # Lazy fallback: keep absolute paths if images weren't preloaded.
        if self.images is None:
            self.paths = [os.path.join(root, rp) for rp in self._rel_paths]

    # -- cache -------------------------------------------------------------
    def _cache_path(self):
        key = "|".join([
            os.path.abspath(self.root), str(self.image_size),
            str(self.bg_color), str(CACHE_VERSION),
        ])
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(
            self.cache_dir, f"cartoon_{self.image_size}px_{h}.pt"
        )

    def _try_load_cache(self, n_expected):
        path = self._cache_path()
        if not os.path.exists(path):
            return None
        try:
            blob = torch.load(path, map_location="cpu")
            if blob.get("version") != CACHE_VERSION:
                return None
            imgs = blob["images"]
            if imgs.shape[0] != n_expected:
                # File count changed on disk -> cache is stale, rebuild.
                if self.verbose:
                    print(f"[dataset100k] cache count {imgs.shape[0]} != "
                          f"{n_expected} on disk; rebuilding.")
                return None
            if self.verbose:
                print(f"[dataset100k] loaded cache {path} "
                      f"({imgs.shape[0]} images)")
            return (imgs, blob["labels_all"], blob["all_attr_names"],
                    blob["all_cards"], blob["rel_paths"])
        except Exception as e:  # noqa: BLE001 - any corruption -> rebuild
            if self.verbose:
                print(f"[dataset100k] cache unreadable ({e}); rebuilding.")
            return None

    def _save_cache(self, images, labels_all, all_attr_names, all_cards,
                    rel_paths):
        path = self._cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".tmp.{os.getpid()}"
        torch.save(
            {
                "version": CACHE_VERSION,
                "image_size": self.image_size,
                "bg_color": self.bg_color,
                "images": images,             # uint8 (N,3,H,W)
                "labels_all": labels_all,     # int16 (N, n_all)
                "all_attr_names": all_attr_names,
                "all_cards": all_cards,
                "rel_paths": rel_paths,
            },
            tmp,
        )
        os.replace(tmp, path)  # atomic: no half-written cache is ever visible
        if self.verbose:
            mb = images.numel() / 1e6
            print(f"[dataset100k] wrote cache {path} (~{mb:.0f} MB)")

    # -- image loading -----------------------------------------------------
    def _load_image_uint8(self, path):
        """RGBA png -> uint8 RGB (3,H,W), alpha composited on bg_color."""
        img = Image.open(path).convert("RGBA")
        bg = Image.new("RGBA", img.size, self.bg_color + (255,))
        img = Image.alpha_composite(bg, img).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        return t.view(self.image_size, self.image_size, 3).permute(2, 0, 1)

    def _preload_images_uint8(self, paths):
        out = torch.empty(
            (len(paths), 3, self.image_size, self.image_size), dtype=torch.uint8
        )
        report_every = max(1, len(paths) // 20)
        for i, p in enumerate(paths):
            out[i] = self._load_image_uint8(p)
            if self.verbose and (i + 1) % report_every == 0:
                print(f"    {i + 1}/{len(paths)}")
        return out

    @staticmethod
    def _to_float(u8):
        """uint8 (3,H,W) in [0,255] -> float in [-1,1]."""
        return u8.float() / 127.5 - 1.0

    # -- Dataset API -------------------------------------------------------
    @property
    def attribute_dims(self):
        return [self.cardinalities[a] for a in self.cond_attributes]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        if self.images is not None:
            img = self._to_float(self.images[i])
        else:
            img = self._to_float(self._load_image_uint8(self.paths[i]))
        return img, self.labels[i]


def build_dataloader(dataset, batch_size=128, num_workers=0, shuffle=True):
    # With the dataset fully preloaded in RAM there is no I/O to parallelise,
    # so num_workers=0 is the fast default here (and avoids Windows' expensive
    # per-worker pickling of the in-memory image tensor). Raise it only for the
    # lazy (preload=False) path.
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
    import time

    root = sys.argv[1] if len(sys.argv) > 1 else "data/cartoonset100k"
    t0 = time.time()
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
    print(f"init took         : {time.time() - t0:.1f}s "
          f"(re-run to see the cache hit)")
