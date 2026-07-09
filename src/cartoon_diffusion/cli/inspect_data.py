"""Inspect a Cartoon Set directory."""

from __future__ import annotations

import argparse
import time

from cartoon_diffusion.data import CartoonSetDataset, build_dataloader


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/cartoonset10k")
    ap.add_argument("--attrs", nargs="+", default=["eye_color", "hair_color", "face_color"])
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--rebuild_cache", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    ds = CartoonSetDataset(
        args.root,
        image_size=args.image_size,
        cond_attributes=tuple(args.attrs),
        limit=args.limit,
        cache=args.cache,
        rebuild_cache=args.rebuild_cache,
    )
    print(f"dataset size      : {len(ds)}")
    print(f"cond attributes   : {ds.cond_attributes}")
    print(f"attribute dims    : {ds.attribute_dims}")
    img, lab = ds[0]
    print(f"image tensor      : {tuple(img.shape)} range [{img.min():.2f}, {img.max():.2f}]")
    print(f"label vector      : {lab.tolist()}")
    xb, yb = next(iter(build_dataloader(ds, batch_size=8, num_workers=0)))
    print(f"batch images      : {tuple(xb.shape)}")
    print(f"batch labels      : {tuple(yb.shape)}")
    print(f"init took         : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
