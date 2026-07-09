"""CLI for applying a trained Cartoon Set upscaler to PNG images."""

from __future__ import annotations

import argparse
from pathlib import Path

from cartoon_diffusion.upscaling import upscale_image_file


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/upscaler_96/checkpoints/latest.pt")
    ap.add_argument("--input", default=None, help="single PNG image to upscale")
    ap.add_argument("--output", default=None, help="output path for --input")
    ap.add_argument("--input_dir", default=None, help="directory of PNG images")
    ap.add_argument("--output_dir", default="outputs/upscaled")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    return ap


def _default_output(input_path: Path, output_dir: str | Path) -> Path:
    return Path(output_dir) / f"{input_path.stem}_upscaled{input_path.suffix}"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.input is None and args.input_dir is None:
        raise SystemExit("provide --input or --input_dir")
    if args.input is not None and args.input_dir is not None:
        raise SystemExit("use only one of --input or --input_dir")

    if args.input is not None:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else _default_output(input_path, args.output_dir)
        upscale_image_file(
            ckpt_path=args.ckpt,
            input_path=input_path,
            output_path=output_path,
            device=args.device,
        )
        print(f"saved {output_path}")
        return

    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"no .png files found under {input_dir}")
    for input_path in paths:
        output_path = _default_output(input_path, args.output_dir)
        upscale_image_file(
            ckpt_path=args.ckpt,
            input_path=input_path,
            output_path=output_path,
            device=args.device,
        )
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
