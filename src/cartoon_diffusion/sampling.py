"""Qualitative sampling grids for trained Cartoon Set DDPM checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from cartoon_diffusion.checkpoints import load_checkpoint
from cartoon_diffusion.data import denormalize
from cartoon_diffusion.diffusion import GaussianDiffusion
from cartoon_diffusion.model import UNet


def _font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text(draw, xy, s, font, fill=(20, 20, 20), anchor="la"):
    draw.text(xy, s, font=font, fill=fill, anchor=anchor)


def _tensor_to_pil(t, size):
    arr = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, "RGB").resize((size, size), Image.BILINEAR)


def _short(attrs):
    return [a[0] for a in attrs]


def build_annotated_grid(cols, labels_per_row, attrs, vary_idx, weights, fixed, cell=112):
    n_cols = len(weights)
    n_rows = labels_per_row.shape[0]
    letters = _short(attrs)
    pad, cap_h, top_hdr_h, left_hdr_w, title_h = 6, 30, 30, 96, 40
    f_title, f_hdr, f_cap = _font(18), _font(16), _font(14)
    cell_w, cell_h = cell, cell + cap_h
    width = left_hdr_w + n_cols * (cell_w + pad) + pad
    height = title_h + top_hdr_h + n_rows * (cell_h + pad) + pad
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    fixed_desc = "  ".join(
        f"{letters[i]}({attrs[i]})={fixed[i]}" for i in range(len(attrs)) if i != vary_idx
    )
    _text(draw, (pad, pad), f"vary: {attrs[vary_idx]}   |   fixed: {fixed_desc}", f_title)
    y_hdr = title_h + top_hdr_h // 2
    for col, w in enumerate(weights):
        x = left_hdr_w + col * (cell_w + pad) + cell_w // 2
        _text(draw, (x, y_hdr), f"w = {w:g}", f_hdr, anchor="mm")
    for row in range(n_rows):
        y0 = title_h + top_hdr_h + row * (cell_h + pad) + pad
        _text(
            draw,
            (left_hdr_w - 8, y0 + cell // 2),
            f"{attrs[vary_idx]}\n= {labels_per_row[row, vary_idx].item()}",
            f_hdr,
            anchor="rm",
        )
        triple = "  ".join(f"{letters[i]}:{labels_per_row[row, i].item()}" for i in range(len(attrs)))
        for col in range(n_cols):
            x0 = left_hdr_w + col * (cell_w + pad) + pad
            canvas.paste(_tensor_to_pil(cols[col][row], cell), (x0, y0))
            _text(
                draw,
                (x0 + cell // 2, y0 + cell + cap_h // 2),
                f"{triple}   w={weights[col]:g}",
                f_cap,
                anchor="mm",
            )
    return canvas


@torch.no_grad()
def sample_one_attribute(
    model,
    diff,
    attrs,
    dims,
    image_size,
    vary_idx,
    fixed,
    weights,
    seed,
    device,
    *,
    sampler="ddpm",
    ddim_steps=50,
    eta=0.0,
):
    n_rows = dims[vary_idx]
    base = torch.tensor(fixed, device=device)
    labels = base[None].repeat(n_rows, 1)
    labels[:, vary_idx] = torch.arange(n_rows, device=device)
    cols = []
    for w in weights:
        torch.manual_seed(seed)
        if sampler == "ddpm":
            imgs = diff.ddpm_sample(model, labels, image_size=image_size, guidance_weight=w)
        elif sampler == "ddim":
            imgs = diff.ddim_sample(
                model,
                labels,
                image_size=image_size,
                guidance_weight=w,
                steps=ddim_steps,
                eta=eta,
            )
        else:
            raise ValueError(f"unknown sampler {sampler!r}")
        cols.append(denormalize(imgs).cpu())
    return cols, labels.cpu()


def sample_grids(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}   sampler: {args.sampler}")
    ck = load_checkpoint(args.ckpt, map_location=device)
    attrs, dims, image_size = ck["attrs"], ck["attribute_dims"], ck["image_size"]
    fixed = args.fixed if args.fixed is not None else [0] * len(attrs)
    if len(fixed) != len(attrs):
        raise ValueError(f"--fixed needs {len(attrs)} values (one per attr {attrs})")

    model = UNet(dims, image_size=image_size).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    diff = GaussianDiffusion(timesteps=ck["timesteps"])

    if args.vary != "all" and args.vary not in attrs:
        raise ValueError(f"--vary must be one of {attrs} or 'all'")
    targets = list(range(len(attrs))) if args.vary == "all" else [attrs.index(args.vary)]

    out_dir = Path(args.out_dir)
    if args.out is None:
        out_dir.mkdir(parents=True, exist_ok=True)
    for vary_idx in targets:
        cols, labels = sample_one_attribute(
            model,
            diff,
            attrs,
            dims,
            image_size,
            vary_idx,
            fixed,
            args.weights,
            args.seed,
            device,
            sampler=args.sampler,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
        )
        grid = build_annotated_grid(cols, labels, attrs, vary_idx, args.weights, fixed, cell=args.cell)
        if args.vary != "all" and args.out is not None:
            out_path = args.out
        else:
            prefix = os.path.splitext(args.out)[0] + "_" if args.out else str(out_dir / "grid_")
            out_path = f"{prefix}{attrs[vary_idx]}.png"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        grid.save(out_path)
        print(f"saved {out_path}")
