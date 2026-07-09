"""
Qualitative sampling from a trained conditional DDPM checkpoint.

For each conditioned attribute it produces an ANNOTATED grid whose ROWS are the
values of that attribute and whose COLUMNS are guidance weights. Every cell is
captioned with the full attribute triple it was CONDITIONED on plus the guidance
weight, so the grid is self-documenting: you can read off exactly what each
generated image was asked to be.

    # one annotated grid per attribute (grid_eye_color.png, grid_hair_color.png, ...)
    python sample.py --ckpt ckpt.pt --vary all --weights 0 1 3 5 --steps 50

    # just one attribute
    python sample.py --ckpt ckpt.pt --vary hair_color --weights 0 1 3 5 --steps 50 --out grid_hair.png

Other (non-varied) attributes are held fixed via --fixed (default all 0); the
grid title records their values. Attribute VALUES are shown as integer indices
(the Cartoon Set has no canonical color names), e.g. "e:2 h:5 f:0".
"""

import argparse
import os

import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import denormalize
from diffusion import GaussianDiffusion
from model import UNet


# --- small text helpers -------------------------------------------------------
def _font(size):
    """A truetype font if available, else PIL's bitmap default (portable)."""
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text(draw, xy, s, font, fill=(20, 20, 20), anchor="la"):
    draw.text(xy, s, font=font, fill=fill, anchor=anchor)


def _tensor_to_pil(t, size):
    """(3,H,W) in [0,1] -> upscaled RGB PIL image."""
    arr = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, "RGB").resize((size, size), Image.BILINEAR)


def _short(attrs):
    """First letter of each attribute name, for compact captions: e/h/f."""
    return [a[0] for a in attrs]


# --- grid builder -------------------------------------------------------------
def build_annotated_grid(cols, labels_per_row, attrs, vary_idx, weights,
                         fixed, cell=112):
    """
    cols            : list (one per weight) of (n_rows,3,H,W) tensors in [0,1]
    labels_per_row  : (n_rows, n_attr) long, the condition used for each row
    """
    n_cols = len(weights)
    n_rows = labels_per_row.shape[0]
    letters = _short(attrs)

    pad = 6
    cap_h = 30                 # per-cell caption strip
    top_hdr_h = 30             # column headers (guidance weights)
    left_hdr_w = 96            # row headers (varying attribute value)
    title_h = 40

    f_title = _font(18)
    f_hdr = _font(16)
    f_cap = _font(14)

    cell_w = cell
    cell_h = cell + cap_h
    W = left_hdr_w + n_cols * (cell_w + pad) + pad
    H = title_h + top_hdr_h + n_rows * (cell_h + pad) + pad

    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    # title: which attribute varies, which are fixed
    fixed_desc = "  ".join(
        f"{letters[i]}({attrs[i]})={fixed[i]}"
        for i in range(len(attrs)) if i != vary_idx
    )
    _text(draw, (pad, pad),
          f"vary: {attrs[vary_idx]}   |   fixed: {fixed_desc}",
          f_title)

    # column headers: guidance weights
    y_hdr = title_h + top_hdr_h // 2
    for c, w in enumerate(weights):
        x = left_hdr_w + c * (cell_w + pad) + cell_w // 2
        _text(draw, (x, y_hdr), f"w = {w:g}", f_hdr, anchor="mm")

    # rows
    for r in range(n_rows):
        y0 = title_h + top_hdr_h + r * (cell_h + pad) + pad
        # row header: the value of the varying attribute for this row
        _text(draw, (left_hdr_w - 8, y0 + cell // 2),
              f"{attrs[vary_idx]}\n= {labels_per_row[r, vary_idx].item()}",
              f_hdr, anchor="rm")
        # caption text (same triple for the whole row, weight differs per column)
        triple = "  ".join(
            f"{letters[i]}:{labels_per_row[r, i].item()}"
            for i in range(len(attrs))
        )
        for c in range(n_cols):
            x0 = left_hdr_w + c * (cell_w + pad) + pad
            im = _tensor_to_pil(cols[c][r], cell)
            canvas.paste(im, (x0, y0))
            # caption strip under the image
            cap_y = y0 + cell + cap_h // 2
            _text(draw, (x0 + cell // 2, cap_y),
                  f"{triple}   w={weights[c]:g}", f_cap, anchor="mm")

    return canvas


@torch.no_grad()
def sample_one_attribute(model, diff, attrs, dims, image_size, vary_idx,
                         fixed, weights, steps, seed, device):
    """Return list-of-columns (denormalized [0,1] tensors) and the row labels."""
    n_rows = dims[vary_idx]
    base = torch.tensor(fixed, device=device)
    labels = base[None].repeat(n_rows, 1)
    labels[:, vary_idx] = torch.arange(n_rows, device=device)

    cols = []
    for w in weights:
        torch.manual_seed(seed)  # same start noise per row across weights
        imgs = diff.ddim_sample(model, labels, image_size=image_size,
                                guidance_weight=w, steps=steps)
        cols.append(denormalize(imgs).cpu())
    return cols, labels.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--vary", default="all",
                    help="attribute to vary down the rows, or 'all' to produce "
                         "one grid per attribute (default: all)")
    ap.add_argument("--weights", type=float, nargs="+", default=[0, 1, 3, 5])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--fixed", type=int, nargs="+", default=None,
                    help="values for the non-varied attributes (default all 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell", type=int, default=112,
                    help="pixel size of each generated image in the grid")
    ap.add_argument("--out", default=None,
                    help="output path. For a single --vary attribute this is the "
                         "filename; with --vary all it is used as a prefix "
                         "(default: grid_<attr>.png)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    ck = torch.load(args.ckpt, map_location=device)
    attrs, dims = ck["attrs"], ck["attribute_dims"]
    image_size = ck["image_size"]

    fixed = args.fixed if args.fixed is not None else [0] * len(attrs)
    if len(fixed) != len(attrs):
        raise ValueError(f"--fixed needs {len(attrs)} values (one per attr {attrs})")

    model = UNet(dims, image_size=image_size).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    diff = GaussianDiffusion(timesteps=ck["timesteps"])

    if args.vary == "all":
        targets = list(range(len(attrs)))
    else:
        if args.vary not in attrs:
            raise ValueError(f"--vary must be one of {attrs} or 'all'")
        targets = [attrs.index(args.vary)]

    for vary_idx in targets:
        cols, labels = sample_one_attribute(
            model, diff, attrs, dims, image_size, vary_idx,
            fixed, args.weights, args.steps, args.seed, device)
        grid = build_annotated_grid(cols, labels, attrs, vary_idx,
                                    args.weights, fixed, cell=args.cell)

        if args.vary != "all" and args.out is not None:
            out_path = args.out
        else:
            prefix = os.path.splitext(args.out)[0] + "_" if args.out else "grid_"
            out_path = f"{prefix}{attrs[vary_idx]}.png"

        grid.save(out_path)
        print(f"saved {out_path}  ("
              f"{dims[vary_idx]} values of '{attrs[vary_idx]}' x "
              f"{len(args.weights)} weights {args.weights}; "
              f"fixed={ {attrs[i]: fixed[i] for i in range(len(attrs)) if i != vary_idx} })")


if __name__ == "__main__":
    main()
