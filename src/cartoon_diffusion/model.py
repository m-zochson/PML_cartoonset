"""
Conditional U-Net for a class-conditional DDPM with classifier-free guidance.

The condition is a small set of categorical attributes. Each attribute gets its
own embedding table with ONE extra "null" row (index = cardinality) used as the
unconditional token for classifier-free guidance. The per-attribute embeddings
are summed and added to the sinusoidal time embedding.

    model = UNet(attribute_dims=[5, 10, 11])
    eps   = model(x_t, t, labels)      # x_t:(B,3,H,W)  t:(B,)  labels:(B,3)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Standard sinusoidal timestep embedding. t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Lightweight self-attention over spatial positions."""

    def __init__(self, ch, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(
        self,
        attribute_dims,
        in_ch=3,
        base=64,
        ch_mult=(1, 2, 2),
        num_res_blocks=1,
        emb_dim=256,
        attn_at=(8,),
        image_size=32,
        groups=8,
    ):
        super().__init__()
        self.attribute_dims = list(attribute_dims)
        self.base = base

        self.time_mlp = nn.Sequential(
            nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.attr_emb = nn.ModuleList(
            [nn.Embedding(k + 1, emb_dim) for k in self.attribute_dims]
        )

        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)

        # ---- encoder: record the channel count of every pushed skip ----
        self.downs = nn.ModuleList()
        skip_chs = [base]          # stem skip
        ch = base
        res = image_size
        for i, mult in enumerate(ch_mult):
            out = base * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(ch, out, emb_dim, groups))
                ch = out
                skip_chs.append(ch)
                if res in attn_at:
                    self.downs.append(AttnBlock(ch, groups))
            if i != len(ch_mult) - 1:
                self.downs.append(Down(ch))
                skip_chs.append(ch)
                res //= 2

        # ---- bottleneck ----
        self.mid1 = ResBlock(ch, ch, emb_dim, groups)
        self.mid_attn = AttnBlock(ch, groups)
        self.mid2 = ResBlock(ch, ch, emb_dim, groups)

        # ---- decoder: num_res_blocks + 1 blocks per level, consume skips ----
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out = base * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock(ch + skip_chs.pop(), out, emb_dim, groups))
                ch = out
                if res in attn_at:
                    self.ups.append(AttnBlock(ch, groups))
            if i != 0:
                self.ups.append(Up(ch))
                res *= 2
        assert not skip_chs, f"unconsumed skips: {skip_chs}"

        self.out = nn.Sequential(
            nn.GroupNorm(groups, ch), nn.SiLU(), nn.Conv2d(ch, in_ch, 3, padding=1)
        )

    def null_labels(self, batch_size, device):
        """The all-null condition (index = cardinality per attribute)."""
        null = torch.tensor(self.attribute_dims, device=device)
        return null[None].expand(batch_size, -1)

    def forward(self, x, t, labels):
        emb = self.time_mlp(timestep_embedding(t, self.base))
        for i, tbl in enumerate(self.attr_emb):
            emb = emb + tbl(labels[:, i])

        h = self.stem(x)
        skips = [h]
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
                skips.append(h)
            elif isinstance(layer, Down):
                h = layer(h)
                skips.append(h)
            else:  # AttnBlock
                h = layer(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
            elif isinstance(layer, Up):
                h = layer(h)
            else:  # AttnBlock
                h = layer(h)

        return self.out(h)
