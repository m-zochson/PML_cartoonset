"""Super-resolution utilities for optional Cartoon Set post-processing."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cartoon_diffusion.data import denormalize
from cartoon_diffusion.training import (
    amp_dtype_for,
    learning_rate_for_step,
    normalize_lr_schedule,
    set_optimizer_lr,
    validate_grad_clip,
    validate_min_lr,
)


@dataclass
class UpscalerConfig:
    root: str = "data/cartoonset10k"
    input_size: int = 32
    target_size: int = 96
    channels: int = 64
    residual_blocks: int = 4
    refinement_blocks: int = 2
    head_kernel: int = 5
    residual_kernel: int = 3
    up_kernel: int = 3
    refinement_kernel: int = 3
    tail_kernel: int = 5
    batch: int = 128
    steps: int = 10000
    lr: float = 2e-4
    lr_schedule: str = "constant"
    min_lr: float = 0.0
    weight_decay: float = 0.0
    l2_weight: float = 1.0
    precision: str = "fp32"
    grad_clip: float | None = 1.0
    limit: int | None = None
    workers: int = 4
    ckpt: str | None = None
    resume: bool = False
    save_every: int = 1000


class ResidualBlock(nn.Module):
    """Small residual block used by the PixelShuffle upscaler."""

    def __init__(self, channels: int, *, kernel_size: int = 3):
        super().__init__()
        padding = _kernel_padding(kernel_size)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PixelShuffleUpscaler(nn.Module):
    """Lightweight 32x32 -> 96x96 RGB upscaler."""

    def __init__(
        self,
        *,
        scale_factor: int = 3,
        channels: int = 64,
        residual_blocks: int = 4,
        refinement_blocks: int = 2,
        head_kernel: int = 5,
        residual_kernel: int = 3,
        up_kernel: int = 3,
        refinement_kernel: int = 3,
        tail_kernel: int = 5,
    ):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.channels = int(channels)
        self.residual_blocks = validate_block_count(residual_blocks, "residual_blocks")
        self.refinement_blocks = validate_block_count(refinement_blocks, "refinement_blocks")
        self.head_kernel = validate_kernel_size(head_kernel)
        self.residual_kernel = validate_kernel_size(residual_kernel)
        self.up_kernel = validate_kernel_size(up_kernel)
        self.refinement_kernel = validate_kernel_size(refinement_kernel)
        self.tail_kernel = validate_kernel_size(tail_kernel)
        self.head = nn.Sequential(
            nn.Conv2d(
                3,
                channels,
                self.head_kernel,
                padding=_kernel_padding(self.head_kernel),
            ),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[
                ResidualBlock(channels, kernel_size=self.residual_kernel)
                for _ in range(self.residual_blocks)
            ]
        )
        self.up = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * scale_factor * scale_factor,
                self.up_kernel,
                padding=_kernel_padding(self.up_kernel),
            ),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            *[
                ResidualBlock(channels, kernel_size=self.refinement_kernel)
                for _ in range(self.refinement_blocks)
            ]
        )
        self.tail = nn.Conv2d(
            channels,
            3,
            self.tail_kernel,
            padding=_kernel_padding(self.tail_kernel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        h = self.body(h)
        h = self.up(h)
        h = self.refine(h)
        return torch.tanh(self.tail(h))


class UpscalePairsDataset(Dataset):
    """Build supervised low/high-resolution pairs from Cartoon Set PNGs."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        input_size: int = 32,
        target_size: int = 96,
        bg_color: tuple[int, int, int] = (255, 255, 255),
        limit: int | None = None,
        recursive: bool = True,
    ):
        self.root = str(root)
        self.input_size = int(input_size)
        self.target_size = int(target_size)
        self.bg_color = tuple(bg_color)
        pattern = "**/*.png" if recursive else "*.png"
        pngs = sorted(glob(str(Path(self.root) / pattern), recursive=recursive))
        self.paths = [p for p in pngs if os.path.exists(p[:-4] + ".csv")]
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"No Cartoon Set .png/.csv pairs found under {self.root!r}.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        img = load_cartoon_rgb(self.paths[index], self.bg_color)
        low = pil_to_normalized_tensor(
            img.resize((self.input_size, self.input_size), Image.BILINEAR)
        )
        target = pil_to_normalized_tensor(
            img.resize((self.target_size, self.target_size), Image.BILINEAR)
        )
        return low, target


def pil_to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    w, h = img.size
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = t.view(h, w, 3).permute(2, 0, 1)
    return t.float() / 127.5 - 1.0


def load_cartoon_rgb(
    path: str | os.PathLike[str],
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, tuple(bg_color) + (255,))
    return Image.alpha_composite(bg, img).convert("RGB")


def load_image_tensor(
    path: str | os.PathLike[str],
    *,
    input_size: int = 32,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> torch.Tensor:
    img = load_cartoon_rgb(path, bg_color)
    img = img.resize((input_size, input_size), Image.BILINEAR)
    return pil_to_normalized_tensor(img)


def save_tensor_image(tensor: torch.Tensor, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = denormalize(tensor.detach().cpu()).clamp(0, 1)
    arr = (img * 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr, "RGB").save(path)


def upscale_tensor(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    squeeze = x.ndim == 3
    if squeeze:
        x = x.unsqueeze(0)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(x)
    if was_training:
        model.train()
    return out.squeeze(0) if squeeze else out


def make_upscaler(config: UpscalerConfig) -> PixelShuffleUpscaler:
    scale_factor = _scale_factor(config.input_size, config.target_size)
    return PixelShuffleUpscaler(
        scale_factor=scale_factor,
        channels=config.channels,
        residual_blocks=config.residual_blocks,
        refinement_blocks=config.refinement_blocks,
        head_kernel=config.head_kernel,
        residual_kernel=config.residual_kernel,
        up_kernel=config.up_kernel,
        refinement_kernel=config.refinement_kernel,
        tail_kernel=config.tail_kernel,
    )


def validate_kernel_size(kernel_size: int) -> int:
    value = int(kernel_size)
    if value < 1 or value % 2 == 0:
        raise ValueError("upscaler convolution kernels must be positive odd integers")
    return value


def validate_block_count(count: int, name: str) -> int:
    value = int(count)
    if value < 0:
        raise ValueError(f"--{name} must be non-negative")
    return value


def _kernel_padding(kernel_size: int) -> int:
    return validate_kernel_size(kernel_size) // 2


def _scale_factor(input_size: int, target_size: int) -> int:
    if target_size % input_size != 0:
        raise ValueError("target_size must be an integer multiple of input_size")
    scale_factor = target_size // input_size
    if scale_factor < 1:
        raise ValueError("target_size must be >= input_size")
    return scale_factor


def _loader(dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )


def _infinite(loader):
    while True:
        yield from loader


def validate_l2_weight(l2_weight: float) -> float:
    value = float(l2_weight)
    if value < 0:
        raise ValueError("--l2_weight must be non-negative")
    return value


def save_upscaler_checkpoint(
    path: str | os.PathLike[str],
    *,
    step: int,
    model: nn.Module,
    opt,
    config: UpscalerConfig,
    precision: str,
    grad_clip: float | None,
    scaler=None,
) -> None:
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    payload = {
        "kind": "cartoon_upscaler",
        "step": int(step),
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "config": asdict(config),
        "input_size": int(config.input_size),
        "target_size": int(config.target_size),
        "scale_factor": _scale_factor(config.input_size, config.target_size),
        "channels": int(config.channels),
        "residual_blocks": int(config.residual_blocks),
        "refinement_blocks": int(config.refinement_blocks),
        "head_kernel": int(config.head_kernel),
        "residual_kernel": int(config.residual_kernel),
        "up_kernel": int(config.up_kernel),
        "refinement_kernel": int(config.refinement_kernel),
        "tail_kernel": int(config.tail_kernel),
        "precision": precision,
        "grad_clip": grad_clip,
        "lr_schedule": config.lr_schedule,
        "min_lr": config.min_lr,
        "l2_weight": config.l2_weight,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_upscaler_checkpoint(path: str | os.PathLike[str], map_location=None):
    ck = torch.load(path, map_location=map_location)
    if ck.get("kind") != "cartoon_upscaler":
        raise ValueError(f"{path!r} is not a cartoon upscaler checkpoint")
    return ck


def model_from_checkpoint(ck) -> PixelShuffleUpscaler:
    return PixelShuffleUpscaler(
        scale_factor=int(ck["scale_factor"]),
        channels=int(ck["channels"]),
        residual_blocks=int(ck["residual_blocks"]),
        refinement_blocks=int(ck.get("refinement_blocks", 0)),
        head_kernel=int(ck.get("head_kernel", 3)),
        residual_kernel=int(ck.get("residual_kernel", 3)),
        up_kernel=int(ck.get("up_kernel", 3)),
        refinement_kernel=int(ck.get("refinement_kernel", 3)),
        tail_kernel=int(ck.get("tail_kernel", 3)),
    )


def train_upscaler(config: UpscalerConfig) -> None:
    if config.ckpt is None:
        if config.resume:
            raise ValueError("--resume needs an explicit --ckpt")
        config.ckpt = time.strftime("upscaler_%Y%m%d_%H%M%S.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision, amp_enabled, amp_dtype = amp_dtype_for(config.precision, device)
    grad_clip = validate_grad_clip(config.grad_clip)
    lr_schedule = normalize_lr_schedule(config.lr_schedule)
    min_lr = validate_min_lr(config.lr, config.min_lr)
    l2_weight = validate_l2_weight(config.l2_weight)
    scale_factor = _scale_factor(config.input_size, config.target_size)
    print(
        f"device: {device}   precision: {precision}   "
        f"scale: x{scale_factor}   grad_clip: {grad_clip}   "
        f"lr_schedule: {lr_schedule}   loss: l1 + {l2_weight:g}*l2"
    )
    print(
        "arch: "
        f"head k{config.head_kernel}, "
        f"res {config.residual_blocks}x k{config.residual_kernel}, "
        f"up k{config.up_kernel}, "
        f"refine {config.refinement_blocks}x k{config.refinement_kernel}, "
        f"tail k{config.tail_kernel}"
    )

    ds = UpscalePairsDataset(
        config.root,
        input_size=config.input_size,
        target_size=config.target_size,
        limit=config.limit,
    )
    loader = _loader(ds, config.batch, config.workers)
    print(f"images: {len(ds)}  input: {config.input_size}  target: {config.target_size}")

    model = make_upscaler(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")
    opt = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    start_step = 0
    if config.resume:
        if not os.path.exists(config.ckpt):
            raise FileNotFoundError(f"--resume set but checkpoint {config.ckpt!r} not found")
        ck = load_upscaler_checkpoint(config.ckpt, map_location=device)
        if ck["input_size"] != config.input_size or ck["target_size"] != config.target_size:
            raise ValueError("checkpoint input/target size does not match config")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if precision == "fp16" and "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_step = int(ck["step"])
        print(f"resumed from {config.ckpt} @ step {start_step}")
        if start_step >= config.steps:
            print(f"already at/after target {config.steps}; raise --steps to train further.")
            return

    print(f"training steps {start_step + 1} -> {config.steps}")
    model.train()
    data = _infinite(loader)
    t_window = time.time()
    window_start = start_step
    for step in range(start_step + 1, config.steps + 1):
        step_lr = learning_rate_for_step(
            base_lr=config.lr,
            min_lr=min_lr,
            schedule=lr_schedule,
            step=step,
            total_steps=config.steps,
        )
        set_optimizer_lr(opt, step_lr)

        low, target = next(data)
        low, target = low.to(device), target.to(device)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
            pred = model(low)
            l1 = l1_loss(pred, target)
            l2 = l2_loss(pred, target)
            loss = l1 + l2_weight * l2

        opt.zero_grad(set_to_none=True)
        if precision == "fp16":
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        if step % 100 == 0 or step == start_step + 1:
            now = time.time()
            done = step - window_start
            sps = done / max(now - t_window, 1e-9)
            eta_s = (config.steps - step) / max(sps, 1e-9)
            eta = time.strftime("%H:%M:%S", time.gmtime(eta_s))
            print(
                f"step {step:>6}/{config.steps}  loss {loss.item():.4f}  "
                f"l1 {l1.item():.4f}  l2 {l2.item():.4f}  "
                f"lr {step_lr:.2e}  {sps:5.2f} step/s  ETA {eta}"
            )
            t_window, window_start = now, step

        if step % config.save_every == 0 or step == config.steps:
            save_upscaler_checkpoint(
                config.ckpt,
                step=step,
                model=model,
                opt=opt,
                config=config,
                precision=precision,
                grad_clip=grad_clip,
                scaler=scaler if precision == "fp16" else None,
            )
            bak = " (prev -> .bak)" if os.path.exists(config.ckpt + ".bak") else ""
            print(f"saved {config.ckpt} @ step {step}{bak}")

    print("done.")


@torch.no_grad()
def upscale_image_file(
    *,
    ckpt_path: str | os.PathLike[str],
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = load_upscaler_checkpoint(ckpt_path, map_location=device)
    model = model_from_checkpoint(ck).to(device)
    model.load_state_dict(ck["model"])
    input_size = int(ck["input_size"])
    x = load_image_tensor(input_path, input_size=input_size).unsqueeze(0).to(device)
    out = upscale_tensor(model, x).squeeze(0)
    save_tensor_image(out, output_path)
