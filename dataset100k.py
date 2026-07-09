"""Backward-compatible data module for the former 100k loader."""

from cartoon_diffusion.data import CartoonSetDataset, build_dataloader, denormalize
from cartoon_diffusion.cli.inspect_data import main

__all__ = ["CartoonSetDataset", "build_dataloader", "denormalize"]


if __name__ == "__main__":
    main()
