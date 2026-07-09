"""Canonical 100k training CLI wrapper."""

import sys

from cartoon_diffusion.cli.train import main


if __name__ == "__main__":
    main(["--dataset_variant", "100k", *sys.argv[1:]])
