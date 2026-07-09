"""Canonical 100k evaluation CLI wrapper."""

import sys

from cartoon_diffusion.cli.evaluate import main


if __name__ == "__main__":
    main(["--dataset_variant", "100k", *sys.argv[1:]])
