"""Command-line entry point: ``slchannel <config.yaml>``."""

import argparse
import os
import sys

from . import __version__
from .solver import SLChannelFlow


def build_parser():
    parser = argparse.ArgumentParser(
        prog="slchannel", description="Semi-Lagrangian DNS of turbulent channel flow."
    )
    parser.add_argument("config", help="Path to the YAML configuration file (see docs/CONFIG.md).")
    parser.add_argument("--version", action="version", version=f"slChannel {__version__}")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.config):
        sys.exit(
            f"slchannel: configuration file not found: {args.config}\n"
            f"Example configurations live in configs/; see docs/CONFIG.md "
            f"for the available keys."
        )

    SLChannelFlow(config_file=args.config).run_simulation()


if __name__ == "__main__":
    main()
