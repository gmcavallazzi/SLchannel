import torch
import argparse
from solver import SLChannelFlow

# Set double precision for stability
torch.set_default_dtype(torch.float64)


def main():
    parser = argparse.ArgumentParser(description='Semi-Lagrangian DNS Channel Flow Simulation')
    parser.add_argument('config',
                        type=str,
                        nargs='?',
                        default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()

    solver = SLChannelFlow(config_file=args.config)
    solver.run_simulation()


if __name__ == "__main__":
    main()
