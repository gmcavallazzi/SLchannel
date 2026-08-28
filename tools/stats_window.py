"""Build finalized statistics for a time window by ladder subtraction.

The solver accumulates statistics as raw sums; the stats ladder
(examples/slurm/m950_stats_ladder.sh) archives the accumulator state every
N samples. Subtracting an archived state from the current one yields the
statistics of everything AFTER that archive point — any retroactive window
without rerunning.

    python tools/stats_window.py <config.yaml> <results_dir> \
        [--subtract results_dir/stats_ladder/state_n000660.npz] \
        [--out results_dir_window]

With no --subtract, the full accumulated window is finalized as-is. The
output directory gets the adjusted turbulence_stats_state.npz and the
finalized turbulence_stats.npz (via tools/refinalize_stats.py).
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def subtract_state(current, ladder, out_path):
    cur = dict(np.load(current))
    lad = dict(np.load(ladder))
    n_sub = 0
    out = {}
    for k, v in cur.items():
        if k in lad and (k.endswith("_sum") or k == "n_samples"):
            out[k] = v - lad[k]
            n_sub += 1
        else:
            out[k] = v
    np.savez(out_path, **out)
    print(
        f"subtracted {n_sub} accumulator keys: {int(lad['n_samples'])} of "
        f"{int(cur['n_samples'])} samples removed -> {int(out['n_samples'])} in window"
        + (f" (+{int(out['n_samples_extra'])} extra-plane)" if "n_samples_extra" in out else "")
    )


def main():
    ap = argparse.ArgumentParser(description="windowed statistics via ladder subtraction")
    ap.add_argument("config")
    ap.add_argument("results_dir")
    ap.add_argument("--subtract", help="archived ladder state to subtract")
    ap.add_argument("--out", help="output dir (default <results_dir>_window)")
    ap.add_argument(
        "--state", default="turbulence_stats_state.npz", help="state file name in results_dir"
    )
    args = ap.parse_args()

    out_dir = args.out or args.results_dir.rstrip("/") + "_window"
    os.makedirs(out_dir, exist_ok=True)
    src_state = os.path.join(args.results_dir, args.state)
    dst_state = os.path.join(out_dir, "turbulence_stats_state.npz")

    if args.subtract:
        subtract_state(src_state, args.subtract, dst_state)
    else:
        shutil.copyfile(src_state, dst_state)
        print(f"copied full-window state ({int(np.load(dst_state)['n_samples'])} samples)")

    rc = subprocess.run(
        [sys.executable, os.path.join(HERE, "refinalize_stats.py"), args.config, out_dir]
    ).returncode
    if rc != 0:
        sys.exit(rc)
    print(f"finalized: {os.path.join(out_dir, 'turbulence_stats.npz')}")


if __name__ == "__main__":
    main()
