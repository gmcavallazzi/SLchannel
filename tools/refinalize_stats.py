"""Re-finalize turbulence_stats.npz from a saved accumulator state.

Use when save-time derived quantities (e.g. the open-channel u_tau fix in
turbstats.save_statistics) changed after a run finished: the accumulated
sums in turbulence_stats_state.npz are unaffected, so the stats file can be
rebuilt without re-running.

Usage: python tools/refinalize_stats.py <config.yaml> <results_folder>
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml

torch.set_default_dtype(torch.float64)

# Imported after the dtype is set: these build tensors at import time.
from slchannel.turbstats import TurbulenceStats  # noqa: E402
from slchannel.utils import generate_grid  # noqa: E402


def main():
    cfg_path, folder = sys.argv[1], sys.argv[2]
    cfg = yaml.safe_load(open(cfg_path))
    g, d, f, s = cfg["grid"], cfg["domain"], cfg["flow"], cfg["statistics"]
    z_f, z_c, dz_f, dz_c = generate_grid(
        f["gamma"], g["nz"], d["Lz"], stretching_type=d.get("stretching_type", "symmetric")
    )
    ts = TurbulenceStats(
        g["nx"],
        g["ny"],
        g["nz"],
        d["Lx"],
        d["Ly"],
        d["Lz"],
        z_c,
        z_f,
        dz_c,
        dz_f,
        d["Lx"] / g["nx"],
        d["Ly"] / g["ny"],
        1.0 / f["Re"],
        f["Re_tau"],
        z_plus_target=s.get("z_plus_target", 15.0),
        device="cpu",
        top_wall_bc_type=cfg.get("boundary_conditions", {})
        .get("top_wall", {})
        .get("type", "dirichlet"),
        spectra_z_planes=s.get("spectra_z_planes", None),
    )
    ts.load_state(os.path.join(folder, s.get("state_file", "turbulence_stats_state.npz")))
    ts.save_statistics(os.path.join(folder, s.get("output_file", "turbulence_stats.npz")))


if __name__ == "__main__":
    main()
