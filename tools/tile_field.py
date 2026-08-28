"""Tile a converged channel field into a larger periodic box.

Builds the initial condition for a big-box run from a small-box seed: the
field is replicated exactly (the directions are periodic, so tiling is not
an approximation) and a smooth solenoidal perturbation is added so the
artificial tile periodicity decorrelates during the warm-up instead of
persisting as a standing pattern. The z grid must match between seed and
target (same nz, gamma, Lz).

The output is a standard fields npz AT THE TILED RESOLUTION with the
target box lengths; point the target config at it with
`initialization: {type: interpolate, field_file: ...}` and the solver's
staggered-aware interpolation brings it onto the target grid (and projects
it divergence-free) at startup.

Example — M950 seed (2pi x pi) -> the Re950 big box (8pi x 3pi):

    python tools/tile_field.py data/m950_seed_768x640x320.npz \
        --config configs/re950_bigbox_sl_dt020.yaml \
        --out results/re950_bigbox_sl_dt020/seed_tiled.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from parallel.decomp import mono_node_view, nodes_to_mono  # noqa: E402
from slchannel.utils import save_flow_fields  # noqa: E402


def solenoidal_perturbation(comp, nx, ny, nz2, Lx, Ly, z_c, amp, rng, n_modes=6):
    """Analytic horizontal solenoidal field (u, v) = (d\\psi/dy, -d\\psi/dx)
    from a few random large-scale streamfunction modes, evaluated at the
    component's staggered node coordinates; zero at the walls."""
    dx, dy = Lx / nx, Ly / ny
    if comp == "u":
        x = (np.arange(nx) * dx)[:, None, None]
        y = ((np.arange(ny) + 0.5) * dy)[None, :, None]
    else:  # v
        x = ((np.arange(nx) + 0.5) * dx)[:, None, None]
        y = (np.arange(ny) * dy)[None, :, None]
    z = np.asarray(z_c)[None, None, :]
    Lz = 2.0
    env = np.sin(np.clip(z, 0.0, Lz) * np.pi / Lz) ** 2
    out = np.zeros((nx, ny, nz2))
    for _ in range(n_modes):
        kx = 2 * np.pi / Lx * rng.integers(1, 5)
        ky = 2 * np.pi / Ly * rng.integers(1, 4)
        ph_x, ph_y = rng.uniform(0, 2 * np.pi, 2)
        a = rng.normal() / n_modes
        if comp == "u":  # d/dy of sin(kx x + phx) sin(ky y + phy)
            out += a * ky * np.sin(kx * x + ph_x) * np.cos(ky * y + ph_y) * env
        else:  # -d/dx
            out += -a * kx * np.cos(kx * x + ph_x) * np.sin(ky * y + ph_y) * env
    rms = np.sqrt((out**2).mean())
    return out * (amp / max(rms, 1e-30))


def main():
    ap = argparse.ArgumentParser(description="tile a periodic channel field into a larger box")
    ap.add_argument("seed", help="source fields npz (small box)")
    ap.add_argument("--config", required=True, help="target-case YAML (box, grid, z-stretching)")
    ap.add_argument("--out", required=True, help="output npz path")
    ap.add_argument("--amp", type=float, default=0.02, help="perturbation rms, U_bulk units")
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    Lx_t, Ly_t = cfg["domain"]["Lx"], cfg["domain"]["Ly"]

    d = np.load(args.seed)
    Lx_s, Ly_s = float(d["Lx"]), float(d["Ly"])
    mx, my = Lx_t / Lx_s, Ly_t / Ly_s
    for name, m in (("x", mx), ("y", my)):
        if abs(m - round(m)) > 1e-9:
            sys.exit(f"target L{name} is not an integer multiple of the seed's ({m:.4f})")
    mx, my = int(round(mx)), int(round(my))

    u = torch.from_numpy(d["u"])
    nx_s, ny_s = u.shape[0] - 1, u.shape[1] - 2
    nz = d["z_c"].shape[0] - 2
    # z grid is carried over from the seed verbatim, so the target must match
    if cfg["grid"]["nz"] != nz:
        sys.exit(f"target nz={cfg['grid']['nz']} != seed nz={nz}: z grids must match")

    print(f"tiling {nx_s}x{ny_s}x{nz} ({Lx_s:.3f}x{Ly_s:.3f}) -> {mx}x in x, {my}x in y")
    rng = np.random.default_rng(args.rng_seed)
    fields = {}
    for c in "uvwp":
        src = torch.from_numpy(d[c]) if c != "u" else u
        nodes = mono_node_view(src, "p" if c == "p" else c, nx_s, ny_s).numpy()
        tiled = np.tile(nodes, (mx, my, 1))
        if c in "uv":
            pert = solenoidal_perturbation(
                c, nx_s * mx, ny_s * my, tiled.shape[2], Lx_t, Ly_t, d["z_c"], args.amp, rng
            )
            tiled = tiled + pert
        fields[c] = nodes_to_mono(
            torch.from_numpy(tiled), "p" if c == "p" else c, nx_s * mx, ny_s * my
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_flow_fields(
        fields["u"],
        fields["v"],
        fields["w"],
        fields["p"],
        torch.from_numpy(d["z_c"]),
        torch.from_numpy(d["z_f"]),
        Lx_t,
        Ly_t,
        0,
        0.0,
        float(d["u_tau"]),
        0.0,
        os.path.dirname(os.path.abspath(args.out)),
        os.path.basename(args.out),
    )
    print(
        f"wrote {args.out}: {nx_s * mx}x{ny_s * my}x{nz} at L = {Lx_t:.3f} x {Ly_t:.3f}, "
        f"perturbation rms {args.amp} U_b (rng seed {args.rng_seed})"
    )


if __name__ == "__main__":
    main()
