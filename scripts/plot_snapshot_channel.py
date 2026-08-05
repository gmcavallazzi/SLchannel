"""Channel snapshot slices from a saved 3D field (fields*.npz), one figure
per velocity component. torChannel figure style (usetex serif, robust
percentile color limits shared across panels, gouraud shading), channel
vocabulary.

Layout per component:
  row 1: x-z cut at mid-span
  row 2: y-z cut at mid-x
  row 3: two x-y cuts, at z+ ~ 15 and z+ ~ 100 (heights set from u_tau)
u uses a sequential map (viridis); v and w are zero-mean and use a
symmetric diverging map (RdBu_r) centred at 0.

Usage (module load texlive first for usetex):
  python scripts/plot_snapshot_channel.py results_fix/kmm_bdf2/fields_t00980.013.npz \
      [--z-plus 15 100] [--out-prefix figures_fix/kmm_snap]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

plt.rcParams.update({'text.usetex': shutil.which('latex') is not None,
                     'font.family': 'serif', 'font.size': 11})


def cell_center(f, comp):
    """Interior cell-centered field (nx, ny, nz) from a ghost-shaped component."""
    if comp == 'u':
        return 0.5 * (f[0:-2, 1:-1, 1:-1] + f[1:-1, 1:-1, 1:-1])
    if comp == 'v':
        return 0.5 * (f[1:-1, 0:-2, 1:-1] + f[1:-1, 1:-1, 1:-1])
    return 0.5 * (f[1:-1, 1:-1, 0:-1] + f[1:-1, 1:-1, 1:])


def main():
    ap = argparse.ArgumentParser(description='channel snapshot slices, all components')
    ap.add_argument('fields')
    ap.add_argument('--z-plus', type=float, nargs=2, default=[15.0, 100.0])
    ap.add_argument('--out-prefix', default='figures_fix/snap')
    args = ap.parse_args()

    d = np.load(args.fields)
    z_c = d['z_c'][1:-1]
    Lx, Ly = float(d['Lx']), float(d['Ly'])
    u_tau = float(d['u_tau'])
    t = float(d['time'])
    step = int(d['step'])
    # nu from Re_b: U_bulk=1, delta=1 convention -> not stored; infer from
    # u_tau and the KMM target is fragile, so read nu via z+ = z*u_tau/nu
    # with nu passed through the grid: use Re from filename config is not
    # available either -- accept nu as u_tau/Re_tau with Re_tau from z range.
    Lz = float(z_c[-1] + z_c[0])          # symmetric grid: z_c[0]+z_c[-1] ~ Lz
    delta = Lz / 2.0
    # z+ of a height z (distance from nearest wall)
    # nu: from the KMM case Re=2792.8 unless overridden by env
    nu = float(os.environ.get('SNAP_NU', 1.0 / 2792.8))

    comps = {'u': (r'u/U_b', 'viridis', False),
             'v': (r'v/U_b', 'RdBu_r', True),
             'w': (r'w/U_b', 'RdBu_r', True)}

    for comp, (label, cmap, sym) in comps.items():
        f = cell_center(d[comp], comp)
        nx, ny, nz = f.shape
        x = (np.arange(nx) + 0.5) * Lx / nx
        y = (np.arange(ny) + 0.5) * Ly / ny

        cuts = []
        for zp in args.z_plus:
            z_target = zp * nu / u_tau                  # distance from bottom wall
            k = int(np.argmin(abs(z_c - z_target)))
            cuts.append((k, zp))

        lo, hi = np.percentile(f, [1, 99])
        if sym:
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m

        fig = plt.figure(figsize=(13.0, 10.5))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.35], hspace=0.42, wspace=0.18)

        ax = fig.add_subplot(gs[0, :])
        pc = ax.pcolormesh(x, z_c, f[:, ny // 2, :].T, cmap=cmap, vmin=lo, vmax=hi,
                           shading='gouraud')
        ax.set_title(rf'${label}$, $x$--$z$ at $y = L_y/2$  (step {step}, $t = {t:.1f}$)')
        ax.set_xlabel(r'$x/\delta$'); ax.set_ylabel(r'$z/\delta$')
        fig.colorbar(pc, ax=ax, pad=0.01)

        ax = fig.add_subplot(gs[1, :])
        pc = ax.pcolormesh(y, z_c, f[nx // 2, :, :].T, cmap=cmap, vmin=lo, vmax=hi,
                           shading='gouraud')
        ax.set_title(rf'${label}$, $y$--$z$ at $x = L_x/2$')
        ax.set_xlabel(r'$y/\delta$'); ax.set_ylabel(r'$z/\delta$')
        fig.colorbar(pc, ax=ax, pad=0.01)

        for col, (k, zp) in enumerate(cuts):
            ax = fig.add_subplot(gs[2, col])
            pc = ax.pcolormesh(x, y, f[:, :, k].T, cmap=cmap, vmin=lo, vmax=hi,
                               shading='gouraud')
            ax.set_title(rf'${label}$, $x$--$y$ at $z^+ \approx {zp:.0f}$'
                         rf'  ($z/\delta = {z_c[k]:.3f}$)')
            ax.set_xlabel(r'$x/\delta$'); ax.set_ylabel(r'$y/\delta$')
            fig.colorbar(pc, ax=ax, pad=0.02)

        out = f"{args.out_prefix}_{comp}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {out}  clim [{lo:.3f}, {hi:.3f}]")


if __name__ == '__main__':
    main()
