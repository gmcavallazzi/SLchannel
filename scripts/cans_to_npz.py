"""Convert a CaNS checkpoint (fld.bin + grid.out) to a torChannel/slChannel
fields.npz, for use as `initialization: {type: interpolate, field_file: ...}`.

CaNS conventions (verified against CaNS_DRL/run0_theory_big):
  fld.bin  : raw float64, Fortran order, u(n1,n2,n3), v, w, p back-to-back,
             then time and istep as two trailing float64 scalars.
  grid.out : 5 columns per z-entry (possibly line-wrapped): junk, z_f, z_c,
             dz-, dz-; entries k = 0..n3+1 (ghosts included).
  Staggering is MAC, identical to torChannel: u at (x_f, y_c, z_c),
  v at (x_c, y_f, z_c), w at (x_c, y_c, z_f). CaNS index k=1..n3 maps to
  torChannel interior index k (one ghost layer on each side).

The output npz stores the SOURCE grid (CaNS z_c/z_f); the solver's
'interpolate' init resamples onto its own tanh grid and re-applies BCs and
the initial projection, so grid mismatch is handled downstream.

Usage:
  python scripts/cans_to_npz.py <cans_data_dir> <out.npz> \
      --fld fld_backup/fld.bin --Lx 10.68 --Ly 3.2 --nu 3.4843e-4
"""

import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', help='CaNS data directory (contains grid.out)')
    ap.add_argument('out', help='output .npz path')
    ap.add_argument('--fld', default='fld.bin', help='field file, relative to data_dir')
    ap.add_argument('--Lx', type=float, required=True)
    ap.add_argument('--Ly', type=float, required=True)
    ap.add_argument('--nu', type=float, required=True,
                    help='kinematic viscosity (for the u_tau diagnostic only)')
    args = ap.parse_args()

    # ---- grid ------------------------------------------------------------
    tokens = np.array(open(os.path.join(args.data_dir, 'grid.out')).read().split(),
                      dtype=np.float64)
    if tokens.size % 5 != 0:
        sys.exit(f"grid.out token count {tokens.size} is not a multiple of 5")
    g = tokens.reshape(-1, 5)          # rows k = 0..n3+1
    n3 = g.shape[0] - 2
    z_f = g[:n3 + 1, 1].copy()         # faces 0..n3 (wall to wall)
    z_c = g[:, 2].copy()               # centers incl. both ghosts (n3+2)
    Lz = z_f[-1]
    assert np.all(np.diff(z_f) > 0), "z_f not monotone"
    assert abs(z_c[0] + z_c[1]) < 1e-12, "bottom ghost center is not mirrored"

    # ---- fields ----------------------------------------------------------
    raw = np.fromfile(os.path.join(args.data_dir, args.fld), dtype=np.float64)
    n12 = raw.size - 2
    if n12 % 4 != 0 or n12 % n3 != 0:
        sys.exit(f"unexpected fld size {raw.size}")
    npts = n12 // 4
    nxy = npts // n3
    n1 = n2 = int(round(np.sqrt(nxy)))
    if n1 * n2 * n3 != npts:
        sys.exit(f"cannot infer square nx=ny from {npts} points with nz={n3}")
    t_chk, istep = raw[-2], int(raw[-1])

    def comp(idx):
        f = raw[idx * npts:(idx + 1) * npts]
        return f.reshape((n1, n2, n3), order='F')

    cu, cv, cw, cp = (comp(i) for i in range(4))

    # ---- torChannel ghost-shaped arrays -----------------------------------
    u = np.zeros((n1 + 1, n2 + 2, n3 + 2))
    v = np.zeros((n1 + 2, n2 + 1, n3 + 2))
    w = np.zeros((n1 + 2, n2 + 2, n3 + 1))
    p = np.zeros((n1 + 2, n2 + 2, n3 + 2))
    u[1:n1 + 1, 1:n2 + 1, 1:n3 + 1] = cu
    v[1:n1 + 1, 1:n2 + 1, 1:n3 + 1] = cv
    w[1:n1 + 1, 1:n2 + 1, 1:n3] = cw[:, :, :n3 - 1]   # cw k=n3 is the top face (0)
    p[1:n1 + 1, 1:n2 + 1, 1:n3 + 1] = cp

    # periodic ghosts in x, y
    for f in (u, v, w, p):
        f[0] = f[-2] if f.shape[0] == n1 + 2 else f[-1]
        f[-1] = f[1]
        f[:, 0] = f[:, -2] if f.shape[1] == n2 + 2 else f[:, -1]
        f[:, -1] = f[:, 1]
    # z ghosts: no-slip bottom, free-slip top (this case), w=0 at walls
    u[:, :, 0] = -u[:, :, 1]
    v[:, :, 0] = -v[:, :, 1]
    u[:, :, -1] = u[:, :, -2]
    v[:, :, -1] = v[:, :, -2]
    w[:, :, 0] = 0.0
    w[:, :, -1] = 0.0
    p[:, :, 0] = p[:, :, 1]
    p[:, :, -1] = p[:, :, -2]

    # ---- diagnostics -------------------------------------------------------
    dz_f = np.diff(z_f)
    u_bulk = (u[1:, 1:-1, 1:-1].mean(axis=(0, 1)) * dz_f).sum() / Lz
    dudz_w = u[1:, 1:-1, 1].mean() / z_c[1]
    u_tau = np.sqrt(args.nu * abs(dudz_w))
    print(f"source: {n1}x{n2}x{n3}, Lz={Lz:.6f}, checkpoint t={t_chk:.3f} istep={istep}")
    print(f"u_bulk = {u_bulk:.6f}   u_tau = {u_tau:.5f}   "
          f"Re_tau = {u_tau * Lz / args.nu:.1f}   dz1+ = {dz_f[0] * u_tau / args.nu:.3f}")

    np.savez(args.out, u=u, v=v, w=w, p=p, z_c=z_c, z_f=z_f,
             Lx=args.Lx, Ly=args.Ly, step=istep, time=t_chk,
             u_tau=u_tau, forcing=0.0)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
