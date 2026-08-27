"""Decomposed eager SL advect vs the monolithic eager advect.

The 1x1 case is the anti-drift pin: LocalSL is a fork of semilag.py and must
reproduce the production result bitwise; it fails if semilag.py changes
underneath the fork. Multi-rank cases validate the halo/ownership machinery
at <= 1e-13.
"""

import torch
from helpers_par import build_ref, default_fields, mono_advect_nodes

from parallel.comm import EmulatedComm
from parallel.decomp import Decomp
from parallel.sl_local import LocalSL, required_halo


def decomposed_advect_nodes(ref, d, nodes, mid_nodes, dt_t):
    comm = EmulatedComm(d)
    ext = {c: d.scatter(nodes[c], c, fill_halos=True) for c in "uvw"}
    mid = {c: d.scatter(mid_nodes[c], c, fill_halos=True) for c in "uvw"}
    sl = {r: LocalSL(ref, d, r) for r in range(d.nranks)}
    star = {c: {} for c in "uvw"}
    H, nxl, nyl, nz = d.H, d.nxl, d.nyl, d.nz
    arr = {
        "u": (slice(H + 1, H + nxl + 1), slice(H, H + nyl), slice(1, nz + 1)),
        "v": (slice(H, H + nxl), slice(H + 1, H + nyl + 1), slice(1, nz + 1)),
        "w": (slice(H, H + nxl), slice(H, H + nyl), slice(1, nz)),
    }
    max_disp = 0.0
    for r in range(d.nranks):
        outs = sl[r].advect({c: ext[c][r] for c in "uvw"}, {c: mid[c][r] for c in "uvw"}, dt_t)
        for c in "uvw":
            e = d.alloc(c)
            e[arr[c]] = outs[c]
            star[c][r] = e
        max_disp = max(max_disp, sl[r].max_disp_cells["x"], sl[r].max_disp_cells["y"])
    comm.pull_minus_edge(star["u"], dim=0)
    comm.pull_minus_edge(star["v"], dim=1)
    return {c: d.gather(star[c]) for c in "uvw"}, max_disp


def test_local_advect(check):
    for order in (4, 6):
        ref, grid = build_ref(nx=24, ny=24, nz=16, order=order)
        nodes = default_fields(grid, seed=0)
        mids = default_fields(grid, seed=1)
        dt = 0.4 * grid["dx"]  # displacement < 1 cell per dt
        for depth, tag in [(dt, "near"), (2.0 * dt, "far")]:
            dt_t = torch.as_tensor(depth, dtype=torch.float64)
            mono = mono_advect_nodes(ref, grid, nodes, mids, dt_t)
            H = required_halo(order, disp_cells=1.0, foot_depth_factor=1.0)
            for px, py in [(1, 1), (2, 1), (1, 2), (2, 2), (4, 1)]:
                d = Decomp(px, py, 24, 24, 16, H=H)
                dec, _ = decomposed_advect_nodes(ref, d, nodes, mids, dt_t)
                for c in "uvw":
                    err = float((dec[c] - mono[c]).abs().max())
                    tol = 0.0 if (px, py) == (1, 1) else 1e-13
                    check(
                        f"advect o{order} {tag} {px}x{py} {c}",
                        err <= tol,
                        f"max|diff|={err:.3e} (tol {tol:g})",
                    )
