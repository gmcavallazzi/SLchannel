"""The halo-width formula holds, the overflow guard fires when the halo is
too small, and the max-displacement diagnostic is exact for uniform
advection."""

import pytest
import torch
from helpers_par import build_ref, default_fields

from parallel.comm import EmulatedComm
from parallel.decomp import Decomp, node_z_len
from parallel.sl_local import HaloOverflowError, LocalSL, required_halo


def _uniform_mids(grid, U0):
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    return {
        "u": torch.full((nx, ny, node_z_len("u", nz)), U0, dtype=torch.float64),
        "v": torch.zeros(nx, ny, node_z_len("v", nz), dtype=torch.float64),
        "w": torch.zeros(nx, ny, node_z_len("w", nz), dtype=torch.float64),
    }


def _advect(ref, d, nodes, mids, dt_t):
    comm = EmulatedComm(d)
    ext = {c: d.scatter(nodes[c], c, fill_halos=True) for c in "uvw"}
    mid = {c: d.scatter(mids[c], c, fill_halos=True) for c in "uvw"}
    disp = 0.0
    for r in range(d.nranks):
        sl = LocalSL(ref, d, r)
        sl.advect({c: ext[c][r] for c in "uvw"}, {c: mid[c][r] for c in "uvw"}, dt_t)
        disp = max(disp, sl.max_disp_cells["x"])
    comm.barrier()
    return disp


def test_halo_width(check):
    order = 6
    ref, grid = build_ref(nx=16, ny=16, nz=12, order=order)
    nodes = default_fields(grid)

    # uniform advecting velocity U0 in +x: displacement = depth*U0/dx exactly
    U0 = 1.25
    mids = _uniform_mids(grid, U0)
    dt = 1.1 * grid["dx"] / U0  # 1.1 cells per dt -> 2.2 at the far (2dt) foot
    dt_t = torch.as_tensor(2.0 * dt, dtype=torch.float64)
    disp_cells = 2.0 * dt * U0 / grid["dx"]

    # the formula's H (per-dt displacement 1.1, far-foot factor 2) passes
    H_ok = required_halo(order, disp_cells=1.1)
    d_ok = Decomp(2, 2, 16, 16, 12, H=H_ok)
    measured = _advect(ref, d_ok, nodes, mids, dt_t)
    check(
        "computed H passes far foot",
        abs(measured - disp_cells) < 1e-10,
        f"H={H_ok}, measured {measured:.6f} vs analytic {disp_cells:.6f} cells",
    )

    # a clearly under-sized halo trips the guard: with a 2.2-cell leftward
    # displacement and the order-6 stencil, indices reach H-4 => H=3 violates
    d_bad = Decomp(2, 2, 16, 16, 12, H=3)
    with pytest.raises(HaloOverflowError):
        _advect(ref, d_bad, nodes, mids, dt_t)
    check("guard fires at undersized H", True, "H=3 raised HaloOverflowError")

    # formula sanity: stencil-only floor at zero displacement
    check(
        "formula floor",
        required_halo(4, 0.0) == 3 and required_halo(6, 0.0) == 4,
        f"o4->{required_halo(4, 0.0)}, o6->{required_halo(6, 0.0)}",
    )
