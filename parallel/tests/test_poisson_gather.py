"""Gathered Poisson: local pressure slabs (halos included) match the
monolithic solve_poisson_fft exactly."""

import torch
from helpers_par import analytic_nodes, build_ref

from parallel.comm import EmulatedComm
from parallel.decomp import Decomp
from parallel.poisson_gather import solve_poisson_gathered
from slchannel.projection import initialize_fft_solver, solve_poisson_fft


def test_poisson_gather(check):
    _, grid = build_ref(nx=16, ny=16, nz=12, order=4)
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    fft_data = initialize_fft_solver(nx, ny, nz, grid["dx"], grid["dy"], grid["dz_c"], grid["dz_f"])

    div_nodes = analytic_nodes("w", grid, lambda X, Y, Z: torch.sin(X) * torch.cos(Y) * (Z - 1.0))[
        :, :, :nz
    ].contiguous()  # (nx, ny, nz) cell-centered rhs
    dt_eff = 0.01
    p_mono = solve_poisson_fft(div_nodes.clone() / dt_eff, fft_data)

    d = Decomp(2, 2, nx, ny, nz, H=4)
    comm = EmulatedComm(d)
    div_local = {
        r: div_nodes[
            d.origin(r)[0] : d.origin(r)[0] + d.nxl,
            d.origin(r)[1] : d.origin(r)[1] + d.nyl,
            :,
        ].clone()
        for r in range(d.nranks)
    }
    p_ext = solve_poisson_gathered(div_local, comm, d, fft_data, dt_eff)

    p_nodes = p_mono[1 : nx + 1, 1 : ny + 1, :]
    expect = d.scatter(p_nodes.contiguous(), "p", fill_halos=True)
    for r in range(d.nranks):
        err = float((p_ext[r] - expect[r]).abs().max())
        check(f"poisson slab rank {r}", err == 0.0, f"max|diff|={err:g}")
