"""Pencil-transposed Poisson vs the monolithic solve and vs the gathered
path, plus a full-step run using it end to end."""

import tempfile

import torch
from helpers import make_config_file
from helpers_par import analytic_nodes, build_ref, default_fields  # noqa: I001 (sys.path setup)

from parallel.comm import EmulatedComm
from parallel.decomp import Decomp, mono_node_view, nodes_to_mono
from parallel.poisson_gather import solve_poisson_gathered
from parallel.poisson_pencil import solve_poisson_pencil
from parallel.sl_local import required_halo
from parallel.step import DecomposedBDF2
from slchannel.projection import initialize_fft_solver
from slchannel.solver import SLChannelFlow


def test_poisson_pencil(check):
    _, grid = build_ref(nx=16, ny=16, nz=12, order=4)
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    fft_data = initialize_fft_solver(nx, ny, nz, grid["dx"], grid["dy"], grid["dz_c"], grid["dz_f"])
    div_nodes = analytic_nodes(
        "w", grid, lambda X, Y, Z: torch.sin(X) * torch.cos(2 * Y) * (Z - 1.0)
    )[:, :, :nz].contiguous()
    dt_eff = 0.01

    for px, py in [(2, 2), (4, 1), (1, 2), (2, 1)]:
        d = Decomp(px, py, nx, ny, nz, H=4)
        comm = EmulatedComm(d)
        div_local = {
            r: div_nodes[
                d.origin(r)[0] : d.origin(r)[0] + d.nxl,
                d.origin(r)[1] : d.origin(r)[1] + d.nyl,
                :,
            ].clone()
            for r in range(d.nranks)
        }
        ref = solve_poisson_gathered(dict(div_local), comm, d, fft_data, dt_eff)
        pen = solve_poisson_pencil(dict(div_local), comm, d, fft_data, dt_eff)
        pmax = max(float(ref[r].abs().max()) for r in ref)
        err = max(float((pen[r] - ref[r]).abs().max()) for r in ref)
        check(
            f"pencil vs gathered {px}x{py}",
            err <= 1e-12 * max(1.0, pmax),
            f"max|diff|={err:.3e} (pmax={pmax:.3f})",
        )


def test_full_step_pencil(check):
    n_steps = 4
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config_file(
            tmp,
            extra={
                "grid": {"nx": 16, "ny": 16, "nz": 24},
                "sl": {"interp_order": 4, "interp_dtype": "fp64"},
                "output": {"results_folder": f"{tmp}/out"},
            },
        )
        mono = SLChannelFlow(config_file=cfg)
    mono.sl._triton = None
    if getattr(mono.sl, "_advect_c", None) is not None:
        mono.sl._advect_c = None
    _, grid = build_ref(
        nx=mono.nx,
        ny=mono.ny,
        nz=mono.nz,
        Lx=mono.Lx,
        Ly=mono.Ly,
        Lz=mono.Lz,
        gamma=mono.gamma,
        order=4,
    )
    fields = default_fields(grid)
    mono.u = nodes_to_mono(fields["u"], "u", mono.nx, mono.ny)
    mono.v = nodes_to_mono(fields["v"], "v", mono.nx, mono.ny)
    mono.w = nodes_to_mono(fields["w"], "w", mono.nx, mono.ny)
    mono.apply_bc_uvw()

    dt = 0.3 * mono.dx
    d = Decomp(2, 2, mono.nx, mono.ny, mono.nz, H=required_halo(4, disp_cells=1.0))
    dec = DecomposedBDF2(mono, d, EmulatedComm(d), poisson="pencil")
    for _ in range(n_steps):
        mono.step_sl_bdf2(dt)
        dec.step(dt)
    umax = float(mono.u.abs().max())
    for c in "uvw":
        err = float(
            (dec.gather_nodes(c) - mono_node_view(getattr(mono, c), c, mono.nx, mono.ny))
            .abs()
            .max()
        )
        check(
            f"pencil {n_steps}-step field {c}",
            err <= 1e-9 * max(1.0, umax),
            f"max|diff|={err:.3e}",
        )
