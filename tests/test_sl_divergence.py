"""Nonlinear robustness of the SL bdf2 step: from a vortex-perturbed initial
field, every step (cubic and quintic) must return a divergence-free field at
solver tolerance with no NaN, and departure-point clamping must stay a tiny
fraction of the grid."""

import tempfile

import torch
from helpers import make_config_file


def run_case(order):
    from slchannel.solver import SLChannelFlow
    from slchannel.utils import compute_divergence

    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config_file(
            tmp,
            nx=16,
            ny=16,
            nz=32,
            Re=1000.0,
            gamma=1.5,
            init_type="vortices",
            pert=0.08,
            dt=0.02,
            extra={"sl": {"interp_order": order}},
        )
        solver = SLChannelFlow(config_file=cfg)
        max_div = 0.0
        for _ in range(25):
            solver.step_sl_bdf2(0.02)
            div = compute_divergence(
                solver.u,
                solver.v,
                solver.w,
                solver.nx,
                solver.ny,
                solver.nz,
                solver.dx,
                solver.dy,
                solver.dz_f,
            )
            max_div = max(max_div, div.abs().max().item())
        finite = all(
            torch.isfinite(t).all().item() for t in (solver.u, solver.v, solver.w, solver.p)
        )
        n_pts = 3 * solver.nx * solver.ny * solver.nz
        clamp_frac = solver.sl.n_clamped_last.item() / n_pts
        return max_div, finite, clamp_frac


def test_sl_divergence(check):
    # note: n_clamped_last covers BOTH bdf2 feet (the 2dt foot travels twice
    # as far), so clamp_frac is over ~6N gather points, not 3N
    for order in [4, 6]:
        max_div, finite, clamp_frac = run_case(order)
        check(
            f"SL bdf2 step order={order}",
            finite and max_div < 1e-10 and clamp_frac < 0.01,
            f"max|div|={max_div:.2e} clamp_frac={clamp_frac:.1e}",
        )
