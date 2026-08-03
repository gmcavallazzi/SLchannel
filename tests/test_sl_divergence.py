"""Nonlinear robustness of the SL step: from a vortex-perturbed initial field,
every step (v1 and v2, cubic and quintic) must return a divergence-free field
at solver tolerance with no NaN, and departure-point clamping must stay a tiny
fraction of the grid."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from _slhelpers import report, make_config_file

torch.set_default_dtype(torch.float64)


def run_case(time_scheme, order):
    from solver import SLChannelFlow
    from utils import compute_divergence

    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config_file(tmp, nx=16, ny=16, nz=32, Re=1000.0, gamma=1.5,
                               init_type='vortices', pert=0.08, dt=0.02,
                               extra={'sl': {'time_scheme': time_scheme,
                                             'interp_order': order}})
        solver = SLChannelFlow(config_file=cfg)
        step = solver.step_sl_bdf2 if time_scheme == 'bdf2' else solver.step_sl
        max_div = 0.0
        for _ in range(25):
            step(0.02)
            div = compute_divergence(solver.u, solver.v, solver.w, solver.nx,
                                     solver.ny, solver.nz, solver.dx, solver.dy,
                                     solver.dz_f)
            max_div = max(max_div, div.abs().max().item())
        finite = all(torch.isfinite(t).all().item()
                     for t in (solver.u, solver.v, solver.w, solver.p))
        n_pts = 3 * solver.nx * solver.ny * solver.nz
        clamp_frac = solver.sl.n_clamped_last.item() / n_pts
        return max_div, finite, clamp_frac


def run():
    ok = True
    # note: under bdf2 n_clamped_last covers BOTH feet (the 2dt foot travels
    # twice as far), so clamp_frac is over ~6N gather points, not 3N
    for time_scheme, order in [('v1', 4), ('v2', 4), ('v1', 6), ('v2', 6),
                               ('bdf2', 4), ('bdf2', 6)]:
        max_div, finite, clamp_frac = run_case(time_scheme, order)
        ok &= report(f"SL step {time_scheme} order={order}",
                     finite and max_div < 1e-10 and clamp_frac < 0.01,
                     f"max|div|={max_div:.2e} clamp_frac={clamp_frac:.1e}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
