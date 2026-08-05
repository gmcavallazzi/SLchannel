"""Laminar Poiseuille flow with the full SL solver: starting from a uniform
profile, the bulk-forcing controller + CN diffusion must converge to the
steady parabola u(z) = 1.5*U_bulk*z*(2-z) (Lz=2, delta=1). Checks steadiness,
profile shape (within FD truncation), bulk-velocity control, and
post-projection divergence."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from _slhelpers import report, make_config_file

torch.set_default_dtype(torch.float64)


def run():
    ok = True
    from solver import SLChannelFlow
    from utils import compute_divergence

    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config_file(tmp, nx=8, ny=8, nz=64, Re=20.0, gamma=1.3,
                               init_type='uniform', pert=0.0, dt=0.05)
        solver = SLChannelFlow(config_file=cfg)

        dt, T = 0.05, 150.0
        n = round(T / dt)
        u_prev_check = None
        for i in range(n):
            u_bulk, _ = solver.step_sl_bdf2(dt)
            if i == n - round(10.0 / dt):
                u_prev_check = solver.u.clone()

        # steadiness over the last 10 time units
        steady = (solver.u - u_prev_check).abs().max().item()
        ok &= report("steady state reached", steady < 1e-8, f"max|du|(10tu)={steady:.2e}")

        # profile vs analytic parabola (FD truncation-limited)
        z = solver.z_c[1:-1]
        exact = 1.5 * solver.U_bulk * z * (2.0 - z)
        got = solver.u[4, 4, 1:-1]
        rel_err = ((got - exact).abs().max() / exact.abs().max()).item()
        ok &= report("parabolic profile", rel_err < 5e-3, f"rel_err={rel_err:.2e}")

        # bulk velocity held by the forcing controller
        bulk_err = abs(u_bulk.item() - solver.U_bulk)
        ok &= report("bulk velocity control", bulk_err < 1e-6, f"|u_bulk-1|={bulk_err:.2e}")

        # divergence at solver tolerance
        div = compute_divergence(solver.u, solver.v, solver.w, solver.nx, solver.ny,
                                 solver.nz, solver.dx, solver.dy, solver.dz_f)
        max_div = div.abs().max().item()
        ok &= report("post-projection divergence", max_div < 1e-10, f"max|div|={max_div:.2e}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
