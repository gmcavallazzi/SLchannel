"""Analytic full-step check: Stokes decay of u(z, t) = sin(pi*z) on Lz=2
(no-slip at both walls, zero bulk). The field is x-independent, so the SL
advection is exact and the step exercises the whole path (SL + explicit xy +
CN z-diffusion + projection). Analytic decay: exp(-nu*pi^2*t), tolerance 1%
at trajectory-CFL-class timesteps."""

import sys, os, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from _slhelpers import report, make_config_file

torch.set_default_dtype(torch.float64)


def run():
    ok = True
    from solver import SLChannelFlow

    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config_file(tmp, nx=8, ny=8, nz=96, Re=100.0, gamma=1.3,
                               init_type='parabolic', pert=0.0, dt=0.02)
        solver = SLChannelFlow(config_file=cfg)

        # overwrite with the Stokes mode: u = sin(pi*z), v = w = 0
        solver.u[:] = torch.sin(math.pi * solver.z_c).view(1, 1, -1)
        solver.v[:] = 0.0
        solver.w[:] = 0.0
        solver.p[:] = 0.0
        solver.apply_bc_uvw()
        solver.forcing = 0.0

        dt, T = 0.02, 1.0
        n = round(T / dt)
        for _ in range(n):
            solver.step_sl(dt)
            solver.forcing = 0.0   # freeze the bulk controller for this test

        decay = math.exp(-solver.nu * math.pi ** 2 * T)
        exact = decay * torch.sin(math.pi * solver.z_c[1:-1])
        got = solver.u[4, 4, 1:-1]
        rel_err = ((got - exact).abs().max() / exact.abs().max()).item()
        ok &= report("Stokes decay rate", rel_err < 0.01, f"rel_err={rel_err:.2e}")

        # v and w must remain identically zero, pressure ~ 0
        vmax = solver.v.abs().max().item()
        wmax = solver.w.abs().max().item()
        ok &= report("v,w remain zero", vmax < 1e-12 and wmax < 1e-12,
                     f"max|v|={vmax:.2e} max|w|={wmax:.2e}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
