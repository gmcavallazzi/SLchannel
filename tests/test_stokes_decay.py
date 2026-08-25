"""Analytic full-step check: Stokes decay of u(z, t) = sin(pi*z) on Lz=2
(no-slip at both walls, zero bulk). The field is x-independent, so the SL
advection is exact and the step exercises the whole path (SL + explicit xy +
CN z-diffusion + projection). Analytic decay: exp(-nu*pi^2*t), tolerance 1%
at trajectory-CFL-class timesteps."""

import math
import tempfile

import torch
from helpers import make_config_file


def test_stokes_decay(check):
    from slchannel.solver import SLChannelFlow

    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config_file(
            tmp, nx=8, ny=8, nz=96, Re=100.0, gamma=1.3, init_type="parabolic", pert=0.0, dt=0.02
        )
        solver = SLChannelFlow(config_file=cfg)

        # overwrite with the Stokes mode: u = sin(pi*z), v = w = 0
        solver.u[:] = torch.sin(math.pi * solver.z_c).view(1, 1, -1)
        solver.v[:] = 0.0
        solver.w[:] = 0.0
        solver.p[:] = 0.0
        solver.apply_bc_uvw()
        solver.forcing = 0.0

        # disable the exact bulk-flux constraint: the Stokes mode must decay
        # freely, including its bulk component (this test exercises the
        # diffusion path, not the mass-flux enforcement)
        solver._apply_bulk_forcing = lambda dt: (0.0, 0.0)

        dt, T = 0.02, 1.0
        n = round(T / dt)
        for _ in range(n):
            solver.step_sl_bdf2(dt)

        decay = math.exp(-solver.nu * math.pi**2 * T)
        exact = decay * torch.sin(math.pi * solver.z_c[1:-1])
        got = solver.u[4, 4, 1:-1]
        rel_err = ((got - exact).abs().max() / exact.abs().max()).item()
        check("Stokes decay rate", rel_err < 0.01, f"rel_err={rel_err:.2e}")

        # v and w must remain identically zero, pressure ~ 0
        vmax = solver.v.abs().max().item()
        wmax = solver.w.abs().max().item()
        check(
            "v,w remain zero", vmax < 1e-12 and wmax < 1e-12, f"max|v|={vmax:.2e} max|w|={wmax:.2e}"
        )
