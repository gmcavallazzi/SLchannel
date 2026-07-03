"""Cross-scheme and self-convergence checks on a nonlinear (vortex) field:
1. SL and the Eulerian IMEX reference, run from the same initial condition at
   the same small dt, must agree to within a few percent (they differ by
   spatial truncation: interpolation vs central fluxes).
2. SL self-convergence in dt (Richardson): ||u_dt - u_dt/2|| / ||u_dt/2 -
   u_dt/4|| >= ~2 for v1 (non-incremental splitting, first order overall) and
   ~4 for v2 (extrapolated pressure in the predictor, trajectory-CN diffusion,
   time-centered xy-RHS, increment projection: O(dt^2) overall). The v2 run
   uses traj_interp_order=4: trilinear trajectory sampling is only C^0, whose
   flow map limits any SL scheme to O(dt) with a small h^2 coefficient."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from _slhelpers import report, make_config_file

torch.set_default_dtype(torch.float64)

T = 0.4


def run_solver(scheme, dt, time_scheme='v1', traj_order=2, freeze_forcing=False):
    from solver import SLChannelFlow
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config_file(tmp, nx=16, ny=16, nz=32, Re=1000.0, gamma=1.5,
                               init_type='vortices', pert=0.08, dt=dt,
                               scheme=scheme,
                               extra={'sl': {'time_scheme': time_scheme,
                                             'traj_interp_order': traj_order}})
        solver = SLChannelFlow(config_file=cfg)
        step = solver.step_sl if scheme == 'sl' else solver.step_imex
        for _ in range(round(T / dt)):
            step(dt)
            if freeze_forcing:
                # the bulk controller has its own O(dt) discrete dynamics that
                # would mask the momentum scheme's temporal order
                solver.forcing = 0.0
        return solver.u.clone(), solver.v.clone(), solver.w.clone()


def field_dist(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def run():
    ok = True

    # --- SL vs Eulerian agreement at small dt ---------------------------
    sl = run_solver('sl', 0.005)
    eu = run_solver('eulerian', 0.005)
    umax = sl[0].abs().max().item()
    rel = field_dist(sl, eu) / umax
    ok &= report("SL vs Eulerian (dt=0.005)", rel < 0.05, f"rel_diff={rel:.2e}")

    # --- temporal self-convergence ---------------------------------------
    for time_scheme, traj_order, min_ratio in [('v1', 2, 1.7), ('v2', 4, 3.0)]:
        u1 = run_solver('sl', 0.02, time_scheme, traj_order, freeze_forcing=True)
        u2 = run_solver('sl', 0.01, time_scheme, traj_order, freeze_forcing=True)
        u4 = run_solver('sl', 0.005, time_scheme, traj_order, freeze_forcing=True)
        d12 = field_dist(u1, u2)
        d24 = field_dist(u2, u4)
        ratio = d12 / d24
        ok &= report(f"SL self-convergence {time_scheme}", ratio > min_ratio,
                     f"|d(dt,dt/2)|={d12:.3e} |d(dt/2,dt/4)|={d24:.3e} ratio={ratio:.2f}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
