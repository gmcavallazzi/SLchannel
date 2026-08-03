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


def run_solver(scheme, dt, time_scheme='v1', traj_order=2, freeze_forcing=False,
               traj_extrap='ab2', sl_extra=None):
    from solver import SLChannelFlow
    with tempfile.TemporaryDirectory() as tmp:
        sl_cfg = {'time_scheme': time_scheme,
                  'traj_interp_order': traj_order,
                  'traj_extrapolation': traj_extrap}
        if sl_extra:
            sl_cfg.update(sl_extra)
        cfg = make_config_file(tmp, nx=16, ny=16, nz=32, Re=1000.0, gamma=1.5,
                               init_type='vortices', pert=0.08, dt=dt,
                               scheme=scheme,
                               extra={'sl': sl_cfg})
        solver = SLChannelFlow(config_file=cfg)
        if freeze_forcing:
            # the exact bulk-flux constraint (uniform shift inside the step)
            # has its own splitting dynamics that would mask the momentum
            # scheme's temporal order — disable it entirely for the
            # self-convergence measurement
            solver._apply_bulk_forcing = lambda dt: (0.0, 0.0)
        if scheme == 'sl':
            step = solver.step_sl_bdf2 if time_scheme == 'bdf2' else solver.step_sl
        else:
            step = solver.step_imex
        for _ in range(round(T / dt)):
            step(dt)
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
    # ('v2', 4, 'pc'): the predictor-corrector mid-velocity must preserve
    # 2nd order (it replaces the AB2 extrapolation, whose amplification of
    # step-decorrelated high-k content is the M3 burst mechanism).
    # ('bdf2', 'inc'): Boukir BDF2 characteristics with incremental pressure
    # is O(dt^2); the 'noninc' variant's projection splitting caps the
    # velocity self-convergence near O(dt).
    for time_scheme, traj_order, extrap, sl_extra, min_ratio in [
            ('v1', 2, 'ab2', None, 1.7),
            ('v2', 4, 'ab2', None, 3.0),
            ('v2', 4, 'pc', None, 3.0),
            ('bdf2', 4, 'ab2', {'bdf2_pressure': 'inc'}, 3.0),
            ('bdf2', 4, 'ab2', {'bdf2_pressure': 'noninc'}, 1.7)]:
        u1 = run_solver('sl', 0.02, time_scheme, traj_order, freeze_forcing=True,
                        traj_extrap=extrap, sl_extra=sl_extra)
        u2 = run_solver('sl', 0.01, time_scheme, traj_order, freeze_forcing=True,
                        traj_extrap=extrap, sl_extra=sl_extra)
        u4 = run_solver('sl', 0.005, time_scheme, traj_order, freeze_forcing=True,
                        traj_extrap=extrap, sl_extra=sl_extra)
        d12 = field_dist(u1, u2)
        d24 = field_dist(u2, u4)
        ratio = d12 / d24
        label = time_scheme if time_scheme == 'bdf2' else f"{time_scheme}/{extrap}"
        if sl_extra:
            label += "/" + "/".join(str(v) for v in sl_extra.values())
        ok &= report(f"SL self-convergence {label}", ratio > min_ratio,
                     f"|d(dt,dt/2)|={d12:.3e} |d(dt/2,dt/4)|={d24:.3e} ratio={ratio:.2f}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
