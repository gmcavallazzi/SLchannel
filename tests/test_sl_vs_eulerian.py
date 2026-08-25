"""Cross-scheme and self-convergence checks on a nonlinear (vortex) field:
1. SL (bdf2) and the Eulerian IMEX reference, run from the same initial
   condition at the same small dt, must agree to within a few percent (they
   differ by spatial truncation: interpolation vs central fluxes).
2. SL self-convergence in dt (Richardson): ||u_dt - u_dt/2|| / ||u_dt/2 -
   u_dt/4|| >= ~4 for bdf2 with incremental pressure (O(dt^2) overall) and
   ~2 for the non-incremental variant (projection splitting caps the
   velocity self-convergence near first order). The convergence runs use
   traj_interp_order=4: trilinear trajectory sampling is only C^0, whose
   flow map limits any SL scheme to O(dt) with a small h^2 coefficient."""

import tempfile

import pytest
from helpers import make_config_file

# ~600 solver steps across 8 solver builds: ~33 s
pytestmark = pytest.mark.slow

T = 0.4


def run_solver(scheme, dt, traj_order=2, freeze_forcing=False, sl_extra=None):
    from slchannel.solver import SLChannelFlow

    with tempfile.TemporaryDirectory() as tmp:
        sl_cfg = {"traj_interp_order": traj_order}
        if sl_extra:
            sl_cfg.update(sl_extra)
        cfg = make_config_file(
            tmp,
            nx=16,
            ny=16,
            nz=32,
            Re=1000.0,
            gamma=1.5,
            init_type="vortices",
            pert=0.08,
            dt=dt,
            scheme=scheme,
            extra={"sl": sl_cfg},
        )
        solver = SLChannelFlow(config_file=cfg)
        if freeze_forcing:
            # the exact bulk-flux constraint (uniform shift inside the step)
            # has its own splitting dynamics that would mask the momentum
            # scheme's temporal order — disable it entirely for the
            # self-convergence measurement
            solver._apply_bulk_forcing = lambda dt: (0.0, 0.0)
        step = solver.step_sl_bdf2 if scheme == "sl" else solver.step_imex
        for _ in range(round(T / dt)):
            step(dt)
        return solver.u.clone(), solver.v.clone(), solver.w.clone()


def field_dist(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def test_sl_vs_eulerian(check):

    # --- SL vs Eulerian agreement at small dt ---------------------------
    sl = run_solver("sl", 0.005)
    eu = run_solver("eulerian", 0.005)
    umax = sl[0].abs().max().item()
    rel = field_dist(sl, eu) / umax
    check("SL vs Eulerian (dt=0.005)", rel < 0.05, f"rel_diff={rel:.2e}")

    # --- temporal self-convergence ---------------------------------------
    # Boukir BDF2 characteristics with incremental pressure is O(dt^2); the
    # 'noninc' variant's projection splitting caps the velocity
    # self-convergence near O(dt).
    for sl_extra, min_ratio in [
        ({"bdf2_pressure": "inc"}, 3.0),
        ({"bdf2_pressure": "noninc"}, 1.7),
    ]:
        u1 = run_solver("sl", 0.02, traj_order=4, freeze_forcing=True, sl_extra=sl_extra)
        u2 = run_solver("sl", 0.01, traj_order=4, freeze_forcing=True, sl_extra=sl_extra)
        u4 = run_solver("sl", 0.005, traj_order=4, freeze_forcing=True, sl_extra=sl_extra)
        d12 = field_dist(u1, u2)
        d24 = field_dist(u2, u4)
        ratio = d12 / d24
        label = "bdf2/" + "/".join(str(v) for v in sl_extra.values())
        check(
            f"SL self-convergence {label}",
            ratio > min_ratio,
            f"|d(dt,dt/2)|={d12:.3e} |d(dt/2,dt/4)|={d24:.3e} ratio={ratio:.2f}",
        )
