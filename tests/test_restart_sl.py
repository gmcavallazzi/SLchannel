"""Restart continuity: 10 steps + checkpoint + 10 restarted steps must match a
20-step uninterrupted run closely. Not bit-identical by design: the restart
re-bootstraps the AB2 trajectory extrapolation (V_mid = V^n for one step),
exactly like torChannel's AB2 bootstrap; the resulting one-step O(dt^2)
perturbation must stay small."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from _slhelpers import report, make_config_file

torch.set_default_dtype(torch.float64)

DT = 0.01


def run_case(time_scheme):
    from solver import SLChannelFlow
    from utils import save_flow_fields

    extra = {'sl': {'time_scheme': time_scheme}}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config_file(tmp, nx=16, ny=16, nz=32, Re=1000.0, gamma=1.5,
                               init_type='vortices', pert=0.08, dt=DT, extra=extra)

        def stepper(s):
            return s.step_sl_bdf2 if time_scheme == 'bdf2' else s.step_sl

        # uninterrupted 20 steps
        ref = SLChannelFlow(config_file=cfg)
        for _ in range(20):
            stepper(ref)(DT)

        # 10 steps, checkpoint, restart, 10 more
        s1 = SLChannelFlow(config_file=cfg)
        for _ in range(10):
            stepper(s1)(DT)
        ckpt = os.path.join(tmp, 'results', 'fields_ckpt.npz')
        save_flow_fields(s1.u, s1.v, s1.w, s1.p, s1.z_c, s1.z_f, s1.Lx, s1.Ly,
                         10, 10 * DT, 0.05, s1.forcing, os.path.join(tmp, 'results'),
                         'fields_ckpt.npz')

        cfg2 = make_config_file(tmp, nx=16, ny=16, nz=32, Re=1000.0, gamma=1.5,
                                init_type='vortices', pert=0.08, dt=DT,
                                extra={**extra, 'initialization': {'field_file': ckpt}},
                                name='config_restart.yaml')
        s2 = SLChannelFlow(config_file=cfg2)
        s2.forcing = s1.forcing  # forcing state carried over (saved in npz)
        for _ in range(10):
            stepper(s2)(DT)

        cont = s2.initial_step == 10 and abs(s2.initial_time - 10 * DT) < 1e-12
        umax = ref.u.abs().max().item()
        rel = max((a - b).abs().max().item()
                  for a, b in [(ref.u, s2.u), (ref.v, s2.v), (ref.w, s2.w)]) / umax
        return cont, s2, rel


def run():
    ok = True
    # bdf2 restart re-bootstraps with one BDF1 step (same one-step O(dt^2)
    # allowance as the AB2 re-bootstrap)
    for time_scheme in ['v1', 'bdf2']:
        cont, s2, rel = run_case(time_scheme)
        ok &= report(f"restart continues from checkpoint [{time_scheme}]",
                     cont, f"step={s2.initial_step} time={s2.initial_time:.4f}")
        ok &= report(f"restart matches uninterrupted run [{time_scheme}]", rel < 1e-4,
                     f"rel_diff={rel:.2e} (bootstrap-limited, not bit-exact)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
