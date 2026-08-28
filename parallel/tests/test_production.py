"""Production driver (parallel/production.py) against the monolithic solver:
same config, same deterministic init, bitwise trajectory; checkpoint /
restart compatibility; STOP-file pause. CPU, emulated backend."""

import os
import tempfile

import numpy as np
import torch
from helpers import make_config_file  # tests/helpers via helpers_par path hook
from helpers_par import build_ref  # noqa: F401  (sys.path setup, import first)

from parallel.decomp import mono_node_view
from parallel.production import ProductionRun
from slchannel.solver import SLChannelFlow

N_STEPS = 6
GRID = {"nx": 16, "ny": 16, "nz": 24}


def _cfg(tmp, name, folder, n_steps=N_STEPS, extra_init=None, n_save=10**8, n_stats=0):
    extra = {
        "grid": GRID,
        "output": {"results_folder": os.path.join(tmp, folder), "n_out": 2, "n_save": n_save},
        "statistics": {"n_stats": n_stats, "t_stats": 0.0},
    }
    if extra_init:
        extra["initialization"] = extra_init
    return make_config_file(tmp, n_steps=n_steps, extra=extra, name=name)


def _run_mono(cfg, n_steps, dt=None):
    mono = SLChannelFlow(config_file=cfg)
    mono.sl._triton = None
    if getattr(mono.sl, "_advect_c", None) is not None:
        mono.sl._advect_c = None
    dt = mono.dt if dt is None else dt
    for _ in range(n_steps):
        mono.step_sl_bdf2(dt)
    return mono


def _max_node_diff(run, mono):
    errs = {}
    for c in "uvw":
        dec_nodes = run.dec.gather_nodes(c)
        mono_nodes = mono_node_view(getattr(mono, c), c, mono.nx, mono.ny)
        errs[c] = float((dec_nodes - mono_nodes).abs().max())
    return errs


def test_production_matches_mono(check):
    with tempfile.TemporaryDirectory() as tmp:
        cfg_p = _cfg(tmp, "prod.yaml", "out_prod", n_stats=2)
        cfg_m = _cfg(tmp, "mono.yaml", "out_mono")

        run = ProductionRun(cfg_p, 2, 2, backend="emulated", poisson="pencil", triton=False)
        res = run.run()
        check("completed", not res["stopped"] and not res["blown"], str(res))
        check("step count", res["step"] == N_STEPS, f"{res['step']}")

        mono = _run_mono(cfg_m, N_STEPS)
        errs = _max_node_diff(run, mono)
        umax = float(mono.u.abs().max())
        for c, err in errs.items():
            check(
                f"{N_STEPS}-step field {c} vs mono",
                err <= 1e-12 * max(1.0, umax),
                f"max|diff|={err:.3e}",
            )

        out = os.path.join(tmp, "out_prod")
        for f in ("fields_init.npz", "fields_final.npz", "timeseries.npz"):
            check(f"{f} written", os.path.exists(os.path.join(out, f)), f)
        ts = np.load(os.path.join(out, "timeseries.npz"))
        check(
            "timeseries keys",
            all(k in ts for k in ("step", "time", "u_bulk", "u_tau", "forcing", "cfl")),
            str(list(ts.keys())),
        )
        check("timeseries rows", len(ts["step"]) == N_STEPS // 2, f"{len(ts['step'])}")
        check(
            "stats accumulated",
            run.mono.turbulence_stats.n_samples == N_STEPS // 2,
            f"{run.mono.turbulence_stats.n_samples}",
        )
        check(
            "stats saved",
            os.path.exists(os.path.join(out, "turbulence_stats.npz")),
            "turbulence_stats.npz",
        )


def test_production_restart_matches_mono_restart(check):
    with tempfile.TemporaryDirectory() as tmp:
        cfg0 = _cfg(tmp, "seed.yaml", "out_seed", n_steps=4, n_save=4)
        run0 = ProductionRun(cfg0, 2, 2, backend="emulated", poisson="pencil", triton=False)
        run0.run()
        ckpt = os.path.join(tmp, "out_seed", "fields.npz")
        check("checkpoint written", os.path.exists(ckpt), ckpt)
        d = np.load(ckpt)
        check(
            "checkpoint keys",
            all(k in d for k in ("u", "v", "w", "p", "time", "step", "forcing")),
            str(list(d.keys())),
        )

        restart_init = {"type": "vortices", "field_file": ckpt}
        cfg_p = _cfg(tmp, "prod_r.yaml", "out_prod_r", n_steps=7, extra_init=restart_init)
        cfg_m = _cfg(tmp, "mono_r.yaml", "out_mono_r", n_steps=7, extra_init=restart_init)

        run = ProductionRun(cfg_p, 2, 2, backend="emulated", poisson="pencil", triton=False)
        res = run.run()
        check("restart continues step count", res["step"] == 7, f"{res['step']}")

        mono = _run_mono(cfg_m, 3)  # 4 done in the checkpoint + 3 more
        errs = _max_node_diff(run, mono)
        umax = float(mono.u.abs().max())
        for c, err in errs.items():
            check(
                f"restart field {c} vs mono restart",
                err <= 1e-12 * max(1.0, umax),
                f"max|diff|={err:.3e}",
            )


def test_production_stop_file(check):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp, "stop.yaml", "out_stop", n_steps=50)
        out = os.path.join(tmp, "out_stop")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "STOP"), "w"):
            pass
        run = ProductionRun(cfg, 2, 2, backend="emulated", poisson="pencil", triton=False)
        res = run.run()
        check("stopped by STOP file", res["stopped"], str(res))
        check("stopped early", res["step"] < 50, f"{res['step']}")
        check("checkpoint written", os.path.exists(os.path.join(out, "fields.npz")), "fields.npz")
        check(
            "no fields_final on pause",
            not os.path.exists(os.path.join(out, "fields_final.npz")),
            "fields_final.npz",
        )


def test_params_only_solver(check):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp, "params.yaml", "out_params")
        s = SLChannelFlow(config_file=cfg, allocate_fields=False)
        check("no fields", s.u is None and s.v is None and s.w is None and s.p is None, "")
        check("grid built", s.z_c is not None and s.fft_data is not None, "")
        check("advector built", s.sl is not None, "")
        check(
            "nothing written",
            not os.path.exists(os.path.join(tmp, "out_params", "fields_init.npz"))
            and not os.path.exists(os.path.join(tmp, "out_params", "grid.csv")),
            "",
        )
        check("dtype default set", torch.get_default_dtype() == torch.float64, "")


def test_local_bulk_reduction_equivalent(check):
    """bulk='local' (allreduced partial sums, no per-step allgather) against
    bulk='gathered': same value to reduction-order rounding, so short-horizon
    fields stay within amplified-epsilon of the gathered trajectory."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_g = _cfg(tmp, "bg.yaml", "out_bg")
        cfg_l = _cfg(tmp, "bl.yaml", "out_bl")
        run_g = ProductionRun(cfg_g, 2, 2, backend="emulated", poisson="pencil", triton=False)
        run_l = ProductionRun(
            cfg_l, 2, 2, backend="emulated", poisson="pencil", triton=False, bulk="local"
        )
        res_g, res_l = run_g.run(), run_l.run()
        check("both completed", res_g["step"] == res_l["step"] == N_STEPS, "")
        for c in "uvw":
            err = float((run_g.dec.gather_nodes(c) - run_l.dec.gather_nodes(c)).abs().max())
            check(
                f"local-vs-gathered field {c}",
                err <= 1e-10,
                f"max|diff|={err:.3e} (reduction-order epsilon, amplified over {N_STEPS} steps)",
            )
        ts_g = np.load(os.path.join(tmp, "out_bg", "timeseries.npz"))
        ts_l = np.load(os.path.join(tmp, "out_bl", "timeseries.npz"))
        du = float(np.abs(ts_g["u_bulk"] - ts_l["u_bulk"]).max())
        check("u_bulk agreement", du <= 1e-12, f"max|diff|={du:.3e}")
