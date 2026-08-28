"""Production driver under real torch.distributed (gloo, 4 processes,
2x2): the fields_final.npz written by rank 0 must match a monolithic run of
the same deterministic config bitwise. Opt-in like the other dist tests:
`-m dist` or SLC_RUN_DIST=1."""

import os
import tempfile

import numpy as np
import pytest
import torch.multiprocessing as mp
from helpers import make_config_file
from helpers_par import build_ref  # noqa: F401  (sys.path setup, import first)

from parallel.production import _dist_test_worker
from slchannel.solver import SLChannelFlow

pytestmark = pytest.mark.dist

N_STEPS = 5


def test_dist_production_matches_mono(check):
    with tempfile.TemporaryDirectory() as tmp:
        extra = {
            "grid": {"nx": 16, "ny": 16, "nz": 24},
            "output": {
                "results_folder": os.path.join(tmp, "out_dist"),
                "n_out": 2,
                "n_save": 10**8,
            },
            "statistics": {"n_stats": 0},
        }
        cfg_d = make_config_file(tmp, n_steps=N_STEPS, extra=extra, name="dist.yaml")
        extra["output"] = dict(extra["output"], results_folder=os.path.join(tmp, "out_mono"))
        cfg_m = make_config_file(tmp, n_steps=N_STEPS, extra=extra, name="mono.yaml")

        init_file = os.path.join(tmp, "store")
        mp.spawn(_dist_test_worker, args=(4, init_file, cfg_d, 2, 2), nprocs=4, join=True)

        final = os.path.join(tmp, "out_dist", "fields_final.npz")
        check("fields_final written by rank 0", os.path.exists(final), final)
        d = np.load(final)

        mono = SLChannelFlow(config_file=cfg_m)
        mono.sl._triton = None
        if getattr(mono.sl, "_advect_c", None) is not None:
            mono.sl._advect_c = None
        for _ in range(N_STEPS):
            mono.step_sl_bdf2(mono.dt)

        umax = float(mono.u.abs().max())
        for c in "uvw":
            err = float(np.abs(d[c] - getattr(mono, c).cpu().numpy()).max())
            check(
                f"dist field {c} vs mono",
                err <= 1e-12 * max(1.0, umax),
                f"max|diff|={err:.3e}",
            )
