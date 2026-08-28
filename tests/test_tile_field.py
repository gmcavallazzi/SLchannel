"""tools/tile_field.py: exact periodic replication into a larger box, and
the target case starts (interpolate + projection) from the tiled seed."""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch
import yaml
from helpers import make_config_file

from parallel.decomp import mono_node_view
from slchannel.solver import SLChannelFlow
from slchannel.utils import save_flow_fields

pytestmark = pytest.mark.slow

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_tile_field(check):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config_file(
            tmp,
            Lx=6.283185307179586,
            Ly=3.141592653589793,
            extra={
                "grid": {"nx": 16, "ny": 12, "nz": 24},
                "output": {"results_folder": f"{tmp}/seed_out"},
            },
        )
        s = SLChannelFlow(config_file=cfg)
        for _ in range(2):
            s.step_sl_bdf2(s.dt)
        save_flow_fields(
            s.u,
            s.v,
            s.w,
            s.p,
            s.z_c,
            s.z_f,
            s.Lx,
            s.Ly,
            2,
            0.02,
            0.06,
            0.0,
            f"{tmp}/seed_out",
            "seed.npz",
        )

        tgt = yaml.safe_load(open(cfg))
        tgt["domain"]["Lx"] = 2 * s.Lx
        tgt["domain"]["Ly"] = 3 * s.Ly
        tgt["grid"] = {"nx": 28, "ny": 30, "nz": 24}
        tgt["initialization"] = {
            "type": "interpolate",
            "field_file": f"{tmp}/tiled.npz",
            "reset_time": True,
        }
        tgt["output"]["results_folder"] = f"{tmp}/tgt_out"
        yaml.safe_dump(tgt, open(f"{tmp}/tgt.yaml", "w"))

        def tile(amp):
            r = subprocess.run(
                [
                    sys.executable,
                    os.path.join(REPO, "tools", "tile_field.py"),
                    f"{tmp}/seed_out/seed.npz",
                    "--config",
                    f"{tmp}/tgt.yaml",
                    "--out",
                    f"{tmp}/tiled.npz",
                    "--amp",
                    str(amp),
                ],
                capture_output=True,
                text=True,
                cwd=REPO,
            )
            check(f"tile_field runs (amp={amp})", r.returncode == 0, r.stderr[-300:])

        tile(0.0)
        d0, d1 = np.load(f"{tmp}/seed_out/seed.npz"), np.load(f"{tmp}/tiled.npz")
        check("tiled box lengths", abs(d1["Lx"] - 2 * s.Lx) + abs(d1["Ly"] - 3 * s.Ly) < 1e-12, "")
        for c in "uvw":
            n0 = mono_node_view(torch.from_numpy(d0[c]), c, 16, 12).numpy()
            n1 = mono_node_view(torch.from_numpy(d1[c]), c, 32, 36).numpy()
            err = np.abs(n1 - np.tile(n0, (2, 3, 1))).max()
            check(f"exact replication {c}", err == 0.0, f"max|diff|={err:.1e}")

        tile(0.02)
        t = SLChannelFlow(config_file=f"{tmp}/tgt.yaml")
        t.step_sl_bdf2(t.dt)
        check("target starts and steps", bool(torch.isfinite(t.u).all()), "")


def test_direct_interpolation_into_bigger_box(check):
    """The campaign-default seeding: `interpolate` init pointed straight at
    a smaller-box field maps it proportionally (a stretch, never a tile or a
    wrap) onto the big box and projects it."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config_file(
            tmp,
            Lx=6.283185307179586,
            Ly=3.141592653589793,
            extra={
                "grid": {"nx": 16, "ny": 12, "nz": 24},
                "output": {"results_folder": f"{tmp}/seed_out"},
            },
        )
        s = SLChannelFlow(config_file=cfg)
        for _ in range(2):
            s.step_sl_bdf2(s.dt)
        save_flow_fields(
            s.u,
            s.v,
            s.w,
            s.p,
            s.z_c,
            s.z_f,
            s.Lx,
            s.Ly,
            2,
            0.02,
            0.06,
            0.0,
            f"{tmp}/seed_out",
            "seed.npz",
        )
        tgt = yaml.safe_load(open(cfg))
        tgt["domain"]["Lx"] = 4 * s.Lx
        tgt["domain"]["Ly"] = 3 * s.Ly
        tgt["grid"] = {"nx": 40, "ny": 30, "nz": 24}
        tgt["initialization"] = {
            "type": "interpolate",
            "field_file": f"{tmp}/seed_out/seed.npz",
            "reset_time": True,
        }
        tgt["output"]["results_folder"] = f"{tmp}/tgt_out"
        yaml.safe_dump(tgt, open(f"{tmp}/tgt.yaml", "w"))
        t = SLChannelFlow(config_file=f"{tmp}/tgt.yaml")
        t.step_sl_bdf2(t.dt)
        check("target starts and steps", bool(torch.isfinite(t.u).all()), "")
        check(
            "bulk preserved after rescale",
            abs(float(t.u[1:41, 1:31, 1:25].mean())) > 0.5,
            f"{float(t.u.mean()):.3f}",
        )
