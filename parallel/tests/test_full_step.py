"""Full decomposed BDF2 steps (BDF1 bootstrap + BDF2) against the monolithic
SLChannelFlow, small closed channel, fixed dt."""

import tempfile

from helpers import make_config_file  # tests/helpers via helpers_par path hook
from helpers_par import build_ref, default_fields  # noqa: F401  (sys.path setup)

from parallel.comm import EmulatedComm
from parallel.decomp import Decomp, mono_node_view, nodes_to_mono
from parallel.sl_local import required_halo
from parallel.step import DecomposedBDF2
from slchannel.solver import SLChannelFlow
from slchannel.utils import compute_divergence


def test_full_step(check):
    n_steps = 6
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config_file(
            tmp,
            extra={
                "grid": {"nx": 16, "ny": 16, "nz": 24},
                "sl": {"interp_order": 4, "interp_dtype": "fp64"},
                "output": {"results_folder": f"{tmp}/out"},
            },
        )
        mono = SLChannelFlow(config_file=cfg)
    mono.sl._triton = None
    if getattr(mono.sl, "_advect_c", None) is not None:
        mono.sl._advect_c = None

    # deterministic analytic seed on both sides
    _, grid = build_ref(
        nx=mono.nx,
        ny=mono.ny,
        nz=mono.nz,
        Lx=mono.Lx,
        Ly=mono.Ly,
        Lz=mono.Lz,
        gamma=mono.gamma,
        order=4,
    )
    fields = default_fields(grid)
    mono.u = nodes_to_mono(fields["u"], "u", mono.nx, mono.ny)
    mono.v = nodes_to_mono(fields["v"], "v", mono.nx, mono.ny)
    mono.w = nodes_to_mono(fields["w"], "w", mono.nx, mono.ny)
    mono.apply_bc_uvw()

    dt = 0.3 * mono.dx  # fixed; displacement well under 1 cell/dt
    H = required_halo(4, disp_cells=1.0)
    d = Decomp(2, 2, mono.nx, mono.ny, mono.nz, H=H)
    comm = EmulatedComm(d)
    dec = DecomposedBDF2(mono, d, comm)

    for k in range(n_steps):
        mono.step_sl_bdf2(dt)  # forcing update included in the step
        dec.step(dt)

    umax = float(mono.u.abs().max())
    for c in "uvw":
        dec_nodes = dec.gather_nodes(c)
        mono_nodes = mono_node_view(getattr(mono, c), c, mono.nx, mono.ny)
        err = float((dec_nodes - mono_nodes).abs().max())
        check(
            f"{n_steps}-step field {c}",
            err <= 1e-10 * max(1.0, umax),
            f"max|diff|={err:.3e} (umax={umax:.3f})",
        )

    # post-projection divergence of the decomposed state
    div = {
        r: compute_divergence(
            d.dview(dec.state["u"][r], "u"),
            d.dview(dec.state["v"][r], "v"),
            d.dview(dec.state["w"][r], "w"),
            d.nxl,
            d.nyl,
            d.nz,
            dec.dx,
            dec.dy,
            dec.dz_f,
        )
        for r in comm.local_ranks
    }
    dmax = max(float(v.abs().max()) for v in div.values())
    check("post-projection divergence", dmax <= 1e-12, f"max|div|={dmax:.3e}")
