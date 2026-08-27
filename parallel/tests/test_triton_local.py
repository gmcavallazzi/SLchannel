"""Localized Triton kernels vs the monolithic Triton path (fp32 pipeline),
on the GPU. Small grid, order 4, seconds of device time (the first call pays
the Triton JIT compile)."""

import pytest
import torch
from helpers_par import build_ref, default_fields, mono_advect_nodes

from parallel.comm import EmulatedComm
from parallel.decomp import Decomp
from parallel.sl_local import LocalSL, required_halo

pytestmark = pytest.mark.gpu


def test_triton_local(check):
    pytest.importorskip("triton", reason="triton not installed")
    from parallel.sl_triton_local import TritonLocalSL
    from slchannel.semilag_triton import TritonSL

    dev = torch.device("cuda")
    order = 4
    ref, grid = build_ref(nx=32, ny=32, nz=16, order=order, interp_dtype="fp32_accum64")
    # move the reference advector to the GPU by rebuilding it there
    from slchannel.semilag import SLAdvector

    ref = SLAdvector(
        grid["nx"],
        grid["ny"],
        grid["nz"],
        grid["dx"],
        grid["dy"],
        grid["Lx"],
        grid["Ly"],
        grid["Lz"],
        grid["z_f"].to(dev),
        grid["z_c"].to(dev),
        1.5,
        stretching_type="symmetric",
        order=order,
        interp_dtype="fp32_accum64",
        device=dev,
    )
    if ref._triton is None:
        ref._triton = TritonSL(ref)
    check("monolithic triton path enabled", ref._triton is not None)

    nodes = {c: t.to(dev) for c, t in default_fields(grid, seed=0).items()}
    mids = {c: t.to(dev) for c, t in default_fields(grid, seed=1).items()}
    dt_t = torch.as_tensor(0.4 * grid["dx"], dtype=torch.float64, device=dev)

    # monolithic Triton result, mapped to node arrays
    mono = mono_advect_nodes(ref, grid, nodes, mids, dt_t)

    H = required_halo(order, disp_cells=1.0, foot_depth_factor=1.0)
    for px, py in [(1, 1), (2, 2)]:
        d = Decomp(px, py, grid["nx"], grid["ny"], grid["nz"], H=H)
        comm = EmulatedComm(d, device=dev)
        ext = {c: d.scatter(nodes[c], c, fill_halos=True) for c in "uvw"}
        mid = {c: d.scatter(mids[c], c, fill_halos=True) for c in "uvw"}
        star = {c: {} for c in "uvw"}
        H_ = d.H
        arr = {
            "u": (slice(H_ + 1, H_ + d.nxl + 1), slice(H_, H_ + d.nyl), slice(1, d.nz + 1)),
            "v": (slice(H_, H_ + d.nxl), slice(H_ + 1, H_ + d.nyl + 1), slice(1, d.nz + 1)),
            "w": (slice(H_, H_ + d.nxl), slice(H_, H_ + d.nyl), slice(1, d.nz)),
        }
        for r in range(d.nranks):
            lsl = LocalSL(ref, d, r)
            tsl = TritonLocalSL(lsl)
            outs = tsl.advect({c: ext[c][r] for c in "uvw"}, {c: mid[c][r] for c in "uvw"}, dt_t)
            for c in "uvw":
                e = d.alloc(c, dtype=torch.float32, device=dev)
                e[arr[c]] = outs[c]
                star[c][r] = e
        comm.pull_minus_edge(star["u"], dim=0)
        comm.pull_minus_edge(star["v"], dim=1)
        umax = float(mono["u"].abs().max())
        for c in "uvw":
            dec = d.gather(star[c]).double()
            err = float((dec - mono[c].double()).abs().max())
            check(
                f"triton local {px}x{py} {c}",
                err <= 2e-6 * max(1.0, umax),
                f"max|diff|={err:.3e} (fp32 pipeline)",
            )
