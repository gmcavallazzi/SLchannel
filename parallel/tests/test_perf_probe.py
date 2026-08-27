"""Informational probe: halo bytes per exchange and the exchange/advect time
ratio for the emulated backend. Numbers are recorded via check() details;
they are NOT performance claims (single device, emulated ranks)."""

import time

import torch
from helpers_par import build_ref, default_fields

from parallel.comm import EmulatedComm
from parallel.decomp import Decomp
from parallel.sl_local import LocalSL, required_halo


def test_perf_probe(check):
    order = 6
    ref, grid = build_ref(nx=48, ny=48, nz=32, order=order)
    H = required_halo(order, disp_cells=1.0, foot_depth_factor=1.0)
    d = Decomp(2, 2, 48, 48, 32, H=H)
    comm = EmulatedComm(d)
    nodes = default_fields(grid, seed=0)
    mids = default_fields(grid, seed=1)
    ext = {c: d.scatter(nodes[c], c, fill_halos=True) for c in "uvw"}
    mid = {c: d.scatter(mids[c], c, fill_halos=True) for c in "uvw"}
    sl = {r: LocalSL(ref, d, r) for r in range(d.nranks)}
    dt_t = torch.as_tensor(0.4 * grid["dx"], dtype=torch.float64)

    # halo bytes for one full u/v/w exchange (both directions, both axes)
    itemsize = 8
    bytes_per_field = 2 * H * (d.nyl + 2 * H) + 2 * H * (d.nxl + 2 * H)
    halo_bytes = sum(bytes_per_field * ext[c][0].shape[2] * itemsize * d.nranks for c in "uvw")

    t0 = time.perf_counter()
    for _ in range(3):
        for c in "uvw":
            comm.halo_exchange(ext[c])
    t_ex = (time.perf_counter() - t0) / 3

    t0 = time.perf_counter()
    for _ in range(3):
        for r in range(d.nranks):
            sl[r].advect({c: ext[c][r] for c in "uvw"}, {c: mid[c][r] for c in "uvw"}, dt_t)
    t_adv = (time.perf_counter() - t0) / 3

    check(
        "halo bytes per exchange",
        halo_bytes > 0,
        f"{halo_bytes / 1e6:.2f} MB total across {d.nranks} ranks (H={H})",
    )
    check(
        "exchange/advect time share",
        t_adv > 0,
        f"exchange {t_ex * 1e3:.1f} ms vs advect {t_adv * 1e3:.1f} ms "
        f"({100 * t_ex / (t_ex + t_adv):.1f}% of combined; emulated CPU, informational)",
    )
