"""Emulated halo exchange fills every halo cell (corners included) with the
periodic extension of the global field, for all components and rank grids."""

from helpers_par import build_ref, default_fields

from parallel.comm import EmulatedComm
from parallel.decomp import Decomp


def test_halo_exchange(check):
    _, grid = build_ref(nx=16, ny=12, nz=10, order=4)
    fields = default_fields(grid)

    for px, py in [(2, 2), (1, 4), (4, 1), (2, 3)]:
        if 12 % py or 16 % px:
            continue
        d = Decomp(px, py, 16, 12, 10, H=3)
        comm = EmulatedComm(d)
        for comp in "uvw":
            locs = d.scatter(fields[comp], comp, fill_halos=False)
            comm.halo_exchange(locs)
            expect = d.scatter(fields[comp], comp, fill_halos=True)
            err = max(float((locs[r] - expect[r]).abs().max()) for r in locs)
            check(f"exchange {comp} {px}x{py}", err == 0.0, f"max|diff|={err:g}")

    # pull_minus_edge: owned leading plane receives the minus neighbor's
    # trailing halo slab content
    d = Decomp(2, 1, 16, 12, 10, H=3)
    comm = EmulatedComm(d)
    locs = d.scatter(fields["u"], "u", fill_halos=True)
    # poison the trailing halo slot of each rank with a marker, pull, verify
    for r in locs:
        locs[r][d.nxl + d.H, :, :] = 100.0 + r
    comm.pull_minus_edge(locs, dim=0)
    ok = all(bool((locs[r][d.H] == 100.0 + d.neighbors(r)["xm"]).all()) for r in locs)
    check("pull_minus_edge x", ok)
