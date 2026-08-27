"""Partition, origins, neighbors, and scatter/gather round-trips."""

import torch
from helpers_par import build_ref, default_fields

from parallel.decomp import Decomp, mono_node_view, nodes_to_mono


def test_decomp(check):
    _, grid = build_ref(nx=16, ny=8, nz=12, order=4)
    fields = default_fields(grid)

    for px, py in [(1, 1), (2, 1), (1, 2), (2, 2), (4, 2)]:
        d = Decomp(px, py, 16, 8, 12, H=2)
        # disjoint, covering partition
        seen = torch.zeros(16, 8, dtype=torch.int64)
        for r in range(d.nranks):
            i0, j0 = d.origin(r)
            seen[i0 : i0 + d.nxl, j0 : j0 + d.nyl] += 1
        check(f"partition {px}x{py}", bool((seen == 1).all()), f"counts {seen.unique().tolist()}")

        # neighbor map is a consistent periodic torus
        ok = all(
            d.neighbors(d.neighbors(r)["xp"])["xm"] == r
            and d.neighbors(d.neighbors(r)["yp"])["ym"] == r
            for r in range(d.nranks)
        )
        check(f"neighbors {px}x{py}", ok)

        # scatter -> gather round trip, all components
        for comp in "uvw":
            locs = d.scatter(fields[comp], comp, fill_halos=True)
            back = d.gather(locs)
            err = float((back - fields[comp]).abs().max())
            check(f"roundtrip {comp} {px}x{py}", err == 0.0, f"max|diff|={err:g}")

        # halo content of the scatter matches the periodic global field
        r = d.nranks - 1
        i0, j0 = d.origin(r)
        gx = torch.arange(i0 - d.H, i0 + d.nxl + d.H) % 16
        gy = torch.arange(j0 - d.H, j0 + d.nyl + d.H) % 8
        expect = fields["w"][gx][:, gy]
        err = float((d.scatter(fields["w"], "w")[r] - expect).abs().max())
        check(f"scatter halos {px}x{py}", err == 0.0, f"max|diff|={err:g}")

    # node view <-> mono ghost assembly round trip
    nodes = fields["u"]
    mono = nodes_to_mono(nodes, "u", 16, 8)
    err = float((mono_node_view(mono, "u", 16, 8) - nodes).abs().max())
    check("nodes<->mono u", err == 0.0, f"max|diff|={err:g}")
    # ghost consistency: u[0] == u[nx] and y wrap
    check("mono u ghosts", bool((mono[0] == mono[16]).all() and (mono[:, 0] == mono[:, 8]).all()))
