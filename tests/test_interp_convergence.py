"""Spatial convergence of the high-order tensor-product Lagrange interpolation:
O(h^4) for tricubic (order=4) and O(h^6) for triquintic (order=6), measured on
the u-component node grid (uniform x,y + tanh-stretched z with nonuniform-node
weights, one-sided stencils at the walls)."""

import math

import torch

from slchannel.semilag import SLAdvector
from slchannel.utils import generate_grid

Lx, Ly, Lz = 2 * math.pi, 2 * math.pi, 2.0
GAMMA = 1.8


def f(x, y, z):
    return (
        torch.sin(2 * math.pi * x / Lx + 0.3)
        * torch.cos(4 * math.pi * y / Ly - 1.1)
        * torch.sin(1.5 * math.pi * z / Lz + 0.4)
    )


def max_interp_error(n, order):
    nx = ny = nz = n
    dx, dy = Lx / nx, Ly / ny
    z_f, z_c, _, _ = generate_grid(GAMMA, nz, Lz, stretching_type="symmetric")
    adv = SLAdvector(nx, ny, nz, dx, dy, Lx, Ly, Lz, z_f, z_c, GAMMA, order=order)
    spec = adv.spec["u"]

    # node values on the u grid (x faces, y centers, z centers incl. ghosts)
    xg = (torch.arange(nx, dtype=torch.float64) * dx).view(-1, 1, 1)
    yg = ((torch.arange(ny, dtype=torch.float64) + 0.5) * dy).view(1, -1, 1)
    zg = z_c.view(1, 1, -1)
    buf = f(xg, yg, zg) * torch.ones(nx, ny, nz + 2)

    # fixed query set (same physical points at every resolution)
    torch.manual_seed(1234)
    npts = 20000
    xq = torch.rand(npts) * Lx
    yq = torch.rand(npts) * Ly
    zq = adv.z_lo + torch.rand(npts) * (adv.z_hi - adv.z_lo)  # includes near-wall

    iw = adv._build_iw(spec, xq, yq, zq, order)
    vals = adv._apply_iw(buf, iw)
    return (vals - f(xq, yq, zq)).abs().max().item()


def test_interp_convergence(check):
    for order, min_ratio in [(4, 10.0), (6, 40.0)]:
        errs = [max_interp_error(n, order) for n in (32, 64, 128)]
        r1, r2 = errs[0] / errs[1], errs[1] / errs[2]
        ideal = 2**order
        check(
            f"interp order={order}",
            r1 > min_ratio and r2 > min_ratio,
            f"errors={[f'{e:.3e}' for e in errs]} ratios={r1:.1f},{r2:.1f} (ideal {ideal})",
        )
