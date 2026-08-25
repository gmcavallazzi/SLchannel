"""Full advect() path under uniform translation: the result must equal the
initial field evaluated at the translated positions. Checks (a) exactness for
a cubic-in-z field (within interpolation order), (b) grid convergence for a
smooth 3D field, (c) zero clamped departure points.

For (b) the shift scales with the grid (DT ~ 1/n) so every resolution samples
the SAME fractional cell offset: the pointwise interpolation-error kernel
varies with that offset (zero at nodes, max mid-cell), and under a fixed
physical shift the offset change between resolutions modulates the 2-point
ratio (measured: x12.6 instead of x16 for the Lagrange remap). The
trajectory is exact for uniform velocity, so DT does not enter the error
otherwise."""

import math

import torch
from helpers import make_field

from slchannel.semilag import SLAdvector
from slchannel.utils import generate_grid

Lx, Ly, Lz = 2 * math.pi, 2 * math.pi, 2.0
GAMMA = 1.5
U0, V0 = 0.9, -0.4
DT = 0.08


def advect_error(n, fn, dt=None):
    dt = DT if dt is None else dt
    nx = ny = n
    nz = n
    dx, dy = Lx / nx, Ly / ny
    z_f, z_c, _, _ = generate_grid(GAMMA, nz, Lz, stretching_type="symmetric")
    adv = SLAdvector(nx, ny, nz, dx, dy, Lx, Ly, Lz, z_f, z_c, GAMMA)

    fields = {c: make_field(c, fn, nx, ny, nz, dx, dy, z_f, z_c) for c in "uvw"}
    mids = {
        c: make_field(c, lambda X, Y, Z: U0 + 0 * X, nx, ny, nz, dx, dy, z_f, z_c) for c in "uvw"
    }
    mids["v"] = make_field("v", lambda X, Y, Z: V0 + 0 * X, nx, ny, nz, dx, dy, z_f, z_c)
    mids["w"] = torch.zeros_like(fields["w"])
    # trajectory velocity components live on their own grids
    mid_u = make_field("u", lambda X, Y, Z: U0 + 0 * X, nx, ny, nz, dx, dy, z_f, z_c)
    mid_v = make_field("v", lambda X, Y, Z: V0 + 0 * X, nx, ny, nz, dx, dy, z_f, z_c)
    mid_w = torch.zeros(nx + 2, ny + 2, nz + 1)

    dt_t = torch.tensor(dt)
    ustar, vstar, wstar = adv.advect(
        fields["u"], fields["v"], fields["w"], mid_u, mid_v, mid_w, dt_t
    )

    errs = []
    for comp, out in (("u", ustar), ("v", vstar), ("w", wstar)):
        xa, ya, za, shape = adv.arrival[comp]
        exact = fn(xa - dt * U0, ya - dt * V0, za).expand(shape)
        if comp == "u" or comp == "v":
            got = out[1 : nx + 1, 1 : ny + 1, 1 : nz + 1]
        else:
            got = out[1 : nx + 1, 1 : ny + 1, 1:nz]
        errs.append((got - exact).abs().max().item())
    return max(errs), adv.n_clamped_last.item()


def test_sl_solid_body(check):

    # cubic in z, constant in x,y: interpolation is exact
    err, ncl = advect_error(16, lambda X, Y, Z: 1.0 + 0.5 * Z - 0.3 * Z**2 + 0.1 * Z**3 + 0 * X)
    check("cubic-in-z exactness", err < 1e-12 and ncl == 0, f"err={err:.2e} clamped={ncl}")

    # smooth 3D field: 4th-order convergence
    def f(X, Y, Z):
        return (
            torch.sin(2 * math.pi * X / Lx)
            * torch.cos(2 * math.pi * Y / Ly)
            * torch.sin(1.2 * math.pi * Z / Lz + 0.2)
        )

    e1, _ = advect_error(24, f, dt=DT)
    e2, _ = advect_error(48, f, dt=DT / 2)  # same fractional cell offset
    check(
        "smooth-field convergence O(h^4)",
        e1 / e2 > 12.0,
        f"errors={e1:.3e},{e2:.3e} ratio={e1 / e2:.1f} (ideal 16)",
    )
