"""Shear advection with analytic characteristics: trajectory velocity
u = S*z (constant along each trajectory, so departure points are exact) acting
on a sinusoidal streamwise field. The advected solution is
u*(x, z) = sin(2*pi*(x - S*z*dt)/Lx); the only error is x-interpolation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
from utils import generate_grid
from semilag import SLAdvector
from _slhelpers import make_field, report, sl_field_interp

torch.set_default_dtype(torch.float64)

Lx, Ly, Lz = 2 * math.pi, 2 * math.pi, 2.0
GAMMA = 1.5
S = 0.8
DT = 0.1


def shear_error(nx, order):
    ny, nz = nx, 32
    dx, dy = Lx / nx, Ly / ny
    z_f, z_c, _, _ = generate_grid(GAMMA, nz, Lz, stretching_type='symmetric')
    adv = SLAdvector(nx, ny, nz, dx, dy, Lx, Ly, Lz, z_f, z_c, GAMMA, order=order,
                     field_interp=sl_field_interp())

    def f(X, Y, Z):
        return torch.sin(2 * math.pi * X / Lx) + 0 * Y + 0 * Z

    fields = {c: make_field(c, f, nx, ny, nz, dx, dy, z_f, z_c) for c in 'uvw'}
    mid_u = make_field('u', lambda X, Y, Z: S * Z + 0 * X, nx, ny, nz, dx, dy, z_f, z_c)
    mid_v = torch.zeros(nx + 2, ny + 1, nz + 2)
    mid_w = torch.zeros(nx + 2, ny + 2, nz + 1)

    ustar, vstar, wstar = adv.advect(fields['u'], fields['v'], fields['w'],
                                     mid_u, mid_v, mid_w, torch.tensor(DT))

    errs = []
    for comp, out in (('u', ustar), ('v', vstar), ('w', wstar)):
        xa, ya, za, shape = adv.arrival[comp]
        exact = torch.sin(2 * math.pi * (xa - S * za * DT) / Lx).expand(shape)
        if comp == 'w':
            got = out[1:nx + 1, 1:ny + 1, 1:nz]
        else:
            got = out[1:nx + 1, 1:ny + 1, 1:nz + 1]
        errs.append((got - exact).abs().max().item())
    return max(errs)


def run():
    ok = True
    e_cubic = shear_error(64, 4)
    ok &= report("shear, tricubic nx=64", e_cubic < 1e-5, f"err={e_cubic:.3e}")

    e1, e2 = shear_error(32, 4), shear_error(64, 4)
    ok &= report("shear convergence O(h^4)", e1 / e2 > 10.0,
                 f"errors={e1:.3e},{e2:.3e} ratio={e1 / e2:.1f} (ideal 16)")

    if sl_field_interp() == 'lagrange':
        e_quintic = shear_error(64, 6)
        ok &= report("shear, triquintic beats tricubic", e_quintic < e_cubic / 10,
                     f"quintic={e_quintic:.3e} cubic={e_cubic:.3e}")
    else:
        print("[skip] triquintic-vs-tricubic: order applies to the Lagrange "
              "remap only (spline mode is a single C^2 cubic interpolant)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
