"""Robustness at the walls: a strong uniform downward/upward trajectory
velocity pushes departure points of near-wall arrival points outside the
domain. They must be clamped (counted in the diagnostic), and the advected
fields must stay finite and bounded by the initial data."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
from utils import generate_grid
from semilag import SLAdvector
from _slhelpers import make_field, report

torch.set_default_dtype(torch.float64)

Lx, Ly, Lz = 2 * math.pi, 2 * math.pi, 2.0
NX = NY = 16
NZ = 48
GAMMA = 2.0


def run():
    ok = True
    dx, dy = Lx / NX, Ly / NY
    z_f, z_c, _, _ = generate_grid(GAMMA, NZ, Lz, stretching_type='symmetric')

    def f(X, Y, Z):
        return torch.cos(2 * math.pi * X / Lx) * torch.sin(math.pi * Z / Lz) + 0 * Y

    for w0, label in [(-0.5, "toward top wall"), (0.5, "toward bottom wall")]:
        adv = SLAdvector(NX, NY, NZ, dx, dy, Lx, Ly, Lz, z_f, z_c, GAMMA)
        fields = {c: make_field(c, f, NX, NY, NZ, dx, dy, z_f, z_c) for c in 'uvw'}
        mid_u = torch.zeros(NX + 1, NY + 2, NZ + 2)
        mid_v = torch.zeros(NX + 2, NY + 1, NZ + 2)
        mid_w = torch.full((NX + 2, NY + 2, NZ + 1), w0)

        ustar, vstar, wstar = adv.advect(fields['u'], fields['v'], fields['w'],
                                         mid_u, mid_v, mid_w, torch.tensor(0.5))

        finite = all(torch.isfinite(t).all().item() for t in (ustar, vstar, wstar))
        bound = max(t.abs().max().item() for t in (ustar, vstar, wstar))
        n_clamped = adv.n_clamped_last.item()
        ok &= report(f"wall clamp {label}",
                     finite and n_clamped > 0 and bound < 1.5,
                     f"clamped={n_clamped} max|field|={bound:.3f}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
