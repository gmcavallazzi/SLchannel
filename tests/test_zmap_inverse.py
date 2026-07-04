"""Round-trip test of the analytic inverse tanh z-map used to locate
semi-Lagrangian departure stencils: _face_coord(z_f[k]) must equal k, and
arbitrary z built from the forward map must invert to the same continuous
face coordinate."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils import generate_grid
from semilag import SLAdvector
from _slhelpers import report, sl_field_interp

torch.set_default_dtype(torch.float64)


def forward_map(kf, gamma, nz, Lz, stretching_type):
    g = torch.tensor(gamma, dtype=torch.float64)
    if stretching_type == 'symmetric':
        xi = 2.0 * kf / nz - 1.0
        return 0.5 * Lz * (1 + torch.tanh(g * xi) / torch.tanh(g))
    else:
        xi = kf / nz
        return Lz * (1.0 - torch.tanh(g * (1.0 - xi)) / torch.tanh(g))


def run():
    ok = True
    Lz = 2.0
    for stretching in ['symmetric', 'bottom']:
        for gamma in [0.8, 1.8, 2.6]:
            for nz in [64, 180]:
                z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, stretching_type=stretching)
                adv = SLAdvector(4, 4, nz, 0.5, 0.5, 2.0, 2.0, Lz, z_f, z_c,
                                 gamma, stretching_type=stretching,
                                 field_interp=sl_field_interp())

                # faces must invert to integer indices
                kf = adv._face_coord(z_f)
                err_faces = (kf - torch.arange(nz + 1, dtype=torch.float64)).abs().max().item()

                # random interior coordinates round-trip through the forward map
                torch.manual_seed(0)
                kf_rand = torch.rand(5000, dtype=torch.float64) * nz
                z_rand = forward_map(kf_rand, gamma, nz, Lz, stretching)
                err_rand = (adv._face_coord(z_rand) - kf_rand).abs().max().item()

                ok &= report(f"zmap {stretching} gamma={gamma} nz={nz}",
                             err_faces < 1e-8 and err_rand < 1e-8,
                             f"face_err={err_faces:.2e} rand_err={err_rand:.2e}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
