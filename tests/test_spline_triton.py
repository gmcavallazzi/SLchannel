"""Triton spline gather vs the eager fp64 spline path (CUDA only; [SKIP] on
CPU-only hosts). The Triton path is fp32 end-to-end, so agreement is checked
at fp32 tolerance: positions good to ~1e-7*L and one 4x4x4 weighted sum give
~1e-5 relative error on O(1) fields.

Run on the GB10 with CC=gcc PYTORCH_JIT=0.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
from utils import generate_grid
from semilag import SLAdvector
from _slhelpers import make_field, report

torch.set_default_dtype(torch.float64)

Lx, Ly, Lz = 2 * math.pi, 2 * math.pi, 2.0
GAMMA = 1.6
NX = NY = 96
NZ = 64
DT = 0.05


def build(field_interp, interp_dtype, device):
    dx, dy = Lx / NX, Ly / NY
    z_f, z_c, _, _ = generate_grid(GAMMA, NZ, Lz, stretching_type='symmetric')
    adv = SLAdvector(NX, NY, NZ, dx, dy, Lx, Ly, Lz, z_f.to(device), z_c.to(device),
                     GAMMA, field_interp=field_interp, interp_dtype=interp_dtype,
                     device=device)
    return adv, z_f, z_c, dx, dy


def fields_on(z_f, z_c, dx, dy, device):
    def f(X, Y, Z):
        return (torch.sin(2 * math.pi * X / Lx + 0.3)
                * torch.cos(4 * math.pi * Y / Ly - 1.1)
                * torch.exp(torch.cos(math.pi * Z / Lz)))

    def g(X, Y, Z):
        return torch.cos(4 * math.pi * X / Lx) * torch.sin(2 * math.pi * Y / Ly) + 0.2 * Z

    flds, mids, rhs = {}, {}, {}
    for c in 'uvw':
        flds[c] = make_field(c, f, NX, NY, NZ, dx, dy, z_f, z_c).to(device)
        rhs[c] = make_field(c, g, NX, NY, NZ, dx, dy, z_f, z_c).to(device)
    mids['u'] = make_field('u', lambda X, Y, Z: 0.7 + 0.2 * torch.sin(2 * math.pi * Y / Ly) + 0 * X,
                           NX, NY, NZ, dx, dy, z_f, z_c).to(device)
    mids['v'] = make_field('v', lambda X, Y, Z: -0.3 + 0.1 * torch.cos(2 * math.pi * X / Lx) + 0 * Y,
                           NX, NY, NZ, dx, dy, z_f, z_c).to(device)
    mids['w'] = make_field('w', lambda X, Y, Z: 0.15 * torch.sin(math.pi * Z / Lz) + 0 * X,
                           NX, NY, NZ, dx, dy, z_f, z_c).to(device)
    return flds, mids, rhs


def advect(adv, flds, mids, rhs):
    dt_t = torch.tensor(DT, device=adv.device)
    us, vs, ws, extras = None, None, None, None
    out = adv.advect(flds['u'], flds['v'], flds['w'],
                     mids['u'], mids['v'], mids['w'], dt_t,
                     extra_rhs=[(rhs['u'], rhs['v'], rhs['w'])])
    us, vs, ws, extras = out
    return ([t.double().cpu() for t in (us, vs, ws)],
            [t.double().cpu() for t in extras[0]],
            int(adv.n_clamped_last))


def run():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; Triton spline path untestable here")
        return True
    dev = torch.device('cuda')
    ok = True

    # reference: eager fp64 spline on CPU (bit-exact reference path)
    advR, z_f, z_c, dx, dy = build('spline', 'fp64', torch.device('cpu'))
    fR, mR, rR = fields_on(z_f, z_c, dx, dy, torch.device('cpu'))
    ref, ref_ex, ncl_ref = advect(advR, fR, mR, rR)

    # Triton fp32 spline on CUDA
    advT, *_ = build('spline', 'fp32_accum64', dev)
    assert advT._triton is not None, "Triton spline path did not activate"
    fT, mT, rT = fields_on(z_f, z_c, dx, dy, dev)
    got, got_ex, ncl_got = advect(advT, fT, mT, rT)

    for name, a, b in zip('uvw', ref, got):
        scale = a.abs().max().item()
        err = (a - b).abs().max().item() / scale
        ok &= report(f"triton spline vs fp64 eager ({name})", err < 5e-5,
                     f"rel err={err:.2e}")
    for name, a, b in zip('uvw', ref_ex, got_ex):
        scale = a.abs().max().item()
        err = (a - b).abs().max().item() / scale
        ok &= report(f"triton spline extra RHS ({name})", err < 5e-5,
                     f"rel err={err:.2e}")
    ok &= report("clamp count agrees", ncl_got == ncl_ref,
                 f"triton={ncl_got} ref={ncl_ref}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
