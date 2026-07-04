"""C^2 spline field-remap (sl.field_interp: spline) — correctness.

Checks, per staggered component grid:
  1. interpolation property: the spline passes through the node values
     (B-spline prefilter inverts the [1/6,4/6,1/6] kernel; Hermite hits c_k);
  2. z-cubic exactness: cubic-in-z data reproduced to roundoff at arbitrary
     points (Lagrange-cubic-clamped ends make the spline cubic-exact);
  3. O(h^4) convergence on a smooth trig field under grid doubling;
  4. C^1 across x cell faces: one-sided slope mismatch vanishes with delta
     (the Lagrange kink does not).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
from utils import generate_grid
from semilag import SLAdvector, _gather_interp
from _slhelpers import make_field, report

torch.set_default_dtype(torch.float64)

Lx, Ly, Lz = 2 * math.pi, 2 * math.pi, 2.0
GAMMA = 1.6


def make_adv(nx, ny, nz, field_interp='spline'):
    dx, dy = Lx / nx, Ly / ny
    z_f, z_c, _, _ = generate_grid(GAMMA, nz, Lz, stretching_type='symmetric')
    adv = SLAdvector(nx, ny, nz, dx, dy, Lx, Ly, Lz, z_f, z_c, GAMMA,
                     field_interp=field_interp)
    return adv, z_f, z_c


def eval_spline(adv, comp, x, y, z):
    spec = adv.spec[comp]
    iw = adv._build_iw_spline_impl(spec, x, y, z, False)
    return _gather_interp(adv.qbuf[comp].reshape(-1),
                          iw[0].long(), iw[1].long(), iw[2].long(),
                          iw[3], iw[4], iw[5])


def fill_and_coeff(adv, comp, fn, nx, ny, nz, z_f, z_c):
    field = make_field(comp, fn, nx, ny, nz, Lx / nx, Ly / ny, z_f, z_c)
    adv._fill(adv.fbuf[comp], comp, field)
    adv._spline_coeffs(adv.fbuf[comp], comp, adv.qbuf[comp])


def node_points(adv, comp, nx, ny):
    """Physical coordinates of a subset of interior nodes of `comp`."""
    spec = adv.spec[comp]
    ii = torch.arange(0, nx, 3)
    jj = torch.arange(0, ny, 3)
    kk = torch.arange(2, spec.NZ - 2, 2)
    I, J, K = torch.meshgrid(ii, jj, kk, indexing='ij')
    x = (I.reshape(-1).double() - spec.shift_x) * adv.dx
    y = (J.reshape(-1).double() - spec.shift_y) * adv.dy
    z = spec.znodes[K.reshape(-1)]
    vals = adv.fbuf[comp][I.reshape(-1), J.reshape(-1), K.reshape(-1)]
    return x, y, z, vals


def run():
    ok = True
    torch.manual_seed(3)

    # ---- 1. interpolation property on random data -------------------------
    nx = ny = 24; nz = 32
    adv, z_f, z_c = make_adv(nx, ny, nz)
    for comp in 'uvw':
        adv._fill(adv.fbuf[comp], comp,
                  torch.randn({'u': (nx + 1, ny + 2, nz + 2),
                               'v': (nx + 2, ny + 1, nz + 2),
                               'w': (nx + 2, ny + 2, nz + 1)}[comp]))
        adv._spline_coeffs(adv.fbuf[comp], comp, adv.qbuf[comp])
        x, y, z, ref = node_points(adv, comp, nx, ny)
        vals = eval_spline(adv, comp, x, y, z)
        err = (vals - ref).abs().max().item()
        ok &= report(f"interpolation property ({comp})", err < 1e-11, f"err={err:.2e}")

    # ---- 2. cubic-in-z exactness ------------------------------------------
    def fcub(X, Y, Z):
        return 2.0 + 0.5 * Z - 0.3 * Z**2 + 0.1 * Z**3 + 0.0 * X + 0.0 * Y

    for comp in 'uvw':
        fill_and_coeff(adv, comp, fcub, nx, ny, nz, z_f, z_c)
        N = 4000
        x = torch.rand(N) * 3 * Lx - Lx           # unwrapped x/y allowed
        y = torch.rand(N) * 3 * Ly - Ly
        z = adv.z_lo + torch.rand(N) * (adv.z_hi - adv.z_lo)
        vals = eval_spline(adv, comp, x, y, z)
        ref = 2.0 + 0.5 * z - 0.3 * z**2 + 0.1 * z**3
        err = (vals - ref).abs().max().item()
        ok &= report(f"cubic-in-z exactness ({comp})", err < 1e-11, f"err={err:.2e}")

    # ---- 3. O(h^4) self-convergence on a smooth field ----------------------
    def fsm(X, Y, Z):
        return (torch.sin(2 * math.pi * X / Lx) * torch.cos(4 * math.pi * Y / Ly)
                * torch.exp(torch.cos(math.pi * Z / Lz)))

    errs = []
    for n in (24, 48):
        a2, zf2, zc2 = make_adv(n, n, n)
        fill_and_coeff(a2, 'u', fsm, n, n, n, zf2, zc2)
        N = 6000
        x = torch.rand(N) * Lx
        y = torch.rand(N) * Ly
        z = a2.z_lo + torch.rand(N) * (a2.z_hi - a2.z_lo)
        vals = eval_spline(a2, 'u', x, y, z)
        ref = fsm(x, y, z)
        errs.append((vals - ref).abs().max().item())
    ratio = errs[0] / errs[1]
    ok &= report("O(h^4) convergence", ratio > 10.0,
                 f"errors={errs[0]:.3e},{errs[1]:.3e} ratio={ratio:.1f} (ideal 16)")

    # ---- 4. C^1 across an x cell face (spline yes, Lagrange no) -----------
    def slope_mismatch(a, comp, x0):
        y = torch.full((1,), 0.37 * Ly)
        z = torch.full((1,), 0.31 * Lz)
        d = 1e-6
        out = []
        for xq in (x0 - d, x0, x0 + d):
            if a.field_interp == 'spline':
                v = eval_spline(a, comp, torch.tensor([xq]), y, z)
            else:
                iw = a._build_iw(a.spec[comp], torch.tensor([xq]), y, z, a.order)
                v = a._apply_iw(a.fbuf[comp], iw)
            out.append(v.item())
        sl_left = (out[1] - out[0]) / d
        sl_right = (out[2] - out[1]) / d
        return abs(sl_right - sl_left)

    x_face = 5 * (Lx / nx)   # a u-node location = cell-face of the u stencil
    fill_and_coeff(adv, 'u', fsm, nx, ny, nz, z_f, z_c)
    mm_spline = slope_mismatch(adv, 'u', x_face)

    advL, _, _ = make_adv(nx, ny, nz, field_interp='lagrange')
    fieldL = make_field('u', fsm, nx, ny, nz, Lx / nx, Ly / ny, z_f, z_c)
    advL._fill(advL.fbuf['u'], 'u', fieldL)
    mm_lagr = slope_mismatch(advL, 'u', x_face)

    ok &= report("C^1 in x (spline)", mm_spline < 1e-4,
                 f"slope jump: spline={mm_spline:.2e} vs lagrange={mm_lagr:.2e}")
    ok &= report("Lagrange kink present (sanity)", mm_lagr > 10 * max(mm_spline, 1e-12),
                 f"kink={mm_lagr:.2e}")

    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
