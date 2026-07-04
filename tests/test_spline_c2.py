"""C^2 continuity of the spline field-remap — the property the whole
spectral-floor argument rests on (C^1 was verified in test_spline_interp;
here the SECOND derivative).

Across every kind of stencil boundary the second derivative from the left
must equal the second derivative from the right as the probe distance
shrinks:
  - a uniform x cell face (B-spline direction),
  - a nonuniform z node in the bulk of the tanh-stretched grid,
  - the wall-adjacent z interval (where the clamped end rows act).
The Lagrange remap must FAIL the same probe (kink present) — otherwise the
probe itself is too blunt to discriminate and the test is vacuous.

Method, two complementary probes:
  1. C^2 (spline): one-sided second differences just left and just right of
     the boundary; the jump for a C^2 interpolant is O(d) (third-derivative
     jump leaking in), so jump(d) -> 0 linearly, small vs curvature ~max|f''|.
  2. kink (Lagrange): the second difference CENTERED ON the boundary sees a
     first-derivative jump as [f']/d -> diverges as d -> 0, while for a C^2
     interpolant it converges to f''. Divergence ratio ~2 under d -> d/2
     identifies the kink; the spline must stay bounded on the SAME probe.
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
NX = NY = 24
NZ = 32


def make_adv(field_interp):
    dx, dy = Lx / NX, Ly / NY
    z_f, z_c, _, _ = generate_grid(GAMMA, NZ, Lz, stretching_type='symmetric')
    adv = SLAdvector(NX, NY, NZ, dx, dy, Lx, Ly, Lz, z_f, z_c, GAMMA,
                     field_interp=field_interp)
    return adv, z_f, z_c


def fsm(X, Y, Z):
    return (torch.sin(2 * math.pi * X / Lx + 0.3)
            * torch.cos(4 * math.pi * Y / Ly - 1.1)
            * torch.exp(torch.cos(math.pi * Z / Lz)))


def evaluate(adv, comp, x, y, z):
    if adv.field_interp == 'spline':
        spec = adv.spec[comp]
        iw = adv._build_iw_spline_impl(spec, x, y, z, False)
        return _gather_interp(adv.qbuf[comp].reshape(-1),
                              iw[0].long(), iw[1].long(), iw[2].long(),
                              iw[3], iw[4], iw[5])
    iw = adv._build_iw(adv.spec[comp], x, y, z, adv.order)
    return adv._apply_iw(adv.fbuf[comp], iw)


def make_probe(adv, comp, axis):
    base = {'x': 0.37 * Lx, 'y': 0.29 * Ly, 'z': 0.23 * Lz}

    def probe(q):
        vals = {ax: torch.full((1,), base[ax]) for ax in 'xyz'}
        vals[axis] = torch.tensor([q])
        return evaluate(adv, comp, vals['x'], vals['y'], vals['z']).item()
    return probe


def d2_jump(adv, comp, axis, q0, d):
    """|f''_right - f''_left| across the boundary point q0 along `axis`,
    centered 2nd differences one step off the boundary on each side
    (stencils never straddle q0 -> pure C^2 diagnostic, blind to C^0 kinks)."""
    probe = make_probe(adv, comp, axis)

    def second(qc):
        return (probe(qc - d) - 2.0 * probe(qc) + probe(qc + d)) / d**2

    return abs(second(q0 + 1.5 * d) - second(q0 - 1.5 * d))


def d2_straddle(adv, comp, axis, q0, d):
    """Second difference centered ON the boundary: -> f''(q0) if C^2 there,
    ~ [f']/d (divergent) across a derivative kink."""
    probe = make_probe(adv, comp, axis)
    return abs(probe(q0 - d) - 2.0 * probe(q0) + probe(q0 + d)) / d**2


def run():
    ok = True
    curv = 4.0 * (2 * math.pi / Lx) ** 2   # curvature scale of fsm, O(max|f''|)

    advS, z_f, z_c = make_adv('spline')
    advL, _, _ = make_adv('lagrange')
    for adv in (advS, advL):
        field = make_field('u', fsm, NX, NY, NZ, Lx / NX, Ly / NY, z_f, z_c)
        adv._fill(adv.fbuf['u'], 'u', field)
        if adv.field_interp == 'spline':
            adv._spline_coeffs(adv.fbuf['u'], 'u', adv.qbuf['u'])

    # boundaries: u-node x face | bulk z node | wall-adjacent z node.
    # The Lagrange-kink sanity runs only where the kink is measurable: it
    # scales as h_local^3, so on the tanh-stretched grid the near-wall z
    # kink (~1e-5 at node 2) is buried under the smooth-curvature signal —
    # the spectral-floor scatterers are the uniform x,y faces and mid-
    # channel z nodes (measured 1/d divergence x1.9 under d halving there,
    # x1.08 at the wall-adjacent node).
    zn = advS.spec['u'].znodes
    cases = [('x face', 'x', 5 * (Lx / NX), True),
             ('z node (bulk)', 'z', float(zn[NZ // 2]), True),
             ('z node (wall-adjacent)', 'z', float(zn[2]), False)]

    for name, axis, q0, kink_measurable in cases:
        # probe 1: one-sided second-derivative jump -> 0 linearly (C^2)
        d1, d2 = 2e-3, 1e-3
        jS1, jS2 = d2_jump(advS, 'u', axis, q0, d1), d2_jump(advS, 'u', axis, q0, d2)
        ok &= report(f"C^2 at {name} (spline)",
                     jS2 < 0.02 * curv and jS2 < 0.75 * jS1,
                     f"jump(d={d1:g})={jS1:.3e} jump(d={d2:g})={jS2:.3e} curv~{curv:.1f}")

        # probe 2: straddling second difference — bounded for the spline,
        # 1/d-divergent for the Lagrange kink (ratio ~2 under d halving)
        e1, e2 = 2e-4, 1e-4
        sS1, sS2 = d2_straddle(advS, 'u', axis, q0, e1), d2_straddle(advS, 'u', axis, q0, e2)
        sL1, sL2 = d2_straddle(advL, 'u', axis, q0, e1), d2_straddle(advL, 'u', axis, q0, e2)
        ok &= report(f"straddle bounded at {name} (spline)",
                     sS1 < 3 * curv and sS2 < 3 * curv,
                     f"S(d={e1:g})={sS1:.3e} S(d={e2:g})={sS2:.3e} curv~{curv:.1f}")
        if kink_measurable:
            ok &= report(f"Lagrange kink diverges at {name} (sanity)",
                         sL2 > 1.5 * sL1 and sL2 > 10 * sS2,
                         f"lagrange S={sL1:.3e}->{sL2:.3e} (x{sL2 / max(sL1, 1e-300):.2f}) "
                         f"vs spline S={sS2:.3e}")
        else:
            print(f"[skip] Lagrange kink at {name}: kink ~ h_local^3 below "
                  f"curvature signal on the stretched grid "
                  f"(S={sL1:.3e}->{sL2:.3e})")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
