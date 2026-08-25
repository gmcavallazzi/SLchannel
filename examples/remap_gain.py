"""Pure-remap spectral gain experiment (CPU, no solver).

Uniform translation preserves spectra exactly, so under repeated advect()
with proper ghost refresh any change in the high-k tail is the interpolant's
own per-step gain. Result: the remap is strictly CONTRACTIVE — per-step tail
gain 0.86-0.91 for tricubic, higher (less dissipative) for triquintic. The
high-k spectral floor therefore CANNOT come from the remap alone; it was
traced to the trajectory-velocity extrapolation. See the report, section 4.2a.

Pitfalls this script encodes (both produced fake 'injection' first):
  - ghosts/periodic images MUST be refreshed after every advect (the solver
    does this in apply_bc_uvw; a bare advect() loop does not);
  - a frozen NON-uniform velocity legitimately cascades energy to fine
    scales (differential straining) — only uniform translation isolates
    the numerical gain.

Usage: python examples/remap_gain.py  (from the repo root; CPU, ~2 min)
"""

import math

import torch

from slchannel.semilag import SLAdvector
from slchannel.utils import generate_grid

torch.set_default_dtype(torch.float64)


def make_field(comp, fn, nx, ny, nz, dx, dy, z_f, z_c):
    """Ghost-shaped staggered field for `comp`, filled from an analytic fn.

    Evaluated at every entry's physical position, ghosts included, so no BC
    machinery is needed to set up an interpolation experiment.
    """
    if comp == "u":
        X = (torch.arange(0, nx + 1, dtype=torch.float64) * dx).view(-1, 1, 1)
        Y = ((torch.arange(0, ny + 2, dtype=torch.float64) - 0.5) * dy).view(1, -1, 1)
        Z = z_c.view(1, 1, -1)
    elif comp == "v":
        X = ((torch.arange(0, nx + 2, dtype=torch.float64) - 0.5) * dx).view(-1, 1, 1)
        Y = (torch.arange(0, ny + 1, dtype=torch.float64) * dy).view(1, -1, 1)
        Z = z_c.view(1, 1, -1)
    elif comp == "w":
        X = ((torch.arange(0, nx + 2, dtype=torch.float64) - 0.5) * dx).view(-1, 1, 1)
        Y = ((torch.arange(0, ny + 2, dtype=torch.float64) - 0.5) * dy).view(1, -1, 1)
        Z = z_f.view(1, 1, -1)
    else:
        raise ValueError(comp)
    shape = {
        "u": (nx + 1, ny + 2, nz + 2),
        "v": (nx + 2, ny + 1, nz + 2),
        "w": (nx + 2, ny + 2, nz + 1),
    }[comp]
    return fn(X, Y, Z) * torch.ones(shape, dtype=torch.float64)


NX, NY, NZ = 128, 32, 32
Lx, Ly, Lz = 2 * math.pi, math.pi, 2.0
dx, dy = Lx / NX, Ly / NY
NSTEPS = 300


def refresh(u, comp):
    """Periodic x image/ghosts + periodic y ghosts + constant-in-z ghosts
    (exact for the z-independent test field)."""
    if comp == "u":
        u[0] = u[NX]
    else:
        u[0] = u[NX]
        u[NX + 1] = u[1]
    if comp == "v":
        u[:, 0] = u[:, NY]
    else:
        u[:, 0] = u[:, NY]
        u[:, NY + 1] = u[:, 1]
    u[..., 0] = u[..., 1]
    u[..., -1] = u[..., -2]


def experiment(order, a_frac, z_f, z_c):
    adv = SLAdvector(NX, NY, NZ, dx, dy, Lx, Ly, Lz, z_f, z_c, 1.5, order=order)
    U0 = a_frac * dx
    mid_u = make_field("u", lambda X, Y, Z: U0 + 0 * X, NX, NY, NZ, dx, dy, z_f, z_c)
    mid_v = torch.zeros(NX + 2, NY + 1, NZ + 2)
    mid_w = torch.zeros(NX + 2, NY + 2, NZ + 1)

    def f0(X, Y, Z):
        return torch.sin(2 * math.pi * X / Lx) * torch.cos(2 * math.pi * Y / Ly) + 0 * Z

    flds = {c: make_field(c, f0, NX, NY, NZ, dx, dy, z_f, z_c) for c in "uvw"}
    for c in "uvw":
        flds[c] += 1e-6 * torch.randn_like(flds[c])
        refresh(flds[c], c)
    dt_t = torch.tensor(1.0)

    def tail(u):
        sl = u[1 : NX + 1, 1 : NY + 1, NZ // 2]
        E = (torch.fft.rfft(sl, dim=0).abs() ** 2).mean(dim=1)
        return E[-10:].mean().item()

    t0 = tail(flds["u"])
    for _ in range(NSTEPS):
        us, vs, ws = adv.advect(flds["u"], flds["v"], flds["w"], mid_u, mid_v, mid_w, dt_t)
        for c, s in zip("uvw", (us, vs, ws)):
            flds[c] = s.clone()
            refresh(flds[c], c)
    return t0, tail(flds["u"])


def main():
    torch.manual_seed(7)
    z_f, z_c, _, _ = generate_grid(1.5, NZ, Lz, stretching_type="symmetric")
    print(f"uniform translation, {NSTEPS} remaps (gain must be <= 1):")
    for order, name in ((4, "tricubic"), (6, "triquintic")):
        for a in (0.1, 0.25, 0.5):
            a0, a1 = experiment(order, a, z_f, z_c)
            g = (a1 / a0) ** (1 / NSTEPS)
            tag = ">1 INJECTION" if g > 1.0000001 else "(contractive)"
            print(
                f"  {name:10s} a={a:4.2f}: tail {a0:.3e} -> {a1:.3e}  per-step gain {g:.6f} {tag}"
            )


if __name__ == "__main__":
    main()
