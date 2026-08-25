"""Triton semi-Lagrangian gather kernels vs the eager reference.

`semilag_triton.TritonSL` is the production interpolation path: it keeps the
3(P+1) Lagrange weights register-resident instead of materialising them. It
must reproduce the eager `advect()` to fp32-interpolation accuracy.

The first check is the important one. The fast path is selected inside a
`try/except` that falls back to eager with a printed message, so a broken
Triton path does not fail anything — it just runs an order of magnitude
slower and silently. That failure mode is exactly what this test exists to
catch, so assert the path is *enabled* before comparing against it.
"""

import pytest
import torch
from helpers import make_field

from slchannel.semilag import SLAdvector
from slchannel.utils import generate_grid

pytestmark = pytest.mark.gpu

NX = NY = 32
NZ = 48
LX = LY = 6.283185307179586
LZ = 2.0
GAMMA = 1.5


def build(order, use_triton):
    dx, dy = LX / NX, LY / NY
    # the advector's grid tensors must live on the compute device, as they
    # do when SLChannelFlow builds it
    z_f, z_c, _, _ = generate_grid(GAMMA, NZ, LZ, device="cuda", stretching_type="symmetric")
    adv = SLAdvector(
        NX,
        NY,
        NZ,
        dx,
        dy,
        LX,
        LY,
        LZ,
        z_f,
        z_c,
        GAMMA,
        order=order,
        traj_order=2,
        interp_dtype="fp32_accum64" if use_triton else "fp64",
        device=torch.device("cuda"),
    )
    if not use_triton:
        adv._triton = None
    return adv, dx, dy, z_f, z_c


def smooth_fields(dx, dy, z_f, z_c):
    import math

    def f(X, Y, Z):
        return torch.sin(2 * math.pi * X / LX) * torch.cos(2 * math.pi * Y / LY) * (1.0 + 0.3 * Z)

    z_f, z_c = z_f.cpu(), z_c.cpu()
    u = make_field("u", f, NX, NY, NZ, dx, dy, z_f, z_c)
    v = make_field("v", f, NX, NY, NZ, dx, dy, z_f, z_c)
    w = make_field("w", lambda X, Y, Z: 0.0 * X + 0.0 * Y + 0.0 * Z, NX, NY, NZ, dx, dy, z_f, z_c)
    return u.cuda(), v.cuda(), w.cuda()


@pytest.mark.parametrize("order", [4, 6])
def test_semilag_triton(check, order):
    pytest.importorskip("triton", reason="Triton fast path needs triton")

    adv_t, dx, dy, z_f, z_c = build(order, use_triton=True)
    check(
        f"Triton fast path enabled (order={order})",
        adv_t._triton is not None,
        "if this fails the solver silently runs the slow eager path",
    )
    if adv_t._triton is None:
        return

    adv_e, *_ = build(order, use_triton=False)
    u, v, w = smooth_fields(dx, dy, z_f, z_c)
    dt = 0.05

    ut, vt, wt = adv_t.advect(u.clone(), v.clone(), w.clone(), u.clone(), v.clone(), w.clone(), dt)
    ue, ve, we = adv_e.advect(u.clone(), v.clone(), w.clone(), u.clone(), v.clone(), w.clone(), dt)

    for name, a, b in (("u", ut, ue), ("v", vt, ve), ("w", wt, we)):
        scale = b.abs().max().item() or 1.0
        err = (a.double() - b.double()).abs().max().item() / scale
        # fp32 interpolation against an fp64 reference: ~1e-6 is the floor
        check(f"Triton vs eager {name} (order={order})", err < 5e-6, f"rel max err = {err:.2e}")

    check(
        f"no departure points clamped (order={order})",
        adv_t.n_clamped_last == 0,
        f"clamped={adv_t.n_clamped_last}",
    )
