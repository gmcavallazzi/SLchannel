"""Departure-point accuracy: exact for uniform and linear-shear velocity
fields (trilinear sampling and the midpoint rule are exact there), and
O(dt^2) global / O(dt^3) local error for a rotating velocity field."""

import math

import torch
from helpers import make_field

from slchannel.semilag import SLAdvector
from slchannel.utils import generate_grid

Lx, Ly, Lz = 2 * math.pi, 2 * math.pi, 2.0
NX = NY = 16
NZ = 32
GAMMA = 1.5


def make_adv(n_traj_iters=2):
    dx, dy = Lx / NX, Ly / NY
    z_f, z_c, _, _ = generate_grid(GAMMA, NZ, Lz, stretching_type="symmetric")
    adv = SLAdvector(NX, NY, NZ, dx, dy, Lx, Ly, Lz, z_f, z_c, GAMMA, n_traj_iters=n_traj_iters)
    return adv, z_f, z_c, dx, dy


def fill_mid(adv, z_f, z_c, fu, fv, fw):
    dx, dy = adv.dx, adv.dy
    adv._fill(adv.mbuf["u"], "u", make_field("u", fu, NX, NY, NZ, dx, dy, z_f, z_c))
    adv._fill(adv.mbuf["v"], "v", make_field("v", fv, NX, NY, NZ, dx, dy, z_f, z_c))
    adv._fill(adv.mbuf["w"], "w", make_field("w", fw, NX, NY, NZ, dx, dy, z_f, z_c))


def test_departure_points(check):

    # ---- uniform translation: exact ------------------------------------
    adv, z_f, z_c, dx, dy = make_adv()
    U0, V0, W0 = 0.7, -0.3, 0.02
    fill_mid(
        adv,
        z_f,
        z_c,
        lambda X, Y, Z: U0 + 0 * X,
        lambda X, Y, Z: V0 + 0 * X,
        lambda X, Y, Z: W0 + 0 * X,
    )
    dt = torch.tensor(0.05)
    for comp in "uvw":
        xa, ya, za, _ = adv.arrival[comp]
        xd, yd, zd = adv.departure_coords(comp, dt)
        err = max(
            (xd - (xa - dt * U0)).abs().max().item(),
            (yd - (ya - dt * V0)).abs().max().item(),
            (zd - (za - dt * W0)).abs().max().item(),
        )
        check(f"uniform translation, comp={comp}", err < 1e-13, f"err={err:.2e}")

    # ---- linear shear u = S*z: exact (velocity constant on trajectory) --
    S = 0.8
    fill_mid(adv, z_f, z_c, lambda X, Y, Z: S * Z, lambda X, Y, Z: 0 * X, lambda X, Y, Z: 0 * X)
    xa, ya, za, _ = adv.arrival["u"]
    xd, yd, zd = adv.departure_coords("u", dt)
    err = max(
        (xd - (xa - dt * S * za)).abs().max().item(),
        (yd - ya).abs().max().item(),
        (zd - za).abs().max().item(),
    )
    check("linear shear", err < 1e-13, f"err={err:.2e}")

    # ---- rotation in x-z: local error O(dt^3) ---------------------------
    x0, z0, omega = math.pi, 1.0, 0.3
    errs = []
    for dt_val in (0.2, 0.1, 0.05):
        adv, z_f, z_c, dx, dy = make_adv(n_traj_iters=8)  # converge the fixed point
        fill_mid(
            adv,
            z_f,
            z_c,
            lambda X, Y, Z: -omega * (Z - z0),
            lambda X, Y, Z: 0 * X,
            lambda X, Y, Z: omega * (X - x0),
        )
        dt_t = torch.tensor(dt_val)
        xa, ya, za, _ = adv.arrival["u"]
        xd, yd, zd = adv.departure_coords("u", dt_t)
        th = omega * dt_val
        xd_ex = x0 + math.cos(th) * (xa - x0) + math.sin(th) * (za - z0)
        zd_ex = z0 - math.sin(th) * (xa - x0) + math.cos(th) * (za - z0)
        # keep away from the walls (no clamping) and from the x seam: the
        # rotational w = omega*(x-x0) is not x-periodic, so stencils that wrap
        # around x=0/Lx sample a discontinuity and are not part of this test
        mask = (((za - z0).abs() < 0.7) & (xa > 1.0) & (xa < Lx - 1.0)).expand(NX, NY, NZ)
        err = torch.maximum((xd - xd_ex).abs(), (zd - zd_ex).abs())[mask].max().item()
        errs.append(err)
    r1, r2 = errs[0] / errs[1], errs[1] / errs[2]
    check(
        "rotation local error O(dt^3)",
        r1 > 6.0 and r2 > 6.0,
        f"errors={[f'{e:.3e}' for e in errs]} ratios={r1:.1f},{r2:.1f} (ideal 8)",
    )
