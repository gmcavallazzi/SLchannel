"""
semilag_triton.py — hand-written Triton kernels for the semi-Lagrangian fast
path (fp32 pipeline, CUDA only).

Why these exist: the interpolation math is pointwise-per-output-element, but
torch.compile/Inductor materializes the (N, order) stencil weight/index
tensors (they are multi-use intermediates), which costs ~10 GB of memory
traffic per interpolated field at production size. Here the whole pipeline —
coordinate -> stencil -> weights -> 64-point gather — lives in registers;
the only global traffic is the departure coordinates, the field, and the
output. fp32 throughout (fp32 flops are ~free on the GB10; fp64 is 1/64).

Kernels:
  _departure_kernel : arrival coords (from the flat index), n_traj_iters
                      midpoint iterations with inline trilinear sampling of
                      the three staggered mid-velocity buffers, writes
                      (xd, yd, zd) and atomically counts wall-clamped points.
  _gather_kernel    : locates the stencil (analytic inverse tanh z-map +
                      one node compare), computes Lagrange weights inline
                      (uniform x/y closed form, nonuniform z from the actual
                      nodes + precomputed inverse denominators), accumulates
                      the ORDER^3 gather.

Grid conventions identical to semilag.SLAdvector (see its docstring).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _z_locate(
    z,
    znodes,
    Lz,
    gamma,
    tanh_g,
    nzf,
    IS_CENTERS: tl.constexpr,
    IS_SYMMETRIC: tl.constexpr,
    NZN: tl.constexpr,
    ORDER: tl.constexpr,
):
    """Base z-node index k0 for the ORDER-point stencil containing z."""
    if IS_SYMMETRIC:
        arg = (2.0 * z / Lz - 1.0) * tanh_g
    else:
        arg = (1.0 - z / Lz) * tanh_g
    arg = tl.minimum(tl.maximum(arg, -0.999999), 0.999999)
    xi = 0.5 * tl.log((1.0 + arg) / (1.0 - arg)) / gamma  # atanh
    if IS_SYMMETRIC:
        kf = (xi + 1.0) * 0.5 * nzf
    else:
        kf = (1.0 - xi) * nzf  # xi here is atanh(...)/gamma of (1 - z/Lz)
    c = tl.math.floor(kf).to(tl.int32)
    c = tl.minimum(tl.maximum(c, 0), NZN - 2)  # face-interval index
    if IS_CENTERS:
        zc1 = tl.load(znodes + c + 1)
        m0 = c + (z >= zc1).to(tl.int32)
    else:
        m0 = c
    k0 = tl.minimum(tl.maximum(m0 - (ORDER // 2 - 1), 0), NZN - ORDER)
    return k0


@triton.jit
def _wrap(i, N: tl.constexpr):
    r = i % N
    return tl.where(r < 0, r + N, r)


@triton.jit
def _trilinear(
    F,
    x,
    y,
    z,
    znodes,
    inv_dx,
    shift_x,
    inv_dy,
    shift_y,
    Lz,
    gamma,
    tanh_g,
    nzf,
    z_lo,
    z_hi,
    NX: tl.constexpr,
    NY: tl.constexpr,
    NZ: tl.constexpr,
    IS_CENTERS: tl.constexpr,
    IS_SYMMETRIC: tl.constexpr,
):
    """Trilinear sample of one node buffer at arbitrary points (registers only)."""
    z = tl.minimum(tl.maximum(z, z_lo), z_hi)
    sx = x * inv_dx + shift_x
    ix0 = tl.math.floor(sx).to(tl.int32)
    txf = sx - ix0
    sy = y * inv_dy + shift_y
    iy0 = tl.math.floor(sy).to(tl.int32)
    tyf = sy - iy0
    k0 = _z_locate(z, znodes, Lz, gamma, tanh_g, nzf, IS_CENTERS, IS_SYMMETRIC, NZ, 2)
    z0 = tl.load(znodes + k0)
    z1 = tl.load(znodes + k0 + 1)
    tz = (z - z0) / (z1 - z0)

    ix_a = _wrap(ix0, NX) * (NY * NZ)
    ix_b = _wrap(ix0 + 1, NX) * (NY * NZ)
    iy_a = _wrap(iy0, NY) * NZ
    iy_b = _wrap(iy0 + 1, NY) * NZ

    acc = (1 - txf) * (
        (1 - tyf)
        * ((1 - tz) * tl.load(F + ix_a + iy_a + k0) + tz * tl.load(F + ix_a + iy_a + k0 + 1))
        + tyf * ((1 - tz) * tl.load(F + ix_a + iy_b + k0) + tz * tl.load(F + ix_a + iy_b + k0 + 1))
    ) + txf * (
        (1 - tyf)
        * ((1 - tz) * tl.load(F + ix_b + iy_a + k0) + tz * tl.load(F + ix_b + iy_a + k0 + 1))
        + tyf * ((1 - tz) * tl.load(F + ix_b + iy_b + k0) + tz * tl.load(F + ix_b + iy_b + k0 + 1))
    )
    return acc


@triton.jit
def _departure_kernel(
    FU,
    FV,
    FW,
    zc,
    zf,
    XD,
    YD,
    ZD,
    NCLAMP,
    dt,
    dx,
    dy,
    Lz,
    gamma,
    tanh_g,
    nzf,
    z_lo,
    z_hi,
    N,
    NYI: tl.constexpr,
    NZI: tl.constexpr,
    NX: tl.constexpr,
    NY: tl.constexpr,
    NZC: tl.constexpr,
    NZF: tl.constexpr,
    COMP: tl.constexpr,
    N_ITERS: tl.constexpr,
    IS_SYMMETRIC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # decode arrival point from the flat interior index (z fastest)
    k = offs % NZI
    j = (offs // NZI) % NYI
    i = offs // (NZI * NYI)

    fi = i.to(tl.float32)
    fj = j.to(tl.float32)
    if COMP == 0:  # u: x-faces, y centers, z centers
        xa = (fi + 1.0) * dx
        ya = (fj + 0.5) * dy
        za = tl.load(zc + k + 1, mask=mask, other=0.5)
    elif COMP == 1:  # v: x centers, y-faces, z centers
        xa = (fi + 0.5) * dx
        ya = (fj + 1.0) * dy
        za = tl.load(zc + k + 1, mask=mask, other=0.5)
    else:  # w: x centers, y centers, interior z faces
        xa = (fi + 0.5) * dx
        ya = (fj + 0.5) * dy
        za = tl.load(zf + k + 1, mask=mask, other=0.5)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    half_dt = 0.5 * dt

    xm = xa
    ym = ya
    zm = za
    for _ in tl.static_range(N_ITERS):
        us = _trilinear(
            FU,
            xm,
            ym,
            zm,
            zc,
            inv_dx,
            0.0,
            inv_dy,
            -0.5,
            Lz,
            gamma,
            tanh_g,
            nzf,
            z_lo,
            z_hi,
            NX,
            NY,
            NZC,
            True,
            IS_SYMMETRIC,
        )
        vs = _trilinear(
            FV,
            xm,
            ym,
            zm,
            zc,
            inv_dx,
            -0.5,
            inv_dy,
            0.0,
            Lz,
            gamma,
            tanh_g,
            nzf,
            z_lo,
            z_hi,
            NX,
            NY,
            NZC,
            True,
            IS_SYMMETRIC,
        )
        ws = _trilinear(
            FW,
            xm,
            ym,
            zm,
            zf,
            inv_dx,
            -0.5,
            inv_dy,
            -0.5,
            Lz,
            gamma,
            tanh_g,
            nzf,
            z_lo,
            z_hi,
            NX,
            NY,
            NZF,
            False,
            IS_SYMMETRIC,
        )
        xm = xa - half_dt * us
        ym = ya - half_dt * vs
        zm = tl.minimum(tl.maximum(za - half_dt * ws, z_lo), z_hi)

    xd = 2.0 * xm - xa
    yd = 2.0 * ym - ya
    zd = 2.0 * zm - za

    clamped = ((zd < z_lo) | (zd > z_hi)) & mask
    tl.atomic_add(NCLAMP, tl.sum(clamped.to(tl.int32), axis=0))
    zd = tl.minimum(tl.maximum(zd, z_lo), z_hi)

    tl.store(XD + offs, xd, mask=mask)
    tl.store(YD + offs, yd, mask=mask)
    tl.store(ZD + offs, zd, mask=mask)


@triton.jit
def _gather_kernel(
    F,
    XD,
    YD,
    ZD,
    OUT,
    znodes,
    dinv,
    udenom,
    dx,
    dy,
    shift_x,
    shift_y,
    Lz,
    gamma,
    tanh_g,
    nzf,
    z_lo,
    z_hi,
    N,
    NX: tl.constexpr,
    NY: tl.constexpr,
    NZ: tl.constexpr,
    IS_CENTERS: tl.constexpr,
    IS_SYMMETRIC: tl.constexpr,
    ORDER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(XD + offs, mask=mask, other=0.0)
    y = tl.load(YD + offs, mask=mask, other=0.0)
    z = tl.load(ZD + offs, mask=mask, other=0.5)
    z = tl.minimum(tl.maximum(z, z_lo), z_hi)

    sx = x / dx + shift_x
    ix0 = tl.math.floor(sx).to(tl.int32)
    txf = sx - ix0
    sy = y / dy + shift_y
    iy0 = tl.math.floor(sy).to(tl.int32)
    tyf = sy - iy0

    k0 = _z_locate(z, znodes, Lz, gamma, tanh_g, nzf, IS_CENTERS, IS_SYMMETRIC, NZ, ORDER)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for ii in tl.static_range(ORDER):
        # uniform Lagrange weight in x for node offset ii - (ORDER/2 - 1)
        wx = tl.load(udenom + ii)
        for ll in tl.static_range(ORDER):
            if ll != ii:
                wx = wx * (txf - (ll - (ORDER // 2 - 1)))
        ix = _wrap(ix0 + (ii - (ORDER // 2 - 1)), NX) * (NY * NZ)
        for jj in tl.static_range(ORDER):
            wy = tl.load(udenom + jj)
            for ll in tl.static_range(ORDER):
                if ll != jj:
                    wy = wy * (tyf - (ll - (ORDER // 2 - 1)))
            iy = _wrap(iy0 + (jj - (ORDER // 2 - 1)), NY) * NZ
            base = ix + iy + k0
            wxy = wx * wy
            for kk in tl.static_range(ORDER):
                # nonuniform z weight from the actual node coordinates
                wz = tl.load(dinv + k0 * ORDER + kk)
                for ll in tl.static_range(ORDER):
                    if ll != kk:
                        wz = wz * (z - tl.load(znodes + k0 + ll))
                acc += wxy * wz * tl.load(F + base + kk, mask=mask, other=0.0)

    tl.store(OUT + offs, acc, mask=mask)


class TritonSL:
    """Launch wrapper bound to one SLAdvector (fp32 pipeline, CUDA)."""

    BLOCK = 128

    def __init__(self, adv):
        self.adv = adv
        dev = adv.device
        self.zc32 = adv.z_c.to(torch.float32).contiguous()
        self.zf32 = adv.z_f.to(torch.float32).contiguous()
        # flat inverse-denominator tables and uniform inverse denominators
        self.dinv = {}
        self.udenom = {}
        from .semilag import _uniform_inv_denominators, _z_denominator_table

        for comp in "uvw":
            spec = adv.spec[comp]
            self.dinv[comp] = (
                _z_denominator_table(spec.znodes.double(), adv.order)
                .to(torch.float32)
                .contiguous()
                .reshape(-1)
            )
        self.udenom[adv.order] = torch.tensor(
            _uniform_inv_denominators(adv.order), dtype=torch.float32, device=dev
        )
        # departure coordinate buffers (shared across components: sized for
        # the largest interior; w uses a prefix)
        n_max = adv.nx * adv.ny * adv.nz
        self.xd = torch.empty(n_max, dtype=torch.float32, device=dev)
        self.yd = torch.empty(n_max, dtype=torch.float32, device=dev)
        self.zd = torch.empty(n_max, dtype=torch.float32, device=dev)
        self.nclamp = torch.zeros(1, dtype=torch.int32, device=dev)
        self.out = {
            c: torch.empty(adv.arrival[c][3], dtype=torch.float32, device=dev) for c in "uvw"
        }

    def departure(self, comp, dt):
        adv = self.adv
        nzi = adv.nz if comp != "w" else adv.nz - 1
        n = adv.nx * adv.ny * nzi
        grid = (triton.cdiv(n, self.BLOCK),)
        _departure_kernel[grid](
            adv.mbuf["u"],
            adv.mbuf["v"],
            adv.mbuf["w"],
            self.zc32,
            self.zf32,
            self.xd,
            self.yd,
            self.zd,
            self.nclamp,
            float(dt),
            adv.dx,
            adv.dy,
            adv.Lz,
            adv.gamma,
            adv._tanh_g,
            float(adv.nz),
            adv.z_lo,
            adv.z_hi,
            n,
            adv.ny,
            nzi,
            adv.nx,
            adv.ny,
            adv.nz + 2,
            adv.nz + 1,
            "uvw".index(comp),
            adv.n_traj_iters,
            adv.stretching_type == "symmetric",
            self.BLOCK,
        )
        return n

    def gather(self, comp, buf, n):
        adv = self.adv
        spec = adv.spec[comp]
        grid = (triton.cdiv(n, self.BLOCK),)
        out = self.out[comp].reshape(-1)
        _gather_kernel[grid](
            buf.reshape(-1),
            self.xd,
            self.yd,
            self.zd,
            out,
            spec.znodes.to(torch.float32) if spec.znodes.dtype != torch.float32 else spec.znodes,
            self.dinv[comp],
            self.udenom[adv.order],
            adv.dx,
            adv.dy,
            spec.shift_x,
            spec.shift_y,
            adv.Lz,
            adv.gamma,
            adv._tanh_g,
            float(adv.nz),
            adv.z_lo,
            adv.z_hi,
            n,
            spec.NX,
            spec.NY,
            spec.NZ,
            spec.ztype == "centers",
            adv.stretching_type == "symmetric",
            adv.order,
            self.BLOCK,
        )
        return self.out[comp]
