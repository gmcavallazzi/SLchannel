"""Localized Triton kernels for the decomposed semi-Lagrangian path.

Fork of slchannel/semilag_triton.py with the three changes documented in
parallel/__init__.py:
  1. the flat-index arrival decode uses the LOCAL interior dims and the rank
     origin (i_start, j_start), so arrival coordinates stay global-physical
     and identical to the monolithic kernels;
  2. `_wrap(i, N) = i % N` is replaced by `i - i_start + H` offset arithmetic
     into the extended (nxl+2H, nyl+2H, NZ) node buffers -- no periodic wrap
     within a rank;
  3. every horizontal stencil index is guarded: out-of-range indices are
     clamped for memory safety and counted into an overflow flag buffer,
     which the host turns into sl_local.HaloOverflowError.

fp32 pipeline, CUDA only, same as production. z machinery (_z_locate,
denominator tables) is imported unchanged.
"""

import torch
import triton
import triton.language as tl

from slchannel.semilag_triton import _z_locate

from .sl_local import HaloOverflowError


@triton.jit
def _loc(i0, ISTART, H, NE: tl.constexpr):
    """Local extended index for global node index i0; returns (index, bad)."""
    il = i0 - ISTART + H
    bad = (il < 0) | (il >= NE)
    il = tl.minimum(tl.maximum(il, 0), NE - 1)
    return il, bad


@triton.jit
def _trilinear_local(
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
    ISTART,
    JSTART,
    H,
    NXE: tl.constexpr,
    NYE: tl.constexpr,
    NZ: tl.constexpr,
    IS_CENTERS: tl.constexpr,
    IS_SYMMETRIC: tl.constexpr,
):
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

    ixa, ba = _loc(ix0, ISTART, H, NXE)
    ixb, bb = _loc(ix0 + 1, ISTART, H, NXE)
    iya, bc = _loc(iy0, JSTART, H, NYE)
    iyb, bd = _loc(iy0 + 1, JSTART, H, NYE)
    bad = ba | bb | bc | bd
    ix_a = ixa * (NYE * NZ)
    ix_b = ixb * (NYE * NZ)
    iy_a = iya * NZ
    iy_b = iyb * NZ

    acc = (1 - txf) * (
        (1 - tyf)
        * ((1 - tz) * tl.load(F + ix_a + iy_a + k0) + tz * tl.load(F + ix_a + iy_a + k0 + 1))
        + tyf * ((1 - tz) * tl.load(F + ix_a + iy_b + k0) + tz * tl.load(F + ix_a + iy_b + k0 + 1))
    ) + txf * (
        (1 - tyf)
        * ((1 - tz) * tl.load(F + ix_b + iy_a + k0) + tz * tl.load(F + ix_b + iy_a + k0 + 1))
        + tyf * ((1 - tz) * tl.load(F + ix_b + iy_b + k0) + tz * tl.load(F + ix_b + iy_b + k0 + 1))
    )
    return acc, bad


@triton.jit
def _departure_kernel_local(
    FU,
    FV,
    FW,
    zc,
    zf,
    XD,
    YD,
    ZD,
    NCLAMP,
    OOB,
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
    ISTART,
    JSTART,
    H,
    NYI: tl.constexpr,
    NZI: tl.constexpr,
    NXE: tl.constexpr,
    NYE: tl.constexpr,
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

    # decode LOCAL arrival index, form GLOBAL-physical arrival coordinates
    # (matches sl_local.LocalSL: faces are the (i_start, i_start+nxl] set)
    k = offs % NZI
    j = (offs // NZI) % NYI
    i = offs // (NZI * NYI)
    fi = (i + ISTART).to(tl.float32)
    fj = (j + JSTART).to(tl.float32)
    if COMP == 0:  # u
        xa = (fi + 1.0) * dx
        ya = (fj + 0.5) * dy
        za = tl.load(zc + k + 1, mask=mask, other=0.5)
    elif COMP == 1:  # v
        xa = (fi + 0.5) * dx
        ya = (fj + 1.0) * dy
        za = tl.load(zc + k + 1, mask=mask, other=0.5)
    else:  # w
        xa = (fi + 0.5) * dx
        ya = (fj + 0.5) * dy
        za = tl.load(zf + k + 1, mask=mask, other=0.5)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    half_dt = 0.5 * dt

    xm = xa
    ym = ya
    zm = za
    bad = offs < 0  # all-False init
    for _ in tl.static_range(N_ITERS):
        us, b1 = _trilinear_local(
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
            ISTART,
            JSTART,
            H,
            NXE,
            NYE,
            NZC,
            True,
            IS_SYMMETRIC,
        )
        vs, b2 = _trilinear_local(
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
            ISTART,
            JSTART,
            H,
            NXE,
            NYE,
            NZC,
            True,
            IS_SYMMETRIC,
        )
        ws, b3 = _trilinear_local(
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
            ISTART,
            JSTART,
            H,
            NXE,
            NYE,
            NZF,
            False,
            IS_SYMMETRIC,
        )
        bad = bad | (mask & (b1 | b2 | b3))
        xm = xa - half_dt * us
        ym = ya - half_dt * vs
        zm = tl.minimum(tl.maximum(za - half_dt * ws, z_lo), z_hi)

    xd = 2.0 * xm - xa
    yd = 2.0 * ym - ya
    zd = 2.0 * zm - za

    clamped = ((zd < z_lo) | (zd > z_hi)) & mask
    tl.atomic_add(NCLAMP, tl.sum(clamped.to(tl.int32), axis=0))
    tl.atomic_add(OOB, tl.sum(bad.to(tl.int32), axis=0))
    zd = tl.minimum(tl.maximum(zd, z_lo), z_hi)

    tl.store(XD + offs, xd, mask=mask)
    tl.store(YD + offs, yd, mask=mask)
    tl.store(ZD + offs, zd, mask=mask)


@triton.jit
def _gather_kernel_local(
    F,
    XD,
    YD,
    ZD,
    OUT,
    znodes,
    dinv,
    udenom,
    OOB,
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
    ISTART,
    JSTART,
    H,
    NXE: tl.constexpr,
    NYE: tl.constexpr,
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

    bad = offs < 0  # all-False init
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for ii in tl.static_range(ORDER):
        wx = tl.load(udenom + ii)
        for ll in tl.static_range(ORDER):
            if ll != ii:
                wx = wx * (txf - (ll - (ORDER // 2 - 1)))
        ixl, bx = _loc(ix0 + (ii - (ORDER // 2 - 1)), ISTART, H, NXE)
        bad = bad | (mask & bx)
        ix = ixl * (NYE * NZ)
        for jj in tl.static_range(ORDER):
            wy = tl.load(udenom + jj)
            for ll in tl.static_range(ORDER):
                if ll != jj:
                    wy = wy * (tyf - (ll - (ORDER // 2 - 1)))
            iyl, by = _loc(iy0 + (jj - (ORDER // 2 - 1)), JSTART, H, NYE)
            bad = bad | (mask & by)
            iy = iyl * NZ
            base = ix + iy + k0
            wxy = wx * wy
            for kk in tl.static_range(ORDER):
                wz = tl.load(dinv + k0 * ORDER + kk)
                for ll in tl.static_range(ORDER):
                    if ll != kk:
                        wz = wz * (z - tl.load(znodes + k0 + ll))
                acc += wxy * wz * tl.load(F + base + kk, mask=mask, other=0.0)

    tl.atomic_add(OOB, tl.sum(bad.to(tl.int32), axis=0))
    tl.store(OUT + offs, acc, mask=mask)


class TritonLocalSL:
    """Launch wrapper for one rank of the decomposition (fp32, CUDA).
    Mirrors semilag_triton.TritonSL bound to a sl_local.LocalSL."""

    BLOCK = 128

    def __init__(self, local_sl):
        self.lsl = local_sl
        ref, d = local_sl.ref, local_sl.d
        dev = ref.spec["u"].znodes.device
        self.zc32 = ref.z_c.to(torch.float32).contiguous()
        self.zf32 = ref.z_f.to(torch.float32).contiguous()
        from slchannel.semilag import _uniform_inv_denominators, _z_denominator_table

        self.dinv = {
            c: _z_denominator_table(ref.spec[c].znodes.double(), ref.order)
            .to(torch.float32)
            .contiguous()
            .reshape(-1)
            .to(dev)
            for c in "uvw"
        }
        self.udenom = torch.tensor(
            _uniform_inv_denominators(ref.order), dtype=torch.float32, device=dev
        )
        n_max = d.nxl * d.nyl * d.nz
        self.xd = torch.empty(n_max, dtype=torch.float32, device=dev)
        self.yd = torch.empty(n_max, dtype=torch.float32, device=dev)
        self.zd = torch.empty(n_max, dtype=torch.float32, device=dev)
        self.nclamp = torch.zeros(1, dtype=torch.int32, device=dev)
        self.oob = torch.zeros(1, dtype=torch.int32, device=dev)

    def advect(self, fields_ext, mids_ext, dt_t):
        """Same contract as LocalSL.advect, on the Triton fp32 path."""
        lsl, ref, d = self.lsl, self.lsl.ref, self.lsl.d
        for c in "uvw":
            lsl.mbuf[c].copy_(mids_ext[c])
            lsl.fbuf[c].copy_(fields_ext[c])
        H = d.H
        nxe, nye = d.nxl + 2 * H, d.nyl + 2 * H
        self.nclamp.zero_()
        self.oob.zero_()
        out = {}
        for comp in "uvw":
            nzi = d.nz if comp != "w" else d.nz - 1
            n = d.nxl * d.nyl * nzi
            grid = (triton.cdiv(n, self.BLOCK),)
            _departure_kernel_local[grid](
                lsl.mbuf["u"],
                lsl.mbuf["v"],
                lsl.mbuf["w"],
                self.zc32,
                self.zf32,
                self.xd,
                self.yd,
                self.zd,
                self.nclamp,
                self.oob,
                float(dt_t),
                ref.dx,
                ref.dy,
                ref.Lz,
                ref.gamma,
                ref._tanh_g,
                float(ref.nz),
                ref.z_lo,
                ref.z_hi,
                n,
                lsl.i_start,
                lsl.j_start,
                H,
                d.nyl,
                nzi,
                nxe,
                nye,
                ref.nz + 2,
                ref.nz + 1,
                "uvw".index(comp),
                ref.n_traj_iters,
                ref.stretching_type == "symmetric",
                self.BLOCK,
            )
            spec = ref.spec[comp]
            vals = torch.empty(n, dtype=torch.float32, device=self.xd.device)
            _gather_kernel_local[grid](
                lsl.fbuf[comp].reshape(-1),
                self.xd,
                self.yd,
                self.zd,
                vals,
                spec.znodes.to(torch.float32),
                self.dinv[comp],
                self.udenom,
                self.oob,
                ref.dx,
                ref.dy,
                spec.shift_x,
                spec.shift_y,
                ref.Lz,
                ref.gamma,
                ref._tanh_g,
                float(ref.nz),
                ref.z_lo,
                ref.z_hi,
                n,
                lsl.i_start,
                lsl.j_start,
                H,
                nxe,
                nye,
                spec.NZ,
                spec.ztype == "centers",
                ref.stretching_type == "symmetric",
                ref.order,
                self.BLOCK,
            )
            out[comp] = vals.reshape((d.nxl, d.nyl, nzi))
        n_oob = int(self.oob.item())
        if n_oob:
            raise HaloOverflowError("triton", "xy", -1, -1, nxe, H)
        lsl.n_clamped_last = self.nclamp[0].long()
        return out
