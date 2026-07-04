"""
eulerian_triton.py — hand-written Triton kernel for the Eulerian IMEX
explicit RHS (advection + xy-diffusion), the baseline counterpart of
semilag_triton.py.

Fair-comparison requirement (docs in CLAUDE.md / research notes): the SL
scheme must not win merely because its implementation was optimized harder,
so the Eulerian baseline gets the same treatment. One pointwise kernel per
velocity component replaces the ~40 sliced eager ops of
operators.compute_momentum_rhs_fused_imex: every face interpolation and
derivative lives in registers; the only global traffic is the three velocity
fields (neighbor loads are L2-cached across adjacent lanes, z is the
contiguous axis) and the one output component.

Runs in the fields' native dtype (fp64). The stencil is bandwidth-bound
(~35 flops per ~60 cached bytes), so the GB10's 1/64 fp64 flop rate does not
bind — unlike the flop-dense SL interpolation, no fp32 pipeline is needed.
Scalar coefficients are passed through a small fp64 tensor because Triton
casts Python-float kernel arguments to fp32.

Numerics are identical to compute_momentum_rhs_fused_imex (divergence-form
advection on the staggered MAC grid, xy Laplacian only — z-diffusion is the
implicit solve's job); agreement is at reassociation-level rounding.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rhs_kernel(U, V, W, OUT, DZF_INV, DZC_INV, PAR,
                N, nx, ny, nz, NYT, NZT,
                SUX, SUY, SVX, SVY, SWX, SWY,
                COMP: tl.constexpr, BLOCK: tl.constexpr):
    """RHS = xy-diffusion - advection for one component, over its full
    (ghost-padded) tensor extent; non-interior lanes store 0."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    K = offs % NZT
    J = (offs // NZT) % NYT
    I = offs // (NZT * NYT)

    if COMP == 0:      # u: x-faces 1..nx-1, y centers 1..ny, z centers 1..nz
        inside = (I >= 1) & (I < nx) & (J >= 1) & (J <= ny) & (K >= 1) & (K <= nz)
    elif COMP == 1:    # v: x centers 1..nx, y-faces 1..ny-1, z centers 1..nz
        inside = (I >= 1) & (I <= nx) & (J >= 1) & (J < ny) & (K >= 1) & (K <= nz)
    else:              # w: x centers 1..nx, y centers 1..ny, z-faces 1..nz-1
        inside = (I >= 1) & (I <= nx) & (J >= 1) & (J <= ny) & (K >= 1) & (K < nz)
    mi = m & inside

    dx_inv = tl.load(PAR + 0)
    dy_inv = tl.load(PAR + 1)
    nu_dx2 = tl.load(PAR + 2)
    nu_dy2 = tl.load(PAR + 3)

    ui = I * SUX + J * SUY + K
    vi = I * SVX + J * SVY + K
    wi = I * SWX + J * SWY + K

    if COMP == 0:
        uc = tl.load(U + ui, mask=mi, other=0.0)
        uxp = tl.load(U + ui + SUX, mask=mi, other=0.0)
        uxm = tl.load(U + ui - SUX, mask=mi, other=0.0)
        uyp = tl.load(U + ui + SUY, mask=mi, other=0.0)
        uym = tl.load(U + ui - SUY, mask=mi, other=0.0)
        uzp = tl.load(U + ui + 1, mask=mi, other=0.0)
        uzm = tl.load(U + ui - 1, mask=mi, other=0.0)

        r = 0.5 * (uc + uxp)
        l = 0.5 * (uxm + uc)
        duudx = (r * r - l * l) * dx_inv

        # v at the four surrounding y-faces of the u-face
        v00 = tl.load(V + vi, mask=mi, other=0.0)
        v10 = tl.load(V + vi + SVX, mask=mi, other=0.0)
        v0m = tl.load(V + vi - SVY, mask=mi, other=0.0)
        v1m = tl.load(V + vi + SVX - SVY, mask=mi, other=0.0)
        dvudy = (0.5 * (v00 + v10) * (0.5 * (uc + uyp))
                 - 0.5 * (v0m + v1m) * (0.5 * (uym + uc))) * dy_inv

        w00 = tl.load(W + wi, mask=mi, other=0.0)
        w10 = tl.load(W + wi + SWX, mask=mi, other=0.0)
        w0m = tl.load(W + wi - 1, mask=mi, other=0.0)
        w1m = tl.load(W + wi + SWX - 1, mask=mi, other=0.0)
        dzfi = tl.load(DZF_INV + K - 1, mask=mi, other=0.0)
        dwudz = (0.5 * (w00 + w10) * (0.5 * (uc + uzp))
                 - 0.5 * (w0m + w1m) * (0.5 * (uzm + uc))) * dzfi

        diff = (uxp - 2.0 * uc + uxm) * nu_dx2 + (uyp - 2.0 * uc + uym) * nu_dy2
        out = diff - (duudx + dvudy + dwudz)

    elif COMP == 1:
        vc = tl.load(V + vi, mask=mi, other=0.0)
        vxp = tl.load(V + vi + SVX, mask=mi, other=0.0)
        vxm = tl.load(V + vi - SVX, mask=mi, other=0.0)
        vyp = tl.load(V + vi + SVY, mask=mi, other=0.0)
        vym = tl.load(V + vi - SVY, mask=mi, other=0.0)
        vzp = tl.load(V + vi + 1, mask=mi, other=0.0)
        vzm = tl.load(V + vi - 1, mask=mi, other=0.0)

        u00 = tl.load(U + ui, mask=mi, other=0.0)
        u01 = tl.load(U + ui + SUY, mask=mi, other=0.0)
        um0 = tl.load(U + ui - SUX, mask=mi, other=0.0)
        um1 = tl.load(U + ui - SUX + SUY, mask=mi, other=0.0)
        duvdx = (0.5 * (u00 + u01) * (0.5 * (vc + vxp))
                 - 0.5 * (um0 + um1) * (0.5 * (vxm + vc))) * dx_inv

        t = 0.5 * (vc + vyp)
        b = 0.5 * (vym + vc)
        dvvdy = (t * t - b * b) * dy_inv

        w00 = tl.load(W + wi, mask=mi, other=0.0)
        w01 = tl.load(W + wi + SWY, mask=mi, other=0.0)
        w0m = tl.load(W + wi - 1, mask=mi, other=0.0)
        w1m = tl.load(W + wi + SWY - 1, mask=mi, other=0.0)
        dzfi = tl.load(DZF_INV + K - 1, mask=mi, other=0.0)
        dwvdz = (0.5 * (w00 + w01) * (0.5 * (vc + vzp))
                 - 0.5 * (w0m + w1m) * (0.5 * (vzm + vc))) * dzfi

        diff = (vxp - 2.0 * vc + vxm) * nu_dx2 + (vyp - 2.0 * vc + vym) * nu_dy2
        out = diff - (duvdx + dvvdy + dwvdz)

    else:
        wc = tl.load(W + wi, mask=mi, other=0.0)
        wxp = tl.load(W + wi + SWX, mask=mi, other=0.0)
        wxm = tl.load(W + wi - SWX, mask=mi, other=0.0)
        wyp = tl.load(W + wi + SWY, mask=mi, other=0.0)
        wym = tl.load(W + wi - SWY, mask=mi, other=0.0)
        wzp = tl.load(W + wi + 1, mask=mi, other=0.0)
        wzm = tl.load(W + wi - 1, mask=mi, other=0.0)

        u00 = tl.load(U + ui, mask=mi, other=0.0)
        u0p = tl.load(U + ui + 1, mask=mi, other=0.0)
        um0 = tl.load(U + ui - SUX, mask=mi, other=0.0)
        ump = tl.load(U + ui - SUX + 1, mask=mi, other=0.0)
        duwdx = (0.5 * (u00 + u0p) * (0.5 * (wc + wxp))
                 - 0.5 * (um0 + ump) * (0.5 * (wxm + wc))) * dx_inv

        v00 = tl.load(V + vi, mask=mi, other=0.0)
        v0p = tl.load(V + vi + 1, mask=mi, other=0.0)
        vm0 = tl.load(V + vi - SVY, mask=mi, other=0.0)
        vmp = tl.load(V + vi - SVY + 1, mask=mi, other=0.0)
        dvwdy = (0.5 * (v00 + v0p) * (0.5 * (wc + wyp))
                 - 0.5 * (vm0 + vmp) * (0.5 * (wym + wc))) * dy_inv

        t = 0.5 * (wc + wzp)
        b = 0.5 * (wzm + wc)
        dzci = tl.load(DZC_INV + K, mask=mi, other=0.0)
        dwwdz = (t * t - b * b) * dzci

        diff = (wxp - 2.0 * wc + wxm) * nu_dx2 + (wyp - 2.0 * wc + wym) * nu_dy2
        out = diff - (duwdx + dvwdy + dwwdz)

    tl.store(OUT + offs, tl.where(mi, out, 0.0), mask=m)


class TritonEulerianRHS:
    """Launch wrapper bound to one grid; returns fresh output tensors each
    call (the solver's AB2 history swaps ownership of the returned RHS)."""

    BLOCK = 256

    def __init__(self, nx, ny, nz, dx, dy, dz_c, dz_f, nu, device):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dzf_inv = (1.0 / dz_f).contiguous()
        self.dzc_inv = (1.0 / dz_c).contiguous()
        self.par = torch.tensor(
            [1.0 / dx, 1.0 / dy, nu / dx ** 2, nu / dy ** 2],
            dtype=torch.float64, device=device)

    def __call__(self, u, v, w):
        nx, ny, nz = self.nx, self.ny, self.nz
        sux, suy = u.stride(0), u.stride(1)
        svx, svy = v.stride(0), v.stride(1)
        swx, swy = w.stride(0), w.stride(1)
        outs = []
        for comp, f in enumerate((u, v, w)):
            out = torch.empty_like(f)
            n = f.numel()
            grid = (triton.cdiv(n, self.BLOCK),)
            _rhs_kernel[grid](
                u, v, w, out, self.dzf_inv, self.dzc_inv, self.par,
                n, nx, ny, nz, f.shape[1], f.shape[2],
                sux, suy, svx, svy, swx, swy,
                comp, self.BLOCK)
            outs.append(out)
        return tuple(outs)
