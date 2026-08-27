"""Per-rank semi-Lagrangian advection with local index arithmetic.

LocalSL is a documented fork of the EAGER path of slchannel.semilag.SLAdvector
(departure_coords at semilag.py:534-552, _build_iw_impl at 471-500), with
exactly two changes to the horizontal index math:

    1. the periodic modulo  ix = remainder(floor(x/dx + shift) + offs, NX)
       becomes the rank-local offset  ix = floor(x/dx + shift) - i_start + H
       (+ offs), with NO wrap;
    2. every stencil index is range-guarded: a departure foot whose stencil
       leaves the extended local buffer raises HaloOverflowError instead of
       silently aliasing a periodic image.

Everything rank-independent (z node tables, denominators, Lagrange weight
routines, clamp bounds, dtypes) is taken from ONE monolithic reference
SLAdvector shared by all ranks, so the z arithmetic is bitwise production
code. The 1x1-rank test in tests/test_local_advect.py pins this fork to the
production result at 0 ulp and fails if semilag.py drifts underneath it.

Arrival coordinates are GLOBAL-physical and reproduce the monolithic
convention exactly, including the wrap of the x=0 u-face (computed at
x = Lx, matching semilag.py:292 `ar = arange(1, nx+1)`): the rank owning
global node 0 computes it at coordinate nx*dx. Consequently each rank
computes arrival nodes (i_start, i_start + nxl] for the u/x and v/y face
directions; the shared face lands in the arrival rank's trailing halo slot
and the OWNING rank pulls it afterwards (comm.pull_minus_edge) -- see
halo.edge_pull_slices.
"""

import math

import torch

from slchannel.semilag import _gather_interp


class HaloOverflowError(RuntimeError):
    def __init__(self, comp, axis, lo, hi, limit, H):
        self.comp, self.axis, self.lo, self.hi, self.limit, self.H = (
            comp,
            axis,
            lo,
            hi,
            limit,
            H,
        )
        super().__init__(
            f"SL stencil left the halo: comp={comp} axis={axis} "
            f"index range [{lo}, {hi}] outside [0, {limit}) at H={H}"
        )


def required_halo(order, disp_cells, foot_depth_factor=2.0):
    """Halo width for a per-dt displacement of `disp_cells` cells.

    The BDF2 far foot integrates over 2*dt (solver.py:847-849), hence the
    default foot_depth_factor. The gather stencil extends offs in
    [-(order//2 - 1), order//2] around floor(x/dx + shift) (semilag.py:478),
    and the staggered half-shift can push the floor one more node left; the
    +1.0 makes the left/right reach symmetric and covers it.
    """
    d = foot_depth_factor * disp_cells
    return int(math.ceil(d + 1.0)) + order // 2


class LocalSL:
    def __init__(self, ref, decomp, rank):
        self.ref = ref
        self.d = decomp
        self.rank = rank
        self.H = decomp.H
        self.i_start, self.j_start = decomp.origin(rank)
        nxl, nyl, nz = decomp.nxl, decomp.nyl, decomp.nz
        dev = ref.spec["u"].znodes.device
        cd = ref._coord_dtype

        # ---- global-physical arrival coordinates for this rank's nodes ----
        # (mirrors semilag.py:292-307 exactly; see module docstring for the
        # (i_start, i_start+nxl] face convention)
        ar = torch.arange(self.i_start + 1, self.i_start + nxl + 1, dtype=cd, device=dev)
        aj = torch.arange(self.j_start + 1, self.j_start + nyl + 1, dtype=cd, device=dev)
        x_face = (ar * ref.dx).view(nxl, 1, 1)
        x_cent = ((ar - 0.5) * ref.dx).view(nxl, 1, 1)
        y_face = (aj * ref.dy).view(1, nyl, 1)
        y_cent = ((aj - 0.5) * ref.dy).view(1, nyl, 1)
        z_cent = ref.arrival["u"][2]
        z_facei = ref.arrival["w"][2]
        self.arrival = {
            "u": (x_face, y_cent, z_cent, (nxl, nyl, nz)),
            "v": (x_cent, y_face, z_cent, (nxl, nyl, nz)),
            "w": (x_cent, y_cent, z_facei, (nxl, nyl, nz - 1)),
        }

        # node buffers in the extended local layout, production buffer dtype
        self.mbuf = {
            c: torch.empty(decomp.ext_shape(c), dtype=ref._buf_dtype, device=dev) for c in "uvw"
        }
        self.fbuf = {
            c: torch.empty(decomp.ext_shape(c), dtype=ref._buf_dtype, device=dev) for c in "uvw"
        }
        self.n_clamped_last = torch.zeros((), dtype=torch.int64, device=dev)
        self.max_disp_cells = {"x": 0.0, "y": 0.0}

    # ---- fork of semilag.SLAdvector._build_iw_impl (semilag.py:471-500) ----
    # changes: local offset instead of remainder, range guard, extended strides
    def _build_iw_local(self, comp, spec, x, y, z, order, count_clamp):
        ref, H = self.ref, self.H
        nxe = self.d.nxl + 2 * H
        nye = self.d.nyl + 2 * H
        if count_clamp:
            n_clamped = ((z < ref.z_lo) | (z > ref.z_hi)).sum()
            z = torch.clamp(z, ref.z_lo, ref.z_hi)
        else:
            n_clamped = torch.zeros((), dtype=torch.int64, device=z.device)

        offs = torch.arange(order, device=x.device) - (order // 2 - 1)
        o_lo, o_hi = int(offs[0]), int(offs[-1])

        sx = x / ref.dx + spec.shift_x
        ix0 = sx.floor()
        tx = sx - ix0
        ixl = ix0.long() - self.i_start + H  # local node index, NO wrap
        lo, hi = int(ixl.min()) + o_lo, int(ixl.max()) + o_hi
        if lo < 0 or hi >= nxe:
            raise HaloOverflowError(comp, "x", lo, hi, nxe, H)
        ix_lin = ((ixl.unsqueeze(1) + offs) * (nye * spec.NZ)).to(ref._idx_dtype)
        wx = ref._uniform_weights(tx, order).to(ref._buf_dtype)

        sy = y / ref.dy + spec.shift_y
        iy0 = sy.floor()
        ty = sy - iy0
        iyl = iy0.long() - self.j_start + H
        lo, hi = int(iyl.min()) + o_lo, int(iyl.max()) + o_hi
        if lo < 0 or hi >= nye:
            raise HaloOverflowError(comp, "y", lo, hi, nye, H)
        iy_lin = ((iyl.unsqueeze(1) + offs) * spec.NZ).to(ref._idx_dtype)
        wy = ref._uniform_weights(ty, order).to(ref._buf_dtype)

        m0 = ref._locate_z(z, spec)
        k0 = torch.clamp(m0 - (order // 2 - 1), 0, spec.NZ - order)
        wz = ref._z_weights(z, k0, spec, order).to(ref._buf_dtype)

        return ix_lin, iy_lin, k0.to(ref._idx_dtype), wx, wy, wz, n_clamped

    # ---- fork of semilag.SLAdvector._sample (semilag.py:512-528) ----
    def _sample(self, comp, x, y, z):
        ref = self.ref
        shape = x.shape
        ix_lin, iy_lin, kz0, wx, wy, wz, _ = self._build_iw_local(
            comp,
            ref.spec[comp],
            x.reshape(-1),
            y.reshape(-1),
            torch.clamp(z, ref.z_lo, ref.z_hi).reshape(-1),
            ref.traj_order,
            False,
        )
        return _gather_interp(
            self.mbuf[comp].reshape(-1), ix_lin.long(), iy_lin.long(), kz0.long(), wx, wy, wz
        ).reshape(shape)

    # ---- fork of semilag.SLAdvector.departure_coords (semilag.py:534-552) ----
    def departure_coords(self, comp, dt_t):
        ref = self.ref
        xa, ya, za, _ = self.arrival[comp]
        half_dt = 0.5 * dt_t
        xm, ym, zm = torch.broadcast_tensors(xa, ya, za)
        for _ in range(ref.n_traj_iters):
            us = self._sample("u", xm, ym, zm)
            vs = self._sample("v", xm, ym, zm)
            ws = self._sample("w", xm, ym, zm)
            xm = xa - half_dt * us
            ym = ya - half_dt * vs
            zm = torch.clamp(za - half_dt * ws, ref.z_lo, ref.z_hi)
        return 2.0 * xm - xa, 2.0 * ym - ya, 2.0 * zm - za

    def advect(self, fields_ext, mids_ext, dt_t):
        """Advect the three components; returns dict comp -> interior values
        of shape self.arrival[comp][3]. `fields_ext`/`mids_ext`: dict comp ->
        extended local node arrays (halos must be freshly exchanged)."""
        ref = self.ref
        for c in "uvw":
            self.mbuf[c].copy_(mids_ext[c])
            self.fbuf[c].copy_(fields_ext[c])
        n_clamped = torch.zeros((), dtype=torch.int64, device=self.mbuf["u"].device)
        out = {}
        for comp in "uvw":
            xa, ya, za, shape = self.arrival[comp]
            xd, yd, zd = self.departure_coords(comp, dt_t)
            self.max_disp_cells["x"] = max(
                self.max_disp_cells["x"], float((xd - xa).abs().max()) / ref.dx
            )
            self.max_disp_cells["y"] = max(
                self.max_disp_cells["y"], float((yd - ya).abs().max()) / ref.dy
            )
            ix_lin, iy_lin, kz0, wx, wy, wz, ncl = self._build_iw_local(
                comp,
                ref.spec[comp],
                xd.reshape(-1),
                yd.reshape(-1),
                zd.reshape(-1),
                ref.order,
                True,
            )
            n_clamped += ncl
            vals = _gather_interp(
                self.fbuf[comp].reshape(-1),
                ix_lin.long(),
                iy_lin.long(),
                kz0.long(),
                wx,
                wy,
                wz,
            )
            out[comp] = vals.reshape(shape)
        self.n_clamped_last = n_clamped
        return out
