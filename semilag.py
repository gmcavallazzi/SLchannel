"""
semilag.py — Semi-Lagrangian advection on the staggered channel grid.

This is the module that makes slChannel different from torChannel: instead of
an explicit Eulerian advection term (CFL-limited), each velocity component is
advanced by tracing the characteristic back from its arrival point (the
component's own staggered face) and interpolating the previous field at the
departure point with high-order tensor-product Lagrange interpolation.

Grid conventions match torChannel exactly (one ghost layer per side,
u:(nx+1,ny+2,nz+2) at x-faces, v:(nx+2,ny+1,nz+2) at y-faces,
w:(nx+2,ny+2,nz+1) at z-faces, periodic x/y, walls in z, tanh-stretched z).

Design notes
------------
- Interpolation nodes per component are a contiguous "node buffer" holding one
  full period in x and y (periodic indexing via modulo) plus the full z column:
  cell centers INCLUDING the two ghost centers for u/v (the ghost value
  -u[...,1] realizes the odd/no-slip extension), and all faces for w (wall
  faces are exact zeros).
- x,y are uniform: Lagrange weights are closed-form polynomials of the
  fractional offset. z is nonuniform: weights are computed against the actual
  node coordinates (z_c / z_f) with precomputed inverse-denominator tables.
  NOTE: z_c centers are arithmetic means of faces, NOT the image of uniform
  computational points, so uniform-xi weights in z would silently lose order.
- The analytic inverse of the tanh face map is used only to LOCATE the z
  stencil (continuous face index), followed by one compare against the actual
  node — exact regardless of the center-vs-face subtlety, no searchsorted.
- Departure points: iterated midpoint rule x_m <- x_a - 0.5*dt*V(x_m) with V
  the time-extrapolated velocity (supplied by the solver), trilinear sampling
  along the trajectory, then x_d = 2*x_m - x_a. Trilinear sampling reuses the
  same index/weight machinery with order=2.
- Departure z is clamped just inside the walls; the number of clamped points
  is accumulated per step as a diagnostic (self.n_clamped_last).
"""

import torch
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Small containers
# ---------------------------------------------------------------------------

@dataclass
class IndexWeights:
    """Departure-point stencil: flat gather indices + per-direction weights.

    ix_lin, iy_lin are pre-multiplied by the node-buffer strides so a flat
    gather index is simply ix_lin[:, i] + iy_lin[:, j] + kz0 (+ z offset).
    The same IndexWeights can interpolate any field living on the same
    component grid (used by the v2 scheme to interpolate the explicit RHS).
    """
    ix_lin: torch.Tensor   # (N, order) int64
    iy_lin: torch.Tensor   # (N, order) int64
    kz0: torch.Tensor      # (N,)      int64 — base z-node index
    wx: torch.Tensor       # (N, order)
    wy: torch.Tensor       # (N, order)
    wz: torch.Tensor       # (N, order)
    order: int
    n_clamped: torch.Tensor  # 0-D int64 — departure points clamped at walls


@dataclass
class _GridSpec:
    """Node-grid description for one velocity component."""
    name: str
    shift_x: float          # node coordinate s_x = x/dx + shift_x
    shift_y: float
    NX: int
    NY: int
    NZ: int
    znodes: torch.Tensor    # (NZ,) physical z of the nodes
    ztype: str              # 'centers' | 'faces'
    denom_inv: dict         # order -> (NZ - order + 1, order) inverse denominators


def _z_denominator_table(znodes: torch.Tensor, order: int) -> torch.Tensor:
    """Inverse Lagrange denominators for every contiguous stencil of `order`
    nodes: denom_inv[b, m] = 1 / prod_{l != m} (z[b+m] - z[b+l])."""
    zn = znodes.unfold(0, order, 1)                    # (n_base, order)
    diff = zn.unsqueeze(2) - zn.unsqueeze(1)           # [b, m, l] = z_m - z_l
    n_base = diff.shape[0]
    eye = torch.eye(order, dtype=diff.dtype, device=diff.device)
    diff = diff + eye.unsqueeze(0)                     # 1 on the diagonal
    return 1.0 / diff.prod(dim=2)


def _uniform_inv_denominators(order: int) -> list:
    """Inverse denominators for integer-offset nodes off_m = m - (order/2 - 1)."""
    offs = [m - (order // 2 - 1) for m in range(order)]
    inv = []
    for m in range(order):
        d = 1.0
        for l in range(order):
            if l != m:
                d *= (offs[m] - offs[l])
        inv.append(1.0 / d)
    return inv


class SLAdvector:
    """Semi-Lagrangian advection of the three staggered velocity components."""

    def __init__(self, nx, ny, nz, dx, dy, Lx, Ly, Lz,
                 z_f, z_c, gamma, stretching_type='symmetric',
                 order=4, n_traj_iters=2, top_wall_bc_type='dirichlet',
                 interp_dtype='fp64', device=torch.device('cpu')):
        if order not in (4, 6):
            raise ValueError(f"sl.interp_order must be 4 (tricubic) or 6 (triquintic), got {order}")
        if stretching_type not in ('symmetric', 'bottom'):
            raise ValueError(f"SLAdvector supports 'symmetric'/'bottom' stretching, got '{stretching_type}'")
        if interp_dtype not in ('fp64', 'fp32_accum64'):
            raise ValueError(f"sl.interp_dtype must be 'fp64' or 'fp32_accum64', got {interp_dtype}")
        if nz + 1 < order:
            raise ValueError(f"nz={nz} too small for interpolation order {order}")

        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy = dx, dy
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.gamma = float(gamma)
        self.stretching_type = stretching_type
        self.order = order
        self.n_traj_iters = n_traj_iters
        self.top_wall_bc_type = top_wall_bc_type
        self.interp_dtype = interp_dtype
        self.device = device

        self.z_f = z_f
        self.z_c = z_c
        self._tanh_g = torch.tanh(torch.tensor(self.gamma, dtype=torch.float64, device=device)).item()

        # Departure z kept strictly inside the channel
        eps_z = 1e-12 * Lz
        self.z_lo = (z_f[0] + eps_z).item()
        self.z_hi = (z_f[-1] - eps_z).item()

        # --- node-grid specs -------------------------------------------------
        # u: nodes x = i*dx (i=0..nx-1), y = (j+1/2)*dy (j=0..ny-1), z = z_c (ghosts incl.)
        # v: nodes x = (i+1/2)*dx,       y = j*dy,               z = z_c
        # w: nodes x = (i+1/2)*dx,       y = (j+1/2)*dy,         z = z_f (walls incl.)
        def make_spec(name, shift_x, shift_y, znodes, ztype):
            denoms = {2: _z_denominator_table(znodes, 2),
                      order: _z_denominator_table(znodes, order)}
            return _GridSpec(name, shift_x, shift_y, nx, ny, len(znodes),
                             znodes, ztype, denoms)

        self.spec = {
            'u': make_spec('u', 0.0, -0.5, z_c, 'centers'),
            'v': make_spec('v', -0.5, 0.0, z_c, 'centers'),
            'w': make_spec('w', -0.5, -0.5, z_f, 'faces'),
        }

        # Uniform-direction inverse denominators (python floats, baked into weights)
        self._udenom = {2: _uniform_inv_denominators(2),
                        order: _uniform_inv_denominators(order)}

        # --- arrival coordinates (broadcastable, physical) -------------------
        ar = torch.arange(1, nx + 1, dtype=torch.float64, device=device)
        aj = torch.arange(1, ny + 1, dtype=torch.float64, device=device)
        x_face = (ar * dx).view(nx, 1, 1)
        x_cent = ((ar - 0.5) * dx).view(nx, 1, 1)
        y_face = (aj * dy).view(1, ny, 1)
        y_cent = ((aj - 0.5) * dy).view(1, ny, 1)
        z_cent = z_c[1:nz + 1].view(1, 1, nz)
        z_facei = z_f[1:nz].view(1, 1, nz - 1)   # interior faces only; walls pinned w=0

        self.arrival = {
            'u': (x_face, y_cent, z_cent, (nx, ny, nz)),
            'v': (x_cent, y_face, z_cent, (nx, ny, nz)),
            'w': (x_cent, y_cent, z_facei, (nx, ny, nz - 1)),
        }

        # --- persistent node buffers -----------------------------------------
        # One set for the advected fields, one for the trajectory (mid) velocity.
        def alloc(spec):
            return torch.empty(spec.NX, spec.NY, spec.NZ, dtype=torch.float64, device=device)

        self.fbuf = {c: alloc(self.spec[c]) for c in 'uvw'}
        self.mbuf = {c: alloc(self.spec[c]) for c in 'uvw'}

        # Preallocated ghost-shaped outputs
        self.ustar = torch.zeros(nx + 1, ny + 2, nz + 2, dtype=torch.float64, device=device)
        self.vstar = torch.zeros(nx + 2, ny + 1, nz + 2, dtype=torch.float64, device=device)
        self.wstar = torch.zeros(nx + 2, ny + 2, nz + 1, dtype=torch.float64, device=device)

        # Diagnostics: departure points clamped at the walls, last step
        self.n_clamped_last = torch.zeros((), dtype=torch.int64, device=device)

        # Last departure stencils (reused for extra-field interpolation, v2)
        self.last_iw = {}

    # ------------------------------------------------------------------
    # Node-buffer filling (views of the ghost-shaped fields)
    # ------------------------------------------------------------------

    def _fill(self, buf, comp, field):
        nx, ny = self.nx, self.ny
        if comp == 'u':
            buf.copy_(field[0:nx, 1:ny + 1, :])
        elif comp == 'v':
            buf.copy_(field[1:nx + 1, 0:ny, :])
        else:  # w
            buf.copy_(field[1:nx + 1, 1:ny + 1, :])

    # ------------------------------------------------------------------
    # z location via the analytic inverse tanh map
    # ------------------------------------------------------------------

    def _face_coord(self, z):
        """Continuous face index kf in [0, nz] for physical z in [0, Lz]."""
        if self.stretching_type == 'symmetric':
            arg = (2.0 * z / self.Lz - 1.0) * self._tanh_g
            arg = torch.clamp(arg, -1.0 + 1e-15, 1.0 - 1e-15)
            xi = torch.atanh(arg) / self.gamma
            return (xi + 1.0) * (0.5 * self.nz)
        else:  # 'bottom'
            arg = (1.0 - z / self.Lz) * self._tanh_g
            arg = torch.clamp(arg, -1.0 + 1e-15, 1.0 - 1e-15)
            xi = 1.0 - torch.atanh(arg) / self.gamma
            return xi * self.nz

    def _locate_z(self, z, spec):
        """Index m0 of the z-node interval [znodes[m0], znodes[m0+1]] containing z."""
        kf = self._face_coord(z)
        c = torch.clamp(kf.floor().long(), 0, self.nz - 1)
        if spec.ztype == 'faces':
            return c
        # centers: cell c has center znodes[c+1]; pick the interval by comparing
        return c + (z >= spec.znodes[c + 1]).long()

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def _uniform_weights(self, t, order):
        """Lagrange weights on integer nodes off_m = m - (order/2 - 1); t in [0,1)."""
        offs = torch.arange(order, dtype=t.dtype, device=t.device) - (order // 2 - 1)
        d = t.unsqueeze(-1) - offs                     # (N, order)
        w = torch.empty_like(d)
        inv = self._udenom[order]
        for m in range(order):
            num = None
            for l in range(order):
                if l == m:
                    continue
                num = d[:, l] if num is None else num * d[:, l]
            w[:, m] = num * inv[m]
        return w

    def _z_weights(self, z, k0, spec, order):
        """Nonuniform Lagrange weights against the actual z nodes."""
        ar = torch.arange(order, device=z.device)
        zn = spec.znodes[k0.unsqueeze(-1) + ar]        # (N, order)
        d = z.unsqueeze(-1) - zn
        w = torch.empty_like(d)
        for m in range(order):
            num = None
            for l in range(order):
                if l == m:
                    continue
                num = d[:, l] if num is None else num * d[:, l]
            w[:, m] = num
        return w * spec.denom_inv[order][k0]

    # ------------------------------------------------------------------
    # Stencil construction + gather
    # ------------------------------------------------------------------

    def _build_iw(self, spec, x, y, z, order, count_clamp=False):
        """IndexWeights for interpolation at flat physical points (x, y, z).

        x, y may be unwrapped (any real); z must lie inside [z_lo, z_hi] unless
        count_clamp=True, in which case it is clamped here and counted.
        """
        if count_clamp:
            n_clamped = ((z < self.z_lo) | (z > self.z_hi)).sum()
            z = torch.clamp(z, self.z_lo, self.z_hi)
        else:
            n_clamped = torch.zeros((), dtype=torch.int64, device=z.device)

        offs = torch.arange(order, device=x.device) - (order // 2 - 1)

        sx = x / self.dx + spec.shift_x
        ix0 = sx.floor()
        tx = sx - ix0
        ix = torch.remainder(ix0.long().unsqueeze(1) + offs, spec.NX)
        ix_lin = ix * (spec.NY * spec.NZ)
        wx = self._uniform_weights(tx, order)

        sy = y / self.dy + spec.shift_y
        iy0 = sy.floor()
        ty = sy - iy0
        iy = torch.remainder(iy0.long().unsqueeze(1) + offs, spec.NY)
        iy_lin = iy * spec.NZ
        wy = self._uniform_weights(ty, order)

        m0 = self._locate_z(z, spec)
        k0 = torch.clamp(m0 - (order // 2 - 1), 0, spec.NZ - order)
        wz = self._z_weights(z, k0, spec, order)

        return IndexWeights(ix_lin, iy_lin, k0, wx, wy, wz, order, n_clamped)

    def _apply_iw(self, buf, iw):
        """Tensor-product gather-and-sum: interpolate `buf` at the stencils in `iw`."""
        order = iw.order
        F = buf.reshape(-1)
        ar = torch.arange(order, device=F.device)
        lowp = (self.interp_dtype == 'fp32_accum64' and order == self.order)
        if lowp:
            F = F.float()
            wx, wy, wz = iw.wx.float(), iw.wy.float(), iw.wz.float()
        else:
            wx, wy, wz = iw.wx, iw.wy, iw.wz
        acc = torch.zeros(iw.kz0.shape[0], dtype=torch.float64, device=F.device)
        for i in range(order):
            for j in range(order):
                idx = iw.ix_lin[:, i] + iw.iy_lin[:, j] + iw.kz0
                g = F[idx.unsqueeze(1) + ar]           # (N, order), z-contiguous
                inner = (g * wz).sum(dim=1)
                pref = wx[:, i] * wy[:, j]
                if lowp:
                    acc += (pref * inner).double()
                else:
                    acc += pref * inner
        return acc

    def _sample(self, comp, x, y, z):
        """Trilinear sample of the trajectory (mid) velocity component `comp`
        at arbitrary points. Input tensors share one broadcast shape."""
        shape = x.shape
        iw = self._build_iw(self.spec[comp], x.reshape(-1), y.reshape(-1),
                            torch.clamp(z, self.z_lo, self.z_hi).reshape(-1), 2)
        return self._apply_iw(self.mbuf[comp], iw).reshape(shape)

    # ------------------------------------------------------------------
    # Departure points
    # ------------------------------------------------------------------

    def departure_coords(self, comp, dt_t):
        """Departure points (x_d, y_d, z_d) for component `comp`, using the
        mid-velocity node buffers (must be filled beforehand). Iterated
        midpoint rule; x/y unwrapped, z NOT yet clamped (clamped in _build_iw
        where the clamp is also counted)."""
        xa, ya, za, _ = self.arrival[comp]
        half_dt = 0.5 * dt_t

        xm, ym, zm = torch.broadcast_tensors(xa, ya, za)
        for _ in range(self.n_traj_iters):
            us = self._sample('u', xm, ym, zm)
            vs = self._sample('v', xm, ym, zm)
            ws = self._sample('w', xm, ym, zm)
            xm = xa - half_dt * us
            ym = ya - half_dt * vs
            zm = torch.clamp(za - half_dt * ws, self.z_lo, self.z_hi)

        # x_d = x_a - dt*V(x_m) with the same V used for the last midpoint update
        return 2.0 * xm - xa, 2.0 * ym - ya, 2.0 * zm - za

    def compute_departure(self, comp, dt_t):
        """Departure-point stencil for component `comp`."""
        xd, yd, zd = self.departure_coords(comp, dt_t)
        return self._build_iw(self.spec[comp], xd.reshape(-1), yd.reshape(-1),
                              zd.reshape(-1), self.order, count_clamp=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def advect(self, u, v, w, u_mid, v_mid, w_mid, dt_t, extra_rhs=None):
        """Semi-Lagrangian advection of (u, v, w).

        u_mid/v_mid/w_mid: trajectory velocity at t^{n+1/2} (BC-consistent
        ghost-shaped fields, e.g. the AB2 extrapolation 1.5*u^n - 0.5*u^{n-1}).

        Returns (ustar, vstar, wstar) — ghost-shaped, interiors filled with the
        field interpolated at the departure points (ghosts are NOT refreshed
        here; call apply_bc afterwards). w wall faces stay 0.

        extra_rhs: optional (Ru, Rv, Rw) ghost-shaped fields interpolated at the
        SAME departure points (for the v2 scheme); returned as a fourth tuple of
        interior-shaped tensors.
        """
        # Trajectory velocity node buffers
        self._fill(self.mbuf['u'], 'u', u_mid)
        self._fill(self.mbuf['v'], 'v', v_mid)
        self._fill(self.mbuf['w'], 'w', w_mid)

        fields = {'u': u, 'v': v, 'w': w}
        outs = {'u': self.ustar, 'v': self.vstar, 'w': self.wstar}
        extras = {} if extra_rhs is not None else None
        if extra_rhs is not None:
            extra_map = {'u': extra_rhs[0], 'v': extra_rhs[1], 'w': extra_rhs[2]}

        n_clamped = torch.zeros((), dtype=torch.int64, device=self.device)
        for comp in 'uvw':
            iw = self.compute_departure(comp, dt_t)
            self.last_iw[comp] = iw
            n_clamped += iw.n_clamped

            self._fill(self.fbuf[comp], comp, fields[comp])
            vals = self._apply_iw(self.fbuf[comp], iw)

            shape = self.arrival[comp][3]
            out = outs[comp]
            if comp == 'u':
                out[1:self.nx + 1, 1:self.ny + 1, 1:self.nz + 1] = vals.reshape(shape)
            elif comp == 'v':
                out[1:self.nx + 1, 1:self.ny + 1, 1:self.nz + 1] = vals.reshape(shape)
            else:
                out[1:self.nx + 1, 1:self.ny + 1, 1:self.nz] = vals.reshape(shape)

            if extras is not None:
                self._fill(self.fbuf[comp], comp, extra_map[comp])
                extras[comp] = self._apply_iw(self.fbuf[comp], iw).reshape(shape)

        self.n_clamped_last = n_clamped

        if extras is not None:
            return self.ustar, self.vstar, self.wstar, (extras['u'], extras['v'], extras['w'])
        return self.ustar, self.vstar, self.wstar
