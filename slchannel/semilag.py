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

from dataclasses import dataclass

import torch

from . import env


def _gather_interp(Ff, ix_lin, iy_lin, kz0, wx, wy, wz):
    """Tensor-product gather-and-sum, written as a fully unrolled pointwise
    expression so torch.compile fuses it into ONE kernel (no materialized
    (N, order^3) intermediates). This is the performance-critical kernel:
    on the GB10 the compiled fp32 version runs ~40x faster than fp64
    (fp64 throughput is 1/64 of fp32 on consumer Blackwell)."""
    order = wx.shape[1]
    out = None
    for i in range(order):
        for j in range(order):
            base = ix_lin[:, i] + iy_lin[:, j] + kz0
            wij = wx[:, i] * wy[:, j]
            for k in range(order):
                term = (wij * wz[:, k]) * Ff[base + k]
                out = term if out is None else out + term
    return out


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

    ix_lin: torch.Tensor  # (N, order) int64
    iy_lin: torch.Tensor  # (N, order) int64
    kz0: torch.Tensor  # (N,)      int64 — base z-node index
    wx: torch.Tensor  # (N, order)
    wy: torch.Tensor  # (N, order)
    wz: torch.Tensor  # (N, order)
    order: int
    n_clamped: torch.Tensor  # 0-D int64 — departure points clamped at walls


@dataclass
class _GridSpec:
    """Node-grid description for one velocity component."""

    name: str
    shift_x: float  # node coordinate s_x = x/dx + shift_x
    shift_y: float
    NX: int
    NY: int
    NZ: int
    znodes: torch.Tensor  # (NZ,) physical z of the nodes
    ztype: str  # 'centers' | 'faces'
    denom_inv: dict  # order -> (NZ - order + 1, order) inverse denominators


def _z_denominator_table(znodes: torch.Tensor, order: int) -> torch.Tensor:
    """Inverse Lagrange denominators for every contiguous stencil of `order`
    nodes: denom_inv[b, m] = 1 / prod_{l != m} (z[b+m] - z[b+l]).

    Computed on CPU (torch.prod on CUDA fp64 goes through an NVRTC-JIT
    reduction kernel that fails on the GB10/sm_121), then moved to the
    node tensor's device. Init-time only, so the round-trip is free."""
    device = znodes.device
    zn = znodes.cpu().unfold(0, order, 1)  # (n_base, order)
    diff = zn.unsqueeze(2) - zn.unsqueeze(1)  # [b, m, l] = z_m - z_l
    eye = torch.eye(order, dtype=diff.dtype)
    diff = diff + eye.unsqueeze(0)  # 1 on the diagonal
    return (1.0 / diff.prod(dim=2)).to(device)


def _uniform_inv_denominators(order: int) -> list:
    """Inverse denominators for integer-offset nodes off_m = m - (order/2 - 1)."""
    offs = [m - (order // 2 - 1) for m in range(order)]
    inv = []
    for m in range(order):
        d = 1.0
        for l in range(order):
            if l != m:
                d *= offs[m] - offs[l]
        inv.append(1.0 / d)
    return inv


class SLAdvector:
    """Semi-Lagrangian advection of the three staggered velocity components."""

    def __init__(
        self,
        nx,
        ny,
        nz,
        dx,
        dy,
        Lx,
        Ly,
        Lz,
        z_f,
        z_c,
        gamma,
        stretching_type="symmetric",
        order=4,
        traj_order=2,
        n_traj_iters=2,
        top_wall_bc_type="dirichlet",
        interp_dtype="fp64",
        device=torch.device("cpu"),
    ):
        """Build the advector for one grid; reusable across steps and fields.

        Parameters
        ----------
        nx, ny, nz : int
            Interior cell counts. x and y are periodic, z is wall-bounded.
        dx, dy : float
            Uniform spacings in the periodic directions.
        Lx, Ly, Lz : float
            Domain extents. Lx and Ly set the periodic wrap.
        z_f, z_c : torch.Tensor
            Wall-normal face and centre coordinates of the tanh-stretched grid,
            as produced by :func:`slchannel.utils.generate_grid`. Interpolation
            weights are built against these actual nodes: `z_c` are face
            midpoints, *not* the image of uniform computational points, so
            uniform-xi weights in z would silently lose an order.
        gamma : float
            Stretching parameter of that tanh map, needed to invert it
            analytically when locating a z stencil.
        stretching_type : {'symmetric', 'bottom'}
            Which tanh map was used.
        order : {4, 6}
            Field-interpolation order: 4 is tricubic, 6 triquintic. This is the
            dominant control on interpolation dissipation, and hence on how
            faithfully near-wall statistics survive.
        traj_order : {2, 4}
            Order used when sampling the velocity *along the trajectory*. The
            default 2 (trilinear) is much faster but its C0 interpolant caps
            overall temporal convergence at O(dt) with a small h^2 coefficient;
            4 restores clean O(dt^2) and is what the convergence tests use.
        n_traj_iters : int
            Fixed-point iterations of the midpoint rule for the departure point.
            2 is enough at trajectory CFL 2-5.
        top_wall_bc_type : {'dirichlet', 'neumann'}
            Sets the ghost-cell extension used at the top boundary.
        interp_dtype : {'fp64', 'fp32_accum64'}
            Precision of the interpolation arithmetic. ``fp32_accum64`` enables
            the Triton fast path on CUDA (with `traj_order=2`) and is the
            production choice: fp64 flops are heavily rate-limited on the target
            GPUs, and the interpolation is flop-dense.
        device : torch.device
            Where the node buffers and index tables live.

        Raises
        ------
        ValueError
            If `order`, `traj_order`, `stretching_type` or `interp_dtype` is not
            one of the supported values, or the grid is too small for the
            stencil.
        """
        if order not in (4, 6):
            raise ValueError(f"sl.interp_order must be 4 (tricubic) or 6 (triquintic), got {order}")
        if traj_order not in (2, 4):
            raise ValueError(
                f"sl.traj_interp_order must be 2 (trilinear) or 4 (tricubic), got {traj_order}"
            )
        if stretching_type not in ("symmetric", "bottom"):
            raise ValueError(
                f"SLAdvector supports 'symmetric'/'bottom' stretching, got '{stretching_type}'"
            )
        if interp_dtype not in ("fp64", "fp32_accum64"):
            raise ValueError(
                f"sl.interp_dtype must be 'fp64' or 'fp32_accum64', got {interp_dtype}"
            )
        if nz + 1 < order:
            raise ValueError(f"nz={nz} too small for interpolation order {order}")

        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy = dx, dy
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.gamma = float(gamma)
        self.stretching_type = stretching_type
        self.order = order
        # Spatial order of the trajectory-velocity sampling. Trilinear (2) is
        # cheap but its C^0 interpolant caps the whole scheme at O(dt) with a
        # small h^2-proportional coefficient (the flow map of the interpolated
        # velocity differs from the exact one); tricubic (4) restores clean
        # O(dt^2) for accuracy studies at ~4x the trajectory-sampling cost.
        self.traj_order = traj_order
        self.n_traj_iters = n_traj_iters
        self.top_wall_bc_type = top_wall_bc_type
        self.interp_dtype = interp_dtype
        self.device = device

        self.z_f = z_f
        self.z_c = z_c
        self._tanh_g = torch.tanh(
            torch.tensor(self.gamma, dtype=torch.float64, device=device)
        ).item()

        # Departure z kept strictly inside the channel
        eps_z = 1e-12 * Lz
        self.z_lo = (z_f[0] + eps_z).item()
        self.z_hi = (z_f[-1] - eps_z).item()

        # In fp32_accum64 mode the ENTIRE interpolation pipeline runs in fp32:
        # node buffers, coordinates, weights, and the trajectory math. On the
        # GB10 fp64 throughput is 1/64 of fp32, and the coordinate/weight math
        # (floor, remainder, atanh, Lagrange products) is what dominates the
        # semi-Lagrangian cost — fp64 there is ~40x slower for accuracy far
        # below the scheme's truncation error (positions good to ~1e-7*L).
        self._buf_dtype = torch.float32 if interp_dtype == "fp32_accum64" else torch.float64
        self._coord_dtype = self._buf_dtype

        # --- node-grid specs -------------------------------------------------
        # u: nodes x = i*dx (i=0..nx-1), y = (j+1/2)*dy (j=0..ny-1), z = z_c (ghosts incl.)
        # v: nodes x = (i+1/2)*dx,       y = j*dy,               z = z_c
        # w: nodes x = (i+1/2)*dx,       y = (j+1/2)*dy,         z = z_f (walls incl.)
        # znodes/denominator tables are stored in the coordinate dtype (the
        # denominators are COMPUTED in fp64 first — near-wall spacings are
        # small — then cast).
        def make_spec(name, shift_x, shift_y, znodes, ztype):
            denoms = {
                o: _z_denominator_table(znodes, o).to(self._coord_dtype)
                for o in {2, traj_order, order}
            }
            return _GridSpec(
                name,
                shift_x,
                shift_y,
                nx,
                ny,
                len(znodes),
                znodes.to(self._coord_dtype),
                ztype,
                denoms,
            )

        self.spec = {
            "u": make_spec("u", 0.0, -0.5, z_c, "centers"),
            "v": make_spec("v", -0.5, 0.0, z_c, "centers"),
            "w": make_spec("w", -0.5, -0.5, z_f, "faces"),
        }

        # Uniform-direction inverse denominators (python floats, baked into weights)
        self._udenom = {o: _uniform_inv_denominators(o) for o in {2, traj_order, order}}

        # --- arrival coordinates (broadcastable, physical) -------------------
        ar = torch.arange(1, nx + 1, dtype=self._coord_dtype, device=device)
        aj = torch.arange(1, ny + 1, dtype=self._coord_dtype, device=device)
        x_face = (ar * dx).view(nx, 1, 1)
        x_cent = ((ar - 0.5) * dx).view(nx, 1, 1)
        y_face = (aj * dy).view(1, ny, 1)
        y_cent = ((aj - 0.5) * dy).view(1, ny, 1)
        z_cent = z_c[1 : nz + 1].to(self._coord_dtype).view(1, 1, nz)
        z_facei = (
            z_f[1:nz].to(self._coord_dtype).view(1, 1, nz - 1)
        )  # interior faces; walls pinned w=0

        self.arrival = {
            "u": (x_face, y_cent, z_cent, (nx, ny, nz)),
            "v": (x_cent, y_face, z_cent, (nx, ny, nz)),
            "w": (x_cent, y_cent, z_facei, (nx, ny, nz - 1)),
        }

        # --- compiled fast path -----------------------------------------------
        # Opt-in via TORCHANNEL_COMPILE=1 (repo convention; needs CC=gcc on the
        # GB10). The gather kernel and the stencil build are compiled; the
        # eager path remains the bit-exact reference used by the CPU tests.
        self._use_compiled = env.USE_COMPILE and device.type == "cuda"
        if self._use_compiled:
            from torch import _dynamo as _torch_dynamo
            from torch import _inductor

            _torch_dynamo.config.cache_size_limit = 64
            # keep multi-use pointwise intermediates virtual (recompute instead
            # of materializing (N, order) weight tensors): ~3x on this workload
            _inductor.config.realize_reads_threshold = 10**9
            _inductor.config.realize_opcount_threshold = 10**9
            # ONE fused graph per component: arrival coords -> midpoint
            # trajectory iterations (inline trilinear gathers) -> stencil
            # build -> high-order gather. Everything is pointwise per output
            # element, so the (N, order) index/weight tensors stay virtual
            # (register-resident) instead of costing ~10 GB of traffic per
            # interpolation at production size.
            self._gather_c = torch.compile(_gather_interp, dynamic=False)
            self._dep_c = torch.compile(self._compute_departure_impl, dynamic=False)
            self._advect_c = torch.compile(self._advect_comp_impl, dynamic=False)
        else:
            self._gather_c = None
            self._dep_c = None
            self._advect_c = None

        # Hand-written Triton kernels: the fastest path (weights/indices fully
        # register-resident). Used automatically for the fp32 pipeline with
        # trilinear trajectories on CUDA; disable with SLCHANNEL_TRITON=0.
        self._triton = None
        if (
            device.type == "cuda"
            and interp_dtype == "fp32_accum64"
            and traj_order == 2
            and env.USE_TRITON
        ):
            try:
                from .semilag_triton import TritonSL

                self._triton = TritonSL(self)
            except Exception as e:
                print(
                    f"[semilag] Triton fast path unavailable ({e}); "
                    f"falling back to torch.compile/eager",
                    flush=True,
                )
        # index dtype: int32 halves index traffic on the compiled path (flat
        # indices stay < 2^31 for any realistic grid)
        self._idx_dtype = torch.int32 if self._use_compiled else torch.int64

        # --- persistent node buffers -----------------------------------------
        # One set for the advected fields, one for the trajectory (mid) velocity.
        def alloc(spec):
            return torch.empty(spec.NX, spec.NY, spec.NZ, dtype=self._buf_dtype, device=device)

        self.fbuf = {c: alloc(self.spec[c]) for c in "uvw"}
        self.mbuf = {c: alloc(self.spec[c]) for c in "uvw"}

        # Preallocated ghost-shaped outputs
        self.ustar = torch.zeros(nx + 1, ny + 2, nz + 2, dtype=torch.float64, device=device)
        self.vstar = torch.zeros(nx + 2, ny + 1, nz + 2, dtype=torch.float64, device=device)
        self.wstar = torch.zeros(nx + 2, ny + 2, nz + 1, dtype=torch.float64, device=device)

        # Diagnostics: departure points clamped at the walls, last step
        self.n_clamped_last = torch.zeros((), dtype=torch.int64, device=device)

        # Last departure stencils (eager path only; the fused fast path never
        # materializes them)
        self.last_iw = {}
        # node buffers for extra RHS fields (v2), allocated on first use
        self._ebuf = []

    # ------------------------------------------------------------------
    # Node-buffer filling (views of the ghost-shaped fields)
    # ------------------------------------------------------------------

    def _fill(self, buf, comp, field):
        nx, ny = self.nx, self.ny
        if comp == "u":
            buf.copy_(field[0:nx, 1 : ny + 1, :])
        elif comp == "v":
            buf.copy_(field[1 : nx + 1, 0:ny, :])
        else:  # w
            buf.copy_(field[1 : nx + 1, 1 : ny + 1, :])

    # ------------------------------------------------------------------
    # z location via the analytic inverse tanh map
    # ------------------------------------------------------------------

    def _face_coord(self, z):
        """Continuous face index kf in [0, nz] for physical z in [0, Lz]."""
        if self.stretching_type == "symmetric":
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
        if spec.ztype == "faces":
            return c
        # centers: cell c has center znodes[c+1]; pick the interval by comparing
        return c + (z >= spec.znodes[c + 1]).long()

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def _uniform_weights(self, t, order):
        """Lagrange weights on integer nodes off_m = m - (order/2 - 1); t in [0,1)."""
        offs = torch.arange(order, dtype=t.dtype, device=t.device) - (order // 2 - 1)
        d = t.unsqueeze(-1) - offs  # (N, order)
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
        zn = spec.znodes[k0.unsqueeze(-1) + ar]  # (N, order)
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
        ix_lin, iy_lin, kz0, wx, wy, wz, n_clamped = self._build_iw_impl(
            spec, x, y, z, order, count_clamp
        )
        return IndexWeights(ix_lin, iy_lin, kz0, wx, wy, wz, order, n_clamped)

    def _build_iw_impl(self, spec, x, y, z, order, count_clamp):
        if count_clamp:
            n_clamped = ((z < self.z_lo) | (z > self.z_hi)).sum()
            z = torch.clamp(z, self.z_lo, self.z_hi)
        else:
            n_clamped = torch.zeros((), dtype=torch.int64, device=z.device)

        offs = torch.arange(order, device=x.device) - (order // 2 - 1)

        # coordinates/weights in fp64 (near-wall z spacings are small), then
        # cast weights to the buffer dtype and indices to the index dtype
        sx = x / self.dx + spec.shift_x
        ix0 = sx.floor()
        tx = sx - ix0
        ix = torch.remainder(ix0.long().unsqueeze(1) + offs, spec.NX)
        ix_lin = (ix * (spec.NY * spec.NZ)).to(self._idx_dtype)
        wx = self._uniform_weights(tx, order).to(self._buf_dtype)

        sy = y / self.dy + spec.shift_y
        iy0 = sy.floor()
        ty = sy - iy0
        iy = torch.remainder(iy0.long().unsqueeze(1) + offs, spec.NY)
        iy_lin = (iy * spec.NZ).to(self._idx_dtype)
        wy = self._uniform_weights(ty, order).to(self._buf_dtype)

        m0 = self._locate_z(z, spec)
        k0 = torch.clamp(m0 - (order // 2 - 1), 0, spec.NZ - order)
        wz = self._z_weights(z, k0, spec, order).to(self._buf_dtype)

        return ix_lin, iy_lin, k0.to(self._idx_dtype), wx, wy, wz, n_clamped

    def _apply_iw(self, buf, iw):
        """Interpolate `buf` at the stencils in `iw` (fused kernel when
        compiled; eager reference loop otherwise)."""
        Ff = buf.reshape(-1)
        if self._gather_c is not None:
            return self._gather_c(Ff, iw.ix_lin, iw.iy_lin, iw.kz0, iw.wx, iw.wy, iw.wz)
        return _gather_interp(
            Ff, iw.ix_lin.long(), iw.iy_lin.long(), iw.kz0.long(), iw.wx, iw.wy, iw.wz
        )

    def _sample(self, comp, x, y, z):
        """Sample the trajectory (mid) velocity component `comp` at arbitrary
        points (order = self.traj_order). Inputs share one broadcast shape.
        Calls the RAW helpers so it inlines cleanly when the whole departure
        computation is traced by torch.compile."""
        shape = x.shape
        ix_lin, iy_lin, kz0, wx, wy, wz, _ = self._build_iw_impl(
            self.spec[comp],
            x.reshape(-1),
            y.reshape(-1),
            torch.clamp(z, self.z_lo, self.z_hi).reshape(-1),
            self.traj_order,
            False,
        )
        return _gather_interp(self.mbuf[comp].reshape(-1), ix_lin, iy_lin, kz0, wx, wy, wz).reshape(
            shape
        )

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
            us = self._sample("u", xm, ym, zm)
            vs = self._sample("v", xm, ym, zm)
            ws = self._sample("w", xm, ym, zm)
            xm = xa - half_dt * us
            ym = ya - half_dt * vs
            zm = torch.clamp(za - half_dt * ws, self.z_lo, self.z_hi)

        # x_d = x_a - dt*V(x_m) with the same V used for the last midpoint update
        return 2.0 * xm - xa, 2.0 * ym - ya, 2.0 * zm - za

    def _compute_departure_impl(self, comp, dt_t):
        xd, yd, zd = self.departure_coords(comp, dt_t)
        return self._build_iw_impl(
            self.spec[comp], xd.reshape(-1), yd.reshape(-1), zd.reshape(-1), self.order, True
        )

    def _advect_comp_impl(self, comp, dt_t, F_flat, extra_flats):
        """End-to-end advection of one component: departure points, stencil,
        gather of the field (and of any extra RHS fields with the same
        stencil). Written as a single pointwise chain for torch.compile."""
        xd, yd, zd = self.departure_coords(comp, dt_t)
        x, y, z = xd.reshape(-1), yd.reshape(-1), zd.reshape(-1)
        ix_lin, iy_lin, kz0, wx, wy, wz, n_clamped = self._build_iw_impl(
            self.spec[comp], x, y, z, self.order, True
        )
        out = _gather_interp(F_flat, ix_lin, iy_lin, kz0, wx, wy, wz)
        extras = tuple(_gather_interp(E, ix_lin, iy_lin, kz0, wx, wy, wz) for E in extra_flats)
        return out, extras, n_clamped

    def compute_departure(self, comp, dt_t):
        """Departure-point stencil for component `comp` (compiled as ONE fused
        graph — trajectory iterations + stencil build — on the fast path)."""
        fn = self._dep_c if self._dep_c is not None else self._compute_departure_impl
        ix_lin, iy_lin, kz0, wx, wy, wz, n_clamped = fn(comp, dt_t)
        return IndexWeights(ix_lin, iy_lin, kz0, wx, wy, wz, self.order, n_clamped)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def advect(self, u, v, w, u_mid, v_mid, w_mid, dt_t, extra_rhs=None):
        """Semi-Lagrangian advection of (u, v, w).

        u_mid/v_mid/w_mid: the advecting (trajectory) velocity — BC-consistent
        ghost-shaped fields. The machinery is agnostic to its time level: the
        v1/v2 schemes pass an estimate of V^{n+1/2} (e.g. the AB2
        extrapolation 1.5*u^n - 0.5*u^{n-1}); the bdf2 scheme passes the
        frozen U* = 2*u^n - u^{n-1} and calls advect() twice (dt_t and
        2*dt_t) for the two feet of the same characteristic.

        Returns (ustar, vstar, wstar) — ghost-shaped, interiors filled with the
        field interpolated at the departure points (ghosts are NOT refreshed
        here; call apply_bc afterwards). w wall faces stay 0.

        extra_rhs: optional LIST of (Ru, Rv, Rw) ghost-shaped field triples,
        each interpolated at the SAME departure points (used by the v2 scheme
        for the explicit RHS and the departure-half of the z-diffusion);
        returned as a fourth element: a list of interior-shaped triples.
        """
        # Trajectory velocity node buffers
        self._fill(self.mbuf["u"], "u", u_mid)
        self._fill(self.mbuf["v"], "v", v_mid)
        self._fill(self.mbuf["w"], "w", w_mid)

        fields = {"u": u, "v": v, "w": w}
        outs = {"u": self.ustar, "v": self.vstar, "w": self.wstar}
        extras = [dict() for _ in extra_rhs] if extra_rhs is not None else None
        if extra_rhs is not None and len(self._ebuf) < len(extra_rhs):
            # lazily allocated node buffers for the extra RHS fields
            while len(self._ebuf) < len(extra_rhs):
                self._ebuf.append({c: torch.empty_like(self.fbuf[c]) for c in "uvw"})

        n_clamped = torch.zeros((), dtype=torch.int64, device=self.device)
        if self._triton is not None:
            self._triton.nclamp.zero_()
            dt_f = float(dt_t)
        for ic, comp in enumerate("uvw"):
            shape = self.arrival[comp][3]
            out = outs[comp]
            self._fill(self.fbuf[comp], comp, fields[comp])
            if extras is not None:
                for t, triple in enumerate(extra_rhs):
                    self._fill(self._ebuf[t][comp], comp, triple[ic])

            if self._triton is not None:
                # hand-written Triton kernels (registers-only stencil path).
                # Extras first (gather() reuses one output buffer per comp, so
                # they must be cloned before the field gather overwrites it).
                n = self._triton.departure(comp, dt_f)
                if extras is not None:
                    for t in range(len(extra_rhs)):
                        extras[t][comp] = (
                            self._triton.gather(comp, self._ebuf[t][comp], n).clone().reshape(shape)
                        )
                vals = self._triton.gather(comp, self.fbuf[comp], n)
            elif self._advect_c is not None:
                # fused fast path: one compiled graph per component
                eflats = (
                    tuple(self._ebuf[t][comp].reshape(-1) for t in range(len(extra_rhs)))
                    if extra_rhs is not None
                    else ()
                )
                vals, evals, ncl = self._advect_c(comp, dt_t, self.fbuf[comp].reshape(-1), eflats)
                n_clamped += ncl
                if extras is not None:
                    for t in range(len(extra_rhs)):
                        extras[t][comp] = evals[t].reshape(shape)
            else:
                iw = self.compute_departure(comp, dt_t)
                self.last_iw[comp] = iw
                n_clamped += iw.n_clamped
                vals = self._apply_iw(self.fbuf[comp], iw)
                if extras is not None:
                    for t in range(len(extra_rhs)):
                        extras[t][comp] = self._apply_iw(self._ebuf[t][comp], iw).reshape(shape)

            if comp == "w":
                out[1 : self.nx + 1, 1 : self.ny + 1, 1 : self.nz] = vals.reshape(shape)
            else:
                out[1 : self.nx + 1, 1 : self.ny + 1, 1 : self.nz + 1] = vals.reshape(shape)

        if self._triton is not None:
            n_clamped = self._triton.nclamp[0].long()
        self.n_clamped_last = n_clamped

        if extras is not None:
            return self.ustar, self.vstar, self.wstar, [(e["u"], e["v"], e["w"]) for e in extras]
        return self.ustar, self.vstar, self.wstar
