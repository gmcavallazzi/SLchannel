"""Decomposed BDF2 step driver, mirroring SLChannelFlow line by line.

Mirrored production code (slchannel/solver.py): the BDF1 bootstrap
(solver.py:703-786), step_sl_bdf2 (solver.py:788-993, non-incremental
pressure), _explicit_xy_rhs via the diffusion_xy_* stencils
(operators.py:293-367, arithmetic order preserved verbatim), the z-wall
closures of apply_bc_all (solver.py:70-107), project_velocity
(projection.py:251-272) and the exact-flux bulk forcing
(solver.py:669-698). The z-implicit solves and the (gathered) Poisson solve
reuse the production functions unchanged.

Per-rank state lives in extended node arrays (see decomp.py). After each SL
advect the shared staggered face (u in x, v in y) is computed by the arrival
rank in its trailing halo slot and pulled by the owning rank
(comm.pull_minus_edge) before anything reads owned data.
"""

import torch

from slchannel.operators import (
    solve_implicit_diffusion_u,
    solve_implicit_diffusion_v,
    solve_implicit_diffusion_w,
)
from slchannel.utils import compute_bulk_velocity, compute_divergence

from .poisson_gather import solve_poisson_gathered
from .poisson_pencil import solve_poisson_pencil
from .sl_local import LocalSL


class DecomposedBDF2:
    def __init__(self, mono, decomp, comm, poisson="gathered", use_triton=False, bulk="gathered"):
        """Seed from a monolithic SLChannelFlow instance (grid, transport
        coefficients, fft_data and the reference advector all come from it;
        its current u/v/w become the initial state).

        bulk: 'gathered' reduces the bulk velocity on the allgathered field
        with production's own reduction order (bit-identical to the
        monolithic solver, but a full-field allgather EVERY step);
        'local' sums each rank's owned faces and allreduces (same value to
        ~1e-15, no allgather — the production choice on real multi-GPU)."""
        from .decomp import mono_node_view

        self.d = decomp
        self.comm = comm
        self.nx, self.ny, self.nz = mono.nx, mono.ny, mono.nz
        self.dx, self.dy = mono.dx, mono.dy
        self.nu, self.U_bulk = mono.nu, mono.U_bulk
        self.dz_c, self.dz_f = mono.dz_c, mono.dz_f
        self.top_wall = mono.top_wall_bc_type
        self.fft_data = mono.fft_data
        self.total_volume = mono.total_volume
        self.cell_vol_ratio = mono.cell_vol_ratio
        self.ref = mono.sl
        assert poisson in ("gathered", "pencil")
        self.poisson = poisson
        assert bulk in ("gathered", "local")
        self.bulk = bulk

        ranks = comm.local_ranks
        self.sl = {r: LocalSL(self.ref, decomp, r) for r in ranks}
        self.tsl = None
        if use_triton:
            from .sl_triton_local import TritonLocalSL

            self.tsl = {r: TritonLocalSL(self.sl[r]) for r in ranks}
        self.state = {}
        if mono.u is not None:
            for c in "uvw":
                nodes = mono_node_view(getattr(mono, c), c, self.nx, self.ny)
                full = decomp.scatter(nodes.contiguous(), c, fill_halos=True)
                self.state[c] = {r: full[r] for r in ranks}
        else:
            # parameters-only mono (production driver): state is seeded
            # afterwards through set_state_nodes
            for c in "uvw":
                self.state[c] = {r: decomp.alloc(c, device=comm.device) for r in ranks}
        self.p = None
        self.nm1 = None
        self.rxy_prev = None
        self.dt_prev = None
        # arrival-region ext slices per component (see sl_local docstring)
        H, nxl, nyl, nz = decomp.H, decomp.nxl, decomp.nyl, self.nz
        self.arr = {
            "u": (slice(H + 1, H + nxl + 1), slice(H, H + nyl), slice(1, nz + 1)),
            "v": (slice(H, H + nxl), slice(H + 1, H + nyl + 1), slice(1, nz + 1)),
            "w": (slice(H, H + nxl), slice(H, H + nyl), slice(1, nz)),
        }
        self.own = (slice(H, H + nxl), slice(H, H + nyl))

    # ---- boundary closures + exchange (apply_bc_all, solver.py:70-107) ----

    def _z_closures(self, fields, comp):
        nz = self.nz
        top = 1.0 if (self.top_wall == "neumann" and comp in "uv") else -1.0
        for ext in fields.values():
            if comp == "w":
                ext[:, :, 0] = 0.0
                ext[:, :, nz] = 0.0
            else:
                ext[:, :, 0] = -ext[:, :, 1]
                ext[:, :, nz + 1] = top * ext[:, :, nz]

    def _bc_exchange(self, comps=("u", "v", "w")):
        for c in comps:
            self._z_closures(self.state[c], c)
            self.comm.halo_exchange(self.state[c])

    # ---- explicit xy diffusion at the arrival slots ----
    # (fork of diffusion_xy_* stencils, operators.py:293-367; same op order)

    def _rhs_arrival(self, comp, fields):
        xs, ys, zs = self.arr[comp]
        out = {}
        for r, f in fields.items():
            x0, x1 = xs.start, xs.stop
            y0, y1 = ys.start, ys.stop
            d2x = (
                f[x0 + 1 : x1 + 1, y0:y1, zs]
                - 2 * f[x0:x1, y0:y1, zs]
                + f[x0 - 1 : x1 - 1, y0:y1, zs]
            ) / self.dx**2
            d2y = (
                f[x0:x1, y0 + 1 : y1 + 1, zs]
                - 2 * f[x0:x1, y0:y1, zs]
                + f[x0:x1, y0 - 1 : y1 - 1, zs]
            ) / self.dy**2
            out[r] = self.nu * (d2x + d2y)
        return out

    # ---- shared plumbing for both step flavours ----

    def _adopt_predictor(self, star):
        """star: dict comp -> dict rank -> ext with values at arrival slots.
        Pull the shared faces, close walls, exchange -> complete new state."""
        self.comm.pull_minus_edge(star["u"], dim=0)
        self.comm.pull_minus_edge(star["v"], dim=1)
        self.state = star
        self._bc_exchange()

    def _z_solve_all(self, dt_eff_t):
        d, nxl, nyl, nz = self.d, self.d.nxl, self.d.nyl, self.nz
        H = d.H
        for r in self.comm.local_ranks:
            res = solve_implicit_diffusion_u(
                d.zview(self.state["u"][r], "u"),
                dt_eff_t,
                nxl,
                nyl,
                nz,
                self.dz_c,
                self.dz_f,
                self.nu,
                theta=1.0,
                top_wall_bc_type=self.top_wall,
            )
            self.state["u"][r][H : H + nxl, H : H + nyl, 1 : nz + 1] = res[
                1 : nxl + 1, 1 : nyl + 1, 1 : nz + 1
            ]
            res = solve_implicit_diffusion_v(
                d.zview(self.state["v"][r], "v"),
                dt_eff_t,
                nxl,
                nyl,
                nz,
                self.dz_c,
                self.dz_f,
                self.nu,
                theta=1.0,
                top_wall_bc_type=self.top_wall,
            )
            self.state["v"][r][H : H + nxl, H : H + nyl, 1 : nz + 1] = res[
                1 : nxl + 1, 1 : nyl + 1, 1 : nz + 1
            ]
            res = solve_implicit_diffusion_w(
                d.zview(self.state["w"][r], "w"),
                dt_eff_t,
                nxl,
                nyl,
                nz,
                self.dz_c,
                self.dz_f,
                self.nu,
                theta=1.0,
            )
            self.state["w"][r][H : H + nxl, H : H + nyl, 1:nz] = res[1 : nxl + 1, 1 : nyl + 1, 1:nz]
        self._bc_exchange()

    def _project(self, dt_eff, dt_eff_t):
        d, nxl, nyl, nz = self.d, self.d.nxl, self.d.nyl, self.nz
        H = d.H
        div = {
            r: compute_divergence(
                d.dview(self.state["u"][r], "u"),
                d.dview(self.state["v"][r], "v"),
                d.dview(self.state["w"][r], "w"),
                nxl,
                nyl,
                nz,
                self.dx,
                self.dy,
                self.dz_f,
            )
            for r in self.comm.local_ranks
        }
        if self.poisson == "pencil":
            p_ext = solve_poisson_pencil(div, self.comm, self.d, self.fft_data, dt_eff)
        else:
            p_ext = solve_poisson_gathered(div, self.comm, self.d, self.fft_data, dt_eff)
        ox, oy = self.own
        for r in self.comm.local_ranks:
            p = p_ext[r]
            u, v, w = self.state["u"][r], self.state["v"][r], self.state["w"][r]
            # fork of project_velocity (projection.py:251-272), node layout
            dp_dx = (p[ox, oy, 1 : nz + 1] - p[H - 1 : H + nxl - 1, oy, 1 : nz + 1]) / self.dx
            u[ox, oy, 1 : nz + 1] -= dt_eff_t * dp_dx
            dp_dy = (p[ox, oy, 1 : nz + 1] - p[ox, H - 1 : H + nyl - 1, 1 : nz + 1]) / self.dy
            v[ox, oy, 1 : nz + 1] -= dt_eff_t * dp_dy
            dp_dz = (p[ox, oy, 2 : nz + 1] - p[ox, oy, 1:nz]) / self.dz_c[1:nz].view(1, 1, -1)
            w[ox, oy, 1:nz] -= dt_eff_t * dp_dz
        self._bc_exchange()
        self.p = p_ext
        return p_ext

    def _bulk_forcing(self, dt):
        """Exact-flux uniform shift (fork of _apply_bulk_forcing,
        solver.py:669-698). bulk='gathered': the volume sum runs on the
        GATHERED monolithic field with production's own
        compute_bulk_velocity, so the reduction order -- and therefore every
        subsequent step -- is bit-identical to the single-GPU solver.
        bulk='local': each rank sums its owned faces and the partial sums
        are allreduced -- no allgather; every rank applies the identical
        correction (collectives return the same bits on every rank), but the
        reduction order differs from the monolithic solver at the 1e-16
        level, which chaotic amplification turns into full field
        decorrelation over O(100) time units (statistics unaffected)."""
        H, nxl, nyl, nz = self.d.H, self.d.nxl, self.d.nyl, self.nz
        if self.bulk == "local":
            w = (self.dx * self.dy) * self.dz_f.view(1, 1, -1)
            partial = {
                r: (self.state["u"][r][H : H + nxl, H : H + nyl, 1 : nz + 1] * w).sum()
                for r in self.comm.local_ranks
            }
            total = self.comm.allreduce(partial, op="sum")
            for r in self.comm.local_ranks:
                # float() so the correction is a device-agnostic scalar; the
                # allreduce returns the same bits on every rank
                corr = self.U_bulk - float(total[r]) / self.total_volume
                self.state["u"][r] += corr
        else:
            from .decomp import nodes_to_mono

            full = self.comm.allgather_nodes(self.state["u"])
            for r in self.comm.local_ranks:
                u_mono = nodes_to_mono(full[r], "u", self.nx, self.ny)
                u_bulk = compute_bulk_velocity(u_mono, self.cell_vol_ratio, self.total_volume)
                corr = self.U_bulk - u_bulk
                self.state["u"][r] += corr
        self._bc_exchange(("u",))
        return corr / dt

    # ---- the two step flavours ----

    def _advect_all(self, fields, mids, dt_t):
        outs = {}
        adv = self.tsl if self.tsl is not None else self.sl
        for r in self.comm.local_ranks:
            o = adv[r].advect(
                {c: fields[c][r] for c in "uvw"},
                {c: mids[c][r] for c in "uvw"},
                dt_t,
            )
            # production writes the fp32 Triton gather into fp64 buffers
            # before the BDF2 combination (semilag.py:660-663) -- mirror that
            outs[r] = {c: v.to(torch.float64) for c, v in o.items()}
        return outs

    def _star_from(self, vals):
        """Fresh ext tensors with `vals[r][comp]` written at the arrival
        slots."""
        star = {c: {} for c in "uvw"}
        for c in "uvw":
            for r in self.comm.local_ranks:
                ext = self.d.alloc(c, dtype=vals[r][c].dtype, device=vals[r][c].device)
                ext[self.arr[c]] = vals[r][c]
                star[c][r] = ext
        return star

    def step(self, dt):
        dt_t = torch.as_tensor(dt, dtype=torch.float64)
        self._bc_exchange()
        if self.nm1 is None or self.dt_prev is None or abs(dt - self.dt_prev) > 1e-12 * dt:
            return self._bootstrap(dt, dt_t)

        # explicit xy diffusion + AB2 extrapolation (solver.py:822-831)
        rhs = {c: self._rhs_arrival(c, self.state[c]) for c in "uvw"}
        Rh = {
            c: {r: 2.0 * rhs[c][r] - self.rxy_prev[c][r] for r in self.comm.local_ranks}
            for c in "uvw"
        }
        self.rxy_prev = rhs

        # frozen U* = 2 V^n - V^{n-1} (solver.py:838-844)
        mids = {
            c: {r: 2.0 * self.state[c][r] - self.nm1[c][r] for r in self.comm.local_ranks}
            for c in "uvw"
        }

        far = self._advect_all(self.nm1, mids, 2.0 * dt_t)
        near = self._advect_all(self.state, mids, dt_t)

        # BDF2 predictor (solver.py:869-881): (4 near - far)/3 + dt_eff * Rh
        dt_eff = (2.0 / 3.0) * dt
        dt_eff_t = (2.0 / 3.0) * dt_t
        vals = {}
        for r in self.comm.local_ranks:
            vals[r] = {}
            for c in "uvw":
                v = near[r][c].mul_(4.0 / 3.0).add_(far[r][c], alpha=-1.0 / 3.0)
                v += dt_eff_t * Rh[c][r]
                vals[r][c] = v
        star = self._star_from(vals)

        # rotate history BEFORE adopting (solver.py:917-922)
        for c in "uvw":
            for r in self.comm.local_ranks:
                self.nm1[c][r].copy_(self.state[c][r])
        self._adopt_predictor(star)
        self._z_solve_all(dt_eff_t)
        self._project(dt_eff, dt_eff_t)
        self.dt_prev = dt
        return self._bulk_forcing(dt)

    def _bootstrap(self, dt, dt_t):
        """BDF1 re-bootstrap (solver.py:703-786): single dt-deep foot with
        U* = V^n; seeds the histories."""
        rhs = {c: self._rhs_arrival(c, self.state[c]) for c in "uvw"}
        self.rxy_prev = rhs
        self.nm1 = {c: {r: self.state[c][r].clone() for r in self.comm.local_ranks} for c in "uvw"}
        outs = self._advect_all(self.state, self.state, dt_t)
        vals = {}
        for r in self.comm.local_ranks:
            vals[r] = {c: outs[r][c] + dt_t * rhs[c][r] for c in "uvw"}
        star = self._star_from(vals)
        self._adopt_predictor(star)
        self._z_solve_all(dt_t)
        self._project(dt, dt_t)
        self.dt_prev = dt
        return self._bulk_forcing(dt)

    # ---- seeding and gathering ----

    def set_state_nodes(self, nodes_by_comp):
        """Seed u/v/w from monolithic node arrays, present on the root rank
        only (None elsewhere under a distributed comm). Resets the BDF2
        history, so the next step re-bootstraps — same policy as a
        monolithic restart."""
        for c in "uvw":
            self.state[c] = self.comm.scatter_nodes(nodes_by_comp.get(c), c)
        self.nm1 = None
        self.rxy_prev = None
        self.dt_prev = None
        self._bc_exchange()

    def gather_nodes(self, comp):
        return self.d.gather(self.state[comp])

    def gather_mono(self, comp):
        """Full monolithic ghost-shaped array of a component (or the last
        pressure for comp='p'), assembled through the communicator so it
        works under both backends. Every rank returns the same tensor."""
        from .decomp import nodes_to_mono

        fields = self.p if comp == "p" else self.state[comp]
        if fields is None:
            return None
        full = self.comm.allgather_nodes(fields)
        nodes = full[self.comm.local_ranks[0]]
        return nodes_to_mono(nodes, "p" if comp == "p" else comp, self.nx, self.ny)
