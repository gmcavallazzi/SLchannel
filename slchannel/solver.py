"""
solver.py — SLChannelFlow: channel-flow DNS driver with semi-Lagrangian
advection (adapted from torChannel's ChannelFlow).

Two advection schemes, selected by `advection.scheme` in the YAML config:
  - 'sl'       : semi-Lagrangian advection (semilag.SLAdvector) + explicit
                 xy-diffusion + Crank-Nicolson implicit z-diffusion + FFT
                 pressure projection. Unconditionally stable in advection:
                 CFL_target is a TRAJECTORY CFL (2-5), dt is limited by
                 accuracy and by the explicit xy-diffusion stability cap.
  - 'eulerian' : torChannel's IMEX (AB2 explicit advection) or FE schemes,
                 kept verbatim as the like-for-like reference.

The SL time scheme is the Boukir et al. (1997) second-order characteristics
scheme (the only survivor of the 2026 M3/fix campaigns — earlier v1/v2/pc
variants were removed after bdf2 won the stability-vs-accuracy map):
BDF2 in time along ONE backward characteristic traced with the frozen
extrapolated U* = 2V^n - V^{n-1}; V^n and V^{n-1} are interpolated at the
feet of depths dt and 2dt (each foot integrated independently from the
arrival point — no "bending"), then (3u^{n+1} - 4 ubar + ubarbar)/(2dt)
with z-diffusion implicit at arrival (theta=1, dt' = 2dt/3) and projection
with dt_eff = 2dt/3. BDF2 damps high-frequency content strongly — it holds
the M3 high-k spectral floor bounded up to dt+ ~ 0.25-0.30 (the two-foot
stochastic gain 17/9 sets the upper boundary). Constant dt assumed (BDF1
re-bootstrap on any dt change; pin dt with dt_update_interval: 0).
"""

import os

import numpy as np
import torch
import yaml

from . import env, operators
from .initflow import initialize_flow, initialize_flow_from_file
from .operators import (
    advection_u,
    advection_v,
    advection_w,
    diffusion_xy_u,
    diffusion_xy_v,
    diffusion_xy_w,
    solve_implicit_diffusion_u,
    solve_implicit_diffusion_v,
    solve_implicit_diffusion_w,
)
from .projection import initialize_fft_solver, project_velocity, solve_poisson_fft
from .semilag import SLAdvector
from .turbstats import TurbulenceStats
from .utils import (
    compute_bulk_velocity,
    compute_divergence,
    compute_u_tau,
    generate_grid,
    plot_grid,
    plot_profile,
    save_flow_fields,
    save_grid_csv,
)

# Layer-2 torch.compile (see operators.py): opt-in via TORCHANNEL_COMPILE=1
# (needs CC=gcc; run under PYTORCH_JIT=0). dt is passed as a 0-D tensor at the
# call sites so the adaptive dt does not trigger recompiles.
if env.USE_COMPILE:
    compute_divergence = torch.compile(compute_divergence)
    compute_bulk_velocity = torch.compile(compute_bulk_velocity)
    project_velocity = torch.compile(project_velocity)


@torch.jit.script
def apply_bc_all(
    u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, top_wall_bc_type: str = "dirichlet"
) -> None:
    """
    Apply boundary conditions to all velocity components in a single fused kernel.

    - Periodic in x and y (with staggered grid adjustments)
    - Bottom wall (z=0): always no-slip
    - Top wall (z=Lz): 'dirichlet' (no-slip) or 'neumann' (free-slip); w=0 always
    """
    # U-velocity
    u[0, :, :] = u[-1, :, :]
    u[:, 0, :] = u[:, -2, :]
    u[:, -1, :] = u[:, 1, :]
    u[:, :, 0] = -u[:, :, 1]
    if top_wall_bc_type == "neumann":
        u[:, :, -1] = u[:, :, -2]
    else:
        u[:, :, -1] = -u[:, :, -2]

    # V-velocity
    v[0, :, :] = v[-2, :, :]
    v[-1, :, :] = v[1, :, :]
    v[:, 0, :] = v[:, -1, :]
    v[:, :, 0] = -v[:, :, 1]
    if top_wall_bc_type == "neumann":
        v[:, :, -1] = v[:, :, -2]
    else:
        v[:, :, -1] = -v[:, :, -2]

    # W-velocity
    w[0, :, :] = w[-2, :, :]
    w[-1, :, :] = w[1, :, :]
    w[:, 0, :] = w[:, -2, :]
    w[:, -1, :] = w[:, 1, :]
    w[:, :, 0] = 0.0
    w[:, :, -1] = 0.0


class SLChannelFlow:
    """Channel-flow DNS driver: builds the grid and initial field from a YAML
    config, then advances it to `time.t_max` or `time.n_steps`.

    The advection scheme is chosen by `advection.scheme`:

    ``sl``
        Semi-Lagrangian BDF2 characteristics (:meth:`step_sl_bdf2`) — the
        production scheme. Unconditionally stable in advection, so `dt` is set
        by accuracy (trajectory CFL 2-5, dt+ <= 0.25) rather than by an
        advective CFL.
    ``eulerian``
        The IMEX reference (:meth:`step_imex`, AB2 advection). Kept for
        like-for-like validation and benchmarking against the SL scheme, not as
        a production path.

    Both share the same wall-normal Crank-Nicolson diffusion solve, FFT
    pressure projection and exact bulk-flux constraint, so a difference between
    them is a difference in advection alone.

    Every YAML key, its type and its default are documented in docs/CONFIG.md.

    Example
    -------
    >>> flow = SLChannelFlow("configs/demo_sl_re180.yaml")
    >>> flow.run_simulation()
    """

    def __init__(self, config_file="config.yaml"):
        """Build a solver from a YAML configuration file.

        Parameters
        ----------
        config_file : str
            Path to the YAML configuration. See docs/CONFIG.md for the keys.

        Notes
        -----
        The constructor is not side-effect free. It

        * sets the process-global default dtype to ``torch.float64`` — the
          operators, grid and statistics all assume it;
        * creates `output.results_folder` and marks it with a
          ``.slchannel_results`` sentinel;
        * writes ``grid.csv`` and a grid plot into that folder;
        * **empties that folder** when `output.clean_results_on_fresh_start` is
          true and this is not a restart (guarded — see
          :meth:`_clean_results_folder`);
        * projects the initial field to be discretely divergence-free.
        """
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Device setup
        device_config = config.get("compute", {}).get("device", "auto")
        if device_config == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_config == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"\n{'=' * 80}", flush=True)
        print(f"Device: {self.device}", flush=True)
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
                flush=True,
            )
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print(f"{'=' * 80}\n", flush=True)

        self.nx = config["grid"]["nx"]
        self.ny = config["grid"]["ny"]
        self.nz = config["grid"]["nz"]

        self.Lx = config["domain"]["Lx"]
        self.Ly = config["domain"]["Ly"]
        self.Lz = config["domain"]["Lz"]
        self.stretching_type = config["domain"].get("stretching_type", "symmetric")
        if self.stretching_type not in ["symmetric", "bottom"]:
            raise ValueError(
                f"Invalid stretching type: {self.stretching_type}. "
                "slChannel supports 'symmetric' or 'bottom'"
            )

        self.nu = 1.0 / config["flow"]["Re"]
        self.Re_tau = config["flow"]["Re_tau"]
        self.U_bulk = config["flow"]["U_bulk"]
        self.gamma = config["flow"]["gamma"]

        bc_config = config.get("boundary_conditions", {})
        self.top_wall_bc_type = bc_config.get("top_wall", {}).get("type", "dirichlet")
        if self.top_wall_bc_type not in ["dirichlet", "neumann"]:
            raise ValueError(f"Invalid top wall BC type: {self.top_wall_bc_type}")

        self.dt = config["time"]["dt"]
        self.n_steps = config["time"]["n_steps"]
        self.t_max = config["time"].get("t_max", 1000.0)
        self.cfl_target = config["time"]["CFL_target"]
        self.dt_update_interval = config["time"].get("dt_update_interval", 0)
        self.dt_max = config["time"].get("dt_max", 0.01)
        self.dt_min = config["time"].get("dt_min", 0.0001)
        self.time_scheme = config["time"].get("scheme", "IMEX")
        # Explicit xy-diffusion stability constant: dt <= C / (nu*(1/dx^2+1/dy^2)).
        # Non-binding at channel-DNS resolutions; verified empirically in tests.
        self.diff_stability_C = config["time"].get("diff_stability_C", 0.2)

        # --- Advection scheme -------------------------------------------------
        self.advection_scheme = config.get("advection", {}).get("scheme", "sl")
        if self.advection_scheme not in ["sl", "eulerian"]:
            raise ValueError(
                f"advection.scheme must be 'sl' or 'eulerian', got {self.advection_scheme}"
            )
        sl_cfg = config.get("sl", {})
        self.sl_order = sl_cfg.get("interp_order", 4)
        if self.sl_order not in (4, 6):
            raise ValueError(
                f"sl.interp_order must be 4 (tricubic) or 6 (triquintic), got {self.sl_order}"
            )
        self.sl_traj_order = sl_cfg.get("traj_interp_order", 2)
        self.sl_traj_iters = sl_cfg.get("n_traj_iters", 2)
        self.sl_interp_dtype = sl_cfg.get("interp_dtype", "fp64")
        # BDF2-characteristics options (Boukir et al. 1997):
        #   bdf2_pressure: 'noninc' — p^{n+1} = phi (robust, splitting caps
        #                  velocity self-convergence near O(dt));
        #                  'inc' — incremental on p_ext = 2p^n - p^{n-1}
        #                  (O(dt^2); p_ext must enter BEFORE the implicit
        #                  solve).
        #   bdf2_xy_rhs:   'extrap' — 2R^n - R^{n-1} at arrival (consistent
        #                  at t^{n+1}); 'lagged' — R^n (diagnostic, O(dt) in
        #                  a term ~100x smaller than the trajectory channel).
        self.sl_bdf2_pressure = sl_cfg.get("bdf2_pressure", "noninc")
        self.sl_bdf2_xy_rhs = sl_cfg.get("bdf2_xy_rhs", "extrap")
        for _removed in ("time_scheme", "traj_extrapolation", "field_interp"):
            if _removed in sl_cfg:
                raise ValueError(
                    f"sl.{_removed} was removed in the bdf2-only cleanup: the SL "
                    f"scheme is always Boukir BDF2 characteristics now (drop the "
                    f"key from the config)"
                )
        if self.sl_bdf2_pressure not in ["noninc", "inc"]:
            raise ValueError(
                f"sl.bdf2_pressure must be 'noninc' or 'inc', got {self.sl_bdf2_pressure}"
            )
        if self.sl_bdf2_xy_rhs not in ["extrap", "lagged"]:
            raise ValueError(
                f"sl.bdf2_xy_rhs must be 'extrap' or 'lagged', got {self.sl_bdf2_xy_rhs}"
            )

        # Output settings
        output_config = config.get("output", {})
        self.results_folder = output_config.get("results_folder", "results")
        self.n_out = output_config.get("n_out", 10)
        self.n_save = output_config.get("n_save", 100)
        self.n_snapshot = output_config.get("n_snapshot", 0)
        # Time-based snapshots: write fields_t*.npz every t_snapshot SIMULATION
        # time units (uniform in t+ across runs with different/adaptive dt,
        # unlike the step-based n_snapshot). Lands on the first step crossing
        # each threshold (jitter <= dt). 0 disables.
        self.t_snapshot = output_config.get("t_snapshot", 0.0)
        self._next_snap_time = None
        os.makedirs(self.results_folder, exist_ok=True)
        self._mark_results_folder()

        field_file = config["initialization"].get("field_file", None)
        init_type_cfg = config["initialization"].get("type", "parabolic")
        is_restart = field_file is not None and init_type_cfg != "interpolate"

        clean_results = output_config.get("clean_results_on_fresh_start", False)
        if not is_restart and clean_results:
            self._clean_results_folder()

        torch.set_default_dtype(torch.float64)

        # Grid
        self.z_f, self.z_c, self.dz_f, self.dz_c = generate_grid(
            self.gamma, self.nz, self.Lz, device=self.device, stretching_type=self.stretching_type
        )

        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.cell_vol = (self.dx * self.dy * self.dz_f.view(1, 1, -1)).expand(
            self.nx, self.ny, self.nz
        )
        self.cell_vol_ratio = self.cell_vol
        self.total_volume = self.Lx * self.Ly * self.Lz

        save_grid_csv(self.z_f, self.z_c, self.dz_f, self.dz_c, self.nz, self.results_folder)
        plot_grid(self.z_f, self.z_c, self.results_folder)

        # Initialize flow
        print("Initializing flow...", flush=True)
        reset_time = config["initialization"].get("reset_time", False)

        if is_restart:
            self.u, self.v, self.w, self.p, self.initial_step, self.time = (
                initialize_flow_from_file(field_file, device=self.device, reset_time=reset_time)
            )
            self.initial_time = self.time
            # restore the bulk-forcing controller state saved in the npz
            # (resetting to 0 forces a ~50-step re-convergence transient that
            # would pollute a resumed statistics window)
            try:
                self.forcing = float(np.load(field_file)["forcing"])
                print(f"Restored forcing state: {self.forcing:.6e}", flush=True)
            except Exception:
                self.forcing = 0.0
        elif init_type_cfg == "interpolate":
            if field_file is None:
                raise ValueError(
                    "initialization.type 'interpolate' requires initialization.field_file"
                )
            self.forcing = 0.0
            from .initflow import initialize_flow_interpolated

            self.u, self.v, self.w, self.p = initialize_flow_interpolated(
                field_file,
                self.nx,
                self.ny,
                self.nz,
                self.Lx,
                self.Ly,
                self.Lz,
                self.z_c,
                self.z_f,
                device=self.device,
                source_half=config["initialization"].get("source_half", "lower"),
            )
            apply_bc_all(self.u, self.v, self.w, self.top_wall_bc_type)
            self.initial_step = 0
            self.time = 0.0
            self.initial_time = 0.0
        else:
            self.forcing = 0.0
            self.u, self.v, self.w, self.p = initialize_flow(
                self.nx,
                self.ny,
                self.nz,
                self.z_c,
                self.Ly,
                self.Lz,
                U_bulk=self.U_bulk,
                init_type=init_type_cfg,
                perturbation_intensity=config["initialization"].get("perturbation_intensity", 0.0),
                n_vortices=config["initialization"].get("n_vortices", 4),
                device=self.device,
                top_wall_bc_type=self.top_wall_bc_type,
            )
            self.initial_step = 0
            self.time = 0.0
            self.initial_time = 0.0

        # Rescale u to match U_bulk exactly (fresh/interpolated starts only)
        if field_file is None or init_type_cfg == "interpolate":
            u_bulk_init = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
            if abs(u_bulk_init) > 1e-9:
                self.u *= self.U_bulk / u_bulk_init
            else:
                print("WARNING: Initial bulk velocity is zero. Skipping rescaling.", flush=True)
        else:
            print("Restarting from file: Skipping velocity rescaling.", flush=True)

        # Poisson solver: FFT in the periodic x,y + tridiagonal solve in z.
        solver_type = config.get("solver", {}).get("type", "fft")
        if solver_type != "fft":
            raise ValueError(
                f"solver.type must be 'fft', got {solver_type!r}. The dense "
                f"direct Poisson matrix was removed: it does not scale to DNS "
                f"grids and was never used by a production case."
            )
        self.fft_data = initialize_fft_solver(
            self.nx,
            self.ny,
            self.nz,
            self.dx,
            self.dy,
            self.dz_c,
            self.dz_f,
            top_wall_bc_type=self.top_wall_bc_type,
        )

        # Semi-Lagrangian advector
        self.sl = None
        if self.advection_scheme == "sl":
            self.sl = SLAdvector(
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                self.dy,
                self.Lx,
                self.Ly,
                self.Lz,
                self.z_f,
                self.z_c,
                self.gamma,
                stretching_type=self.stretching_type,
                order=self.sl_order,
                traj_order=self.sl_traj_order,
                n_traj_iters=self.sl_traj_iters,
                top_wall_bc_type=self.top_wall_bc_type,
                interp_dtype=self.sl_interp_dtype,
                device=self.device,
            )
            print(
                f"Semi-Lagrangian advection: order={self.sl_order}, "
                f"traj_order={self.sl_traj_order}, "
                f"n_traj_iters={self.sl_traj_iters}, scheme=bdf2, "
                f"interp_dtype={self.sl_interp_dtype}",
                flush=True,
            )
        # AB2-extrapolation history (V^{n-1}) and midpoint buffers, lazily
        # allocated on the first SL step (restart re-bootstraps: V_mid = V^n)
        self.u_nm1 = None
        self.v_nm1 = None
        self.w_nm1 = None
        self.u_mid = None
        self.v_mid = None
        self.w_mid = None
        # v2 pressure history at half time levels: P_curr = p^{n-1/2},
        # P_prev = p^{n-3/2} (lazily allocated; restart re-bootstraps)
        self._P_curr = None
        self._P_prev = None
        self._p_ext = None
        self._gp_u = None
        self._gp_v = None
        self._gp_w = None
        # v2 history of the explicit xy-RHS (for AB2 time-centering to n+1/2)
        self._Rxy_u_nm1 = None
        self._Rxy_v_nm1 = None
        self._Rxy_w_nm1 = None
        # bdf2: far-foot copy (advect() output buffers are persistent and the
        # second call overwrites them) and constant-dt guard. Under bdf2 the
        # _P_curr/_P_prev history holds FULL levels p^n / p^{n-1} (not v2's
        # half levels); all bdf2 pressure logic is self-contained below.
        self._bdf2_acc_u = None
        self._bdf2_acc_v = None
        self._bdf2_acc_w = None
        self._dt_prev_step = None

        # Triton fast path for the Eulerian explicit RHS (fair-comparison
        # baseline: same hand-written-kernel treatment as the SL advector).
        # Disable with SLCHANNEL_TRITON=0.
        self._triton_eul = None
        if self.advection_scheme == "eulerian" and self.device.type == "cuda" and env.USE_TRITON:
            try:
                from .eulerian_triton import TritonEulerianRHS

                self._triton_eul = TritonEulerianRHS(
                    self.nx,
                    self.ny,
                    self.nz,
                    self.dx,
                    self.dy,
                    self.dz_c,
                    self.dz_f,
                    self.nu,
                    self.device,
                )
                print("Eulerian explicit RHS: Triton fast path enabled", flush=True)
            except Exception as e:
                print(
                    f"[eulerian] Triton fast path unavailable ({e}); using fused eager kernel",
                    flush=True,
                )

        # Statistics
        stats_config = config.get("statistics", {})
        # `enabled` is an explicit off switch; `n_stats: 0` also disables.
        stats_enabled = stats_config.get("enabled", True)
        self.n_stats = stats_config.get("n_stats", 0) if stats_enabled else 0
        self.t_stats = stats_config.get("t_stats", 10.0)
        if self.n_stats > 0:
            z_plus_target = stats_config.get("z_plus_target", 15.0)
            self.stats_output_path = os.path.join(
                self.results_folder, stats_config.get("output_file", "turbulence_stats.npz")
            )
            self.stats_state_path = os.path.join(
                self.results_folder, stats_config.get("state_file", "turbulence_stats_state.npz")
            )
            stats_restart_file = stats_config.get("restart_state_file", None)
            print(
                f"\nStatistics: start t={self.t_stats:.2f}, every {self.n_stats} steps", flush=True
            )
            self.turbulence_stats = TurbulenceStats(
                self.nx,
                self.ny,
                self.nz,
                self.Lx,
                self.Ly,
                self.Lz,
                self.z_c,
                self.z_f,
                self.dz_c,
                self.dz_f,
                self.dx,
                self.dy,
                self.nu,
                self.Re_tau,
                z_plus_target=z_plus_target,
                device=self.device,
                top_wall_bc_type=self.top_wall_bc_type,
            )
            if stats_restart_file is not None:
                print(f"  Loading statistics state from: {stats_restart_file}", flush=True)
                self.turbulence_stats.load_state(stats_restart_file)
        else:
            self.turbulence_stats = None

        # Save initial fields
        u_tau_init = compute_u_tau(
            self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type
        )
        save_flow_fields(
            self.u,
            self.v,
            self.w,
            self.p,
            self.z_c,
            self.z_f,
            self.Lx,
            self.Ly,
            0,
            0.0,
            u_tau_init,
            0.0,
            self.results_folder,
            "fields_init.npz",
        )

        # Project initial field to divergence-free
        div = compute_divergence(
            self.u, self.v, self.w, self.nx, self.ny, self.nz, self.dx, self.dy, self.dz_f
        )
        self.p = solve_poisson_fft(div, self.fft_data)
        self.u, self.v, self.w = project_velocity(
            self.u,
            self.v,
            self.w,
            self.p,
            self.nx,
            self.ny,
            self.nz,
            self.dx,
            self.dy,
            self.dz_c,
            self.dz_f,
            1.0,
        )
        self.apply_bc_uvw()
        div_final = compute_divergence(
            self.u, self.v, self.w, self.nx, self.ny, self.nz, self.dx, self.dy, self.dz_f
        )
        print(
            f"Initial divergence after projection: max(|div|) = {torch.max(torch.abs(div_final)):.6e}",
            flush=True,
        )

        # AB2 buffers for the Eulerian reference scheme
        self.rhs_u_prev = None
        self.rhs_v_prev = None
        self.rhs_w_prev = None
        self.rhs_u_curr = None
        self.rhs_v_curr = None
        self.rhs_w_curr = None

        self.time = self.initial_time
        self.current_step = self.initial_step

        # Optional CUDA-graph capture of the FFT-Poisson solve (torChannel's
        # machinery, unchanged; the SL region itself is NOT graphable — it
        # builds index tensors)
        self._pgraph = None
        self._pg_cudagraph = env.USE_POISSON_CUDAGRAPH and self.device.type == "cuda"

    def _poisson_fft_graphed(self, rhs):
        """Replay (or first capture) the FFT-Poisson solve as a CUDA graph."""
        if self._pgraph is None:
            self._pg_in = rhs.clone()
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(3):
                    solve_poisson_fft(self._pg_in, self.fft_data)
            torch.cuda.current_stream().wait_stream(warm)
            self._pgraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._pgraph):
                self._pg_out = solve_poisson_fft(self._pg_in, self.fft_data)
        self._pg_in.copy_(rhs)
        self._pgraph.replay()
        return self._pg_out

    def _solve_poisson(self, rhs):
        if self._pg_cudagraph:
            return self._poisson_fft_graphed(rhs)
        return solve_poisson_fft(rhs, self.fft_data)

    def apply_bc_uvw(self):
        apply_bc_all(self.u, self.v, self.w, self.top_wall_bc_type)

    def compute_cfl_dt(self):
        """Adaptive dt. For 'eulerian' this is the standard advective CFL; for
        'sl' the CFL_target is a trajectory CFL (typically 2-5) and the explicit
        xy-diffusion stability cap is applied."""
        dti = operators.compute_cfl_fused(
            self.u,
            self.v,
            self.w,
            self.nx,
            self.ny,
            self.nz,
            self.dx,
            self.dy,
            self.dz_f,
            self.dz_c,
        )
        if dti < 1e-10:
            dti = 1.0
        dt_new = self.cfl_target / dti
        if self.advection_scheme == "sl":
            dt_diff = self.diff_stability_C / (self.nu * (1.0 / self.dx**2 + 1.0 / self.dy**2))
            dt_new = min(dt_new, dt_diff)
        return min(max(dt_new, self.dt_min), self.dt_max)

    # ------------------------------------------------------------------
    # Semi-Lagrangian step
    # ------------------------------------------------------------------

    def _explicit_xy_rhs(self):
        """Explicit horizontal diffusion + bulk forcing, evaluated on V^n.
        Full-shaped tensors (interior filled, ghosts zero)."""
        rhs_u = diffusion_xy_u(self.u, self.nx, self.ny, self.nz, self.dx, self.dy, self.nu)
        rhs_v = diffusion_xy_v(self.v, self.nx, self.ny, self.nz, self.dx, self.dy, self.nu)
        rhs_w = diffusion_xy_w(self.w, self.nx, self.ny, self.nz, self.dx, self.dy, self.nu)
        # NOTE: no bulk forcing here -- it is applied as an exact uniform shift
        # at the end of the step (see _apply_bulk_forcing).
        return rhs_u, rhs_v, rhs_w

    # ------------------------------------------------------------------
    # BDF2 characteristics step (Boukir et al. 1997)
    # ------------------------------------------------------------------

    def _apply_bulk_forcing(self, dt):
        """Constant mass flux, enforced EXACTLY (CaNS convention).

        A uniform shift of u puts the bulk velocity on target in a single step.
        Being spatially constant its divergence is zero, so it can be applied
        after the projection without spoiling it, and the force is DIAGNOSED
        from the correction rather than steered towards it. CaNS does the same
        (cmpt_bulk_forcing in rk.f90:304 sets f = velf - mean, bulk_forcing in
        mom.f90 adds it uniformly).

        This replaces an integral controller of gain 0.1/dt. That controller had
        no proportional term, so its error equation was
        d2(du_b)/dt2 = -(0.1/dt^2) du_b -- an undamped oscillator of period ~20
        steps, damped only incidentally by wall drag, holding the bulk velocity
        to only ~1.5% rms in the forcing. Worse for this repo specifically, its
        gain scaled as 1/dt, so its closed-loop response was itself
        dt-dependent: any dt sweep or SL-vs-Eulerian comparison at differing dt
        was partly measuring the controller rather than the scheme.

        The body force is no longer added to the momentum RHS either -- it is
        this shift. Adding it there put it through the AB2 combination, applying
        1.5*f_n - 0.5*f_{n-1}: an extrapolated control signal.
        """
        u_bulk_star = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
        correction = self.U_bulk - u_bulk_star
        self.u[1 : self.nx + 1, 1 : self.ny + 1, 1 : self.nz + 1] += correction
        self.apply_bc_uvw()
        self.forcing = correction / dt
        u_bulk_current = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
        return u_bulk_current, self.forcing

    def _bdf2_forcing_update(self, dt):
        """Exact mass-flux constraint at the physical dt (see _apply_bulk_forcing)."""
        return self._apply_bulk_forcing(dt)

    def _step_sl_bdf1_bootstrap(self, dt, dt_t):
        """One BDF1 (backward-Euler characteristics) step: U* = V^n, single
        foot at depth dt, theta=1 z-solve and projection with the full dt.
        Used on the first step, after restart, and on a dt change (the BDF2
        coefficients assume constant dt). Seeds the u_nm1 / R histories.
        Same policy as the documented AB2 restart re-bootstrap: one O(dt^2)
        step, not bit-exact across restarts."""
        nx, ny, nz = self.nx, self.ny, self.nz
        rhs_u, rhs_v, rhs_w = self._explicit_xy_rhs()

        # seed histories from the CURRENT state before it is advanced
        if self.u_nm1 is None:
            self.u_nm1 = self.u.clone()
            self.v_nm1 = self.v.clone()
            self.w_nm1 = self.w.clone()
        else:
            self.u_nm1.copy_(self.u)
            self.v_nm1.copy_(self.v)
            self.w_nm1.copy_(self.w)
        if self._Rxy_u_nm1 is None:
            self._Rxy_u_nm1 = rhs_u.clone()
            self._Rxy_v_nm1 = rhs_v.clone()
            self._Rxy_w_nm1 = rhs_w.clone()
        else:
            self._Rxy_u_nm1.copy_(rhs_u)
            self._Rxy_v_nm1.copy_(rhs_v)
            self._Rxy_w_nm1.copy_(rhs_w)

        ustar, vstar, wstar = self.sl.advect(self.u, self.v, self.w, self.u, self.v, self.w, dt_t)
        ustar[1 : nx + 1, 1 : ny + 1, 1 : nz + 1] += (
            dt_t * rhs_u[1 : nx + 1, 1 : ny + 1, 1 : nz + 1]
        )
        vstar[1 : nx + 1, 1 : ny + 1, 1 : nz + 1] += (
            dt_t * rhs_v[1 : nx + 1, 1 : ny + 1, 1 : nz + 1]
        )
        wstar[1 : nx + 1, 1 : ny + 1, 1:nz] += dt_t * rhs_w[1 : nx + 1, 1 : ny + 1, 1:nz]
        self.u, self.v, self.w = ustar, vstar, wstar
        self.apply_bc_uvw()

        self.u = solve_implicit_diffusion_u(
            self.u,
            dt_t,
            nx,
            ny,
            nz,
            self.dz_c,
            self.dz_f,
            self.nu,
            theta=1.0,
            top_wall_bc_type=self.top_wall_bc_type,
        )
        self.v = solve_implicit_diffusion_v(
            self.v,
            dt_t,
            nx,
            ny,
            nz,
            self.dz_c,
            self.dz_f,
            self.nu,
            theta=1.0,
            top_wall_bc_type=self.top_wall_bc_type,
        )
        self.w = solve_implicit_diffusion_w(
            self.w, dt_t, nx, ny, nz, self.dz_c, self.dz_f, self.nu, theta=1.0
        )
        self.apply_bc_uvw()

        div = compute_divergence(self.u, self.v, self.w, nx, ny, nz, self.dx, self.dy, self.dz_f)
        phi = self._solve_poisson(div / dt)
        self.u, self.v, self.w = project_velocity(
            self.u, self.v, self.w, phi, nx, ny, nz, self.dx, self.dy, self.dz_c, self.dz_f, dt_t
        )
        if self.sl_bdf2_pressure == "inc":
            if self._P_curr is None:
                self._P_curr = phi.clone()
            else:
                self._P_curr.copy_(phi)
            self._P_prev = None
        self.p = phi
        self.apply_bc_uvw()
        self._dt_prev_step = dt
        return self._bdf2_forcing_update(dt)

    def step_sl_bdf2(self, dt):
        """BDF2 along characteristics (Boukir et al. 1997):
            (3 u^{n+1} - 4 ubar^n + ubarbar^{n-1}) / (2 dt)
                = nu Lap u^{n+1} - grad p + f,
        with ubar^n / ubarbar^{n-1} the values of V^n / V^{n-1} at the feet
        (depths dt and 2dt) of ONE characteristic traced backward from the
        arrival face with the frozen extrapolated U* = 2 V^n - V^{n-1}
        (both feet MUST use the same U*; each foot is an independent
        iterated-midpoint integration from the arrival point — never
        continue the far foot from the near foot, which drops the global
        order to 1, Boukir Remark 4i). Unlike v2's trajectory-CN, BDF2 is
        strongly damping at high frequency — the property that should
        absorb the U* extrapolation noise (the M3 spectral-floor
        mechanism) instead of remapping it neutrally."""
        nx, ny, nz = self.nx, self.ny, self.nz
        self.apply_bc_uvw()
        dt_t = torch.as_tensor(dt, device=self.device, dtype=torch.float64)

        # constant-dt guard: BDF2 coefficients and the 2dt foot depth assume
        # a fixed step; re-bootstrap with BDF1 on the first step, after
        # restart, or whenever dt changes (pin dt in bdf2 configs)
        if (
            self.u_nm1 is None
            or self._dt_prev_step is None
            or abs(dt - self._dt_prev_step) > 1e-12 * dt
        ):
            if self._dt_prev_step is not None and self.u_nm1 is not None:
                print(
                    f"[bdf2] dt changed ({self._dt_prev_step:.6g} -> {dt:.6g}): BDF1 re-bootstrap",
                    flush=True,
                )
            return self._step_sl_bdf1_bootstrap(dt, dt_t)

        # ---- explicit xy-diffusion + forcing at arrival ----
        rhs_u, rhs_v, rhs_w = self._explicit_xy_rhs()
        if self.sl_bdf2_xy_rhs == "extrap":
            # AB2 extrapolation to t^{n+1} (interior-only use, no gathers)
            Rh_u = 2.0 * rhs_u - self._Rxy_u_nm1
            Rh_v = 2.0 * rhs_v - self._Rxy_v_nm1
            Rh_w = 2.0 * rhs_w - self._Rxy_w_nm1
        else:
            Rh_u, Rh_v, Rh_w = rhs_u, rhs_v, rhs_w
        self._Rxy_u_nm1.copy_(rhs_u)
        self._Rxy_v_nm1.copy_(rhs_v)
        self._Rxy_w_nm1.copy_(rhs_w)

        # ---- frozen advecting velocity U* = 2 V^n - V^{n-1} (both feet) ----
        # ghosts of a linear combination of BC-consistent fields are
        # BC-consistent (BCs are linear)
        if self.u_mid is None:
            self.u_mid = self.u.clone()
            self.v_mid = self.v.clone()
            self.w_mid = self.w.clone()
        self.u_mid.copy_(self.u).mul_(2.0).add_(self.u_nm1, alpha=-1.0)
        self.v_mid.copy_(self.v).mul_(2.0).add_(self.v_nm1, alpha=-1.0)
        self.w_mid.copy_(self.w).mul_(2.0).add_(self.w_nm1, alpha=-1.0)

        # ---- far foot FIRST (depth 2dt, field V^{n-1}); advect() returns
        # its persistent buffers, so copy out before the near-foot call ----
        ub2u, ub2v, ub2w = self.sl.advect(
            self.u_nm1, self.v_nm1, self.w_nm1, self.u_mid, self.v_mid, self.w_mid, 2.0 * dt_t
        )
        nc_far = self.sl.n_clamped_last
        if self._bdf2_acc_u is None:
            self._bdf2_acc_u = torch.empty_like(ub2u)
            self._bdf2_acc_v = torch.empty_like(ub2v)
            self._bdf2_acc_w = torch.empty_like(ub2w)
        self._bdf2_acc_u.copy_(ub2u)
        self._bdf2_acc_v.copy_(ub2v)
        self._bdf2_acc_w.copy_(ub2w)

        # ---- near foot (depth dt, field V^n) ----
        ustar, vstar, wstar = self.sl.advect(
            self.u, self.v, self.w, self.u_mid, self.v_mid, self.w_mid, dt_t
        )
        # diagnostic: report clamped points over BOTH feet
        self.sl.n_clamped_last = self.sl.n_clamped_last + nc_far

        # ---- BDF2 predictor: u_hat = (4 ubar - ubarbar)/3 + dt_eff*(R - gp) ----
        dt_eff = (2.0 / 3.0) * dt
        dt_eff_t = (2.0 / 3.0) * dt_t
        ustar.mul_(4.0 / 3.0).add_(self._bdf2_acc_u, alpha=-1.0 / 3.0)
        vstar.mul_(4.0 / 3.0).add_(self._bdf2_acc_v, alpha=-1.0 / 3.0)
        wstar.mul_(4.0 / 3.0).add_(self._bdf2_acc_w, alpha=-1.0 / 3.0)
        ustar[1 : nx + 1, 1 : ny + 1, 1 : nz + 1] += (
            dt_eff_t * Rh_u[1 : nx + 1, 1 : ny + 1, 1 : nz + 1]
        )
        vstar[1 : nx + 1, 1 : ny + 1, 1 : nz + 1] += (
            dt_eff_t * Rh_v[1 : nx + 1, 1 : ny + 1, 1 : nz + 1]
        )
        wstar[1 : nx + 1, 1 : ny + 1, 1:nz] += dt_eff_t * Rh_w[1 : nx + 1, 1 : ny + 1, 1:nz]

        p_ext = None
        if self.sl_bdf2_pressure == "inc" and self._P_curr is not None:
            # extrapolated pressure p_ext = 2 p^n - p^{n-1} (FULL levels);
            # must enter the predictor BEFORE the implicit solve (same rule
            # as v2, verified analytically: adding it after the solve is
            # algebraically identical to non-incremental)
            if self._P_prev is None:
                p_ext = self._P_curr
            else:
                if self._p_ext is None:
                    self._p_ext = torch.empty_like(self._P_curr)
                self._p_ext.copy_(self._P_curr).mul_(2.0).add_(self._P_prev, alpha=-1.0)
                p_ext = self._p_ext
            if self._gp_u is None:
                self._gp_u = torch.zeros_like(self.u)
                self._gp_v = torch.zeros_like(self.v)
                self._gp_w = torch.zeros_like(self.w)
            self._gp_u.zero_()
            self._gp_v.zero_()
            self._gp_w.zero_()
            project_velocity(
                self._gp_u,
                self._gp_v,
                self._gp_w,
                p_ext,
                nx,
                ny,
                nz,
                self.dx,
                self.dy,
                self.dz_c,
                self.dz_f,
                1.0,
            )
            ustar += dt_eff_t * self._gp_u
            vstar += dt_eff_t * self._gp_v
            wstar += dt_eff_t * self._gp_w

        # ---- rotate velocity history BEFORE adopting the advector buffers
        # (ustar aliases them; self.u must still be V^n when copied) ----
        self.u_nm1.copy_(self.u)
        self.v_nm1.copy_(self.v)
        self.w_nm1.copy_(self.w)
        self.u, self.v, self.w = ustar, vstar, wstar
        self.apply_bc_uvw()

        # ---- implicit z-diffusion with the BDF2 coefficient:
        # (I - (2dt/3) nu Dzz) u = u_hat, i.e. theta=1 with dt' = 2dt/3 ----
        self.u = solve_implicit_diffusion_u(
            self.u,
            dt_eff_t,
            nx,
            ny,
            nz,
            self.dz_c,
            self.dz_f,
            self.nu,
            theta=1.0,
            top_wall_bc_type=self.top_wall_bc_type,
        )
        self.v = solve_implicit_diffusion_v(
            self.v,
            dt_eff_t,
            nx,
            ny,
            nz,
            self.dz_c,
            self.dz_f,
            self.nu,
            theta=1.0,
            top_wall_bc_type=self.top_wall_bc_type,
        )
        self.w = solve_implicit_diffusion_w(
            self.w, dt_eff_t, nx, ny, nz, self.dz_c, self.dz_f, self.nu, theta=1.0
        )
        self.apply_bc_uvw()

        # ---- projection with dt_eff (u^{n+1} = u** - (2dt/3) grad phi) ----
        div = compute_divergence(self.u, self.v, self.w, nx, ny, nz, self.dx, self.dy, self.dz_f)
        phi = self._solve_poisson(div / dt_eff)
        self.u, self.v, self.w = project_velocity(
            self.u,
            self.v,
            self.w,
            phi,
            nx,
            ny,
            nz,
            self.dx,
            self.dy,
            self.dz_c,
            self.dz_f,
            dt_eff_t,
        )
        if self.sl_bdf2_pressure == "inc":
            # p^{n+1} = p_ext + phi (phi lives in the Poisson workspace,
            # materialize into the history buffers)
            if self._P_curr is None:
                self._P_curr = phi.clone()
            elif self._P_prev is None:
                self._P_prev = torch.empty_like(self._P_curr)
                self._P_prev.copy_(self._P_curr)
                self._P_curr.copy_(self._P_curr + phi)  # p_ext was P_curr
            else:
                # p_ext = 2*P_curr - P_prev is in self._p_ext
                self._P_prev.copy_(self._P_curr)
                self._P_curr.copy_(self._p_ext).add_(phi)
            self.p = self._P_curr
        else:
            self.p = phi
        self.apply_bc_uvw()
        self._dt_prev_step = dt

        return self._bdf2_forcing_update(dt)

    # ------------------------------------------------------------------
    # Eulerian reference schemes (torChannel-identical)
    # ------------------------------------------------------------------

    def compute_momentum_rhs_explicit_imex(self):
        if self._triton_eul is not None:
            return self._triton_eul(self.u, self.v, self.w)
        if self.device.type == "cuda" and hasattr(operators, "compute_momentum_rhs_fused_imex"):
            return operators.compute_momentum_rhs_fused_imex(
                self.u,
                self.v,
                self.w,
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                self.dy,
                self.dz_c,
                self.dz_f,
                self.nu,
            )
        adv_u = advection_u(
            self.u, self.v, self.w, self.nx, self.ny, self.nz, self.dx, self.dy, self.dz_f
        )
        adv_v = advection_v(
            self.u, self.v, self.w, self.nx, self.ny, self.nz, self.dx, self.dy, self.dz_f
        )
        adv_w = advection_w(
            self.u, self.v, self.w, self.nx, self.ny, self.nz, self.dx, self.dy, self.dz_c
        )
        diff_xy_u = diffusion_xy_u(self.u, self.nx, self.ny, self.nz, self.dx, self.dy, self.nu)
        diff_xy_v = diffusion_xy_v(self.v, self.nx, self.ny, self.nz, self.dx, self.dy, self.nu)
        diff_xy_w = diffusion_xy_w(self.w, self.nx, self.ny, self.nz, self.dx, self.dy, self.nu)
        return diff_xy_u - adv_u, diff_xy_v - adv_v, diff_xy_w - adv_w

    def step_imex(self, dt):
        """IMEX: AB2 explicit advection + xy-diffusion, CN implicit z-diffusion."""
        self.apply_bc_uvw()

        rhs_u_explicit, rhs_v_explicit, rhs_w_explicit = self.compute_momentum_rhs_explicit_imex()
        # NOTE: no bulk forcing here -- see _apply_bulk_forcing.

        if self.rhs_u_curr is None:
            self.rhs_u_curr = rhs_u_explicit
            self.rhs_v_curr = rhs_v_explicit
            self.rhs_w_curr = rhs_w_explicit
        else:
            self.rhs_u_curr[:] = rhs_u_explicit
            self.rhs_v_curr[:] = rhs_v_explicit
            self.rhs_w_curr[:] = rhs_w_explicit

        if self.rhs_u_prev is None:
            self.u += dt * self.rhs_u_curr
            self.v += dt * self.rhs_v_curr
            self.w += dt * self.rhs_w_curr
        else:
            self.u += dt * (1.5 * self.rhs_u_curr - 0.5 * self.rhs_u_prev)
            self.v += dt * (1.5 * self.rhs_v_curr - 0.5 * self.rhs_v_prev)
            self.w += dt * (1.5 * self.rhs_w_curr - 0.5 * self.rhs_w_prev)

        self.rhs_u_prev, self.rhs_u_curr = self.rhs_u_curr, self.rhs_u_prev
        self.rhs_v_prev, self.rhs_v_curr = self.rhs_v_curr, self.rhs_v_prev
        self.rhs_w_prev, self.rhs_w_curr = self.rhs_w_curr, self.rhs_w_prev

        self.apply_bc_uvw()

        dt_t = torch.as_tensor(dt, device=self.device, dtype=torch.float64)
        self.u = solve_implicit_diffusion_u(
            self.u,
            dt_t,
            self.nx,
            self.ny,
            self.nz,
            self.dz_c,
            self.dz_f,
            self.nu,
            top_wall_bc_type=self.top_wall_bc_type,
        )
        self.v = solve_implicit_diffusion_v(
            self.v,
            dt_t,
            self.nx,
            self.ny,
            self.nz,
            self.dz_c,
            self.dz_f,
            self.nu,
            top_wall_bc_type=self.top_wall_bc_type,
        )
        self.w = solve_implicit_diffusion_w(
            self.w, dt_t, self.nx, self.ny, self.nz, self.dz_c, self.dz_f, self.nu
        )
        self.apply_bc_uvw()

        div = compute_divergence(
            self.u, self.v, self.w, self.nx, self.ny, self.nz, self.dx, self.dy, self.dz_f
        )
        self.p = self._solve_poisson(div / dt)
        self.u, self.v, self.w = project_velocity(
            self.u,
            self.v,
            self.w,
            self.p,
            self.nx,
            self.ny,
            self.nz,
            self.dx,
            self.dy,
            self.dz_c,
            self.dz_f,
            dt_t,
        )
        self.apply_bc_uvw()

        return self._apply_bulk_forcing(dt)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    # Sentinel written into every results folder slChannel creates. Only a
    # folder carrying it may be emptied by clean_results_on_fresh_start, so a
    # mistyped results_folder pointing at $HOME or a source tree cannot be
    # wiped: the marker is absent and the cleanup refuses.
    _RESULTS_MARKER = ".slchannel_results"

    def _mark_results_folder(self):
        """Record that `results_folder` is a slChannel-managed output folder."""
        marker = os.path.join(self.results_folder, self._RESULTS_MARKER)
        if not os.path.exists(marker):
            with open(marker, "w") as fh:
                fh.write(
                    "Created by slChannel. Its presence allows "
                    "output.clean_results_on_fresh_start to empty this "
                    "folder. Delete it to protect the contents.\n"
                )

    def _clean_results_folder(self):
        """Empty `results_folder` on a fresh (non-restart) start.

        Only ever touches a folder that slChannel itself created, identified by
        the `.slchannel_results` marker, and never a path that is the working
        directory, an ancestor of it, or a filesystem/home root.
        """
        target = os.path.realpath(self.results_folder)
        cwd = os.path.realpath(os.getcwd())
        home = os.path.realpath(os.path.expanduser("~"))

        unsafe = None
        if target in (os.path.sep, home, cwd):
            unsafe = "it is the filesystem root, the home directory, or the working directory"
        elif cwd.startswith(target + os.path.sep):
            unsafe = "it is an ancestor of the working directory"
        elif not os.path.exists(os.path.join(target, self._RESULTS_MARKER)):
            unsafe = (
                f"it carries no {self._RESULTS_MARKER} marker, so it was not created by slChannel"
            )

        if unsafe is not None:
            print(
                f"Refusing to clean results folder {target}: {unsafe}. "
                f"Remove output.clean_results_on_fresh_start, or point "
                f"results_folder at a folder slChannel created.",
                flush=True,
            )
            return

        print(f"Fresh start: Cleaning results folder: {self.results_folder}", flush=True)
        for filename in os.listdir(target):
            if filename == self._RESULTS_MARKER:
                continue
            file_path = os.path.join(target, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil

                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}", flush=True)

    def run_simulation(self):
        """Advance the flow until `time.t_max` or `time.n_steps` is reached.

        Progress is printed as a table: step, time, dt, max|div|, bulk velocity,
        u_tau and the driving forcing, plus a `clamped` column under the SL
        scheme counting departure points clamped at the walls.

        Writes into `output.results_folder`:

        ``timeseries.npz``
            Per-step history of the scalar diagnostics.
        ``fields.npz``
            Latest checkpoint, overwritten every `output.n_save` steps; this is
            what a restart reads.
        ``fields_t*.npz``
            Snapshots at fixed simulation-time intervals (`output.t_snapshot`).
        ``turbulence_stats.npz``
            Accumulated statistics, when `statistics.enabled` is set.
        ``fields_error.npz``
            Written only if the solution goes non-finite, for post-mortem.

        Returns
        -------
        None
        """
        import time

        if self.advection_scheme == "sl":
            interp_name = "tricubic" if self.sl_order == 4 else "triquintic"
            step_function = self.step_sl_bdf2
            scheme_name = (
                f"Semi-Lagrangian ({interp_name}, bdf2/"
                f"{self.sl_bdf2_pressure}) + implicit z-diffusion"
            )
        elif self.time_scheme == "IMEX":
            step_function = self.step_imex
            scheme_name = "Eulerian IMEX (AB2 + Implicit z-diffusion)"
        else:
            raise ValueError(
                f"Unknown scheme combination: advection={self.advection_scheme}, "
                f"time.scheme={self.time_scheme}"
            )

        delta = self.Lz if self.top_wall_bc_type == "neumann" else self.Lz / 2.0
        u_tau_target = self.Re_tau * self.nu / delta

        dz_min = torch.min(self.dz_f).item()
        dz_max = torch.max(self.dz_f).item()
        dx_plus = self.dx * u_tau_target / self.nu
        dy_plus = self.dy * u_tau_target / self.nu

        print("=" * 90, flush=True)
        print(f"Time stepping scheme: {scheme_name}", flush=True)
        print(f"Performance layers: {env.summary()}", flush=True)
        print(
            f"Grid: {self.nx}x{self.ny}x{self.nz}  |  Domain: {self.Lx:.2f}x{self.Ly:.2f}x{self.Lz:.2f}  |  "
            f"dx={self.dx:.3f}, dy={self.dy:.3f}, dz_min={dz_min:.3f}, dz_max={dz_max:.3f}  |  "
            f"dx+={dx_plus:.1f}, dy+={dy_plus:.1f}, "
            f"dz+_min={dz_min * u_tau_target / self.nu:.1f}, dz+_max={dz_max * u_tau_target / self.nu:.1f}",
            flush=True,
        )
        print("=" * 90, flush=True)
        header = (
            f"{'Step':>6} {'Time':>10} {'dt':>10} {'max(div)':>12} "
            f"{'u_bulk':>10} {'u_tau':>10} {'forcing':>12}"
        )
        if self.advection_scheme == "sl":
            header += f" {'clamped':>9}"
        print(header, flush=True)
        print("=" * 90, flush=True)

        start_time = time.time()
        last_walltime_print = start_time

        chunk_size = self.n_save // self.n_out + 1
        timeseries_data = {
            "step": np.zeros(chunk_size, dtype=np.int32),
            "time": np.zeros(chunk_size, dtype=np.float64),
            "u_bulk": np.zeros(chunk_size, dtype=np.float64),
            "u_tau": np.zeros(chunk_size, dtype=np.float64),
            "forcing": np.zeros(chunk_size, dtype=np.float64),
            "index": 0,
        }

        step = self.initial_step
        while step < self.n_steps and self.time < self.t_max:
            step += 1
            self.current_step = step
            if step > 0 and step % (10 * self.n_out) == 0:
                print(header, flush=True)
                current_time = time.time()
                elapsed = current_time - last_walltime_print
                total_elapsed = current_time - start_time
                print(
                    f"  Wall-time: {elapsed:.2f}s (last {10 * self.n_out} steps), "
                    f"{total_elapsed:.2f}s (total)",
                    flush=True,
                )
                last_walltime_print = current_time

            if self.dt_update_interval > 0 and step % self.dt_update_interval == 0 and step > 0:
                dt_new = self.compute_cfl_dt()
                if abs(dt_new - self.dt) / self.dt > 0.05:
                    self.dt = dt_new

            u_bulk, forcing = step_function(self.dt)
            self.time += self.dt

            if step % self.n_out == 0:
                div_final = compute_divergence(
                    self.u, self.v, self.w, self.nx, self.ny, self.nz, self.dx, self.dy, self.dz_f
                )
                max_div = torch.max(torch.abs(div_final)).item()
                u_tau = compute_u_tau(
                    self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type
                )

                u_bulk_scalar = u_bulk.item() if torch.is_tensor(u_bulk) else u_bulk
                u_tau_scalar = u_tau.item() if torch.is_tensor(u_tau) else u_tau
                forcing_scalar = forcing.item() if torch.is_tensor(forcing) else forcing

                idx = timeseries_data["index"]
                timeseries_data["step"][idx] = step
                timeseries_data["time"][idx] = self.time
                timeseries_data["u_bulk"][idx] = u_bulk_scalar
                timeseries_data["u_tau"][idx] = u_tau_scalar
                timeseries_data["forcing"][idx] = forcing_scalar
                timeseries_data["index"] += 1

            if self.n_stats > 0 and self.time >= self.t_stats and step % self.n_stats == 0:
                if step % self.n_out == 0:
                    u_tau_for_stats = u_tau
                else:
                    u_tau_for_stats = compute_u_tau(
                        self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type
                    )
                if self.turbulence_stats.n_samples == 0:
                    print(
                        f"  [Stats] Starting statistics collection at t = {self.time:.3f}",
                        flush=True,
                    )
                self.turbulence_stats.accumulate_statistics(self.u, self.v, self.w, u_tau_for_stats)

            if step % self.n_save == 0:
                if (
                    torch.any(torch.isnan(self.u))
                    or torch.any(torch.isinf(self.u))
                    or torch.any(torch.isnan(self.v))
                    or torch.any(torch.isinf(self.v))
                    or torch.any(torch.isnan(self.w))
                    or torch.any(torch.isinf(self.w))
                    or torch.any(torch.isnan(self.p))
                    or torch.any(torch.isinf(self.p))
                ):
                    print(f"\n{'=' * 90}", flush=True)
                    print(
                        f"ERROR: NaN or Inf detected at step {step}, time = {self.time:.6f}",
                        flush=True,
                    )
                    print(f"{'=' * 90}\n", flush=True)
                    u_tau_error = compute_u_tau(
                        self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type
                    )
                    u_bulk_error = compute_bulk_velocity(
                        self.u, self.cell_vol_ratio, self.total_volume
                    )
                    forcing_error = (self.U_bulk - u_bulk_error) / self.dt
                    save_flow_fields(
                        self.u,
                        self.v,
                        self.w,
                        self.p,
                        self.z_c,
                        self.z_f,
                        self.Lx,
                        self.Ly,
                        step,
                        self.time,
                        u_tau_error.item() if torch.is_tensor(u_tau_error) else u_tau_error,
                        forcing_error.item() if torch.is_tensor(forcing_error) else forcing_error,
                        self.results_folder,
                        "fields_error.npz",
                    )
                    print("Error state saved to fields_error.npz", flush=True)
                    break

                if step % self.n_out != 0:
                    u_tau = compute_u_tau(
                        self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type
                    )
                    u_tau_scalar = u_tau.item() if torch.is_tensor(u_tau) else u_tau
                    forcing_scalar = forcing.item() if torch.is_tensor(forcing) else forcing

                if timeseries_data["index"] > 0:
                    npz_file = os.path.join(self.results_folder, "timeseries.npz")
                    n_filled = timeseries_data["index"]
                    chunks = {
                        k: timeseries_data[k][:n_filled]
                        for k in ("step", "time", "u_bulk", "u_tau", "forcing")
                    }
                    if os.path.exists(npz_file):
                        existing = np.load(npz_file)
                        chunks = {k: np.concatenate([existing[k], chunks[k]]) for k in chunks}
                    np.savez_compressed(npz_file, **chunks)
                    timeseries_data["index"] = 0

                save_flow_fields(
                    self.u,
                    self.v,
                    self.w,
                    self.p,
                    self.z_c,
                    self.z_f,
                    self.Lx,
                    self.Ly,
                    step,
                    self.time,
                    u_tau_scalar,
                    forcing_scalar,
                    self.results_folder,
                    "fields.npz",
                )

                if self.turbulence_stats is not None and self.turbulence_stats.n_samples > 0:
                    self.turbulence_stats.save_state(self.stats_state_path)

            snap_due = self.n_snapshot > 0 and step % self.n_snapshot == 0
            if self.t_snapshot > 0:
                if self._next_snap_time is None:
                    self._next_snap_time = self.initial_time + self.t_snapshot
                if self.time >= self._next_snap_time - 1e-12:
                    snap_due = True
                    while self._next_snap_time <= self.time + 1e-12:
                        self._next_snap_time += self.t_snapshot
            if snap_due:
                u_tau_snap = compute_u_tau(
                    self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type
                )
                forcing_snap = forcing.item() if torch.is_tensor(forcing) else forcing
                snap_name = f"fields_t{self.time:09.3f}.npz"
                save_flow_fields(
                    self.u,
                    self.v,
                    self.w,
                    self.p,
                    self.z_c,
                    self.z_f,
                    self.Lx,
                    self.Ly,
                    step,
                    self.time,
                    u_tau_snap.item() if torch.is_tensor(u_tau_snap) else u_tau_snap,
                    forcing_snap,
                    self.results_folder,
                    snap_name,
                )
                print(f"  [Snapshot] Saved {snap_name}", flush=True)

            if step % self.n_out == 0:
                row = (
                    f"{step:6d} {self.time:10.6f} {self.dt:10.6f} {max_div:12.3e} "
                    f"{u_bulk_scalar:10.6f} {u_tau_scalar:10.6f} {forcing_scalar:12.3e}"
                )
                if self.advection_scheme == "sl":
                    row += f" {self.sl.n_clamped_last.item():9d}"
                print(row, flush=True)

        total_wall_time = time.time() - start_time
        print(f"{'=' * 90}", flush=True)
        print(
            f"Simulation complete: {step} steps, total time = {self.time:.6f}, final dt = {self.dt:.6f}",
            flush=True,
        )
        print(f"Total wall time: {total_wall_time:.2f}s", flush=True)
        print("=" * 90 + "\n", flush=True)

        if self.turbulence_stats is not None and self.turbulence_stats.n_samples > 0:
            print(
                f"\nFinalizing turbulence statistics ({self.turbulence_stats.n_samples} samples)...",
                flush=True,
            )
            self.turbulence_stats.save_state(self.stats_state_path)
            self.turbulence_stats.save_statistics(self.stats_output_path)

        u_profile = self.u[0, 0, :]
        plot_profile(
            u_profile,
            self.z_c,
            "u",
            "z",
            "Final velocity profile",
            "u_profile_final.png",
            self.results_folder,
        )

        u_tau_final = compute_u_tau(
            self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type
        )
        u_bulk_final = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
        forcing_final = (self.U_bulk - u_bulk_final) / self.dt
        save_flow_fields(
            self.u,
            self.v,
            self.w,
            self.p,
            self.z_c,
            self.z_f,
            self.Lx,
            self.Ly,
            step,
            self.time,
            u_tau_final,
            forcing_final,
            self.results_folder,
            "fields_final.npz",
        )
