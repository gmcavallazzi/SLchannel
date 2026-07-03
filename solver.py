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

SL step (v1), per step of size dt:
  1. V^{n+1/2} = 1.5 V^n - 0.5 V^{n-1}  (AB2 extrapolation; first step: V^n)
  2. (u*,v*,w*) = SL advection of V^n along V^{n+1/2} characteristics
  3. u* += dt*(explicit xy-diffusion(V^n) + bulk forcing)   [arrival point]
  4. Crank-Nicolson implicit z-diffusion
  5. FFT Poisson projection to divergence-free
v2 ('sl.time_scheme: v2') is the characteristic-consistent 2nd-order variant:
all explicit terms (xy-diffusion, forcing, and the AB-extrapolated pressure
gradient) are averaged between departure and arrival points, the explicit
half of the z-diffusion is evaluated at the DEPARTURE point (the implicit
half at arrival, i.e. Crank-Nicolson along the trajectory), and the
projection solves for a pressure INCREMENT on the extrapolated pressure.
Note: subtracting the old-pressure gradient AFTER the diffusion solve would
be algebraically identical to the non-incremental scheme (by linearity of
the Poisson solve) — the pressure must enter the predictor BEFORE the
implicit solve, which is what this implementation does.
"""

import os
import math
import torch
import yaml
import numpy as np
import operators
from utils import generate_grid, plot_grid, save_grid_csv, plot_profile, compute_u_tau, compute_bulk_velocity, compute_divergence, save_flow_fields
from initflow import initialize_flow, initialize_flow_from_file
from operators import advection_u, advection_v, advection_w, diffusion_u, diffusion_v, diffusion_w, diffusion_xy_u, diffusion_xy_v, diffusion_xy_w, solve_implicit_diffusion_u, solve_implicit_diffusion_v, solve_implicit_diffusion_w
from projection import build_poisson_matrix, solve_poisson, project_velocity
from projection_fft import initialize_fft_solver, solve_poisson_fft
from turbstats import TurbulenceStats
from semilag import SLAdvector

# Layer-2 torch.compile (see operators.py): opt-in via TORCHANNEL_COMPILE=1
# (needs CC=gcc; run under PYTORCH_JIT=0). dt is passed as a 0-D tensor at the
# call sites so the adaptive dt does not trigger recompiles.
if os.environ.get("TORCHANNEL_COMPILE", "0") == "1":
    compute_divergence = torch.compile(compute_divergence)
    compute_bulk_velocity = torch.compile(compute_bulk_velocity)
    project_velocity = torch.compile(project_velocity)


@torch.jit.script
def apply_bc_all(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, top_wall_bc_type: str = 'dirichlet') -> None:
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
    if top_wall_bc_type == 'neumann':
        u[:, :, -1] = u[:, :, -2]
    else:
        u[:, :, -1] = -u[:, :, -2]

    # V-velocity
    v[0, :, :] = v[-2, :, :]
    v[-1, :, :] = v[1, :, :]
    v[:, 0, :] = v[:, -1, :]
    v[:, :, 0] = -v[:, :, 1]
    if top_wall_bc_type == 'neumann':
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

    def __init__(self, config_file='config.yaml'):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Device setup
        device_config = config.get('compute', {}).get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print(f"\n{'='*80}", flush=True)
        print(f"Device: {self.device}", flush=True)
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print(f"{'='*80}\n", flush=True)

        self.nx = config['grid']['nx']
        self.ny = config['grid']['ny']
        self.nz = config['grid']['nz']

        self.Lx = config['domain']['Lx']
        self.Ly = config['domain']['Ly']
        self.Lz = config['domain']['Lz']
        self.stretching_type = config['domain'].get('stretching_type', 'symmetric')
        if self.stretching_type not in ['symmetric', 'bottom']:
            raise ValueError(f"Invalid stretching type: {self.stretching_type}. "
                             "slChannel supports 'symmetric' or 'bottom'")

        self.nu = 1.0 / config['flow']['Re']
        self.Re_tau = config['flow']['Re_tau']
        self.U_bulk = config['flow']['U_bulk']
        self.gamma = config['flow']['gamma']

        bc_config = config.get('boundary_conditions', {})
        self.top_wall_bc_type = bc_config.get('top_wall', {}).get('type', 'dirichlet')
        if self.top_wall_bc_type not in ['dirichlet', 'neumann']:
            raise ValueError(f"Invalid top wall BC type: {self.top_wall_bc_type}")

        self.dt = config['time']['dt']
        self.n_steps = config['time']['n_steps']
        self.t_max = config['time'].get('t_max', 1000.0)
        self.cfl_target = config['time']['CFL_target']
        self.dt_update_interval = config['time'].get('dt_update_interval', 0)
        self.dt_max = config['time'].get('dt_max', 0.01)
        self.dt_min = config['time'].get('dt_min', 0.0001)
        self.time_scheme = config['time'].get('scheme', 'IMEX')
        # Explicit xy-diffusion stability constant: dt <= C / (nu*(1/dx^2+1/dy^2)).
        # Non-binding at channel-DNS resolutions; verified empirically in tests.
        self.diff_stability_C = config['time'].get('diff_stability_C', 0.2)

        # --- Advection scheme -------------------------------------------------
        self.advection_scheme = config.get('advection', {}).get('scheme', 'sl')
        if self.advection_scheme not in ['sl', 'eulerian']:
            raise ValueError(f"advection.scheme must be 'sl' or 'eulerian', got {self.advection_scheme}")
        sl_cfg = config.get('sl', {})
        self.sl_order = sl_cfg.get('interp_order', 4)
        self.sl_traj_order = sl_cfg.get('traj_interp_order', 2)
        self.sl_traj_iters = sl_cfg.get('n_traj_iters', 2)
        self.sl_time_scheme = sl_cfg.get('time_scheme', 'v1')
        self.sl_interp_dtype = sl_cfg.get('interp_dtype', 'fp64')
        if self.sl_time_scheme not in ['v1', 'v2']:
            raise ValueError(f"sl.time_scheme must be 'v1' or 'v2', got {self.sl_time_scheme}")

        # Output settings
        output_config = config.get('output', {})
        self.results_folder = output_config.get('results_folder', 'results')
        self.n_out = output_config.get('n_out', 10)
        self.n_save = output_config.get('n_save', 100)
        self.n_snapshot = output_config.get('n_snapshot', 0)
        os.makedirs(self.results_folder, exist_ok=True)

        field_file = config['initialization'].get('field_file', None)
        init_type_cfg = config['initialization'].get('type', 'parabolic')
        is_restart = field_file is not None and init_type_cfg != 'interpolate'

        clean_results = output_config.get('clean_results_on_fresh_start', False)
        if not is_restart and clean_results:
            print(f"Fresh start: Cleaning results folder: {self.results_folder}", flush=True)
            for filename in os.listdir(self.results_folder):
                file_path = os.path.join(self.results_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}", flush=True)

        torch.set_default_dtype(torch.float64)

        # Grid
        self.z_f, self.z_c, self.dz_f, self.dz_c = generate_grid(
            self.gamma, self.nz, self.Lz, device=self.device,
            stretching_type=self.stretching_type)

        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.cell_vol = (self.dx * self.dy * self.dz_f.view(1, 1, -1)).expand(self.nx, self.ny, self.nz)
        self.cell_vol_ratio = self.cell_vol
        self.total_volume = self.Lx * self.Ly * self.Lz

        save_grid_csv(self.z_f, self.z_c, self.dz_f, self.dz_c, self.nz, self.results_folder)
        plot_grid(self.z_f, self.z_c, self.results_folder)

        # Initialize flow
        print("Initializing flow...", flush=True)
        reset_time = config['initialization'].get('reset_time', False)

        if is_restart:
            self.u, self.v, self.w, self.p, self.initial_step, self.time = \
                initialize_flow_from_file(field_file, device=self.device, reset_time=reset_time)
            self.initial_time = self.time
            self.forcing = 0.0
        elif init_type_cfg == 'interpolate':
            if field_file is None:
                raise ValueError("initialization.type 'interpolate' requires initialization.field_file")
            self.forcing = 0.0
            from initflow import initialize_flow_interpolated
            self.u, self.v, self.w, self.p = initialize_flow_interpolated(
                field_file, self.nx, self.ny, self.nz, self.Lx, self.Ly, self.Lz,
                self.z_c, self.z_f, device=self.device,
                source_half=config['initialization'].get('source_half', 'lower'))
            apply_bc_all(self.u, self.v, self.w, self.top_wall_bc_type)
            self.initial_step = 0
            self.time = 0.0
            self.initial_time = 0.0
        else:
            self.forcing = 0.0
            self.u, self.v, self.w, self.p = initialize_flow(
                self.nx, self.ny, self.nz, self.z_c, self.Ly, self.Lz,
                U_bulk=self.U_bulk,
                init_type=init_type_cfg,
                perturbation_intensity=config['initialization'].get('perturbation_intensity', 0.0),
                n_vortices=config['initialization'].get('n_vortices', 4),
                device=self.device,
                top_wall_bc_type=self.top_wall_bc_type)
            self.initial_step = 0
            self.time = 0.0
            self.initial_time = 0.0

        # Rescale u to match U_bulk exactly (fresh/interpolated starts only)
        if field_file is None or init_type_cfg == 'interpolate':
            u_bulk_init = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
            if abs(u_bulk_init) > 1e-9:
                self.u *= (self.U_bulk / u_bulk_init)
            else:
                print(f"WARNING: Initial bulk velocity is zero. Skipping rescaling.", flush=True)
        else:
            print("Restarting from file: Skipping velocity rescaling.", flush=True)

        # Poisson solver
        self.solver_type = config.get('solver', {}).get('type', 'fft')
        if self.solver_type == 'direct':
            self.poisson_matrix = build_poisson_matrix(self.nx, self.ny, self.nz,
                                                       self.dx, self.dy, self.dz_c, self.dz_f,
                                                       top_wall_bc_type=self.top_wall_bc_type)
        elif self.solver_type == 'fft':
            self.fft_data = initialize_fft_solver(self.nx, self.ny, self.nz,
                                                  self.dx, self.dy, self.dz_c, self.dz_f,
                                                  top_wall_bc_type=self.top_wall_bc_type)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")

        # Semi-Lagrangian advector
        self.sl = None
        if self.advection_scheme == 'sl':
            self.sl = SLAdvector(self.nx, self.ny, self.nz, self.dx, self.dy,
                                 self.Lx, self.Ly, self.Lz, self.z_f, self.z_c,
                                 self.gamma, stretching_type=self.stretching_type,
                                 order=self.sl_order, traj_order=self.sl_traj_order,
                                 n_traj_iters=self.sl_traj_iters,
                                 top_wall_bc_type=self.top_wall_bc_type,
                                 interp_dtype=self.sl_interp_dtype, device=self.device)
            print(f"Semi-Lagrangian advection: order={self.sl_order}, "
                  f"traj_order={self.sl_traj_order}, "
                  f"n_traj_iters={self.sl_traj_iters}, scheme={self.sl_time_scheme}, "
                  f"interp_dtype={self.sl_interp_dtype}", flush=True)
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

        # Statistics
        stats_config = config.get('statistics', {})
        self.n_stats = stats_config.get('n_stats', 0)
        self.t_stats = stats_config.get('t_stats', 10.0)
        if self.n_stats > 0:
            z_plus_target = stats_config.get('z_plus_target', 15.0)
            self.stats_output_path = os.path.join(self.results_folder,
                                                  stats_config.get('output_file', 'turbulence_stats.npz'))
            self.stats_state_path = os.path.join(self.results_folder,
                                                 stats_config.get('state_file', 'turbulence_stats_state.npz'))
            stats_restart_file = stats_config.get('restart_state_file', None)
            print(f"\nStatistics: start t={self.t_stats:.2f}, every {self.n_stats} steps", flush=True)
            self.turbulence_stats = TurbulenceStats(
                self.nx, self.ny, self.nz, self.Lx, self.Ly, self.Lz,
                self.z_c, self.z_f, self.dz_c, self.dz_f,
                self.dx, self.dy, self.nu,
                self.Re_tau, z_plus_target=z_plus_target, device=self.device)
            if stats_restart_file is not None:
                print(f"  Loading statistics state from: {stats_restart_file}", flush=True)
                self.turbulence_stats.load_state(stats_restart_file)
        else:
            self.turbulence_stats = None

        # Save initial fields
        u_tau_init = compute_u_tau(self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type)
        save_flow_fields(self.u, self.v, self.w, self.p, self.z_c, self.z_f,
                         self.Lx, self.Ly, 0, 0.0, u_tau_init, 0.0,
                         self.results_folder, 'fields_init.npz')

        # Project initial field to divergence-free
        div = compute_divergence(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                                 self.dx, self.dy, self.dz_f)
        if self.solver_type == 'direct':
            self.p = solve_poisson(self.poisson_matrix, div, self.nx, self.ny, self.nz, self.top_wall_bc_type)
        else:
            self.p = solve_poisson_fft(div, self.fft_data)
        self.u, self.v, self.w = project_velocity(self.u, self.v, self.w, self.p,
                                                  self.nx, self.ny, self.nz,
                                                  self.dx, self.dy, self.dz_c, self.dz_f, 1.0)
        self.apply_bc_uvw()
        div_final = compute_divergence(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                                       self.dx, self.dy, self.dz_f)
        print(f"Initial divergence after projection: max(|div|) = {torch.max(torch.abs(div_final)):.6e}", flush=True)

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
        self._pg_cudagraph = (os.environ.get("TORCHANNEL_POISSON_CUDAGRAPH", "0") == "1"
                              and self.device.type == 'cuda'
                              and self.solver_type == 'fft')

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
        if self.solver_type == 'direct':
            return solve_poisson(self.poisson_matrix, rhs, self.nx, self.ny, self.nz, self.top_wall_bc_type)
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
            self.u, self.v, self.w, self.nx, self.ny, self.nz,
            self.dx, self.dy, self.dz_f, self.dz_c)
        if dti < 1e-10:
            dti = 1.0
        dt_new = self.cfl_target / dti
        if self.advection_scheme == 'sl':
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
        rhs_u[1:self.nx + 1, 1:self.ny + 1, 1:self.nz + 1] += self.forcing
        return rhs_u, rhs_v, rhs_w

    def step_sl(self, dt):
        """One semi-Lagrangian projection step (see module docstring)."""
        nx, ny, nz = self.nx, self.ny, self.nz
        self.apply_bc_uvw()
        dt_t = torch.as_tensor(dt, device=self.device, dtype=torch.float64)

        # ---- trajectory velocity at t^{n+1/2} (AB2 extrapolation) ----
        if self.u_nm1 is None:
            # bootstrap (first step / after restart): V^{n+1/2} ~ V^n
            self.u_nm1 = self.u.clone()
            self.v_nm1 = self.v.clone()
            self.w_nm1 = self.w.clone()
            self.u_mid = self.u.clone()
            self.v_mid = self.v.clone()
            self.w_mid = self.w.clone()
            u_mid, v_mid, w_mid = self.u, self.v, self.w
        else:
            # 1.5*V^n - 0.5*V^{n-1}; ghosts of a linear combination of
            # BC-consistent fields are BC-consistent (BCs are linear)
            self.u_mid.copy_(self.u).mul_(1.5).add_(self.u_nm1, alpha=-0.5)
            self.v_mid.copy_(self.v).mul_(1.5).add_(self.v_nm1, alpha=-0.5)
            self.w_mid.copy_(self.w).mul_(1.5).add_(self.w_nm1, alpha=-0.5)
            self.u_nm1.copy_(self.u)
            self.v_nm1.copy_(self.v)
            self.w_nm1.copy_(self.w)
            u_mid, v_mid, w_mid = self.u_mid, self.v_mid, self.w_mid

        # ---- explicit horizontal terms on V^n ----
        rhs_u, rhs_v, rhs_w = self._explicit_xy_rhs()

        # ---- semi-Lagrangian advection + explicit terms ----
        if self.sl_time_scheme == 'v2':
            # time-center the viscous xy-RHS at t^{n+1/2} via AB2 extrapolation
            # (the pressure below is already extrapolated to n+1/2, and the
            # z-diffusion is trajectory-CN; without this the xy half would
            # contribute a residual O(dt) error)
            if self._Rxy_u_nm1 is None:
                self._Rxy_u_nm1 = rhs_u.clone()
                self._Rxy_v_nm1 = rhs_v.clone()
                self._Rxy_w_nm1 = rhs_w.clone()
                # bootstrap: R^{n+1/2} ~ R^n
            else:
                rhs_u_half = 1.5 * rhs_u - 0.5 * self._Rxy_u_nm1
                rhs_v_half = 1.5 * rhs_v - 0.5 * self._Rxy_v_nm1
                rhs_w_half = 1.5 * rhs_w - 0.5 * self._Rxy_w_nm1
                self._Rxy_u_nm1.copy_(rhs_u)
                self._Rxy_v_nm1.copy_(rhs_v)
                self._Rxy_w_nm1.copy_(rhs_w)
                rhs_u, rhs_v, rhs_w = rhs_u_half, rhs_v_half, rhs_w_half

            # extrapolated pressure at t^{n+1/2}: 2*p^{n-1/2} - p^{n-3/2}
            p_ext = None
            if self._P_curr is not None:
                if self._P_prev is None:
                    p_ext = self._P_curr
                else:
                    if self._p_ext is None:
                        self._p_ext = torch.empty_like(self._P_curr)
                    self._p_ext.copy_(self._P_curr).mul_(2.0).add_(self._P_prev, alpha=-1.0)
                    p_ext = self._p_ext
            if p_ext is not None:
                # -grad(p_ext) on each component grid via project_velocity on
                # zero fields (it computes u -= dt*grad(p) on the interiors)
                if self._gp_u is None:
                    self._gp_u = torch.zeros_like(self.u)
                    self._gp_v = torch.zeros_like(self.v)
                    self._gp_w = torch.zeros_like(self.w)
                self._gp_u.zero_(); self._gp_v.zero_(); self._gp_w.zero_()
                project_velocity(self._gp_u, self._gp_v, self._gp_w, p_ext,
                                 nx, ny, nz, self.dx, self.dy,
                                 self.dz_c, self.dz_f, 1.0)
                rhs_u += self._gp_u
                rhs_v += self._gp_v
                rhs_w += self._gp_w

            # z-diffusion explicit half, evaluated at the departure point
            # (z-part = full Laplacian - xy Laplacian)
            rz_u = diffusion_u(self.u, nx, ny, nz, self.dx, self.dy, self.dz_c, self.dz_f, self.nu) \
                - diffusion_xy_u(self.u, nx, ny, nz, self.dx, self.dy, self.nu)
            rz_v = diffusion_v(self.v, nx, ny, nz, self.dx, self.dy, self.dz_c, self.dz_f, self.nu) \
                - diffusion_xy_v(self.v, nx, ny, nz, self.dx, self.dy, self.nu)
            rz_w = diffusion_w(self.w, nx, ny, nz, self.dx, self.dy, self.dz_c, self.dz_f, self.nu) \
                - diffusion_xy_w(self.w, nx, ny, nz, self.dx, self.dy, self.nu)

            # BC-consistent ghosts so the departure-point interpolation of the
            # RHS is meaningful near the walls / across the periodic seams
            apply_bc_all(rhs_u, rhs_v, rhs_w, self.top_wall_bc_type)
            apply_bc_all(rz_u, rz_v, rz_w, self.top_wall_bc_type)

            ustar, vstar, wstar, ((Rud, Rvd, Rwd), (Rzud, Rzvd, Rzwd)) = self.sl.advect(
                self.u, self.v, self.w, u_mid, v_mid, w_mid, dt_t,
                extra_rhs=[(rhs_u, rhs_v, rhs_w), (rz_u, rz_v, rz_w)])
            # explicit terms averaged along the characteristic; z-diffusion
            # departure half only (arrival half is the implicit solve below)
            ustar[1:nx + 1, 1:ny + 1, 1:nz + 1] += dt_t * (
                0.5 * (Rud + rhs_u[1:nx + 1, 1:ny + 1, 1:nz + 1]) + 0.5 * Rzud)
            vstar[1:nx + 1, 1:ny + 1, 1:nz + 1] += dt_t * (
                0.5 * (Rvd + rhs_v[1:nx + 1, 1:ny + 1, 1:nz + 1]) + 0.5 * Rzvd)
            wstar[1:nx + 1, 1:ny + 1, 1:nz] += dt_t * (
                0.5 * (Rwd + rhs_w[1:nx + 1, 1:ny + 1, 1:nz]) + 0.5 * Rzwd)
        else:
            ustar, vstar, wstar = self.sl.advect(self.u, self.v, self.w,
                                                 u_mid, v_mid, w_mid, dt_t)
            # v1: explicit terms at the arrival index (first-order splitting,
            # acceptable: nu is small). Full-array adds; ghosts fixed below.
            ustar += dt_t * rhs_u
            vstar += dt_t * rhs_v
            wstar += dt_t * rhs_w

        # advect() returns the advector's persistent buffers; safe to adopt
        # them because the implicit diffusion below returns fresh tensors
        self.u, self.v, self.w = ustar, vstar, wstar
        self.apply_bc_uvw()

        # ---- implicit z-diffusion ----
        if self.sl_time_scheme == 'v2':
            # arrival half of the trajectory-CN: (I - 0.5*dt*nu*Dzz) u = u*,
            # i.e. theta=1 with dt/2 (the departure half is already in u*)
            half_dt_t = 0.5 * dt_t
            self.u = solve_implicit_diffusion_u(self.u, half_dt_t, nx, ny, nz,
                                                self.dz_c, self.dz_f, self.nu, theta=1.0,
                                                top_wall_bc_type=self.top_wall_bc_type)
            self.v = solve_implicit_diffusion_v(self.v, half_dt_t, nx, ny, nz,
                                                self.dz_c, self.dz_f, self.nu, theta=1.0,
                                                top_wall_bc_type=self.top_wall_bc_type)
            self.w = solve_implicit_diffusion_w(self.w, half_dt_t, nx, ny, nz,
                                                self.dz_c, self.dz_f, self.nu, theta=1.0)
        else:
            # v1: standard Crank-Nicolson at the arrival column
            self.u = solve_implicit_diffusion_u(self.u, dt_t, nx, ny, nz,
                                                self.dz_c, self.dz_f, self.nu,
                                                top_wall_bc_type=self.top_wall_bc_type)
            self.v = solve_implicit_diffusion_v(self.v, dt_t, nx, ny, nz,
                                                self.dz_c, self.dz_f, self.nu,
                                                top_wall_bc_type=self.top_wall_bc_type)
            self.w = solve_implicit_diffusion_w(self.w, dt_t, nx, ny, nz,
                                                self.dz_c, self.dz_f, self.nu)
        self.apply_bc_uvw()

        # ---- projection ----
        div = compute_divergence(self.u, self.v, self.w, nx, ny, nz,
                                 self.dx, self.dy, self.dz_f)
        phi = self._solve_poisson(div / dt)
        self.u, self.v, self.w = project_velocity(self.u, self.v, self.w, phi,
                                                  nx, ny, nz, self.dx, self.dy,
                                                  self.dz_c, self.dz_f, dt_t)
        if self.sl_time_scheme == 'v2':
            # pressure increment: p^{n+1/2} = p_ext + phi (phi lives in the
            # Poisson workspace, so materialize into the history buffers)
            if self._P_curr is None:
                self._P_curr = phi.clone()
            else:
                if self._P_prev is None:
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

        # ---- bulk forcing controller (lagged relaxation, torChannel-identical) ----
        u_bulk_current = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
        relaxation = 0.1
        self.forcing += (self.U_bulk - u_bulk_current) / dt * relaxation

        return u_bulk_current, self.forcing

    # ------------------------------------------------------------------
    # Eulerian reference schemes (torChannel-identical)
    # ------------------------------------------------------------------

    def compute_momentum_rhs_explicit_imex(self):
        if self.device.type == 'cuda' and hasattr(operators, 'compute_momentum_rhs_fused_imex'):
            return operators.compute_momentum_rhs_fused_imex(
                self.u, self.v, self.w, self.nx, self.ny, self.nz,
                self.dx, self.dy, self.dz_c, self.dz_f, self.nu)
        adv_u = advection_u(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                            self.dx, self.dy, self.dz_f)
        adv_v = advection_v(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                            self.dx, self.dy, self.dz_f)
        adv_w = advection_w(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                            self.dx, self.dy, self.dz_c)
        diff_xy_u = diffusion_xy_u(self.u, self.nx, self.ny, self.nz, self.dx, self.dy, self.nu)
        diff_xy_v = diffusion_xy_v(self.v, self.nx, self.ny, self.nz, self.dx, self.dy, self.nu)
        diff_xy_w = diffusion_xy_w(self.w, self.nx, self.ny, self.nz, self.dx, self.dy, self.nu)
        return diff_xy_u - adv_u, diff_xy_v - adv_v, diff_xy_w - adv_w

    def step_imex(self, dt):
        """IMEX: AB2 explicit advection + xy-diffusion, CN implicit z-diffusion."""
        self.apply_bc_uvw()

        rhs_u_explicit, rhs_v_explicit, rhs_w_explicit = self.compute_momentum_rhs_explicit_imex()
        rhs_u_explicit += self.forcing

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
        self.u = solve_implicit_diffusion_u(self.u, dt_t, self.nx, self.ny, self.nz,
                                            self.dz_c, self.dz_f, self.nu,
                                            top_wall_bc_type=self.top_wall_bc_type)
        self.v = solve_implicit_diffusion_v(self.v, dt_t, self.nx, self.ny, self.nz,
                                            self.dz_c, self.dz_f, self.nu,
                                            top_wall_bc_type=self.top_wall_bc_type)
        self.w = solve_implicit_diffusion_w(self.w, dt_t, self.nx, self.ny, self.nz,
                                            self.dz_c, self.dz_f, self.nu)
        self.apply_bc_uvw()

        div = compute_divergence(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                                 self.dx, self.dy, self.dz_f)
        self.p = self._solve_poisson(div / dt)
        self.u, self.v, self.w = project_velocity(self.u, self.v, self.w, self.p,
                                                  self.nx, self.ny, self.nz,
                                                  self.dx, self.dy, self.dz_c, self.dz_f, dt_t)
        self.apply_bc_uvw()

        u_bulk_current = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
        relaxation = 0.1
        self.forcing += (self.U_bulk - u_bulk_current) / dt * relaxation
        return u_bulk_current, self.forcing

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_simulation(self):
        import time

        if self.advection_scheme == 'sl':
            step_function = self.step_sl
            interp_name = 'tricubic' if self.sl_order == 4 else 'triquintic'
            scheme_name = f"Semi-Lagrangian ({interp_name}, {self.sl_time_scheme}) + CN z-diffusion"
        elif self.time_scheme == 'IMEX':
            step_function = self.step_imex
            scheme_name = "Eulerian IMEX (AB2 + Implicit z-diffusion)"
        else:
            raise ValueError(f"Unknown scheme combination: advection={self.advection_scheme}, "
                             f"time.scheme={self.time_scheme}")

        delta = self.Lz if self.top_wall_bc_type == 'neumann' else self.Lz / 2.0
        u_tau_target = self.Re_tau * self.nu / delta

        dz_min = torch.min(self.dz_f).item()
        dz_max = torch.max(self.dz_f).item()
        dx_plus = self.dx * u_tau_target / self.nu
        dy_plus = self.dy * u_tau_target / self.nu

        print("=" * 90, flush=True)
        print(f"Time stepping scheme: {scheme_name}", flush=True)
        print(f"Grid: {self.nx}x{self.ny}x{self.nz}  |  Domain: {self.Lx:.2f}x{self.Ly:.2f}x{self.Lz:.2f}  |  "
              f"dx={self.dx:.3f}, dy={self.dy:.3f}, dz_min={dz_min:.3f}, dz_max={dz_max:.3f}  |  "
              f"dx+={dx_plus:.1f}, dy+={dy_plus:.1f}, "
              f"dz+_min={dz_min * u_tau_target / self.nu:.1f}, dz+_max={dz_max * u_tau_target / self.nu:.1f}", flush=True)
        print("=" * 90, flush=True)
        header = (f"{'Step':>6} {'Time':>10} {'dt':>10} {'max(div)':>12} "
                  f"{'u_bulk':>10} {'u_tau':>10} {'forcing':>12}")
        if self.advection_scheme == 'sl':
            header += f" {'clamped':>9}"
        print(header, flush=True)
        print("=" * 90, flush=True)

        start_time = time.time()
        last_walltime_print = start_time

        chunk_size = self.n_save // self.n_out + 1
        timeseries_data = {
            'step': np.zeros(chunk_size, dtype=np.int32),
            'time': np.zeros(chunk_size, dtype=np.float64),
            'u_bulk': np.zeros(chunk_size, dtype=np.float64),
            'u_tau': np.zeros(chunk_size, dtype=np.float64),
            'forcing': np.zeros(chunk_size, dtype=np.float64),
            'index': 0
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
                print(f"  Wall-time: {elapsed:.2f}s (last {10 * self.n_out} steps), "
                      f"{total_elapsed:.2f}s (total)", flush=True)
                last_walltime_print = current_time

            if self.dt_update_interval > 0 and step % self.dt_update_interval == 0 and step > 0:
                dt_new = self.compute_cfl_dt()
                if abs(dt_new - self.dt) / self.dt > 0.05:
                    self.dt = dt_new

            u_bulk, forcing = step_function(self.dt)
            self.time += self.dt

            if step % self.n_out == 0:
                div_final = compute_divergence(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                                               self.dx, self.dy, self.dz_f)
                max_div = torch.max(torch.abs(div_final)).item()
                u_tau = compute_u_tau(self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type)

                u_bulk_scalar = u_bulk.item() if torch.is_tensor(u_bulk) else u_bulk
                u_tau_scalar = u_tau.item() if torch.is_tensor(u_tau) else u_tau
                forcing_scalar = forcing.item() if torch.is_tensor(forcing) else forcing

                idx = timeseries_data['index']
                timeseries_data['step'][idx] = step
                timeseries_data['time'][idx] = self.time
                timeseries_data['u_bulk'][idx] = u_bulk_scalar
                timeseries_data['u_tau'][idx] = u_tau_scalar
                timeseries_data['forcing'][idx] = forcing_scalar
                timeseries_data['index'] += 1

            if self.n_stats > 0 and self.time >= self.t_stats and step % self.n_stats == 0:
                if step % self.n_out == 0:
                    u_tau_for_stats = u_tau
                else:
                    u_tau_for_stats = compute_u_tau(self.u, self.z_c, self.nu,
                                                    top_wall_bc_type=self.top_wall_bc_type)
                if self.turbulence_stats.n_samples == 0:
                    print(f"  [Stats] Starting statistics collection at t = {self.time:.3f}", flush=True)
                self.turbulence_stats.accumulate_statistics(self.u, self.v, self.w, u_tau_for_stats)

            if step % self.n_save == 0:
                if (torch.any(torch.isnan(self.u)) or torch.any(torch.isinf(self.u)) or
                        torch.any(torch.isnan(self.v)) or torch.any(torch.isinf(self.v)) or
                        torch.any(torch.isnan(self.w)) or torch.any(torch.isinf(self.w)) or
                        torch.any(torch.isnan(self.p)) or torch.any(torch.isinf(self.p))):
                    print(f"\n{'='*90}", flush=True)
                    print(f"ERROR: NaN or Inf detected at step {step}, time = {self.time:.6f}", flush=True)
                    print(f"{'='*90}\n", flush=True)
                    u_tau_error = compute_u_tau(self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type)
                    u_bulk_error = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
                    forcing_error = (self.U_bulk - u_bulk_error) / self.dt
                    save_flow_fields(self.u, self.v, self.w, self.p, self.z_c, self.z_f,
                                     self.Lx, self.Ly, step, self.time,
                                     u_tau_error.item() if torch.is_tensor(u_tau_error) else u_tau_error,
                                     forcing_error.item() if torch.is_tensor(forcing_error) else forcing_error,
                                     self.results_folder, 'fields_error.npz')
                    print(f"Error state saved to fields_error.npz", flush=True)
                    break

                if step % self.n_out != 0:
                    u_tau = compute_u_tau(self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type)
                    u_tau_scalar = u_tau.item() if torch.is_tensor(u_tau) else u_tau
                    forcing_scalar = forcing.item() if torch.is_tensor(forcing) else forcing

                if timeseries_data['index'] > 0:
                    npz_file = os.path.join(self.results_folder, 'timeseries.npz')
                    n_filled = timeseries_data['index']
                    chunks = {k: timeseries_data[k][:n_filled]
                              for k in ('step', 'time', 'u_bulk', 'u_tau', 'forcing')}
                    if os.path.exists(npz_file):
                        existing = np.load(npz_file)
                        chunks = {k: np.concatenate([existing[k], chunks[k]]) for k in chunks}
                    np.savez_compressed(npz_file, **chunks)
                    timeseries_data['index'] = 0

                save_flow_fields(self.u, self.v, self.w, self.p, self.z_c, self.z_f,
                                 self.Lx, self.Ly, step, self.time, u_tau_scalar, forcing_scalar,
                                 self.results_folder, 'fields.npz')

                if self.turbulence_stats is not None and self.turbulence_stats.n_samples > 0:
                    self.turbulence_stats.save_state(self.stats_state_path)

            if self.n_snapshot > 0 and step % self.n_snapshot == 0:
                u_tau_snap = compute_u_tau(self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type)
                forcing_snap = forcing.item() if torch.is_tensor(forcing) else forcing
                snap_name = f'fields_t{self.time:09.3f}.npz'
                save_flow_fields(self.u, self.v, self.w, self.p, self.z_c, self.z_f,
                                 self.Lx, self.Ly, step, self.time,
                                 u_tau_snap.item() if torch.is_tensor(u_tau_snap) else u_tau_snap,
                                 forcing_snap, self.results_folder, snap_name)
                print(f"  [Snapshot] Saved {snap_name}", flush=True)

            if step % self.n_out == 0:
                row = (f"{step:6d} {self.time:10.6f} {self.dt:10.6f} {max_div:12.3e} "
                       f"{u_bulk_scalar:10.6f} {u_tau_scalar:10.6f} {forcing_scalar:12.3e}")
                if self.advection_scheme == 'sl':
                    row += f" {self.sl.n_clamped_last.item():9d}"
                print(row, flush=True)

        total_wall_time = time.time() - start_time
        print(f"{'='*90}", flush=True)
        print(f"Simulation complete: {step} steps, total time = {self.time:.6f}, final dt = {self.dt:.6f}", flush=True)
        print(f"Total wall time: {total_wall_time:.2f}s", flush=True)
        print("=" * 90 + "\n", flush=True)

        if self.turbulence_stats is not None and self.turbulence_stats.n_samples > 0:
            print(f"\nFinalizing turbulence statistics ({self.turbulence_stats.n_samples} samples)...", flush=True)
            self.turbulence_stats.save_state(self.stats_state_path)
            self.turbulence_stats.save_statistics(self.stats_output_path)

        u_profile = self.u[0, 0, :]
        plot_profile(u_profile, self.z_c, 'u', 'z', 'Final velocity profile',
                     'u_profile_final.png', self.results_folder)

        u_tau_final = compute_u_tau(self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type)
        u_bulk_final = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
        forcing_final = (self.U_bulk - u_bulk_final) / self.dt
        save_flow_fields(self.u, self.v, self.w, self.p, self.z_c, self.z_f,
                         self.Lx, self.Ly, step, self.time, u_tau_final, forcing_final,
                         self.results_folder, 'fields_final.npz')
