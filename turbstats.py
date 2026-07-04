import os
import torch
import numpy as np

class TurbulenceStats:
    """
    Class for computing and accumulating turbulence statistics on-the-fly.

    Computes:
    - U(z): mean velocity profile
    - u'u'(z), v'v'(z), w'w'(z), u'w'(z): Reynolds stresses
    - 2D energy spectra at z+ = z_plus_target: E(kx,ky) for uu, vv, ww, uw
      (premultiplication by kx*ky is done during plotting)

    Note: dU/dz is computed in post-processing from U(z) for better accuracy.

    Uses hybrid accumulation: accumulates sums, divides by n_samples at end.
    """

    def __init__(self, nx, ny, nz, Lx, Ly, Lz, z_c, z_f, dz_c, dz_f,
                 dx, dy, nu, Re_tau_target, z_plus_target=15.0, device='cpu'):
        """
        Initialize statistics accumulator.

        Args:
            nx, ny, nz: Grid dimensions
            Lx, Ly, Lz: Domain sizes
            z_c, z_f: Cell centers and faces in z-direction
            dz_c, dz_f: Grid spacings in z
            dx, dy: Grid spacings in x, y
            nu: Kinematic viscosity
            Re_tau_target: Target friction Reynolds number (for z+ calculation)
            z_plus_target: Target height in wall units for 2D spectra
            device: 'cpu' or 'cuda'
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.z_c = z_c
        self.z_f = z_f
        self.dz_c = dz_c
        self.dz_f = dz_f
        self.dx = dx
        self.dy = dy
        self.nu = nu
        self.device = device

        # Compute target u_tau and find z+ grid indices
        delta = Lz / 2.0  # Half-channel height
        self.u_tau_target = Re_tau_target * nu / delta

        # Find grid indices closest to z_plus_target from each wall
        # Bottom wall: z = 0, find z_c closest to z_plus_target * nu / u_tau_target
        # Top wall: z = Lz, find z_c closest to Lz - z_plus_target * nu / u_tau_target
        z_target_phys = z_plus_target * nu / self.u_tau_target

        # Interior z_c points (excluding ghost cells)
        z_c_interior = z_c[1:nz+1]

        # Find index for bottom wall (closest to z_target_phys)
        idx_bot = torch.argmin(torch.abs(z_c_interior - z_target_phys)).item()
        self.k_bot = idx_bot + 1  # Offset by 1 for ghost cell

        # Find index for top wall (closest to Lz - z_target_phys)
        idx_top = torch.argmin(torch.abs(z_c_interior - (Lz - z_target_phys))).item()
        self.k_top = idx_top + 1  # Offset by 1 for ghost cell

        print(f"\nStatistics initialization:", flush=True)
        print(f"  Target z+ = {z_plus_target:.1f}", flush=True)
        print(f"  Bottom wall: z_c[{self.k_bot}] = {z_c[self.k_bot]:.6f}, z+ = {z_c[self.k_bot] * self.u_tau_target / nu:.1f}", flush=True)
        print(f"  Top wall: z_c[{self.k_top}] = {z_c[self.k_top]:.6f}, z+ = {(Lz - z_c[self.k_top]) * self.u_tau_target / nu:.1f}", flush=True)

        # Sample counter
        self.n_samples = 0

        # Initialize accumulators for 1D profiles (over z)
        # All profiles are on interior points (nz points)
        self.U_sum = torch.zeros(nz, device=device)
        self.uu_sum = torch.zeros(nz, device=device)
        self.vv_sum = torch.zeros(nz, device=device)
        self.ww_sum = torch.zeros(nz, device=device)
        self.uw_sum = torch.zeros(nz, device=device)


        # Initialize accumulators for 2D energy spectra at z+
        # Shape: (nx//2, ny//2) for each of uu, vv, ww, uw
        # Store raw spectra, premultiplication done during plotting
        self.E_uu_2d_sum = torch.zeros(nx//2, ny//2, device=device)
        self.E_vv_2d_sum = torch.zeros(nx//2, ny//2, device=device)
        self.E_ww_2d_sum = torch.zeros(nx//2, ny//2, device=device)
        self.E_uw_2d_sum = torch.zeros(nx//2, ny//2, device=device)

        # Wavenumber arrays for 2D spectra (for plotting/saving)
        # dx, dy are already grid spacings (Lx/nx, Ly/ny)
        self.kx = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)[1:]  # Skip DC component
        self.ky = 2 * np.pi * np.fft.rfftfreq(ny, d=dy)[1:]

    def accumulate_statistics(self, u, v, w, u_tau_current):
        """
        Accumulate statistics from one snapshot.

        Args:
            u, v, w: Velocity fields (staggered grid, including ghost cells)
            u_tau_current: Current friction velocity (for diagnostics)
        """
        # Extract interior points
        # u: shape (nx+1, ny+2, nz+2) -> interior: (nx+1, ny, nz) at [0:nx+1, 1:ny+1, 1:nz+1]
        # v: shape (nx+2, ny+1, nz+2) -> interior: (nx, ny+1, nz) at [1:nx+1, 0:ny+1, 1:nz+1]
        # w: shape (nx+2, ny+2, nz+1) -> interior: (nx, ny, nz+1) at [1:nx+1, 1:ny+1, 0:nz+1]

        u_int = u[0:self.nx+1, 1:self.ny+1, 1:self.nz+1]  # (nx+1, ny, nz)
        v_int = v[1:self.nx+1, 0:self.ny+1, 1:self.nz+1]  # (nx, ny+1, nz)
        w_int = w[1:self.nx+1, 1:self.ny+1, 0:self.nz+1]  # (nx, ny, nz+1)

        # Compute mean velocity profile U(z) by averaging over x,y
        # Average u at cell centers by interpolating in x
        u_cell_center = 0.5 * (u_int[:-1, :, :] + u_int[1:, :, :])  # (nx, ny, nz)
        U = torch.mean(u_cell_center, dim=(0, 1))  # (nz,)
        self.U_sum += U

        # Compute fluctuations u'(x,y,z) = u(x,y,z) - U(z)
        # Broadcast U(z) to (nx, ny, nz)
        u_fluct = u_cell_center - U.view(1, 1, -1)  # (nx, ny, nz)

        # For v and w, compute fluctuations similarly
        # v is already at cell centers in x,z, interpolate in y
        v_cell_center = 0.5 * (v_int[:, :-1, :] + v_int[:, 1:, :])  # (nx, ny, nz)
        V = torch.mean(v_cell_center, dim=(0, 1))  # (nz,)
        v_fluct = v_cell_center - V.view(1, 1, -1)

        # w is at cell faces in z, interpolate to cell centers
        w_cell_center = 0.5 * (w_int[:, :, :-1] + w_int[:, :, 1:])  # (nx, ny, nz)
        W = torch.mean(w_cell_center, dim=(0, 1))  # (nz,)
        w_fluct = w_cell_center - W.view(1, 1, -1)

        # Compute Reynolds stresses by averaging over x,y
        uu = torch.mean(u_fluct * u_fluct, dim=(0, 1))  # (nz,)
        vv = torch.mean(v_fluct * v_fluct, dim=(0, 1))  # (nz,)
        ww = torch.mean(w_fluct * w_fluct, dim=(0, 1))  # (nz,)
        uw = torch.mean(u_fluct * w_fluct, dim=(0, 1))  # (nz,)

        self.uu_sum += uu
        self.vv_sum += vv
        self.ww_sum += ww
        self.uw_sum += uw

        # Compute 2D premultiplied spectra at z+ locations
        # Extract planes at k_bot and k_top
        # Average between both walls

        # Bottom wall plane
        u_bot = u_fluct[:, :, self.k_bot - 1]  # (nx, ny)
        v_bot = v_fluct[:, :, self.k_bot - 1]
        w_bot = w_fluct[:, :, self.k_bot - 1]

        # Top wall plane
        u_top = u_fluct[:, :, self.k_top - 1]
        v_top = v_fluct[:, :, self.k_top - 1]
        w_top = w_fluct[:, :, self.k_top - 1]

        # Compute 2D FFTs for both walls separately
        u_fft_bot = torch.fft.rfft2(u_bot)
        v_fft_bot = torch.fft.rfft2(v_bot)
        w_fft_bot = torch.fft.rfft2(w_bot)

        # For top wall, negate w to account for coordinate system (z points into wall)
        # This is crucial for uw correlation
        u_fft_top = torch.fft.rfft2(u_top)
        v_fft_top = torch.fft.rfft2(v_top)
        w_fft_top = torch.fft.rfft2(-w_top)

        # Energy spectra via explicit real/imag arithmetic: complex
        # elementwise CUDA ops (abs, conj-multiply) are jiterator kernels
        # compiled with nvrtc at runtime, which fails on the GB10 (sm_121)
        # regardless of PYTORCH_JIT=0. .real/.imag are views; the math below
        # runs as plain real-dtype eager kernels. Same formulation as
        # torChannel's multi-plane spectra.
        norm = (self.nx * self.ny)**2

        def _auto(F):
            return (F.real * F.real + F.imag * F.imag) / norm

        def _cross_re(F, G):
            # (F * conj(G)).real
            return (F.real * G.real + F.imag * G.imag) / norm

        E_uu_bot = _auto(u_fft_bot)
        E_vv_bot = _auto(v_fft_bot)
        E_ww_bot = _auto(w_fft_bot)
        E_uw_bot = _cross_re(u_fft_bot, w_fft_bot)

        E_uu_top = _auto(u_fft_top)
        E_vv_top = _auto(v_fft_top)
        E_ww_top = _auto(w_fft_top)
        E_uw_top = _cross_re(u_fft_top, w_fft_top)

        # Average the spectra (no flips needed for auto-spectra)
        E_uu_2d = 0.5 * (E_uu_bot + E_uu_top)
        E_vv_2d = 0.5 * (E_vv_bot + E_vv_top)
        E_ww_2d = 0.5 * (E_ww_bot + E_ww_top)
        E_uw_2d = 0.5 * (E_uw_bot + E_uw_top)

        # Keep only positive wavenumbers (skip DC) and fold negative wavenumbers
        # Use symmetric spectrum for negative kx: E(kx) + E(-kx) for kx > 0
        # rfft2 output has shape (nx, nky+1)
        # kx indices: 0, 1, ..., nx/2, -nx/2+1, ..., -1
        nkx = self.nx // 2
        nky = self.ny // 2

        # Helper to fold a spectrum
        def fold_spectrum(E):
            # E shape: (nx, nky+1)
            # Positive kx part (indices 1 to nkx)
            E_pos = E[1:nkx+1, 1:nky+1]

            # Negative kx part (indices nx-nkx to nx)
            # We need to flip this to match the order of E_pos (1 to nkx)
            # indices in E: nx-1 (k=-1), nx-2 (k=-2), ..., nx-nkx (k=-nkx)
            # We want to add E[nx-1] to E[1], E[nx-2] to E[2], etc.
            # So we take E[nx-nkx : nx] and flip it.
            E_neg = E[self.nx-nkx:self.nx, 1:nky+1]
            E_neg_flipped = torch.flip(E_neg, dims=[0])

            return E_pos + E_neg_flipped

        E_uu_2d_sym = fold_spectrum(E_uu_2d)
        E_vv_2d_sym = fold_spectrum(E_vv_2d)
        E_ww_2d_sym = fold_spectrum(E_ww_2d)
        E_uw_2d_sym = fold_spectrum(E_uw_2d)

        # Accumulate raw spectra (premultiplication done during plotting)
        self.E_uu_2d_sum += E_uu_2d_sym[:nkx, :nky]
        self.E_vv_2d_sum += E_vv_2d_sym[:nkx, :nky]
        self.E_ww_2d_sum += E_ww_2d_sym[:nkx, :nky]
        self.E_uw_2d_sum += E_uw_2d_sym[:nkx, :nky]

        # Increment sample counter
        self.n_samples += 1

    def finalize_statistics(self):
        """
        Finalize statistics by dividing accumulated sums by n_samples.

        Returns:
            Dictionary of computed statistics (all on CPU as numpy arrays)
        """
        if self.n_samples == 0:
            raise RuntimeError("No samples accumulated - cannot finalize statistics")

        print(f"\nFinalizing statistics from {self.n_samples} samples...", flush=True)

        # Compute averages on GPU first (keep as tensors)
        U_mean_gpu = self.U_sum / self.n_samples
        uu_mean_gpu = self.uu_sum / self.n_samples
        vv_mean_gpu = self.vv_sum / self.n_samples
        ww_mean_gpu = self.ww_sum / self.n_samples
        uw_mean_gpu = self.uw_sum / self.n_samples

        E_uu_2d_gpu = self.E_uu_2d_sum / self.n_samples
        E_vv_2d_gpu = self.E_vv_2d_sum / self.n_samples
        E_ww_2d_gpu = self.E_ww_2d_sum / self.n_samples
        E_uw_2d_gpu = self.E_uw_2d_sum / self.n_samples

        # Compute u_tau from velocity gradient at wall
        # At the wall: tau_wall = nu * dU/dz|_wall = u_tau^2
        # Simple approach: use first interior point distance from wall

        # Distance from wall to first interior cell center
        dz_utau = float(self.z_c[1])  # z_c[1] is first interior point (z_c[0] is ghost)

        # Velocity gradient: average of first points from both walls divided by distance
        # U_mean_gpu[0]: first interior point from bottom wall
        # U_mean_gpu[-1]: first interior point from top wall
        dUdz_wall = float(0.5 * (U_mean_gpu[0] + U_mean_gpu[-1]) / dz_utau)

        # Compute u_tau from wall shear stress: tau_wall = nu * dU/dz|_wall = u_tau^2
        u_tau_computed = float(np.sqrt(self.nu * dUdz_wall))

        # Convert to numpy only at the end for saving
        U_mean = np.asarray(U_mean_gpu.detach().cpu().numpy())
        uu_mean = np.asarray(uu_mean_gpu.detach().cpu().numpy())
        vv_mean = np.asarray(vv_mean_gpu.detach().cpu().numpy())
        ww_mean = np.asarray(ww_mean_gpu.detach().cpu().numpy())
        uw_mean = np.asarray(uw_mean_gpu.detach().cpu().numpy())

        E_uu_2d = np.asarray(E_uu_2d_gpu.detach().cpu().numpy())
        E_vv_2d = np.asarray(E_vv_2d_gpu.detach().cpu().numpy())
        E_ww_2d = np.asarray(E_ww_2d_gpu.detach().cpu().numpy())
        E_uw_2d = np.asarray(E_uw_2d_gpu.detach().cpu().numpy())

        # Package into dictionary
        stats = {
            'n_samples': self.n_samples,
            'z_c': np.asarray(self.z_c[1:self.nz+1].detach().cpu().numpy()),  # Interior points
            'U_mean': U_mean,
            'uu_mean': uu_mean,
            'vv_mean': vv_mean,
            'ww_mean': ww_mean,
            'uw_mean': uw_mean,
            'kx': self.kx[:self.nx//2],
            'ky': self.ky[:self.ny//2],
            'E_uu_2d': E_uu_2d,
            'E_vv_2d': E_vv_2d,
            'E_ww_2d': E_ww_2d,
            'E_uw_2d': E_uw_2d,
            'z_plus_target': self.k_bot,  # Store for reference
            'Re_tau_target': self.u_tau_target * (self.Lz / 2) / self.nu,
            'nu': self.nu,  # Kinematic viscosity
            'Re': 1.0 / self.nu,  # Reynolds number
            'u_tau': u_tau_computed  # Friction velocity from Reynolds stress
        }

        return stats

    def save_statistics(self, filepath):
        """
        Save finalized statistics to NPZ file.

        Args:
            filepath: Path to save NPZ file
        """
        stats = self.finalize_statistics()
        np.savez_compressed(filepath, **stats)
        print(f"\nStatistics saved to: {filepath}", flush=True)

    @staticmethod
    def load_statistics(filepath):
        """
        Load statistics from NPZ file.

        Args:
            filepath: Path to NPZ file

        Returns:
            Dictionary of statistics
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Statistics file not found: {filepath}")

        data = np.load(filepath)
        stats = {key: data[key] for key in data.files}
        return stats

    def save_state(self, filepath):
        """
        Save the current accumulator state (running sums + n_samples) to NPZ file.
        This allows restarting statistics accumulation from a checkpoint.

        Args:
            filepath: Path to save state file (e.g., 'stats_state.npz')
        """
        # Convert all GPU tensors to CPU numpy arrays for saving
        state = {
            'n_samples': self.n_samples,
            # Running sums
            'U_sum': np.asarray(self.U_sum.detach().cpu().numpy()),
            'uu_sum': np.asarray(self.uu_sum.detach().cpu().numpy()),
            'vv_sum': np.asarray(self.vv_sum.detach().cpu().numpy()),
            'ww_sum': np.asarray(self.ww_sum.detach().cpu().numpy()),
            'uw_sum': np.asarray(self.uw_sum.detach().cpu().numpy()),
            'E_uu_2d_sum': np.asarray(self.E_uu_2d_sum.detach().cpu().numpy()),
            'E_vv_2d_sum': np.asarray(self.E_vv_2d_sum.detach().cpu().numpy()),
            'E_ww_2d_sum': np.asarray(self.E_ww_2d_sum.detach().cpu().numpy()),
            'E_uw_2d_sum': np.asarray(self.E_uw_2d_sum.detach().cpu().numpy()),
            # Grid parameters for validation on load
            'nx': self.nx,
            'ny': self.ny,
            'nz': self.nz,
        }

        np.savez_compressed(filepath, **state)
        print(f"\nStatistics state saved: {self.n_samples} samples accumulated -> {filepath}", flush=True)

    def load_state(self, filepath):
        """
        Load accumulator state from a previously saved checkpoint.
        Restores all running sums and n_samples to continue accumulation.

        Args:
            filepath: Path to state file (e.g., 'stats_state.npz')
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Statistics state file not found: {filepath}")

        print(f"\nLoading statistics state from: {filepath}", flush=True)
        data = np.load(filepath)

        # Validate grid dimensions match
        if data['nx'] != self.nx or data['ny'] != self.ny or data['nz'] != self.nz:
            raise ValueError(
                f"Grid dimension mismatch! State file has (nx={data['nx']}, ny={data['ny']}, nz={data['nz']}) "
                f"but current simulation has (nx={self.nx}, ny={self.ny}, nz={self.nz})"
            )

        # Restore n_samples
        self.n_samples = int(data['n_samples'])

        # Restore running sums (convert back to GPU tensors)
        self.U_sum = torch.tensor(data['U_sum'], device=self.device)
        self.uu_sum = torch.tensor(data['uu_sum'], device=self.device)
        self.vv_sum = torch.tensor(data['vv_sum'], device=self.device)
        self.ww_sum = torch.tensor(data['ww_sum'], device=self.device)
        self.uw_sum = torch.tensor(data['uw_sum'], device=self.device)
        self.E_uu_2d_sum = torch.tensor(data['E_uu_2d_sum'], device=self.device)
        self.E_vv_2d_sum = torch.tensor(data['E_vv_2d_sum'], device=self.device)
        self.E_ww_2d_sum = torch.tensor(data['E_ww_2d_sum'], device=self.device)
        self.E_uw_2d_sum = torch.tensor(data['E_uw_2d_sum'], device=self.device)

        # Backward compatibility: ignore dUdz_sum and omega_y_sum if they exist in old files
        # (they are no longer computed or needed)

        print(f"  Restored state with {self.n_samples} accumulated samples", flush=True)


def compute_statistics_from_snapshot(field_file, config_file, output_file, overrides=None):
    """
    Compute turbulence statistics from a single snapshot (for testing).

    Args:
        field_file: Path to field snapshot (.npz file)
        config_file: Path to configuration file (.yaml)
        output_file: Path to save computed statistics (.npz)
        overrides: Optional dictionary of parameter overrides (e.g., {'Re': 5000, 'Re_tau': 180})

    Returns:
        Dictionary of computed statistics
    """
    import yaml
    from utils import load_flow_fields, compute_u_tau

    print("="*80)
    print("Computing statistics from single snapshot")
    print("="*80)

    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides if provided
    if overrides is not None:
        if 'Re' in overrides:
            config['flow']['Re'] = overrides['Re']
            print(f"  Override: Re = {overrides['Re']:.1f} (nu = {1.0/overrides['Re']:.6e})")
        if 'Re_tau' in overrides:
            config['flow']['Re_tau'] = overrides['Re_tau']
            print(f"  Override: Re_tau = {overrides['Re_tau']:.1f}")

    # Extract flow parameters
    nu = 1.0 / config['flow']['Re']
    Re_tau_target = config['flow']['Re_tau']

    # Device setup
    device_config = config.get('compute', {}).get('device', 'auto')
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_config == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"\nDevice: {device}")

    # Load flow fields (includes grid information)
    print(f"\nLoading flow fields from: {field_file}")
    fields = load_flow_fields(field_file, device=device)

    u = fields['u']
    v = fields['v']
    w = fields['w']

    # Extract grid from field file (contains the actual stretched grid used)
    z_c = fields['z_c']
    z_f = fields['z_f']
    Lx = fields['Lx']
    Ly = fields['Ly']

    # Infer grid dimensions from field shapes
    # u: (nx+1, ny+2, nz+2), v: (nx+2, ny+1, nz+2), w: (nx+2, ny+2, nz+1)
    nx = u.shape[0] - 1
    ny = u.shape[1] - 2
    nz = w.shape[2] - 1

    print(f"  Inferred grid dimensions from field shapes:")
    print(f"    u.shape = {u.shape} -> nx = {nx}")
    print(f"    v.shape = {v.shape} -> ny = {ny}")
    print(f"    w.shape = {w.shape} -> nz = {nz}")

    # Compute grid spacings (dz_c, dz_f)
    # z_c includes ghost cells, so interior is z_c[1:nz+1]
    # z_f has nz+1 face locations
    dz_f = z_f[1:] - z_f[:-1]  # Length nz
    dz_c = z_c[1:] - z_c[:-1]  # Length nz+1

    # Compute dx, dy
    dx = Lx / nx
    dy = Ly / ny

    # Update Lz from grid
    Lz = z_f[-1].item()

    print(f"  Loaded from step {fields['step']}, time = {fields['time']:.6f}")
    print(f"  Grid: nx={nx}, ny={ny}, nz={nz}")
    print(f"  Domain: Lx={Lx:.4f}, Ly={Ly:.4f}, Lz={Lz:.4f}")
    print(f"  z-grid: stretched, z_min={z_f[0]:.6f}, z_max={z_f[-1]:.6f}")
    print(f"  Grid spacings: dx={dx:.6f}, dy={dy:.6f}, dz_min={dz_f.min():.6f}, dz_max={dz_f.max():.6f}")

    # Compute u_tau
    u_tau = compute_u_tau(u, z_c, nu)
    print(f"  u_tau = {u_tau:.6e}")

    # Initialize statistics accumulator
    z_plus_target = config.get('statistics', {}).get('z_plus_target', 15.0)

    stats_computer = TurbulenceStats(
        nx, ny, nz, Lx, Ly, Lz, z_c, z_f, dz_c, dz_f, dx, dy, nu,
        Re_tau_target, z_plus_target=z_plus_target, device=device
    )

    # Accumulate statistics from this single snapshot
    print("\nComputing statistics...")
    stats_computer.accumulate_statistics(u, v, w, u_tau)

    # Finalize and save
    stats = stats_computer.finalize_statistics()

    np.savez_compressed(output_file, **stats)
    print(f"\nStatistics saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Mean velocity profile U(z):")
    print(f"  min = {stats['U_mean'].min():.6e}, max = {stats['U_mean'].max():.6e}")
    print(f"\nReynolds stresses:")
    print(f"  u'u': min = {stats['uu_mean'].min():.6e}, max = {stats['uu_mean'].max():.6e}")
    print(f"  v'v': min = {stats['vv_mean'].min():.6e}, max = {stats['vv_mean'].max():.6e}")
    print(f"  w'w': min = {stats['ww_mean'].min():.6e}, max = {stats['ww_mean'].max():.6e}")
    print(f"  u'w': min = {stats['uw_mean'].min():.6e}, max = {stats['uw_mean'].max():.6e}")
    print(f"\n2D Energy spectra at z+ ≈ {z_plus_target}:")
    print(f"  E_uu: min = {stats['E_uu_2d'].min():.6e}, max = {stats['E_uu_2d'].max():.6e}")
    print(f"  E_vv: min = {stats['E_vv_2d'].min():.6e}, max = {stats['E_vv_2d'].max():.6e}")
    print(f"  E_ww: min = {stats['E_ww_2d'].min():.6e}, max = {stats['E_ww_2d'].max():.6e}")
    print(f"  E_uw: min = {stats['E_uw_2d'].min():.6e}, max = {stats['E_uw_2d'].max():.6e}")
    print("="*80)

    return stats
