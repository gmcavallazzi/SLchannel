"""utils.py — Grid construction and run diagnostics.

The tanh-stretched wall-normal grid, u_tau / bulk-velocity / divergence
diagnostics, and the npz field checkpoint reader and writer.

Provenance
----------
Inherited from torChannel, the Eulerian parent solver by the same author
(MIT), imported verbatim in slChannel commit 05a1b30 and kept conceptually
in sync since. Changes here are limited to package-relative imports, the
removal of code paths slChannel does not use, and the local divergences
noted above. Deliberately NOT reformatted, so it stays diffable against
upstream. See docs/PROVENANCE.md.
"""

import os
import math
import torch
import matplotlib.pyplot as plt

def generate_grid(gamma, nz, Lz, device='cpu', stretching_type='symmetric'):
    """
    Generate stretched grid in z-direction using hyperbolic tangent stretching.

    Args:
        gamma: Stretching parameter (higher = more clustering)
        nz: Number of cells in z-direction
        Lz: Domain height
        device: Device for tensor allocation
        stretching_type: 'symmetric' (cluster at both walls) or 'bottom' (cluster at bottom only)

    Returns:
        z_f: Face coordinates (nz+1 points)
        z_c: Cell center coordinates (nz+2 points, includes ghost cells)
        dz_f: Face spacing (nz points)
        dz_c: Center spacing (nz+1 points)
    """
    k = torch.linspace(0, nz, nz+1, device=device)

    if stretching_type == 'bottom':
        # One-sided stretching: cluster near bottom wall only
        # Maps k ∈ [0, nz] → xi ∈ [0, 1] → z_f ∈ [0, Lz]
        # Fine spacing at z=0, coarse spacing at z=Lz
        xi = k / nz
        gamma_tensor = torch.tensor(gamma, device=device)
        z_f = Lz * (1.0 - torch.tanh(gamma * (1.0 - xi)) / torch.tanh(gamma_tensor))
    else:  # 'symmetric' (default)
        # Two-sided stretching: cluster near both walls
        # Maps k ∈ [0, nz] → xi ∈ [-1, 1] → z_f ∈ [0, Lz]
        xi = (2 * k / nz) - 1
        z_f = 0.5 * Lz * (1 + torch.tanh(gamma*xi)/torch.tanh(torch.tensor(gamma, device=device)))

    z_c_inn = 0.5 * (z_f[:-1] + z_f[1:])
    z_c = torch.cat([torch.tensor([-z_c_inn[0]], device=device), z_c_inn,
                     torch.tensor([2*z_f[-1] -z_c_inn[-1]], device=device)])

    # Original definitions (names are confusing but match operators.py expectations!)
    dz_f = z_f[1:] - z_f[:-1]  # Length nz
    dz_c = z_c[1:] - z_c[:-1]  # Length nz+1

    return z_f, z_c, dz_f, dz_c


def save_grid_csv(z_f, z_c, dz_f, dz_c, nz, results_folder):
    import csv
    import numpy as np

    max_len = max(len(z_f), len(z_c), len(dz_f), len(dz_c))

    # Pad tensors to same length with NaN, ensuring all on same device
    device = z_f.device
    z_f_padded = torch.cat([z_f, torch.full((max_len - len(z_f),), float('nan'), device=device)])
    z_c_padded = torch.cat([z_c, torch.full((max_len - len(z_c),), float('nan'), device=device)])
    dz_f_padded = torch.cat([dz_f, torch.full((max_len - len(dz_f),), float('nan'), device=device)])
    dz_c_padded = torch.cat([dz_c, torch.full((max_len - len(dz_c),), float('nan'), device=device)])

    # Convert to CPU numpy arrays
    z_f_np = z_f_padded.cpu().numpy()
    z_c_np = z_c_padded.cpu().numpy()
    dz_f_np = dz_f_padded.cpu().numpy()
    dz_c_np = dz_c_padded.cpu().numpy()

    filepath = os.path.join(results_folder, 'grid.csv')
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['z_f', 'z_c', 'dz_f', 'dz_c'])
        for i in range(max_len):
            writer.writerow([z_f_np[i], z_c_np[i], dz_f_np[i], dz_c_np[i]])

def plot_grid(z_f, z_c, results_folder):
    import numpy as np
    plt.figure()
    # Convert to CPU for plotting
    plt.plot(np.arange(len(z_f)), z_f.cpu().numpy(), 'o-', label='z_f (faces)')
    plt.plot(np.arange(len(z_c)), z_c.cpu().numpy(), 'x-', label='z_c (centers)')
    plt.xlabel('Index')
    plt.ylabel('z')
    plt.title('Grid points in z')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_folder, 'grid.png'))
    plt.close()

def plot_profile(data, coord, data_label, coord_label, title, filename, results_folder):
    plt.figure()
    plt.plot(data.cpu().numpy(), coord.cpu().numpy())
    plt.xlabel(data_label)
    plt.ylabel(coord_label)
    plt.title(title)
    plt.grid()
    filepath = os.path.join(results_folder, filename)
    plt.savefig(filepath)
    plt.close()


def compute_u_tau(u, z_c, nu, top_wall_bc_type='dirichlet'):
    """
    Compute friction velocity u_tau from wall shear stress.
    Uses one-sided difference to approximate du/dz at the wall.
    
    Args:
        u: Velocity field (u component)
        z_c: Cell center coordinates
        nu: Kinematic viscosity
        top_wall_bc_type: 'dirichlet' (no-slip) or 'neumann' (free-slip)
    """
    u_mean_bot = torch.mean(u[:, :, 1])
    dist = z_c[1]

    # Wall shear stress: tau_wall = nu * du/dz at wall
    # Approximate: du/dz ≈ u[1] / dist (since u[0]=0 at wall by no-slip BC)
    tau_bot = nu * u_mean_bot / dist
    u_tau_bot = torch.sqrt(torch.abs(tau_bot))

    if top_wall_bc_type == 'neumann':
        # Free-slip top wall: shear stress is zero at top.
        # Use only bottom wall for u_tau
        return u_tau_bot
    else:
        # No-slip top wall: Average of bottom and top
        u_mean_top = torch.mean(u[:, :, -2])
        tau_top = nu * u_mean_top / dist
        u_tau_top = torch.sqrt(torch.abs(tau_top))
        return 0.5*(u_tau_bot + u_tau_top)


@torch.jit.script
def compute_bulk_velocity(u: torch.Tensor, cell_vol_ratio: torch.Tensor,
                         total_volume: float) -> torch.Tensor:
    """
    Compute bulk (volume-averaged) velocity for staggered grid.

    Uses u-values at cell right faces (u[1:nx+1]) which is consistent
    with how forcing is applied to u[1:nx+1, 1:ny+1, 1:nz+1].

    For periodic BC in x, this gives the correct volume average since
    the u-faces correspond to the cell volumes in cell_vol_ratio.
    JIT-compiled for GPU performance.
    """
    nx, ny, nz = cell_vol_ratio.shape
    u_bulk = torch.sum(u[1:nx+1, 1:ny+1, 1:nz+1] * cell_vol_ratio) / total_volume
    #u_bulk = torch.sum(0.5*(u[1:nx+1, 1:ny+1, 1:nz+1]+u[0:nx, 1:ny+1, 1:nz+1]) * cell_vol_ratio) / total_volume
    return u_bulk


@torch.jit.script
def compute_divergence(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                      nx: int, ny: int, nz: int,
                      dx: float, dy: float, dz_f: torch.Tensor) -> torch.Tensor:
    """
    Compute divergence of velocity field on staggered grid.
    JIT-compiled for GPU performance.
    """
    # Vectorized computation using PyTorch slicing (GPU-compatible)
    du_dx = (u[1:nx+1, 1:ny+1, 1:nz+1] - u[0:nx, 1:ny+1, 1:nz+1]) / dx
    dv_dy = (v[1:nx+1, 1:ny+1, 1:nz+1] - v[1:nx+1, 0:ny, 1:nz+1]) / dy
    dw_dz = (w[1:nx+1, 1:ny+1, 1:nz+1] - w[1:nx+1, 1:ny+1, 0:nz]) / dz_f[0:nz].view(1, 1, -1)

    div = du_dx + dv_dy + dw_dz
    return div


def save_flow_fields(u, v, w, p, z_c, z_f, Lx, Ly, step, time, u_tau, forcing, results_folder, filename='fields.npz'):
    """
    Save flow fields to npz file (silent, no screen output).
    Intended to be overwritten during simulation for quick inspection.
    """
    import numpy as np
    import torch
    filepath = os.path.join(results_folder, filename)

    # Convert tensors to numpy (handle both GPU and CPU tensors)
    u_np = u.cpu().numpy() if torch.is_tensor(u) else u
    v_np = v.cpu().numpy() if torch.is_tensor(v) else v
    w_np = w.cpu().numpy() if torch.is_tensor(w) else w
    p_np = p.cpu().numpy() if torch.is_tensor(p) else p
    z_c_np = z_c.cpu().numpy() if torch.is_tensor(z_c) else z_c
    z_f_np = z_f.cpu().numpy() if torch.is_tensor(z_f) else z_f
    u_tau_val = u_tau.item() if torch.is_tensor(u_tau) else u_tau
    forcing_val = forcing.item() if torch.is_tensor(forcing) else forcing

    np.savez(filepath,
             u=u_np,
             v=v_np,
             w=w_np,
             p=p_np,
             z_c=z_c_np,
             z_f=z_f_np,
             Lx=Lx,
             Ly=Ly,
             step=step,
             time=time,
             u_tau=u_tau_val,
             forcing=forcing_val)

def load_flow_fields(filepath, device='cpu'):
    """
    Load flow fields from npz file.

    Args:
        filepath: Path to the .npz file to load
        device: Device to load tensors to ('cpu' or 'cuda')

    Returns:
        Dictionary containing:
            - u, v, w, p: velocity and pressure fields as torch tensors
            - z_c, z_f: grid coordinates as torch tensors
            - Lx, Ly: domain sizes (floats)
            - step: timestep number (int)
            - time: simulation time (float)
            - u_tau, forcing: flow statistics (floats)
    """
    import numpy as np

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Field file not found: {filepath}")

    # Load npz file
    data = np.load(filepath)

    # Convert numpy arrays to torch tensors on the specified device
    # Use torch.tensor() with explicit dtype=torch.float32 for consistency
    fields = {
        'u': torch.tensor(data['u'], device=device),
        'v': torch.tensor(data['v'], device=device),
        'w': torch.tensor(data['w'], device=device),
        'p': torch.tensor(data['p'], device=device),
        'z_c': torch.tensor(data['z_c'], device=device),
        'z_f': torch.tensor(data['z_f'], device=device),
        'Lx': float(data['Lx']),
        'Ly': float(data['Ly']),
        'step': int(data['step']),
        'time': float(data['time']),
        'u_tau': float(data['u_tau']),
        'forcing': float(data['forcing'])
    }

    return fields
