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

def generate_hybrid_grid(nz_uniform, nz_stretched, z_transition, Lz, gamma, device='cpu'):
    """
    Generate hybrid grid: uniform [0, z_transition] + stretched [z_transition, Lz].
    Designed for canopy simulations with uniform grid in canopy region and
    stretched grid above for efficient boundary layer resolution.

    Args:
        nz_uniform: Number of cells in uniform region [0, z_transition]
        nz_stretched: Number of cells in stretched region [z_transition, Lz]
        z_transition: Height of transition (typically canopy height)
        Lz: Total domain height
        gamma: Stretching parameter for upper region (higher = more clustering at bottom)
        device: Device for tensor allocation

    Returns:
        z_f: Face coordinates (nz_uniform + nz_stretched + 1 points)
        z_c: Cell center coordinates (nz_uniform + nz_stretched + 2 points, includes ghost cells)
        dz_f: Face spacing (nz_uniform + nz_stretched points)
        dz_c: Center spacing (nz_uniform + nz_stretched + 1 points)
    """
    # Region 1: Uniform grid [0, z_transition]
    dz_uniform = z_transition / nz_uniform
    z_f_uniform = torch.linspace(0, z_transition, nz_uniform + 1, device=device)

    # Region 2: Stretched grid [z_transition, Lz] with one-sided stretching
    # Use one-sided stretching: fine at z=z_transition, coarse at z=Lz
    H_stretched = Lz - z_transition
    k = torch.linspace(0, nz_stretched, nz_stretched + 1, device=device)
    xi = k / nz_stretched  # ξ ∈ [0, 1]

    # One-sided tanh stretching: fine at ξ=0 (bottom), coarse at ξ=1 (top)
    gamma_tensor = torch.tensor(gamma, device=device)
    z_stretched_local = H_stretched * (1.0 - torch.tanh(gamma * (1.0 - xi)) / torch.tanh(gamma_tensor))

    # Concatenate grids (skip first point of stretched to avoid duplicate at transition)
    z_f = torch.cat([z_f_uniform, z_transition + z_stretched_local[1:]])

    # Compute cell centers and spacings (same procedure as generate_grid)
    z_c_inn = 0.5 * (z_f[:-1] + z_f[1:])
    z_c = torch.cat([torch.tensor([-z_c_inn[0]], device=device), z_c_inn,
                     torch.tensor([2*z_f[-1] - z_c_inn[-1]], device=device)])

    dz_f = z_f[1:] - z_f[:-1]  # Length: nz_uniform + nz_stretched
    dz_c = z_c[1:] - z_c[:-1]  # Length: nz_uniform + nz_stretched + 1

    # Validate C1 continuity at transition
    idx_transition = nz_uniform
    dz_before = dz_f[idx_transition - 1].item()
    dz_after = dz_f[idx_transition].item()
    discontinuity = abs(dz_after - dz_before) / dz_before

    if discontinuity > 0.01:  # > 1% discontinuity
        print(f"WARNING: C1 discontinuity at z={z_transition:.4f}: ", flush=True)
        print(f"  dz before = {dz_before:.6e}", flush=True)
        print(f"  dz after  = {dz_after:.6e}", flush=True)
        print(f"  Relative jump = {discontinuity*100:.2f}%", flush=True)
        print(f"  Consider adjusting gamma (current: {gamma:.3f})", flush=True)
    else:
        print(f"Hybrid grid C1 continuity check: {discontinuity*100:.3f}% (OK)", flush=True)

    # Validate monotonicity
    if torch.any(dz_f <= 0):
        raise ValueError("Non-positive cell spacing detected in hybrid grid!")
    if torch.any(z_f[1:] <= z_f[:-1]):
        raise ValueError("Non-monotonic grid detected!")

    return z_f, z_c, dz_f, dz_c

def generate_double_stretched_grid(nz_canopy, nz_outer, z_transition, Lz,
                                   gamma_canopy, gamma_outer, device='cpu'):
    """
    Generate double-stretched grid for canopy flows: tanh clustering at BOTH
    the canopy bed (z=0) and the filament tips (z=z_transition), where the
    highest shear is expected, plus one-sided stretching above the canopy
    (fine at the tips, coarse at the top boundary).

    Follows Monti et al. (2022): "two tangent-hyperbolic functions that
    concentrate the nodes ... at the edge of the canopy layer and close to
    the solid wall".

    Args:
        nz_canopy: Number of cells in canopy region [0, z_transition]
        nz_outer: Number of cells in outer region [z_transition, Lz]
        z_transition: Canopy height h
        Lz: Total domain height
        gamma_canopy: Stretching parameter inside the canopy (symmetric tanh,
                      higher = stronger clustering at bed and tips)
        gamma_outer: Stretching parameter above the canopy (one-sided tanh,
                     higher = stronger clustering at the tips), or 'auto' to
                     solve for C1 continuity at the transition (recommended)
        device: Device for tensor allocation

    Returns:
        z_f: Face coordinates (nz_canopy + nz_outer + 1 points)
        z_c: Cell center coordinates (nz_canopy + nz_outer + 2 points, includes ghost cells)
        dz_f: Face spacing (nz_canopy + nz_outer points)
        dz_c: Center spacing (nz_canopy + nz_outer + 1 points)
    """
    # Region 1: canopy [0, z_transition], symmetric tanh (cluster at both ends)
    k_c = torch.linspace(0, nz_canopy, nz_canopy + 1, device=device)
    xi_c = (2 * k_c / nz_canopy) - 1  # xi ∈ [-1, 1]
    gamma_c = torch.tensor(gamma_canopy, device=device)
    z_f_canopy = 0.5 * z_transition * (1 + torch.tanh(gamma_c * xi_c) / torch.tanh(gamma_c))

    # Region 2: outer [z_transition, Lz], one-sided tanh (fine at tips, coarse at top)
    H_outer = Lz - z_transition
    k_o = torch.linspace(0, nz_outer, nz_outer + 1, device=device)
    xi_o = k_o / nz_outer  # xi ∈ [0, 1]

    if gamma_outer == 'auto':
        # Solve for the gamma that makes the first outer spacing equal the
        # last canopy spacing (C1 continuity). The first spacing
        # H_outer * (1 - tanh(g*(1-1/nz))/tanh(g)) decreases monotonically
        # with g, so bisection is safe.
        target = (z_f_canopy[-1] - z_f_canopy[-2]).item()

        def first_spacing(g):
            return H_outer * (1.0 - math.tanh(g * (1.0 - 1.0/nz_outer)) / math.tanh(g))

        g_lo, g_hi = 1e-3, 20.0
        if first_spacing(g_lo) < target:
            raise ValueError(
                f"Cannot match dz={target:.3e} at the transition: even gamma_outer→0 "
                f"gives {first_spacing(g_lo):.3e}. Increase nz_outer or gamma_canopy.")
        for _ in range(100):
            g_mid = 0.5 * (g_lo + g_hi)
            if first_spacing(g_mid) > target:
                g_lo = g_mid
            else:
                g_hi = g_mid
        gamma_outer = 0.5 * (g_lo + g_hi)
        print(f"  gamma_outer='auto' resolved to {gamma_outer:.6f}", flush=True)

    gamma_o = torch.tensor(gamma_outer, device=device)
    z_f_outer = H_outer * (1.0 - torch.tanh(gamma_o * (1.0 - xi_o)) / torch.tanh(gamma_o))

    # Concatenate (skip duplicate face at the transition)
    z_f = torch.cat([z_f_canopy, z_transition + z_f_outer[1:]])

    # Cell centers and spacings (same procedure as generate_grid)
    z_c_inn = 0.5 * (z_f[:-1] + z_f[1:])
    z_c = torch.cat([torch.tensor([-z_c_inn[0]], device=device), z_c_inn,
                     torch.tensor([2*z_f[-1] - z_c_inn[-1]], device=device)])

    dz_f = z_f[1:] - z_f[:-1]  # Length nz_canopy + nz_outer
    dz_c = z_c[1:] - z_c[:-1]  # Length nz_canopy + nz_outer + 1

    # Diagnostics: spacings at the shear-critical locations
    dz_bed = dz_f[0].item()
    dz_tip_below = dz_f[nz_canopy - 1].item()
    dz_tip_above = dz_f[nz_canopy].item()
    dz_max_canopy = dz_f[:nz_canopy].max().item()
    dz_max_outer = dz_f[nz_canopy:].max().item()
    print(f"Double-stretched grid:", flush=True)
    print(f"  Canopy [0, {z_transition}]: nz={nz_canopy}, gamma={gamma_canopy}, "
          f"dz_bed={dz_bed:.3e}, dz_tip={dz_tip_below:.3e}, dz_max={dz_max_canopy:.3e}", flush=True)
    print(f"  Outer  [{z_transition}, {Lz}]: nz={nz_outer}, gamma={gamma_outer}, "
          f"dz_tip={dz_tip_above:.3e}, dz_max={dz_max_outer:.3e}", flush=True)

    # C1 continuity check at the transition (same criterion as hybrid grid)
    discontinuity = abs(dz_tip_above - dz_tip_below) / dz_tip_below
    if discontinuity > 0.01:
        print(f"WARNING: C1 discontinuity at z={z_transition:.4f}:", flush=True)
        print(f"  dz below = {dz_tip_below:.6e}", flush=True)
        print(f"  dz above = {dz_tip_above:.6e}", flush=True)
        print(f"  Relative jump = {discontinuity*100:.2f}%", flush=True)
        print(f"  Consider adjusting gamma_canopy/gamma_outer", flush=True)
    else:
        print(f"  C1 continuity at z={z_transition}: {discontinuity*100:.3f}% (OK)", flush=True)

    if torch.any(dz_f <= 0):
        raise ValueError("Non-positive cell spacing detected in double-stretched grid!")
    if torch.any(z_f[1:] <= z_f[:-1]):
        raise ValueError("Non-monotonic grid detected!")

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

def print_poisson_matrix(A, nx, ny, nz, results_folder):
    import numpy as np
    N = nx * ny * nz
    print(f"\nPoisson matrix shape: ({N}, {N})", flush=True)
    print(f"Matrix for grid: nx={nx}, ny={ny}, nz={nz}", flush=True)

    csv_path = os.path.join(results_folder, 'poisson_matrix.csv')
    A_dense = A.to_dense().cpu().numpy()
    np.savetxt(csv_path, A_dense, delimiter=',')

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

def print_divergence_field(div, nx, ny, nz):
    print("\nDivergence field:", flush=True)
    for k in range(nz):
        print(f"  z-layer k={k}:", flush=True)
        for j in range(ny):
            row = "    "
            for i in range(nx):
                row += f"{div[i,j,k]:+.6e} "
            print(row, flush=True)

def print_velocity_summary(u, v, w, nx, ny, nz):
    print("\nVelocity field summary:", flush=True)
    print(f"  u: min={torch.min(u):.6e}, max={torch.max(u):.6e}, mean={torch.mean(u):.6e}", flush=True)
    print(f"  v: min={torch.min(v):.6e}, max={torch.max(v):.6e}, mean={torch.mean(v):.6e}", flush=True)
    print(f"  w: min={torch.min(w):.6e}, max={torch.max(w):.6e}, mean={torch.mean(w):.6e}", flush=True)
    print(f"  Interior u[1:{nx},1:{ny+1},1:{nz+1}]: min={torch.min(u[1:nx+1,1:ny+1,1:nz+1]):.6e}, max={torch.max(u[1:nx+1,1:ny+1,1:nz+1]):.6e}", flush=True)
    print(f"  Interior v[1:{nx+1},1:{ny},1:{nz+1}]: min={torch.min(v[1:nx+1,1:ny+1,1:nz+1]):.6e}, max={torch.max(v[1:nx+1,1:ny+1,1:nz+1]):.6e}", flush=True)
    print(f"  Interior w[1:{nx+1},1:{ny+1},1:{nz}]: min={torch.min(w[1:nx+1,1:ny+1,1:nz]):.6e}, max={torch.max(w[1:nx+1,1:ny+1,1:nz]):.6e}", flush=True)

def test_poisson_matrix_indexing(nx, ny, nz):
    print("\nTesting Poisson matrix indexing:", flush=True)
    print(f"Grid: nx={nx}, ny={ny}, nz={nz}", flush=True)
    print(f"Total interior points: {nx*ny*nz}", flush=True)
    print("\nIndex mapping (i,j,k) -> flat_index:", flush=True)
    for i in range(min(2, nx)):
        for j in range(min(2, ny)):
            for k in range(min(2, nz)):
                idx = i + j*nx + k*nx*ny
                print(f"  ({i},{j},{k}) -> {idx}", flush=True)

    print("\nPyTorch reshape ordering test:", flush=True)
    test_tensor = torch.arange(nx*ny*nz).reshape(nx, ny, nz)
    print(f"  Tensor shape: {test_tensor.shape}", flush=True)
    print(f"  test_tensor[0,0,0] = {test_tensor[0,0,0]}", flush=True)
    print(f"  test_tensor[1,0,0] = {test_tensor[1,0,0] if nx > 1 else 'N/A'}", flush=True)
    print(f"  test_tensor[0,1,0] = {test_tensor[0,1,0] if ny > 1 else 'N/A'}", flush=True)
    print(f"  test_tensor[0,0,1] = {test_tensor[0,0,1] if nz > 1 else 'N/A'}", flush=True)
    print(f"  Flattened (first 8): {test_tensor.reshape(-1)[:8]}", flush=True)
    
def save_flow_slices(u, v, w, z_c, Lx, Ly, results_folder, step_name):
    """
    Save xy, xz, and yz slices of the velocity field with physical coordinates.
    Silent - no print output.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    nx, ny, nz = u.shape[0]-2, u.shape[1]-2, w.shape[2]
    
    # Mid-plane indices
    ix = nx // 2
    iy = ny // 2
    iz = nz // 2
    
    # XY slice at z_mid
    u_xy = u[0:nx+1, 1:ny+1, iz+1].cpu().numpy().T
    v_xy = v[1:nx+1, 1:ny+1, iz+1].cpu().numpy().T
    w_xy = w[1:nx+1, 1:ny+1, iz].cpu().numpy().T
    z_mid = z_c[iz+1].item()
    
    # XZ slice at y_mid
    u_xz = u[0:nx+1, iy+1, 1:nz+1].cpu().numpy().T
    v_xz = v[1:nx+1, iy, 1:nz+1].cpu().numpy().T
    w_xz = w[1:nx+1, iy+1, 1:nz].cpu().numpy().T
    y_mid = iy * Ly / ny
    
    # YZ slice at x_mid  
    u_yz = u[ix+1, 1:ny+1, 1:nz+1].cpu().numpy().T
    v_yz = v[ix+1, 1:ny+1, 1:nz+1].cpu().numpy().T
    w_yz = w[ix+1, 1:ny+1, 1:nz].cpu().numpy().T
    x_mid = ix * Lx / nx
    
    # Get z coordinates for plotting
    z_plot_u = z_c[1:nz+1].cpu().numpy()  # For u,v (nz points)
    z_plot_w = z_c[1:nz].cpu().numpy()     # For w (nz-1 points)
    
    # Create x and y coordinates
    x_plot_u = torch.linspace(0, Lx, nx+1).cpu().numpy()
    x_plot_v = torch.linspace(0, Lx, nx).cpu().numpy()
    y_plot = torch.linspace(0, Ly, ny).cpu().numpy()
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # XY slices (row 0)
    cf = axes[0, 0].contourf(x_plot_u, y_plot, u_xy, levels=20, cmap='RdBu_r')
    axes[0, 0].set_title(f'u (XY, z={z_mid:.3f})')
    plt.colorbar(cf, ax=axes[0, 0])
    
    cf = axes[0, 1].contourf(x_plot_v, y_plot, v_xy, levels=20, cmap='RdBu_r')
    axes[0, 1].set_title(f'v (XY, z={z_mid:.3f})')
    plt.colorbar(cf, ax=axes[0, 1])
    
    cf = axes[0, 2].contourf(x_plot_v, y_plot, w_xy, levels=20, cmap='RdBu_r')
    axes[0, 2].set_title(f'w (XY, z={z_mid:.3f})')
    plt.colorbar(cf, ax=axes[0, 2])
    
    # XZ slices (row 1)
    cf = axes[1, 0].contourf(x_plot_u, z_plot_u, u_xz, levels=20, cmap='RdBu_r')
    axes[1, 0].set_title(f'u (XZ, y={y_mid:.3f})')
    plt.colorbar(cf, ax=axes[1, 0])
    
    cf = axes[1, 1].contourf(x_plot_v, z_plot_u, v_xz, levels=20, cmap='RdBu_r')
    axes[1, 1].set_title(f'v (XZ, y={y_mid:.3f})')
    plt.colorbar(cf, ax=axes[1, 1])
    
    cf = axes[1, 2].contourf(x_plot_v, z_plot_w, w_xz, levels=20, cmap='RdBu_r')
    axes[1, 2].set_title(f'w (XZ, y={y_mid:.3f})')
    plt.colorbar(cf, ax=axes[1, 2])
    
    # YZ slices (row 2)
    cf = axes[2, 0].contourf(y_plot, z_plot_u, u_yz, levels=20, cmap='RdBu_r')
    axes[2, 0].set_title(f'u (YZ, x={x_mid:.3f})')
    plt.colorbar(cf, ax=axes[2, 0])
    
    cf = axes[2, 1].contourf(y_plot, z_plot_u, v_yz, levels=20, cmap='RdBu_r')
    axes[2, 1].set_title(f'v (YZ, x={x_mid:.3f})')
    plt.colorbar(cf, ax=axes[2, 1])
    
    cf = axes[2, 2].contourf(y_plot, z_plot_w, w_yz, levels=20, cmap='RdBu_r')
    axes[2, 2].set_title(f'w (YZ, x={x_mid:.3f})')
    plt.colorbar(cf, ax=axes[2, 2])
    
    plt.tight_layout()
    filepath = os.path.join(results_folder, f'flow_slices_{step_name}.png')
    plt.savefig(filepath, dpi=100)
    plt.close()

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
