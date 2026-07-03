import torch

def build_poisson_matrix(nx, ny, nz, dx, dy, dz_c, dz_f, pin_pressure=True, top_wall_bc_type='dirichlet'):
    """
    Build Laplacian matrix for Poisson equation: ∇²p = div/dt

    Grid: nx×ny×nz interior cells
    BCs: Periodic in x,y; Variable in z
    - Bottom wall: always Neumann (dp/dz=0) for pressure
    - Top wall: Neumann (dp/dz=0) or Dirichlet (p=0) based on top_wall_bc_type

    Args:
        top_wall_bc_type: 'dirichlet' (no-slip velocity) or 'neumann' (free-slip velocity)

    Indexing: idx = (i-1) + (j-1)*nx + (k-1)*nx*ny
    where i,j,k ∈ [1, nx]×[1, ny]×[1, nz]
    """
    N = nx * ny * nz
    A = torch.zeros(N, N)
    
    for k in range(1, nz+1):
        for j in range(1, ny+1):
            for i in range(1, nx+1):
                # Current point index
                idx = (i-1) + (j-1)*nx + (k-1)*nx*ny
                diag = 0.0
                
                # ===== X-direction (periodic) =====
                # Left neighbor
                i_left = nx if i == 1 else i-1
                idx_left = (i_left-1) + (j-1)*nx + (k-1)*nx*ny
                A[idx, idx_left] = 1.0 / dx**2
                diag -= 1.0 / dx**2
                
                # Right neighbor
                i_right = 1 if i == nx else i+1
                idx_right = (i_right-1) + (j-1)*nx + (k-1)*nx*ny
                A[idx, idx_right] = 1.0 / dx**2
                diag -= 1.0 / dx**2
                
                # ===== Y-direction (periodic) =====
                # Bottom neighbor
                j_bottom = ny if j == 1 else j-1
                idx_bottom = (i-1) + (j_bottom-1)*nx + (k-1)*nx*ny
                A[idx, idx_bottom] = 1.0 / dy**2
                diag -= 1.0 / dy**2
                
                # Top neighbor
                j_top = 1 if j == ny else j+1
                idx_top = (i-1) + (j_top-1)*nx + (k-1)*nx*ny
                A[idx, idx_top] = 1.0 / dy**2
                diag -= 1.0 / dy**2
                
                # ===== Z-direction (stretched, Neumann BC) =====
                # Consistent finite difference matching compute_divergence + project_velocity:
                # div(grad(p))_k = 1/dz_f[k-1] * [ (p[k+1]-p[k])/dz_c[k] - (p[k]-p[k-1])/dz_c[k-1] ]
                #
                # Coefficients:
                # p[k+1]: 1 / (dz_f[k-1] * dz_c[k])
                # p[k-1]: 1 / (dz_f[k-1] * dz_c[k-1])
                
                # Coefficient for p[k-1]
                if k > 1:
                    idx_down = (i-1) + (j-1)*nx + (k-2)*nx*ny
                    coeff = 1.0 / (dz_f[k-1] * dz_c[k-1])
                    A[idx, idx_down] = coeff
                    diag -= coeff
                
                # Coefficient for p[k+1]
                if k < nz:
                    idx_up = (i-1) + (j-1)*nx + (k)*nx*ny
                    coeff = 1.0 / (dz_f[k-1] * dz_c[k])
                    A[idx, idx_up] = coeff
                    diag -= coeff
                
                # Set diagonal
                A[idx, idx] = diag
    
    # Fix pressure at one point to resolve singularity (Neumann/Periodic BCs)
    if pin_pressure:
        A[0, :] = 0.0
        A[0, 0] = 1.0
    
    return A


def solve_poisson(A, div, nx, ny, nz, top_wall_bc_type='dirichlet'):
    """
    Solve Ap = b for pressure.
    Returns p with ghost cells.

    Args:
        top_wall_bc_type: 'dirichlet' (no-slip velocity) or 'neumann' (free-slip velocity)

    Note: Matrix A uses index ordering where i varies fastest: idx = (i-1) + (j-1)*nx + (k-1)*nx*ny
    PyTorch reshape uses C-order where k varies fastest.
    Must permute to match: (nx,ny,nz) -> (nz,ny,nx) before flatten, then reverse after.
    """
    # Permute to match matrix indexing: i varies fastest
    div_permuted = div.permute(2, 1, 0)  # (nx,ny,nz) -> (nz,ny,nx)
    b = div_permuted.reshape(-1)
    
    # Handle pinned pressure node (idx=0)
    # We modified A such that A[0,:]=[1,0,...]. So we must set b[0]=0 to get p[0]=0.
    # Note: b is a view or copy? reshape returns a view usually.
    # But we want to modify it.
    # b = b.clone() # Make sure we don't modify the original div if it's used elsewhere
    # b[0] = 0.0

    # Use least squares to solve singular system Ap = b
    # This finds solution with minimum norm (effectively setting mean p close to 0)
    # It avoids the need for manual pinning which can cause local conservation errors
    solution = torch.linalg.lstsq(A, b)
    p_flat = solution.solution

    # Reshape and permute back
    p_permuted = p_flat.reshape(nz, ny, nx)
    p_interior = p_permuted.permute(2, 1, 0)  # (nz,ny,nx) -> (nx,ny,nz)

    # Get device from input
    device = div.device
    p = torch.zeros(nx+2, ny+2, nz+2, device=device)
    p[1:nx+1, 1:ny+1, 1:nz+1] = p_interior

    # Periodic BC in x,y
    p[0, :, :] = p[nx, :, :]
    p[nx+1, :, :] = p[1, :, :]
    p[:, 0, :] = p[:, ny, :]
    p[:, ny+1, :] = p[:, 1, :]

    # BC in z: bottom wall always Neumann, top wall depends on BC type
    p[:, :, 0] = p[:, :, 1]  # Bottom: Neumann (dp/dz = 0)
    if top_wall_bc_type == 'neumann':
        # Free-slip velocity BC → Dirichlet pressure BC (p = 0)
        p[:, :, nz+1] = 0.0
    else:  # 'dirichlet'
        # No-slip velocity BC → Neumann pressure BC (dp/dz = 0)
        p[:, :, nz+1] = p[:, :, nz]

    return p


@torch.jit.script
def project_velocity(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                     p: torch.Tensor, nx: int, ny: int, nz: int,
                     dx: float, dy: float, dz_c: torch.Tensor, dz_f: torch.Tensor,
                     dt: float):
    """
    Velocity correction: u = u* - dt*grad(p)
    JIT-compiled for GPU performance
    NOTE: Must use same grid spacing as compute_divergence for consistency
    """
    # Vectorized correction for u (x-faces) - GPU compatible
    dp_dx = (p[2:nx+2, 1:ny+1, 1:nz+1] - p[1:nx+1, 1:ny+1, 1:nz+1]) / dx
    u[1:nx+1, 1:ny+1, 1:nz+1] -= dt * dp_dx

    # Vectorized correction for v (y-faces)
    dp_dy = (p[1:nx+1, 2:ny+2, 1:nz+1] - p[1:nx+1, 1:ny+1, 1:nz+1]) / dy
    v[1:nx+1, 1:ny+1, 1:nz+1] -= dt * dp_dy

    # Vectorized correction for w (z-faces)
    dp_dz = (p[1:nx+1, 1:ny+1, 2:nz+1] - p[1:nx+1, 1:ny+1, 1:nz]) / dz_c[1:nz].view(1, 1, -1)
    w[1:nx+1, 1:ny+1, 1:nz] -= dt * dp_dz

    return u, v, w
