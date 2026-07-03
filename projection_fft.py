import torch

def initialize_fft_solver(nx, ny, nz, dx, dy, dz_c, dz_f, top_wall_bc_type='dirichlet'):
    """
    Precompute wavenumbers and tridiagonal matrices for FFT-based Poisson solver.
    Returns a dictionary with precomputed data.

    Fully vectorized implementation - no Python loops.

    Args:
        nx, ny, nz: Grid dimensions
        dx, dy: Grid spacings in x, y
        dz_c, dz_f: Grid spacings in z (center and face)
        top_wall_bc_type: Boundary condition type at top wall
            - 'dirichlet': no-slip velocity BC (u=v=w=0) → Neumann pressure BC (∂p/∂z=0)
            - 'neumann': free-slip velocity BC (∂u/∂z=∂v/∂z=0, w=0) → Dirichlet pressure BC (p=0)
    
    Note on velocity-pressure BC coupling:
        The pressure BC is determined by the incompressibility constraint:
        - No-slip (u=v=w=0): Taking ∂/∂z of ∇·u=0 at wall gives ∂²w/∂z²=0, and since w=∂w/∂z=0,
          we get ∂p/∂z=0 (Neumann pressure BC)
        - Free-slip (∂u/∂z=∂v/∂z=0, w=0): Momentum equation at wall gives p=0 (Dirichlet pressure BC)
    """
    device = dz_c.device

    kx = torch.fft.fftfreq(nx, d=dx/(2*torch.pi), device=device)
    ky = torch.fft.rfftfreq(ny, d=dy/(2*torch.pi), device=device)

    # Use MODIFIED wavenumbers to match central finite difference scheme
    kx_mod = (2.0 / dx) * torch.sin(kx * dx / 2.0)
    ky_mod = (2.0 / dy) * torch.sin(ky * dy / 2.0)

    nkx = len(kx)
    nky = len(ky)  # This is ny//2 + 1

    # Vectorized: Create meshgrid of wavenumbers
    # Shape: (nkx, nky)
    kx_grid, ky_grid = torch.meshgrid(kx_mod, ky_mod, indexing='ij')
    k_horiz_sq = kx_grid**2 + ky_grid**2  # Shape: (nkx, nky)

    # Vectorized tridiagonal matrix construction
    # Build coefficients for all z-levels at once
    # Shape: (nz,)
    dz_left = dz_c[:-1]   # dz_c[k] for k=0..nz-1
    dz_right = dz_c[1:]   # dz_c[k+1] for k=0..nz-1
    dz_avg = dz_f         # dz_f[k] for k=0..nz-1

    coeff_left = 1.0 / (dz_left * dz_avg)   # Shape: (nz,)
    coeff_right = 1.0 / (dz_right * dz_avg)  # Shape: (nz,)

    # Expand k_horiz_sq to include z dimension: (nkx, nky, nz)
    k_horiz_sq_3d = k_horiz_sq.unsqueeze(-1).expand(nkx, nky, nz)

    # Build tridiagonal diagonals
    # Lower diagonal: a[k] = coeff_left[k] for k > 0, else 0
    tri_a = torch.zeros(nkx, nky, nz, device=device)
    tri_a[:, :, 1:] = coeff_left[1:].view(1, 1, -1)

    # Upper diagonal: c[k] = coeff_right[k] for k < nz-1, else 0
    tri_c = torch.zeros(nkx, nky, nz, device=device)
    tri_c[:, :, :-1] = coeff_right[:-1].view(1, 1, -1)

    # Main diagonal: b[k] = -(coeff_left[k] + coeff_right[k] + k_horiz_sq)
    coeff_sum = coeff_left + coeff_right  # Shape: (nz,)
    tri_b = -(coeff_sum.view(1, 1, -1) + k_horiz_sq_3d)  # Shape: (nkx, nky, nz)

    # Apply boundary conditions at walls
    # Bottom wall: always Neumann pressure BC (no-slip velocity → ∂p/∂z = 0)
    # Costa's approach: modify stencil for ghost cell BC
    tri_b[:, :, 0] += coeff_left[0]

    # Top wall:
    # Always use Neumann pressure BC (∂p/∂z = 0) for rigid walls (w=0).
    # This applies to both 'dirichlet' (no-slip) and 'neumann' (free-slip) velocity BCs.
    #
    # Physical reasoning:
    # For a rigid wall (w=0), the wall-normal momentum equation reduces to ∂p/∂z ≈ 0 
    # (assuming viscous terms are small/zero at the wall).
    # The previous assumption of Dirichlet pressure (p=0) for free-slip is typically for 
    # free surfaces, not rigid lids. Using p=0 with w=0 leads to uncorrectable divergence 
    # at the boundary cell because the boundary velocity is fixed.
    tri_b[:, :, -1] += coeff_right[-1]

    # Pre-allocate workspace for pressure field (GPU optimization)
    # Reusing this workspace avoids repeated allocations every timestep
    workspace_p = torch.zeros(nx+2, ny+2, nz+2, device=device)

    return {
        'kx': kx,
        'ky': ky,
        'tri_a': tri_a,
        'tri_b': tri_b,
        'tri_c': tri_c,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'workspace_p': workspace_p,
        'top_wall_bc_type': top_wall_bc_type,
    }

def solve_poisson_fft(div, fft_data):
    """
    Solve Poisson equation using precomputed FFT data.
    Periodic BC in x,y. Pressure BC in z depends on velocity BC type.

    Returns p with ghost cells.
    """
    nx = fft_data['nx']
    ny = fft_data['ny']
    nz = fft_data['nz']
    tri_a = fft_data['tri_a']
    tri_b = fft_data['tri_b']
    tri_c = fft_data['tri_c']
    top_wall_bc_type = fft_data['top_wall_bc_type']

    # FFT in x and y directions
    # rfft2 on dim=(0,1) means real input, complex output.
    div_hat = torch.fft.rfft2(div, dim=(0, 1))

    # div_hat shape is (nx, ny//2+1, nz)
    nkx, nky = div_hat.shape[0], div_hat.shape[1]

    # Flatten batch dimensions (nx * nky) to vectorize the tridiagonal solve
    # New shape: (nx * nky, nz)
    div_hat_flat = div_hat.reshape(-1, nz)
    
    tri_a_flat = tri_a.reshape(-1, nz)
    tri_b_flat = tri_b.reshape(-1, nz)
    tri_c_flat = tri_c.reshape(-1, nz)

    # Solve all tridiagonal systems in parallel
    p_hat_flat = solve_tridiagonal(tri_a_flat, tri_b_flat, tri_c_flat, div_hat_flat)

    # Reshape back to (nx, nky, nz)
    p_hat = p_hat_flat.reshape(nkx, nky, nz)

    # Inverse FFT to get pressure in physical space
    # PyTorch FFTs are normalized by default (irfft2(rfft2(x)) == x)
    p_interior = torch.fft.irfft2(p_hat, s=(nx, ny), dim=(0, 1))

    # Use preallocated workspace (GPU optimization - avoids allocation every timestep)
    p = fft_data['workspace_p']
    p.zero_()  # CRITICAL: Clear workspace before reuse to avoid contamination

    # Fill interior
    p[1:nx+1, 1:ny+1, 1:nz+1] = p_interior

    # Periodic BC in x (plain slices, not list-fancy-indexing: the latter builds
    # host index tensors, which break CUDA-graph capture of this solve).
    p[0] = p[nx]
    p[nx+1] = p[1]

    # Periodic BC in y
    p[:, 0] = p[:, ny]
    p[:, ny+1] = p[:, 1]

    # Pressure BC in z-direction: depends on velocity BC type
    # Bottom wall: always Neumann (∂p/∂z = 0)
    p[:, :, 0] = p[:, :, 1]
    
    # Top wall ghost cell
    # Always Neumann pressure BC (∂p/∂z = 0) → p[nz+1] = p[nz]
    p[:, :, nz+1] = p[:, :, nz]

    return p

@torch.jit.script
def solve_tridiagonal(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                     d: torch.Tensor) -> torch.Tensor:
    """
    Solve batched tridiagonal systems using Thomas algorithm (vectorized).
    JIT-compiled for GPU performance.

    Args:
        a: Lower diagonal (batch, n)
        b: Main diagonal (batch, n)
        c: Upper diagonal (batch, n)
        d: RHS (batch, n)

    Returns:
        x: Solution (batch, n)
    """
    # Dimensions
    batch_size, n = d.shape

    # Get device/dtype
    device = d.device
    dtype = d.dtype

    # Allocate temporaries
    c_prime = torch.zeros((batch_size, n), dtype=dtype, device=device)
    d_prime = torch.zeros((batch_size, n), dtype=dtype, device=device)
    x = torch.zeros((batch_size, n), dtype=dtype, device=device)

    # Forward sweep
    # i = 0
    c_prime[:, 0] = c[:, 0] / b[:, 0]
    d_prime[:, 0] = d[:, 0] / b[:, 0]

    for i in range(1, n):
        denom = b[:, i] - a[:, i] * c_prime[:, i-1]
        # Avoid division by zero if any denom is 0 (unlikely for Poisson)
        # For now assume well-posed

        if i < n - 1:
            c_prime[:, i] = c[:, i] / denom

        d_prime[:, i] = (d[:, i] - a[:, i] * d_prime[:, i-1]) / denom

    # Back substitution
    x[:, n-1] = d_prime[:, n-1]
    for i in range(n-2, -1, -1):
        x[:, i] = d_prime[:, i] - c_prime[:, i] * x[:, i+1]

    return x
