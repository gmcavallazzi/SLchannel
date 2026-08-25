"""operators.py — Staggered-MAC finite-difference operators.

Advection and diffusion stencils, the fused IMEX momentum RHS, the CFL
reduction, and the Crank-Nicolson implicit z-diffusion solves. Used by the
Eulerian reference scheme and, for the diffusion half, by the SL scheme too.

Provenance
----------
Inherited from torChannel, the Eulerian parent solver by the same author
(MIT), imported verbatim in slChannel commit 05a1b30 and kept conceptually
in sync since. Changes here are limited to package-relative imports, the
removal of code paths slChannel does not use, and the local divergences
noted above. Deliberately NOT reformatted, so it stays diffable against
upstream. See docs/PROVENANCE.md.
"""

import torch
from .tridiag import pcr_solve
from . import env

"""
Staggered grid operators for incompressible Navier-Stokes equations.

Grid stencil example for adv_u[1,1,1]:
    u-points needed: [0,1,1], [1,1,1], [2,1,1], [1,0,1], [1,2,1], [1,1,0], [1,1,2]
    v-points needed: [1,1,1], [2,1,1], [1,0,1], [2,0,1]
    w-points needed: [1,1,1], [2,1,1], [1,1,0], [2,1,0]

Velocities are staggered:
    - u lives at x-faces, interpolated in y and z when needed
    - v lives at y-faces, interpolated in x and z when needed
    - w lives at z-faces, interpolated in x and y when needed
"""

@torch.jit.script
def diffusion_u(u: torch.Tensor, nx: int, ny: int, nz: int,
                dx: float, dy: float, dz_c: torch.Tensor, dz_f: torch.Tensor,
                nu: float) -> torch.Tensor:
    """Compute diffusion term for u-component: nu * laplacian(u). JIT-compiled."""
    diff_u = torch.zeros_like(u)

    # Second derivative in x (uniform spacing, periodic)
    # u is staggered in x, indices 0..nx. u[0]=u[nx].
    # Construct extended u in x: [u[0]...u[nx], u[1]]
    u_ext_x = torch.cat([u, u[1:2, :, :]], dim=0)
    
    d2u_dx2 = (u_ext_x[2:nx+2, 1:ny+1, 1:nz+1] -
               2*u_ext_x[1:nx+1, 1:ny+1, 1:nz+1] +
               u_ext_x[0:nx, 1:ny+1, 1:nz+1]) / dx**2

    # Second derivative in y (uniform spacing)
    # u is NOT staggered in y. Indices 0..ny+1. Ghosts at 0, ny+1.
    # Update 1..ny.
    d2u_dy2 = (u[1:nx+1, 2:ny+2, 1:nz+1] -
               2*u[1:nx+1, 1:ny+1, 1:nz+1] +
               u[1:nx+1, 0:ny, 1:nz+1]) / dy**2

    # Second derivative in z (stretched grid, non-uniform spacing)
    # u is at cell centers. Inner diffs are across faces (dz_c). Outer diff is across cell (dz_f).
    dz_left = dz_c[0:nz].view(1, 1, -1)
    dz_right = dz_c[1:nz+1].view(1, 1, -1)
    # dz_avg = 0.5 * (dz_left + dz_right) # Incorrect
    dz_cell = dz_f.view(1, 1, -1) # Correct: cell height
    
    d2u_dz2 = ((u[1:nx+1, 1:ny+1, 2:nz+2] -
                u[1:nx+1, 1:ny+1, 1:nz+1])/dz_right -
               (u[1:nx+1, 1:ny+1, 1:nz+1] -
                u[1:nx+1, 1:ny+1, 0:nz])/dz_left) / dz_cell

    diff_u[1:nx+1, 1:ny+1, 1:nz+1] = nu * (d2u_dx2 + d2u_dy2 + d2u_dz2)

    return diff_u

@torch.jit.script
def diffusion_v(v: torch.Tensor, nx: int, ny: int, nz: int,
                dx: float, dy: float, dz_c: torch.Tensor, dz_f: torch.Tensor,
                nu: float) -> torch.Tensor:
    """Compute diffusion term for v-component: nu * laplacian(v). JIT-compiled."""
    diff_v = torch.zeros_like(v)

    # Second derivative in x (uniform spacing)
    # v is NOT staggered in x. Indices 0..nx+1. Ghosts at 0, nx+1.
    # Update 1..nx.
    d2v_dx2 = (v[2:nx+2, 1:ny+1, 1:nz+1] -
               2*v[1:nx+1, 1:ny+1, 1:nz+1] +
               v[0:nx, 1:ny+1, 1:nz+1]) / dx**2

    # Second derivative in y (uniform spacing, periodic)
    # v is staggered in y, indices 0..ny. v[0]=v[ny].
    # Construct extended v in y: [v[0]...v[ny], v[1]]
    v_ext_y = torch.cat([v, v[:, 1:2, :]], dim=1)

    d2v_dy2 = (v_ext_y[1:nx+1, 2:ny+2, 1:nz+1] -
               2*v_ext_y[1:nx+1, 1:ny+1, 1:nz+1] +
               v_ext_y[1:nx+1, 0:ny, 1:nz+1]) / dy**2

    # Second derivative in z (stretched grid, non-uniform spacing)
    # v is at cell centers. Same as u.
    dz_left = dz_c[0:nz].view(1, 1, -1)
    dz_right = dz_c[1:nz+1].view(1, 1, -1)
    # dz_avg = 0.5 * (dz_left + dz_right)
    dz_cell = dz_f.view(1, 1, -1) # Correct: cell height (dz_f is length nz)
    
    d2v_dz2 = ((v[1:nx+1, 1:ny+1, 2:nz+2] -
                v[1:nx+1, 1:ny+1, 1:nz+1])/dz_right -
               (v[1:nx+1, 1:ny+1, 1:nz+1] -
                v[1:nx+1, 1:ny+1, 0:nz])/dz_left) / dz_cell

    diff_v[1:nx+1, 1:ny+1, 1:nz+1] = nu * (d2v_dx2 + d2v_dy2 + d2v_dz2)

    return diff_v

@torch.jit.script
def diffusion_w(w: torch.Tensor, nx: int, ny: int, nz: int, 
                dx: float, dy: float, dz_c: torch.Tensor, dz_f: torch.Tensor, 
                nu: float) -> torch.Tensor:
    """
    Compute diffusion term for w-component: nu * laplacian(w)
    JIT-compiled for GPU performance
    """
    diff_w = torch.zeros_like(w)
    
    # d2w/dx2
    d2w_dx2 = (w[2:nx+2, 1:ny+1, 1:nz] - 
               2*w[1:nx+1, 1:ny+1, 1:nz] + 
               w[0:nx, 1:ny+1, 1:nz]) / (dx**2)
    
    # d2w/dy2
    d2w_dy2 = (w[1:nx+1, 2:ny+2, 1:nz] - 
               2*w[1:nx+1, 1:ny+1, 1:nz] + 
               w[1:nx+1, 0:ny, 1:nz]) / (dy**2)
    
    # d2w/dz2 (non-uniform grid)
    d2w_dz2 = ((w[1:nx+1, 1:ny+1, 2:nz+1] - 
                w[1:nx+1, 1:ny+1, 1:nz]) / dz_f[1:nz].view(1, 1, -1) -
               (w[1:nx+1, 1:ny+1, 1:nz] - 
                w[1:nx+1, 1:ny+1, 0:nz-1]) / dz_f[0:nz-1].view(1, 1, -1)) / dz_c[0:nz-1].view(1, 1, -1)
    
    diff_w[1:nx+1, 1:ny+1, 1:nz] = nu * (d2w_dx2 + d2w_dy2 + d2w_dz2)

    return diff_w

@torch.jit.script
def advection_u(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                nx: int, ny: int, nz: int,
                dx: float, dy: float, dz_f: torch.Tensor) -> torch.Tensor:
    """
    Compute advection term for u-component using conservative flux form
    JIT-compiled for GPU performance

    Evaluated on x-faces i = 1..nx, which is EVERY distinct face. u is stored
    with shape nx+1 and u[0] is the periodic ghost of u[nx] (apply_bc_all does
    ``u[0] = u[-1]``), so i = nx is a physical face, not a halo. Computing only
    1..nx-1 leaves that plane without advection: the operator then fails to
    telescope and injects a spurious 1/nx of the streamwise momentum budget.
    Face nx needs a right neighbour, hence the one-column periodic extension
    ``ue[nx+1] == u[1]``; v and w already carry that ghost at index nx+1.
    """
    adv_u = torch.zeros_like(u)
    ue = torch.cat([u, u[1:2, :, :]], dim=0)

    # d(uu)/dx: u already at x-faces, interpolate to get u at face centers
    u_interp = 0.5 * (ue[1:nx+1, 1:ny+1, 1:nz+1] + ue[2:nx+2, 1:ny+1, 1:nz+1])
    uu_right = u_interp * u_interp # Fixed: symmetric flux
    u_interp = 0.5 * (ue[0:nx, 1:ny+1, 1:nz+1] + ue[1:nx+1, 1:ny+1, 1:nz+1])
    uu_left = u_interp * u_interp # Fixed: symmetric flux
    duudx = (uu_right - uu_left) / dx

    # d(vu)/dy: v at y-faces, interpolate in x only to get v at u-location
    v_interp = 0.5 * (v[1:nx+1, 1:ny+1, 1:nz+1] + v[2:nx+2, 1:ny+1, 1:nz+1])
    u_interp_y = 0.5 * (ue[1:nx+1, 1:ny+1, 1:nz+1] + ue[1:nx+1, 2:ny+2, 1:nz+1])
    vu_top = v_interp * u_interp_y
    v_interp = 0.5 * (v[1:nx+1, 0:ny, 1:nz+1] + v[2:nx+2, 0:ny, 1:nz+1])
    u_interp_y = 0.5 * (ue[1:nx+1, 0:ny, 1:nz+1] + ue[1:nx+1, 1:ny+1, 1:nz+1])
    vu_bottom = v_interp * u_interp_y
    dvudy = (vu_top - vu_bottom) / dy

    # d(wu)/dz: w at z-faces, interpolate in x only to get w at u-location
    w_interp = 0.5 * (w[1:nx+1, 1:ny+1, 1:nz+1] + w[2:nx+2, 1:ny+1, 1:nz+1])
    u_interp_z = 0.5 * (ue[1:nx+1, 1:ny+1, 1:nz+1] + ue[1:nx+1, 1:ny+1, 2:nz+2])
    wu_top = w_interp * u_interp_z
    w_interp = 0.5 * (w[1:nx+1, 1:ny+1, 0:nz] + w[2:nx+2, 1:ny+1, 0:nz])
    u_interp_z = 0.5 * (ue[1:nx+1, 1:ny+1, 0:nz] + ue[1:nx+1, 1:ny+1, 1:nz+1])
    wu_bottom = w_interp * u_interp_z

    dz_avg = dz_f[0:nz].view(1, 1, -1)
    dwudz = (wu_top - wu_bottom) / dz_avg

    adv_u[1:nx+1, 1:ny+1, 1:nz+1] = duudx + dvudy + dwudz

    return adv_u

@torch.jit.script
def advection_v(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, 
                nx: int, ny: int, nz: int, 
                dx: float, dy: float, dz_f: torch.Tensor) -> torch.Tensor:
    """
    Compute advection term for v-component using conservative flux form
    JIT-compiled for GPU performance

    Evaluated on y-faces j = 1..ny -- the same argument as advection_u, one axis
    over. v has shape ny+1 in y and v[:, 0] is the periodic ghost of v[:, ny],
    so j = ny is a physical face and must be advected. ``ve[:, ny+1] == v[:, 1]``
    supplies its upper neighbour.
    """
    adv_v = torch.zeros_like(v)
    ve = torch.cat([v, v[:, 1:2, :]], dim=1)

    # d(uv)/dx: u at x-faces, interpolate in y only to get u at v-location
    u_interp = 0.5 * (u[1:nx+1, 1:ny+1, 1:nz+1] + u[1:nx+1, 2:ny+2, 1:nz+1])
    v_interp_x = 0.5 * (ve[1:nx+1, 1:ny+1, 1:nz+1] + ve[2:nx+2, 1:ny+1, 1:nz+1])
    uv_right = u_interp * v_interp_x
    u_interp = 0.5 * (u[0:nx, 1:ny+1, 1:nz+1] + u[0:nx, 2:ny+2, 1:nz+1])
    v_interp_x = 0.5 * (ve[0:nx, 1:ny+1, 1:nz+1] + ve[1:nx+1, 1:ny+1, 1:nz+1])
    uv_left = u_interp * v_interp_x
    duvdx = (uv_right - uv_left) / dx

    # d(vv)/dy: v already at y-faces, interpolate to get v at face centers
    v_interp = 0.5 * (ve[1:nx+1, 1:ny+1, 1:nz+1] + ve[1:nx+1, 2:ny+2, 1:nz+1])
    vv_top = v_interp * v_interp # Fixed: symmetric flux
    v_interp = 0.5 * (ve[1:nx+1, 0:ny, 1:nz+1] + ve[1:nx+1, 1:ny+1, 1:nz+1])
    vv_bottom = v_interp * v_interp # Fixed: symmetric flux
    dvvdy = (vv_top - vv_bottom) / dy

    # d(wv)/dz: w at z-faces, interpolate in y only to get w at v-location
    w_interp = 0.5 * (w[1:nx+1, 1:ny+1, 1:nz+1] + w[1:nx+1, 2:ny+2, 1:nz+1])
    v_interp_z = 0.5 * (ve[1:nx+1, 1:ny+1, 1:nz+1] + ve[1:nx+1, 1:ny+1, 2:nz+2])
    wv_top = w_interp * v_interp_z
    w_interp = 0.5 * (w[1:nx+1, 1:ny+1, 0:nz] + w[1:nx+1, 2:ny+2, 0:nz])
    v_interp_z = 0.5 * (ve[1:nx+1, 1:ny+1, 0:nz] + ve[1:nx+1, 1:ny+1, 1:nz+1])
    wv_bottom = w_interp * v_interp_z

    dz_avg = dz_f[0:nz].view(1, 1, -1)
    dwvdz = (wv_top - wv_bottom) / dz_avg

    adv_v[1:nx+1, 1:ny+1, 1:nz+1] = duvdx + dvvdy + dwvdz

    return adv_v

@torch.jit.script
def advection_w(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, 
                nx: int, ny: int, nz: int, 
                dx: float, dy: float, dz_c: torch.Tensor) -> torch.Tensor:
    """
    Compute advection term for w-component using conservative flux form
    JIT-compiled for GPU performance
    """
    adv_w = torch.zeros_like(w)

    # d(uw)/dx: u at x-faces, interpolate in z only to get u at w-location
    u_interp = 0.5 * (u[1:nx+1, 1:ny+1, 1:nz] + u[1:nx+1, 1:ny+1, 2:nz+1])
    w_interp_x = 0.5 * (w[1:nx+1, 1:ny+1, 1:nz] + w[2:nx+2, 1:ny+1, 1:nz])
    uw_right = u_interp * w_interp_x
    u_interp = 0.5 * (u[0:nx, 1:ny+1, 1:nz] + u[0:nx, 1:ny+1, 2:nz+1])
    w_interp_x = 0.5 * (w[0:nx, 1:ny+1, 1:nz] + w[1:nx+1, 1:ny+1, 1:nz])
    uw_left = u_interp * w_interp_x
    duwdx = (uw_right - uw_left) / dx

    # d(vw)/dy: v at y-faces, interpolate in z only to get v at w-location
    v_interp = 0.5 * (v[1:nx+1, 1:ny+1, 1:nz] + v[1:nx+1, 1:ny+1, 2:nz+1])
    w_interp_y = 0.5 * (w[1:nx+1, 1:ny+1, 1:nz] + w[1:nx+1, 2:ny+2, 1:nz])
    vw_top = v_interp * w_interp_y
    v_interp = 0.5 * (v[1:nx+1, 0:ny, 1:nz] + v[1:nx+1, 0:ny, 2:nz+1])
    w_interp_y = 0.5 * (w[1:nx+1, 0:ny, 1:nz] + w[1:nx+1, 1:ny+1, 1:nz])
    vw_bottom = v_interp * w_interp_y
    dvwdy = (vw_top - vw_bottom) / dy

    # d(ww)/dz: w already at z-faces, interpolate to get w at face centers
    w_interp = 0.5 * (w[1:nx+1, 1:ny+1, 1:nz] + w[1:nx+1, 1:ny+1, 2:nz+1])
    ww_top = w_interp * w_interp # Fixed: symmetric flux
    w_interp = 0.5 * (w[1:nx+1, 1:ny+1, 0:nz-1] + w[1:nx+1, 1:ny+1, 1:nz])
    ww_bottom = w_interp * w_interp # Fixed: symmetric flux

    dz_avg = dz_c[1:nz].view(1, 1, -1)
    dwwdz = (ww_top - ww_bottom) / dz_avg

    adv_w[1:nx+1, 1:ny+1, 1:nz] = duwdx + dvwdy + dwwdz

    return adv_w


import torch

# ==============================================================================
# FUSED KERNEL (GPU Optimization - Phase 3)
# ==============================================================================


# ==============================================================================
# IMEX SCHEME OPERATORS
# ==============================================================================

@torch.jit.script
def diffusion_xy_u(u: torch.Tensor, nx: int, ny: int, nz: int,
                   dx: float, dy: float, nu: float) -> torch.Tensor:
    """
    Compute explicit diffusion terms for u-component in x and y directions only.
    Used in IMEX scheme where z-diffusion is treated implicitly.
    JIT-compiled for GPU performance.
    """
    diff_u = torch.zeros_like(u)

    # Second derivative in x (uniform spacing, periodic)
    u_ext_x = torch.cat([u, u[1:2, :, :]], dim=0)
    d2u_dx2 = (u_ext_x[2:nx+2, 1:ny+1, 1:nz+1] -
               2*u_ext_x[1:nx+1, 1:ny+1, 1:nz+1] +
               u_ext_x[0:nx, 1:ny+1, 1:nz+1]) / dx**2

    # Second derivative in y (uniform spacing)
    d2u_dy2 = (u[1:nx+1, 2:ny+2, 1:nz+1] -
               2*u[1:nx+1, 1:ny+1, 1:nz+1] +
               u[1:nx+1, 0:ny, 1:nz+1]) / dy**2

    diff_u[1:nx+1, 1:ny+1, 1:nz+1] = nu * (d2u_dx2 + d2u_dy2)

    return diff_u


@torch.jit.script
def diffusion_xy_v(v: torch.Tensor, nx: int, ny: int, nz: int,
                   dx: float, dy: float, nu: float) -> torch.Tensor:
    """
    Compute explicit diffusion terms for v-component in x and y directions only.
    Used in IMEX scheme where z-diffusion is treated implicitly.
    JIT-compiled for GPU performance.
    """
    diff_v = torch.zeros_like(v)

    # Second derivative in x (uniform spacing)
    d2v_dx2 = (v[2:nx+2, 1:ny+1, 1:nz+1] -
               2*v[1:nx+1, 1:ny+1, 1:nz+1] +
               v[0:nx, 1:ny+1, 1:nz+1]) / dx**2

    # Second derivative in y (uniform spacing, periodic)
    v_ext_y = torch.cat([v, v[:, 1:2, :]], dim=1)
    d2v_dy2 = (v_ext_y[1:nx+1, 2:ny+2, 1:nz+1] -
               2*v_ext_y[1:nx+1, 1:ny+1, 1:nz+1] +
               v_ext_y[1:nx+1, 0:ny, 1:nz+1]) / dy**2

    diff_v[1:nx+1, 1:ny+1, 1:nz+1] = nu * (d2v_dx2 + d2v_dy2)

    return diff_v


@torch.jit.script
def diffusion_xy_w(w: torch.Tensor, nx: int, ny: int, nz: int,
                   dx: float, dy: float, nu: float) -> torch.Tensor:
    """
    Compute explicit diffusion terms for w-component in x and y directions only.
    Used in IMEX scheme where z-diffusion is treated implicitly.
    JIT-compiled for GPU performance.
    """
    diff_w = torch.zeros_like(w)

    # d2w/dx2
    d2w_dx2 = (w[2:nx+2, 1:ny+1, 1:nz] -
               2*w[1:nx+1, 1:ny+1, 1:nz] +
               w[0:nx, 1:ny+1, 1:nz]) / (dx**2)

    # d2w/dy2
    d2w_dy2 = (w[1:nx+1, 2:ny+2, 1:nz] -
               2*w[1:nx+1, 1:ny+1, 1:nz] +
               w[1:nx+1, 0:ny, 1:nz]) / (dy**2)

    diff_w[1:nx+1, 1:ny+1, 1:nz] = nu * (d2w_dx2 + d2w_dy2)

    return diff_w


def solve_tridiagonal_batch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                             d: torch.Tensor) -> torch.Tensor:
    """
    Solve batched tridiagonal systems via parallel cyclic reduction (see
    tridiag.pcr_solve). a, b, c are 1-D (n,), shared across the batch; d is
    (batch, n). PCR replaces the old serial Thomas sweep (~2n sequential kernel
    launches) with ~log2(n) vectorized steps — a large win on GPUs without the
    JIT fuser (GB10/sm_121). a[0] and c[n-1] are ignored.
    """
    return pcr_solve(a, b, c, d)


@torch.jit.script
def solve_implicit_diffusion_u(u: torch.Tensor, dt: float, nx: int, ny: int, nz: int,
                                dz_c: torch.Tensor, dz_f: torch.Tensor, nu: float,
                                theta: float = 0.5, top_wall_bc_type: str = 'dirichlet') -> torch.Tensor:
    """
    Solve implicit diffusion in z for u-component using theta-method.

    theta = 0.5 → Crank-Nicolson (2nd order accurate, recommended)
    theta = 1.0 → Backward Euler (1st order accurate)

    Solves: (I - theta*dt*nu*d²/dz²)u^(n+1) = u^* + (1-theta)*dt*nu*d²u^*/dz²

    Vectorized implementation using batched tridiagonal solver.
    Solves all (nx*ny) columns in parallel - GPU optimized.

    Returns updated u with implicit z-diffusion applied.
    """
    u_new = u.clone()
    alpha = theta * dt * nu  # Implicit part coefficient

    # Build tridiagonal coefficients for all k (vectorized)
    # Shape: (nz,) for each coefficient
    dz_left = dz_c[0:nz]     # Spacing from k-1 to k
    dz_right = dz_c[1:nz+1]  # Spacing from k to k+1
    dz_cell = dz_f[0:nz]     # Cell height at k

    # Tridiagonal coefficients (interior formula)
    coeff_lower = -alpha / (dz_left * dz_cell)  # a[k]
    coeff_center = 1.0 + alpha * (1.0/dz_left + 1.0/dz_right) / dz_cell  # b[k]
    coeff_upper = -alpha / (dz_right * dz_cell)  # c[k]

    # Build tridiagonal matrix with boundary conditions
    # Shape: (nz,) for each diagonal
    a = coeff_lower.clone()
    b = coeff_center.clone()
    c = coeff_upper.clone()

    # Bottom boundary (k=0): u[0] = -u[1] (no-slip at z=0)
    # The ghost cell u[0] is eliminated: u[0] = -u[1]
    # This modifies only the diagonal coefficient, upper coupling stays the same
    a[0] = 0.0  # No coupling to point below (it's been eliminated)
    b[0] = coeff_center[0] - coeff_lower[0]  # Modified diagonal: adds 2*coeff_lower effect
    # c[0] remains unchanged (coeff_upper[0])

    # Top boundary (k=nz-1): depends on BC type
    if top_wall_bc_type == 'neumann':
        # Free-slip: u[nz] = u[nz-1] (Neumann BC: du/dz = 0)
        # Ghost cell u[nz] is eliminated: u[nz] = u[nz-1]
        # d2u/dz2 term: (u[nz]-u[nz-1])/dz_right becomes 0
        # In tridiagonal system: -alpha * u[nz] becomes -alpha * u[nz-1]
        # This adds to the diagonal term
        b[nz-1] = coeff_center[nz-1] + coeff_upper[nz-1]
    else:
        # No-slip: u[nz] = -u[nz-1] (Dirichlet BC: u = 0)
        # Ghost cell u[nz] is eliminated: u[nz] = -u[nz-1]
        b[nz-1] = coeff_center[nz-1] - coeff_upper[nz-1]
        
    c[nz-1] = 0.0  # No coupling to point above (it's been eliminated)

    # RHS: u at current time step
    # Shape: (nx, ny, nz)
    d = u[1:nx+1, 1:ny+1, 1:nz+1].clone()

    # Add explicit diffusion term for Crank-Nicolson: (1-theta)*dt*nu*d²u/dz²
    if theta < 1.0:
        beta = (1.0 - theta) * dt * nu
        U = u[1:nx+1, 1:ny+1, :]  # Shape: (nx, ny, nz+2) with ghost cells

        # Vectorized d²u/dz² for all k (1 stencil op; broadcast dz over last axis).
        # The generic interior stencil uses the ghosts U[0], U[nz+1]; the two wall
        # columns are then overwritten with their exact BC closures so the result
        # is independent of the ghost values (bit-identical to the old per-k loop).
        d2u = ((U[:, :, 2:nz+2] - U[:, :, 1:nz+1]) / dz_right
               - (U[:, :, 1:nz+1] - U[:, :, 0:nz]) / dz_left) / dz_cell
        d2u[:, :, 0] = ((U[:, :, 2] - U[:, :, 1]) / dz_right[0]
                        - 2.0 * U[:, :, 1] / dz_left[0]) / dz_cell[0]
        if top_wall_bc_type == 'neumann':
            d2u[:, :, nz-1] = (-(U[:, :, nz] - U[:, :, nz-1]) / dz_left[nz-1]) / dz_cell[nz-1]
        else:
            d2u[:, :, nz-1] = (-2.0 * U[:, :, nz] / dz_right[nz-1]
                               - (U[:, :, nz] - U[:, :, nz-1]) / dz_left[nz-1]) / dz_cell[nz-1]
        d += beta * d2u

    # Solve batched tridiagonal system for all (i,j) columns
    # Reshape to (nx*ny, nz) for batch processing
    d_batch = d.reshape(nx*ny, nz)
    x_batch = solve_tridiagonal_batch(a, b, c, d_batch)

    # Reshape back and assign
    u_new[1:nx+1, 1:ny+1, 1:nz+1] = x_batch.reshape(nx, ny, nz)

    return u_new


@torch.jit.script
def solve_implicit_diffusion_v(v: torch.Tensor, dt: float, nx: int, ny: int, nz: int,
                                dz_c: torch.Tensor, dz_f: torch.Tensor, nu: float,
                                theta: float = 0.5, top_wall_bc_type: str = 'dirichlet') -> torch.Tensor:
    """
    Solve implicit diffusion in z for v-component using theta-method.

    theta = 0.5 → Crank-Nicolson (2nd order accurate, recommended)
    theta = 1.0 → Backward Euler (1st order accurate)

    Solves: (I - theta*dt*nu*d²/dz²)v^(n+1) = v^* + (1-theta)*dt*nu*d²v^*/dz²

    Vectorized implementation using batched tridiagonal solver.
    Solves all (nx*ny) columns in parallel - GPU optimized.

    Returns updated v with implicit z-diffusion applied.
    """
    v_new = v.clone()
    alpha = theta * dt * nu  # Implicit part coefficient

    # Build tridiagonal coefficients (same as u-component)
    dz_left = dz_c[0:nz]
    dz_right = dz_c[1:nz+1]
    dz_cell = dz_f[0:nz]

    coeff_lower = -alpha / (dz_left * dz_cell)
    coeff_center = 1.0 + alpha * (1.0/dz_left + 1.0/dz_right) / dz_cell
    coeff_upper = -alpha / (dz_right * dz_cell)

    a = coeff_lower.clone()
    b = coeff_center.clone()
    c = coeff_upper.clone()

    # Bottom boundary (k=0): v[0] = -v[1] (no-slip at z=0)
    a[0] = 0.0
    b[0] = coeff_center[0] - coeff_lower[0]
    # c[0] unchanged

    # Top boundary (k=nz-1): depends on BC type
    if top_wall_bc_type == 'neumann':
        # Free-slip: v[nz] = v[nz-1] (Neumann BC: dv/dz = 0)
        b[nz-1] = coeff_center[nz-1] + coeff_upper[nz-1]
    else:
        # No-slip: v[nz] = -v[nz-1] (Dirichlet BC: v = 0)
        b[nz-1] = coeff_center[nz-1] - coeff_upper[nz-1]
        
    c[nz-1] = 0.0

    # RHS: v at current time step
    d = v[1:nx+1, 1:ny+1, 1:nz+1].clone()

    # Add explicit diffusion term for Crank-Nicolson: (1-theta)*dt*nu*d²v/dz²
    if theta < 1.0:
        beta = (1.0 - theta) * dt * nu
        V = v[1:nx+1, 1:ny+1, :]  # Shape: (nx, ny, nz+2) with ghost cells

        # Vectorized d²v/dz² for all k (see solve_implicit_diffusion_u).
        d2v = ((V[:, :, 2:nz+2] - V[:, :, 1:nz+1]) / dz_right
               - (V[:, :, 1:nz+1] - V[:, :, 0:nz]) / dz_left) / dz_cell
        d2v[:, :, 0] = ((V[:, :, 2] - V[:, :, 1]) / dz_right[0]
                        - 2.0 * V[:, :, 1] / dz_left[0]) / dz_cell[0]
        if top_wall_bc_type == 'neumann':
            d2v[:, :, nz-1] = (-(V[:, :, nz] - V[:, :, nz-1]) / dz_left[nz-1]) / dz_cell[nz-1]
        else:
            d2v[:, :, nz-1] = (-2.0 * V[:, :, nz] / dz_right[nz-1]
                               - (V[:, :, nz] - V[:, :, nz-1]) / dz_left[nz-1]) / dz_cell[nz-1]
        d += beta * d2v

    # Solve batched tridiagonal system
    d_batch = d.reshape(nx*ny, nz)
    x_batch = solve_tridiagonal_batch(a, b, c, d_batch)

    v_new[1:nx+1, 1:ny+1, 1:nz+1] = x_batch.reshape(nx, ny, nz)

    return v_new


@torch.jit.script
def solve_implicit_diffusion_w(w: torch.Tensor, dt: float, nx: int, ny: int, nz: int,
                                dz_c: torch.Tensor, dz_f: torch.Tensor, nu: float,
                                theta: float = 0.5) -> torch.Tensor:
    """
    Solve implicit diffusion in z for w-component using theta-method.

    theta = 0.5 → Crank-Nicolson (2nd order accurate, recommended)
    theta = 1.0 → Backward Euler (1st order accurate)

    Solves: (I - theta*dt*nu*d²/dz²)w^(n+1) = w^* + (1-theta)*dt*nu*d²w^*/dz²

    Vectorized implementation using batched tridiagonal solver.
    Solves all (nx*ny) columns in parallel - GPU optimized.

    Note: w is staggered in z (lives at z-faces), so it uses different grid spacings.

    Returns updated w with implicit z-diffusion applied.
    """
    w_new = w.clone()
    alpha = theta * dt * nu  # Implicit part coefficient

    # w has (nz-1) interior points in z-direction
    n_interior = nz - 1

    # Build tridiagonal coefficients for w (staggered in z)
    # w[k] lives at z-faces, different from u/v
    # Use dz_f for spacing between w-points, dz_c for cell centers
    dz_left = dz_f[0:n_interior]      # Spacing to left
    dz_right = dz_f[1:nz]              # Spacing to right
    dz_cv = dz_c[1:nz]                 # Cell center spacing

    coeff_lower = -alpha / (dz_left * dz_cv)
    coeff_center = 1.0 + alpha * (1.0/dz_left + 1.0/dz_right) / dz_cv
    coeff_upper = -alpha / (dz_right * dz_cv)

    a = coeff_lower.clone()
    b = coeff_center.clone()
    c = coeff_upper.clone()

    # Bottom boundary (k=0): w[0] = 0 (impermeability at z=0, Dirichlet BC)
    # Since w[0]=0, there's no coupling to point below
    a[0] = 0.0
    # b[0] and c[0] remain unchanged

    # Top boundary (k=nz-2, last interior point): w[nz] = 0 (impermeability at z=Lz, Dirichlet BC)
    # Since w[nz]=0, there's no coupling to point above
    c[n_interior-1] = 0.0
    # a[n_interior-1] and b[n_interior-1] remain unchanged

    # RHS: w at current time step
    d = w[1:nx+1, 1:ny+1, 1:nz].clone()

    # Add explicit diffusion term for Crank-Nicolson: (1-theta)*dt*nu*d²w/dz²
    if theta < 1.0:
        beta = (1.0 - theta) * dt * nu
        W = w[1:nx+1, 1:ny+1, :]  # Shape: (nx, ny, nz+1), faces 0..nz (walls at 0, nz)
        ni = n_interior

        # Vectorized d²w/dz² over the interior faces (bit-identical to the old
        # per-k loop, including its index convention). Middle by slicing; the two
        # wall-adjacent faces use w=0 at the wall.
        d2w = torch.empty((nx, ny, ni), dtype=w.dtype, device=w.device)
        d2w[:, :, 1:ni-1] = ((W[:, :, 2:ni] - W[:, :, 1:ni-1]) / dz_right[1:ni-1]
                             - (W[:, :, 1:ni-1] - W[:, :, 0:ni-2]) / dz_left[1:ni-1]) / dz_cv[1:ni-1]
        d2w[:, :, 0] = ((W[:, :, 1] - W[:, :, 0]) / dz_right[0]
                        - W[:, :, 0] / dz_left[0]) / dz_cv[0]
        d2w[:, :, ni-1] = (-W[:, :, ni-1] / dz_right[ni-1]
                           - (W[:, :, ni-1] - W[:, :, ni-2]) / dz_left[ni-1]) / dz_cv[ni-1]
        d += beta * d2w

    # Solve batched tridiagonal system
    d_batch = d.reshape(nx*ny, n_interior)
    x_batch = solve_tridiagonal_batch(a, b, c, d_batch)

    w_new[1:nx+1, 1:ny+1, 1:nz] = x_batch.reshape(nx, ny, n_interior)

    return w_new


# ==============================================================================
# ENHANCED FUSED KERNELS (GPU Optimization - Phase 4)
# ==============================================================================


@torch.jit.script
def compute_momentum_rhs_fused_imex(
    u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
    nx: int, ny: int, nz: int,
    dx: float, dy: float,
    dz_c: torch.Tensor, dz_f: torch.Tensor,
    nu: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused kernel for IMEX scheme: advection + XY-diffusion only.
    Z-diffusion is handled implicitly in a separate step.

    This kernel combines 6 separate operations (3 advection + 3 xy-diffusion)
    into a single GPU kernel launch, reducing memory bandwidth by ~40%.

    Returns: (rhs_u, rhs_v, rhs_w) = diffusion_xy - advection for each component

    Performance target: 30-50% speedup over separate kernel calls in IMEX
    """
    # Allocate output tensors
    rhs_u = torch.zeros_like(u)
    rhs_v = torch.zeros_like(v)
    rhs_w = torch.zeros_like(w)

    # Precompute reciprocals
    dx_inv = 1.0 / dx
    dy_inv = 1.0 / dy
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    nu_dx2 = nu * dx2_inv
    nu_dy2 = nu * dy2_inv

    # Precompute z-grid reciprocals for advection
    dz_f_inv = 1.0 / dz_f.view(1, 1, -1)
    dz_c_inv = 1.0 / dz_c.view(1, 1, -1)

    # ==================================================================
    # U-COMPONENT: Fused advection + XY-diffusion (no Z-diffusion)
    # ==================================================================

    # --- ADVECTION (same as v2 kernel) ---
    # On ALL x-faces [1:nx+1]: u[0] is the periodic ghost of u[nx], so face nx
    # is physical. Omitting it costs the flux form its telescoping property and
    # leaks ~1/nx of the streamwise momentum budget (see advection_u).
    u_ext_x = torch.cat([u, u[1:2, :, :]], dim=0)

    # d(uu)/dx
    u_interp = 0.5 * (u_ext_x[1:nx+1, 1:ny+1, 1:nz+1] + u_ext_x[2:nx+2, 1:ny+1, 1:nz+1])
    uu_right = u_interp * u_interp
    u_interp = 0.5 * (u_ext_x[0:nx, 1:ny+1, 1:nz+1] + u_ext_x[1:nx+1, 1:ny+1, 1:nz+1])
    uu_left = u_interp * u_interp
    duudx = (uu_right - uu_left) * dx_inv

    # d(vu)/dy
    v_interp = 0.5 * (v[1:nx+1, 1:ny+1, 1:nz+1] + v[2:nx+2, 1:ny+1, 1:nz+1])
    u_interp_y = 0.5 * (u_ext_x[1:nx+1, 1:ny+1, 1:nz+1] + u_ext_x[1:nx+1, 2:ny+2, 1:nz+1])
    vu_top = v_interp * u_interp_y
    v_interp = 0.5 * (v[1:nx+1, 0:ny, 1:nz+1] + v[2:nx+2, 0:ny, 1:nz+1])
    u_interp_y = 0.5 * (u_ext_x[1:nx+1, 0:ny, 1:nz+1] + u_ext_x[1:nx+1, 1:ny+1, 1:nz+1])
    vu_bottom = v_interp * u_interp_y
    dvudy = (vu_top - vu_bottom) * dy_inv

    # d(wu)/dz
    w_interp = 0.5 * (w[1:nx+1, 1:ny+1, 1:nz+1] + w[2:nx+2, 1:ny+1, 1:nz+1])
    u_interp_z = 0.5 * (u_ext_x[1:nx+1, 1:ny+1, 1:nz+1] + u_ext_x[1:nx+1, 1:ny+1, 2:nz+2])
    wu_top = w_interp * u_interp_z
    w_interp = 0.5 * (w[1:nx+1, 1:ny+1, 0:nz] + w[2:nx+2, 1:ny+1, 0:nz])
    u_interp_z = 0.5 * (u_ext_x[1:nx+1, 1:ny+1, 0:nz] + u_ext_x[1:nx+1, 1:ny+1, 1:nz+1])
    wu_bottom = w_interp * u_interp_z
    dz_avg = dz_f_inv[0, 0, 0:nz]
    dwudz = (wu_top - wu_bottom) * dz_avg

    advection_u = duudx + dvudy + dwudz

    # --- XY-DIFFUSION ONLY (no Z-diffusion) ---

    # d2u/dx2 - periodic
    d2u_dx2 = (u_ext_x[2:nx+2, 1:ny+1, 1:nz+1] -
               2.0 * u_ext_x[1:nx+1, 1:ny+1, 1:nz+1] +
               u_ext_x[0:nx, 1:ny+1, 1:nz+1]) * nu_dx2

    # d2u/dy2
    d2u_dy2 = (u[1:nx+1, 2:ny+2, 1:nz+1] -
               2.0 * u[1:nx+1, 1:ny+1, 1:nz+1] +
               u[1:nx+1, 0:ny, 1:nz+1]) * nu_dy2

    diffusion_xy_u = d2u_dx2 + d2u_dy2  # Note: NO d2u_dz2

    # Combine: RHS = diffusion_xy - advection
    rhs_u[1:nx+1, 1:ny+1, 1:nz+1] = diffusion_xy_u - advection_u

    # ==================================================================
    # V-COMPONENT: Fused advection + XY-diffusion
    # ==================================================================

    # --- ADVECTION ---
    # On ALL y-faces [1:ny+1] -- same reasoning as the u block, one axis over.
    v_ext_y = torch.cat([v, v[:, 1:2, :]], dim=1)

    # d(uv)/dx
    u_interp = 0.5 * (u[1:nx+1, 1:ny+1, 1:nz+1] + u[1:nx+1, 2:ny+2, 1:nz+1])
    v_interp_x = 0.5 * (v_ext_y[1:nx+1, 1:ny+1, 1:nz+1] + v_ext_y[2:nx+2, 1:ny+1, 1:nz+1])
    uv_right = u_interp * v_interp_x
    u_interp = 0.5 * (u[0:nx, 1:ny+1, 1:nz+1] + u[0:nx, 2:ny+2, 1:nz+1])
    v_interp_x = 0.5 * (v_ext_y[0:nx, 1:ny+1, 1:nz+1] + v_ext_y[1:nx+1, 1:ny+1, 1:nz+1])
    uv_left = u_interp * v_interp_x
    duvdx = (uv_right - uv_left) * dx_inv

    # d(vv)/dy
    v_interp = 0.5 * (v_ext_y[1:nx+1, 1:ny+1, 1:nz+1] + v_ext_y[1:nx+1, 2:ny+2, 1:nz+1])
    vv_top = v_interp * v_interp
    v_interp = 0.5 * (v_ext_y[1:nx+1, 0:ny, 1:nz+1] + v_ext_y[1:nx+1, 1:ny+1, 1:nz+1])
    vv_bottom = v_interp * v_interp
    dvvdy = (vv_top - vv_bottom) * dy_inv

    # d(wv)/dz
    w_interp = 0.5 * (w[1:nx+1, 1:ny+1, 1:nz+1] + w[1:nx+1, 2:ny+2, 1:nz+1])
    v_interp_z = 0.5 * (v_ext_y[1:nx+1, 1:ny+1, 1:nz+1] + v_ext_y[1:nx+1, 1:ny+1, 2:nz+2])
    wv_top = w_interp * v_interp_z
    w_interp = 0.5 * (w[1:nx+1, 1:ny+1, 0:nz] + w[1:nx+1, 2:ny+2, 0:nz])
    v_interp_z = 0.5 * (v_ext_y[1:nx+1, 1:ny+1, 0:nz] + v_ext_y[1:nx+1, 1:ny+1, 1:nz+1])
    wv_bottom = w_interp * v_interp_z
    dz_avg = dz_f_inv[0, 0, 0:nz]
    dwvdz = (wv_top - wv_bottom) * dz_avg

    advection_v = duvdx + dvvdy + dwvdz

    # --- XY-DIFFUSION ONLY ---

    # d2v/dx2
    d2v_dx2 = (v[2:nx+2, 1:ny+1, 1:nz+1] -
               2.0 * v[1:nx+1, 1:ny+1, 1:nz+1] +
               v[0:nx, 1:ny+1, 1:nz+1]) * nu_dx2

    # d2v/dy2 - periodic
    d2v_dy2 = (v_ext_y[1:nx+1, 2:ny+2, 1:nz+1] -
               2.0 * v_ext_y[1:nx+1, 1:ny+1, 1:nz+1] +
               v_ext_y[1:nx+1, 0:ny, 1:nz+1]) * nu_dy2

    diffusion_xy_v = d2v_dx2 + d2v_dy2  # Note: NO d2v_dz2

    # Combine
    rhs_v[1:nx+1, 1:ny+1, 1:nz+1] = diffusion_xy_v - advection_v

    # ==================================================================
    # W-COMPONENT: Fused advection + XY-diffusion
    # ==================================================================

    # --- ADVECTION ---
    # d(uw)/dx
    u_interp = 0.5 * (u[1:nx+1, 1:ny+1, 1:nz] + u[1:nx+1, 1:ny+1, 2:nz+1])
    w_interp_x = 0.5 * (w[1:nx+1, 1:ny+1, 1:nz] + w[2:nx+2, 1:ny+1, 1:nz])
    uw_right = u_interp * w_interp_x
    u_interp = 0.5 * (u[0:nx, 1:ny+1, 1:nz] + u[0:nx, 1:ny+1, 2:nz+1])
    w_interp_x = 0.5 * (w[0:nx, 1:ny+1, 1:nz] + w[1:nx+1, 1:ny+1, 1:nz])
    uw_left = u_interp * w_interp_x
    duwdx = (uw_right - uw_left) * dx_inv

    # d(vw)/dy
    v_interp = 0.5 * (v[1:nx+1, 1:ny+1, 1:nz] + v[1:nx+1, 1:ny+1, 2:nz+1])
    w_interp_y = 0.5 * (w[1:nx+1, 1:ny+1, 1:nz] + w[1:nx+1, 2:ny+2, 1:nz])
    vw_top = v_interp * w_interp_y
    v_interp = 0.5 * (v[1:nx+1, 0:ny, 1:nz] + v[1:nx+1, 0:ny, 2:nz+1])
    w_interp_y = 0.5 * (w[1:nx+1, 0:ny, 1:nz] + w[1:nx+1, 1:ny+1, 1:nz])
    vw_bottom = v_interp * w_interp_y
    dvwdy = (vw_top - vw_bottom) * dy_inv

    # d(ww)/dz
    w_interp = 0.5 * (w[1:nx+1, 1:ny+1, 1:nz] + w[1:nx+1, 1:ny+1, 2:nz+1])
    ww_top = w_interp * w_interp
    w_interp = 0.5 * (w[1:nx+1, 1:ny+1, 0:nz-1] + w[1:nx+1, 1:ny+1, 1:nz])
    ww_bottom = w_interp * w_interp
    dz_avg = dz_c[1:nz].view(1, 1, -1)
    dwwdz = (ww_top - ww_bottom) / dz_avg

    advection_w = duwdx + dvwdy + dwwdz

    # --- XY-DIFFUSION ONLY ---

    # d2w/dx2
    d2w_dx2 = (w[2:nx+2, 1:ny+1, 1:nz] -
               2.0 * w[1:nx+1, 1:ny+1, 1:nz] +
               w[0:nx, 1:ny+1, 1:nz]) * nu_dx2

    # d2w/dy2
    d2w_dy2 = (w[1:nx+1, 2:ny+2, 1:nz] -
               2.0 * w[1:nx+1, 1:ny+1, 1:nz] +
               w[1:nx+1, 0:ny, 1:nz]) * nu_dy2

    diffusion_xy_w = d2w_dx2 + d2w_dy2  # Note: NO d2w_dz2

    # Combine
    rhs_w[1:nx+1, 1:ny+1, 1:nz] = diffusion_xy_w - advection_w

    return rhs_u, rhs_v, rhs_w


# ==============================================================================
# CFL COMPUTATION (GPU Optimization)
# ==============================================================================

@torch.jit.script
def compute_cfl_fused(
    u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
    nx: int, ny: int, nz: int,
    dx: float, dy: float,
    dz_f: torch.Tensor, dz_c: torch.Tensor
) -> float:
    """
    Fused CUDA kernel for CFL computation combining all velocity interpolations
    and max reduction into a single GPU kernel launch.

    Computes CFL number for staggered grid velocities:
    - X-direction: ux + vx + wx (velocities interpolated in x)
    - Y-direction: uy + vy + wy (velocities interpolated in y)
    - Z-direction: uz + vz + wz (velocities interpolated in z)

    Returns: Maximum CFL inverse (1/dti) across all cells and directions

    Performance: Reduces ~15 kernel launches to 1, saving memory bandwidth
    """
    # Precompute reciprocals
    dxi = 1.0 / dx
    dyi = 1.0 / dy
    dzfi = 1.0 / dz_f  # Shape: (nz,)
    dzci = 1.0 / dz_c  # Shape: (nz+1,)

    # ==================================================================
    # X-direction CFL: ux + vx + wx
    # ==================================================================
    # ux: u is at x-faces, directly use
    ux = torch.abs(u[1:nx+1, 1:ny+1, 1:nz+1])

    # vx: interpolate v in x (4-point average)
    vx = 0.25 * torch.abs(
        v[1:nx+1, 1:ny+1, 1:nz+1] + v[1:nx+1, 0:ny, 1:nz+1] +
        v[2:nx+2, 1:ny+1, 1:nz+1] + v[2:nx+2, 0:ny, 1:nz+1]
    )

    # wx: interpolate w in x and z (4-point average)
    wx = 0.25 * torch.abs(
        w[1:nx+1, 1:ny+1, 1:nz+1] + w[1:nx+1, 1:ny+1, 0:nz] +
        w[2:nx+2, 1:ny+1, 1:nz+1] + w[2:nx+2, 1:ny+1, 0:nz]
    )

    # CFL contribution in x-direction (element-wise)
    dtix = ux * dxi + vx * dyi + wx * dzfi[0:nz].view(1, 1, -1)

    # ==================================================================
    # Y-direction CFL: uy + vy + wy
    # ==================================================================
    # uy: interpolate u in y (4-point average)
    uy = 0.25 * torch.abs(
        u[1:nx+1, 1:ny+1, 1:nz+1] + u[1:nx+1, 2:ny+2, 1:nz+1] +
        u[0:nx, 2:ny+2, 1:nz+1] + u[0:nx, 1:ny+1, 1:nz+1]
    )

    # vy: v is at y-faces, directly use
    vy = torch.abs(v[1:nx+1, 1:ny+1, 1:nz+1])

    # wy: interpolate w in y and z (4-point average)
    wy = 0.25 * torch.abs(
        w[1:nx+1, 1:ny+1, 1:nz+1] + w[1:nx+1, 2:ny+2, 1:nz+1] +
        w[1:nx+1, 2:ny+2, 0:nz] + w[1:nx+1, 1:ny+1, 0:nz]
    )

    # CFL contribution in y-direction (element-wise)
    dtiy = uy * dxi + vy * dyi + wy * dzfi[0:nz].view(1, 1, -1)

    # ==================================================================
    # Z-direction CFL: uz + vz + wz
    # ==================================================================
    # uz: interpolate u in z (4-point average) - only nz-1 points
    uz = 0.25 * torch.abs(
        u[1:nx+1, 1:ny+1, 1:nz] + u[0:nx, 1:ny+1, 1:nz] +
        u[0:nx, 1:ny+1, 2:nz+1] + u[1:nx+1, 1:ny+1, 2:nz+1]
    )

    # vz: interpolate v in z (4-point average) - only nz-1 points
    vz = 0.25 * torch.abs(
        v[1:nx+1, 1:ny+1, 1:nz] + v[1:nx+1, 0:ny, 1:nz] +
        v[1:nx+1, 0:ny, 2:nz+1] + v[1:nx+1, 1:ny+1, 2:nz+1]
    )

    # wz: w is at z-faces, directly use - only nz-1 points
    wz = torch.abs(w[1:nx+1, 1:ny+1, 1:nz])

    # CFL contribution in z-direction (element-wise)
    dtiz = uz * dxi + vz * dyi + wz * dzci[1:nz].view(1, 1, -1)

    # ==================================================================
    # Final max reduction across all directions and cells
    # ==================================================================
    # Find maximum CFL across all three directions
    # Note: dtiz has one fewer z-point (nz-1 vs nz), so we need separate max
    max_dtix = torch.max(dtix)
    max_dtiy = torch.max(dtiy)
    max_dtiz = torch.max(dtiz)

    # Overall maximum
    dti = torch.max(torch.stack([max_dtix, max_dtiy, max_dtiz]))

    return dti.item()


# ==============================================================================
# OPTIONAL torch.compile (Inductor/Triton) — Layer 2 GPU speedup
# ==============================================================================
# On GB10 (sm_121) the legacy TorchScript fuser cannot NVRTC-compile, so this
# code runs with PYTORCH_JIT=0 (the @torch.jit.script decorators become
# passthroughs). Inductor/Triton DOES support sm_121 (via ptxas + driver), so we
# torch.compile the launch-bound hot functions — the vectorized CN + PCR diffusion
# solves and the fused momentum RHS — fusing many eager kernels into a few.
# Opt-in via TORCHANNEL_COMPILE=1 (needs CC=gcc so Inductor's host compiler isn't
# nvc, which rejects -Wno-psabi).
if env.USE_COMPILE:
    compute_momentum_rhs_fused_imex = torch.compile(compute_momentum_rhs_fused_imex)
    solve_implicit_diffusion_u = torch.compile(solve_implicit_diffusion_u)
    solve_implicit_diffusion_v = torch.compile(solve_implicit_diffusion_v)
    solve_implicit_diffusion_w = torch.compile(solve_implicit_diffusion_w)
