"""initflow.py — Initial conditions.

Vortex-perturbed and parabolic initial fields, restart from a stored field,
and interpolation of a stored field onto a different grid.

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
from .utils import load_flow_fields


def _interp1d_weights(coords, nodes):
    """Linear interpolation indices/weights of query coords into 1-D sorted nodes.
    Queries outside the node range are clamped (constant extrapolation)."""
    k = torch.searchsorted(nodes, coords).clamp(1, len(nodes) - 1)
    k0 = k - 1
    denom = (nodes[k] - nodes[k0]).clamp(min=1e-300)
    t = ((coords - nodes[k0]) / denom).clamp(0.0, 1.0)
    return k0, t


def _trilinear(field, x_nodes, y_nodes, z_nodes, xq, yq, zq):
    """Trilinear interpolation of field (len(x_nodes), len(y_nodes), len(z_nodes))
    onto the tensor-product query points xq, yq, zq. Returns (len(xq), len(yq), len(zq))."""
    ix, tx = _interp1d_weights(xq, x_nodes)
    iy, ty = _interp1d_weights(yq, y_nodes)
    iz, tz = _interp1d_weights(zq, z_nodes)
    out = torch.zeros(len(xq), len(yq), len(zq), dtype=field.dtype, device=field.device)
    for a in (0, 1):
        wx = (tx if a else 1.0 - tx).view(-1, 1, 1)
        for b in (0, 1):
            wy = (ty if b else 1.0 - ty).view(1, -1, 1)
            for c in (0, 1):
                wz = (tz if c else 1.0 - tz).view(1, 1, -1)
                out += wx * wy * wz * field[ix + a][:, iy + b][:, :, iz + c]
    return out


def initialize_flow_interpolated(field_file, nx, ny, nz, Lx, Ly, Lz, z_c, z_f,
                                 device='cpu', source_half='lower'):
    """
    Initialize the flow by interpolating a previously saved turbulent field
    (possibly on a DIFFERENT grid/domain) onto the current staggered grid.

    The source grid is fully recovered from the npz file (z_c/z_f arrays, Lx/Ly
    and the array shapes; x/y are uniform). Staggered-aware trilinear
    interpolation per component, periodic in x/y via the source ghost layers.
    If the source domain is wider/longer, the target box is mapped
    proportionally onto it; if it is taller (full channel -> open channel),
    the lower part of the source is used (source_half='lower').

    The caller must rescale to the target bulk velocity, re-apply boundary
    conditions and project (the solver's init already does all three).

    Returns u, v, w, p on the target grid (p is zero; the initial projection
    rebuilds it). Time/step should be reset by the caller.
    """
    print(f"Interpolating initial field from: {field_file}", flush=True)
    src = load_flow_fields(field_file, device=device)
    u_s, v_s, w_s = src['u'], src['v'], src['w']
    z_c_s, z_f_s = src['z_c'], src['z_f']
    Lx_s, Ly_s = float(src['Lx']), float(src['Ly'])
    nxs, nys = u_s.shape[0] - 1, v_s.shape[1] - 1
    nzs = len(z_f_s) - 1
    Lz_s = float(z_f_s[-1])
    dxs, dys = Lx_s / nxs, Ly_s / nys

    if source_half != 'lower' and Lz_s > Lz:
        raise ValueError(f"source_half='{source_half}' not supported (only 'lower')")
    print(f"  source: {nxs}x{nys}x{nzs}, L = ({Lx_s:.4f}, {Ly_s:.4f}, {Lz_s:.4f})", flush=True)
    print(f"  target: {nx}x{ny}x{nz}, L = ({Lx:.4f}, {Ly:.4f}, {Lz:.4f})"
          + (f" [lower {Lz/Lz_s:.2f} of source height]" if Lz_s > Lz else ""), flush=True)

    # Refresh periodic ghost layers of the source (cheap insurance; z ghosts
    # are re-applied by the loader already)
    u_s[0, :, :] = u_s[-1, :, :]
    u_s[:, 0, :] = u_s[:, -2, :]; u_s[:, -1, :] = u_s[:, 1, :]
    v_s[0, :, :] = v_s[-2, :, :]; v_s[-1, :, :] = v_s[1, :, :]
    v_s[:, 0, :] = v_s[:, -1, :]
    w_s[0, :, :] = w_s[-2, :, :]; w_s[-1, :, :] = w_s[1, :, :]
    w_s[:, 0, :] = w_s[:, -2, :]; w_s[:, -1, :] = w_s[:, 1, :]

    # Source node coordinates (full arrays incl. ghosts/duplicates, so the
    # periodic wrap is covered by construction: faces span [0, L], centers
    # span [-dx/2, L+dx/2])
    dev, dt = u_s.device, u_s.dtype
    x_face_s = torch.arange(nxs + 1, dtype=dt, device=dev) * dxs
    x_cent_s = (torch.arange(nxs + 2, dtype=dt, device=dev) - 0.5) * dxs
    y_face_s = torch.arange(nys + 1, dtype=dt, device=dev) * dys
    y_cent_s = (torch.arange(nys + 2, dtype=dt, device=dev) - 0.5) * dys

    # Target node coordinates: map x/y proportionally onto the source domain
    # (handles different Lx/Ly by stretching the periodic box), z directly
    # (source_half='lower': target z already addresses the lower source region)
    dx, dy = Lx / nx, Ly / ny
    rx, ry = Lx_s / Lx, Ly_s / Ly
    x_face_t = torch.arange(nx + 1, dtype=dt, device=dev) * dx * rx
    x_cent_t = ((torch.arange(nx + 2, dtype=dt, device=dev) - 0.5) * dx * rx)
    y_face_t = torch.arange(ny + 1, dtype=dt, device=dev) * dy * ry
    y_cent_t = ((torch.arange(ny + 2, dtype=dt, device=dev) - 0.5) * dy * ry)
    # clamp x/y ghost coords into the source coordinate span (periodicity is
    # honoured because the span includes both images of the seam)
    x_cent_t = x_cent_t.clamp(x_cent_s[0], x_cent_s[-1])
    y_cent_t = y_cent_t.clamp(y_cent_s[0], y_cent_s[-1])

    u = _trilinear(u_s, x_face_s, y_cent_s, z_c_s, x_face_t, y_cent_t, z_c.to(dev))
    v = _trilinear(v_s, x_cent_s, y_face_s, z_c_s, x_cent_t, y_face_t, z_c.to(dev))
    w = _trilinear(w_s, x_cent_s, y_cent_s, z_f_s, x_cent_t, y_cent_t, z_f.to(dev))
    p = torch.zeros(nx + 2, ny + 2, nz + 2, dtype=dt, device=dev)

    print(f"  interpolated: u in [{u.min().item():.4f}, {u.max().item():.4f}]", flush=True)
    return u.contiguous(), v.contiguous(), w.contiguous(), p


def initialize_flow(nx, ny, nz, z_c, Ly, Lz, U_bulk=1.0, init_type='parabolic', perturbation_intensity=0.0, n_vortices=4, device='cpu', top_wall_bc_type='dirichlet'):
    """Initialize velocity and pressure fields. Creates tensors on specified device (CPU or CUDA)."""
    u = torch.zeros(nx+1, ny+2, nz+2, device=device)
    v = torch.zeros(nx+2, ny+1, nz+2, device=device)
    w = torch.zeros(nx+2, ny+2, nz+1, device=device)
    p = torch.zeros(nx+2, ny+2, nz+2, device=device)

    if init_type == 'parabolic' or init_type == 'vortices':
        # Base parabolic profile
        if top_wall_bc_type == 'neumann':
            # Free-slip top wall: Half-parabola with max velocity at top
            # u(z) = U_max * (z/Lz) * (2 - z/Lz)
            # U_max = 1.5 * U_bulk (same as full parabola)
            U_max = 1.5 * U_bulk
            z_norm = z_c / Lz  # z/Lz in [0, 1]
            u_profile = U_max * z_norm * (2 - z_norm)
        else:
            # No-slip top wall: Full parabola with max velocity at center
            U_max = 1.5 * U_bulk
            z_norm = 2 * z_c / Lz - 1
            u_profile = U_max * (1 - z_norm**2)
            
        u[:, :, :] = u_profile[None, None, :]
    
    if init_type == 'uniform':
        u[:, :, :] = U_bulk
    
    elif init_type == 'vortices':
        # Base profile already set above
        pass # Continue to vortex addition

        # Add counter-rotating vortices using streamfunction
        # psi(y, z) = A * sin(ky * y) * sin^2(pi * z / Lz)
        # v = -dpsi/dz, w = dpsi/dy
        # u perturbation = 0 (or small random noise)
        
        A = perturbation_intensity * U_bulk
        # ky determines the number of vortices in y
        # n_vortices = 2 means one pair (one positive, one negative lobe) -> ky = 2*pi/Ly
        # n_vortices = 4 means two pairs -> ky = 4*pi/Ly
        ky = n_vortices * torch.pi / Ly
        
        # Get device from input tensors
        device = z_c.device
        
        # Grid for v (staggered in y)
        dy = Ly / ny
        y_v = torch.linspace(0, Ly, ny+1, device=device)  # y-faces
        
        # Grid for w (staggered in z)
        y_c = 0.5 * (y_v[:-1] + y_v[1:])
        
        # Compute streamfunction and derivatives
        # For v: at y_v, z_c (use only interior z_c points)
        z_c_interior = z_c[1:nz+1]  # Shape (nz) - interior points only
        yy_v, zz_v = torch.meshgrid(y_v, z_c_interior, indexing='ij')
        psi_v = A * torch.sin(ky * yy_v) * torch.sin(torch.pi * zz_v / Lz)**2
        
        # v = -dpsi/dz
        v_pert = -A * torch.sin(ky * yy_v) * 2 * torch.sin(torch.pi * zz_v / Lz) * torch.cos(torch.pi * zz_v / Lz) * (torch.pi / Lz)
        
        # Broadcast v_pert (shape ny+1, nz) to v (shape nx+2, ny+1, nz+2)
        v[1:nx+1, :, 1:nz+1] += v_pert[None, :, :]
        
        # For w: at y_c, z_f (z-faces)
        # Approximate z_f for w perturbation using interior z_c
        z_f_local = torch.cat([torch.tensor([0.0], device=device), 
                               0.5 * (z_c_interior[:-1] + z_c_interior[1:]), 
                               torch.tensor([Lz], device=device)])
        
        yy_w, zz_w = torch.meshgrid(y_c, z_f_local, indexing='ij')
        psi_w = A * torch.sin(ky * yy_w) * torch.sin(torch.pi * zz_w / Lz)**2
        
        # w = dpsi/dy
        w_pert = A * torch.cos(ky * yy_w) * ky * torch.sin(torch.pi * zz_w / Lz)**2
        
        # Broadcast w_pert (shape ny, nz+1) to w (shape nx+2, ny+2, nz+1)
        w[1:nx+1, 1:ny+1, :] += w_pert[None, :, :]

    # Add random perturbations
    if perturbation_intensity > 0.0 and init_type != 'vortices':
        print(f"Adding random perturbation with intensity {perturbation_intensity} * U_bulk", flush=True)
        noise_scale = perturbation_intensity * U_bulk
        
        # Add noise to internal cells, avoiding boundaries
        # For Z (last dim), use 2:-2 to avoid the first and last fluid points next to the wall
        u[1:-1, 1:-1, 2:-2] += (torch.rand_like(u[1:-1, 1:-1, 2:-2]) - 0.5) * 2 * noise_scale
        v[1:-1, 1:-1, 2:-2] += (torch.rand_like(v[1:-1, 1:-1, 2:-2]) - 0.5) * 2 * noise_scale
        w[1:-1, 1:-1, 2:-2] += (torch.rand_like(w[1:-1, 1:-1, 2:-2]) - 0.5) * 2 * noise_scale

    # Apply BCs to ghost cells
    u[:, :, 0] = -u[:, :, 1]
    u[:, :, -1] = -u[:, :, -2]

    v[:, :, 0] = -v[:, :, 1]
    v[:, :, -1] = -v[:, :, -2]

    w[:, :, 0] = 0.0
    w[:, :, -1] = 0.0

    return u, v, w, p

def initialize_flow_from_file(field_file, device='cpu', reset_time=False):
    """
    Initialize velocity and pressure fields from a previously saved field file.

    This function loads a saved flow field (without projections or any post-processing)
    and returns the raw velocity and pressure fields ready for use as initial conditions.

    Args:
        field_file: Path to the .npz file containing saved flow fields
        device: Device to load tensors to ('cpu' or 'cuda')
        reset_time: If True, reset step and time to 0; if False, continue from saved values

    Returns:
        u, v, w, p: velocity and pressure fields as torch tensors on the specified device
        initial_step: Starting step number (0 if reset_time=True, otherwise from file)
        initial_time: Starting time (0.0 if reset_time=True, otherwise from file)

    Note:
        The loaded fields will have boundary conditions already applied from the saved state.
        However, it's recommended to re-apply boundary conditions after loading to ensure
        consistency with the current simulation setup.
    """
    print(f"Loading flow fields from: {field_file}", flush=True)

    # Load the saved fields
    fields = load_flow_fields(field_file, device=device)

    # Extract velocity and pressure fields
    u = fields['u']
    v = fields['v']
    w = fields['w']
    p = fields['p']

    # Extract time information
    if reset_time:
        initial_step = 0
        initial_time = 0.0
        print(f"Loaded fields from step {fields['step']}, time = {fields['time']:.6f}", flush=True)
        print(f"  Resetting step and time to 0", flush=True)
    else:
        initial_step = int(fields['step'])
        initial_time = float(fields['time'])
        print(f"Loaded fields from step {fields['step']}, time = {fields['time']:.6f}", flush=True)
        print(f"  Continuing from step {initial_step}, time = {initial_time:.6f}", flush=True)

    print(f"  u_tau = {fields['u_tau']:.6e}, forcing = {fields['forcing']:.6e}", flush=True)
    print(f"  Field shapes: u={u.shape}, v={v.shape}, w={w.shape}, p={p.shape}", flush=True)

    # Re-apply boundary conditions to ensure consistency
    # Dirichlet BC in z for u
    u[:, :, 0] = -u[:, :, 1]
    u[:, :, -1] = -u[:, :, -2]

    # Dirichlet BC in z for v
    v[:, :, 0] = -v[:, :, 1]
    v[:, :, -1] = -v[:, :, -2]

    # Dirichlet BC in z for w (w=0 at walls)
    w[:, :, 0] = 0.0
    w[:, :, -1] = 0.0

    return u, v, w, p, initial_step, initial_time
