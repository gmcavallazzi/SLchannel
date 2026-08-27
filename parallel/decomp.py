"""Rank grid, index maps and layout conventions for the z-aligned pencils.

Global node numbering per component (matches the SL node buffers filled by
slchannel/semilag.py:387-394 from the monolithic ghost-shaped fields):

    u : nodes m = 0..nx-1 at x = m*dx (x-faces), y-centers, z = z_c (nz+2)
    v : x-centers, nodes m = 0..ny-1 at y = m*dy (y-faces),  z = z_c (nz+2)
    w : x-centers, y-centers, z = z_f incl. walls (nz+1)
    p : x-centers, y-centers, z = z_c (nz+2)

A rank (cx, cy) owns nodes m in [i_start, i_start + nx_loc) x
[j_start, j_start + ny_loc), stored in an extended local array of shape
(nx_loc + 2H, ny_loc + 2H, NZ): local index a <-> global node
(i_start + a - H) mod nx. ALL components share this one layout; the
staggered off-by-ones live only in the slice tables below.
"""

from dataclasses import dataclass

import torch


def node_z_len(comp, nz):
    return nz + 1 if comp == "w" else nz + 2


def mono_node_view(field, comp, nx, ny):
    """View of a monolithic ghost-shaped field as its global node array."""
    if comp == "u":
        return field[0:nx, 1 : ny + 1, :]
    if comp == "v":
        return field[1 : nx + 1, 0:ny, :]
    return field[1 : nx + 1, 1 : ny + 1, :]  # w, p


def nodes_to_mono(nodes, comp, nx, ny):
    """Assemble a monolithic ghost-shaped field from a global node array,
    filling the x/y periodic ghosts exactly as apply_bc_all does
    (solver.py:70-107); z entries are carried as data."""
    NZ = nodes.shape[2]
    if comp == "u":
        f = torch.zeros(nx + 1, ny + 2, NZ, dtype=nodes.dtype, device=nodes.device)
        f[0:nx, 1 : ny + 1, :] = nodes
        f[nx, 1 : ny + 1, :] = nodes[0]
        f[:, 0, :] = f[:, ny, :]
        f[:, ny + 1, :] = f[:, 1, :]
    elif comp == "v":
        f = torch.zeros(nx + 2, ny + 1, NZ, dtype=nodes.dtype, device=nodes.device)
        f[1 : nx + 1, 0:ny, :] = nodes
        f[1 : nx + 1, ny, :] = nodes[:, 0, :]
        f[0, :, :] = f[nx, :, :]
        f[nx + 1, :, :] = f[1, :, :]
    else:  # w, p
        f = torch.zeros(nx + 2, ny + 2, NZ, dtype=nodes.dtype, device=nodes.device)
        f[1 : nx + 1, 1 : ny + 1, :] = nodes
        f[0, :, :] = f[nx, :, :]
        f[nx + 1, :, :] = f[1, :, :]
        f[:, 0, :] = f[:, ny, :]
        f[:, ny + 1, :] = f[:, 1, :]
    return f


@dataclass(frozen=True)
class Decomp:
    px: int
    py: int
    nx: int
    ny: int
    nz: int
    H: int

    def __post_init__(self):
        assert self.nx % self.px == 0, "nx must divide by px"
        assert self.ny % self.py == 0, "ny must divide by py"
        assert self.nxl >= self.H and self.nyl >= self.H, (
            f"local size must be >= halo width (nxl={self.nxl}, nyl={self.nyl}, H={self.H})"
        )
        assert self.H >= 1

    @property
    def nranks(self):
        return self.px * self.py

    @property
    def nxl(self):
        return self.nx // self.px

    @property
    def nyl(self):
        return self.ny // self.py

    def coords(self, rank):
        return rank // self.py, rank % self.py

    def rank_of(self, cx, cy):
        return (cx % self.px) * self.py + (cy % self.py)

    def origin(self, rank):
        cx, cy = self.coords(rank)
        return cx * self.nxl, cy * self.nyl

    def neighbors(self, rank):
        cx, cy = self.coords(rank)
        return {
            "xm": self.rank_of(cx - 1, cy),
            "xp": self.rank_of(cx + 1, cy),
            "ym": self.rank_of(cx, cy - 1),
            "yp": self.rank_of(cx, cy + 1),
        }

    def ext_shape(self, comp):
        return (self.nxl + 2 * self.H, self.nyl + 2 * self.H, node_z_len(comp, self.nz))

    def alloc(self, comp, dtype=torch.float64, device="cpu"):
        return torch.zeros(self.ext_shape(comp), dtype=dtype, device=device)

    # -- scatter / gather against global node arrays -----------------------

    def scatter(self, nodes, comp, fill_halos=True):
        """Global node array -> dict rank -> extended local array. Halos are
        filled from the periodic global field when fill_halos (test setup /
        Poisson scatter); left zero otherwise (to exercise the exchange)."""
        H = self.H
        out = {}
        for rank in range(self.nranks):
            i0, j0 = self.origin(rank)
            ext = self.alloc(comp, dtype=nodes.dtype, device=nodes.device)
            if fill_halos:
                gx = torch.arange(i0 - H, i0 + self.nxl + H) % self.nx
                gy = torch.arange(j0 - H, j0 + self.nyl + H) % self.ny
                ext.copy_(nodes[gx][:, gy])
            else:
                ext[H : H + self.nxl, H : H + self.nyl, :] = nodes[
                    i0 : i0 + self.nxl, j0 : j0 + self.nyl, :
                ]
            out[rank] = ext
        return out

    def gather(self, locals_, comp=None):
        """dict rank -> extended local array  ->  global node array."""
        H = self.H
        r0 = next(iter(locals_.values()))
        nodes = torch.zeros(self.nx, self.ny, r0.shape[2], dtype=r0.dtype, device=r0.device)
        for rank, ext in locals_.items():
            i0, j0 = self.origin(rank)
            nodes[i0 : i0 + self.nxl, j0 : j0 + self.nyl, :] = ext[
                H : H + self.nxl, H : H + self.nyl, :
            ]
        return nodes

    # -- views for the production z-solves (interior [1:n+1] <-> owned) ----

    def zview(self, ext, comp):
        """Monolithic-style ghost-shaped VIEW for solve_implicit_diffusion_*
        called with nx=nxl, ny=nyl. Interior [1:n+1] lands exactly on the
        owned region; the width-1 'ghosts' are halo data."""
        H, nxl, nyl = self.H, self.nxl, self.nyl
        if comp == "u":
            return ext[H - 1 : H + nxl, H - 1 : H + nyl + 1, :]
        if comp == "v":
            return ext[H - 1 : H + nxl + 1, H - 1 : H + nyl, :]
        return ext[H - 1 : H + nxl + 1, H - 1 : H + nyl + 1, :]  # w, p

    def dview(self, ext, comp):
        """Views aligned so compute_divergence(nx=nxl, ny=nyl) returns the
        divergence of exactly the owned cells."""
        H, nxl, nyl = self.H, self.nxl, self.nyl
        if comp == "u":
            return ext[H : H + nxl + 1, H - 1 : H + nyl + 1, :]
        if comp == "v":
            return ext[H - 1 : H + nxl + 1, H : H + nyl + 1, :]
        return ext[H - 1 : H + nxl + 1, H - 1 : H + nyl + 1, :]  # w

    # -- owned-region slices ------------------------------------------------

    def owned(self, ext):
        H = self.H
        return ext[H : H + self.nxl, H : H + self.nyl, :]
