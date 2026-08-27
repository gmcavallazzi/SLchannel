"""Shared builders for the decomposition tests (CPU, fp64, eager SL path)."""

import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (_REPO, os.path.join(_REPO, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from parallel.decomp import node_z_len, nodes_to_mono  # noqa: E402
from slchannel.semilag import SLAdvector  # noqa: E402
from slchannel.utils import generate_grid  # noqa: E402

TWO_PI = 6.283185307179586


def build_ref(
    nx=32, ny=32, nz=32, Lx=TWO_PI, Ly=TWO_PI, Lz=2.0, gamma=1.5, order=6, interp_dtype="fp64", **kw
):
    dx, dy = Lx / nx, Ly / ny
    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)
    adv = SLAdvector(
        nx,
        ny,
        nz,
        dx,
        dy,
        Lx,
        Ly,
        Lz,
        z_f,
        z_c,
        gamma,
        stretching_type="symmetric",
        order=order,
        interp_dtype="fp64",
        device=torch.device("cpu"),
        **kw,
    )
    adv._triton = None  # deterministic eager reference (test_semilag_triton pattern)
    if getattr(adv, "_advect_c", None) is not None:
        adv._advect_c = None
    grid = dict(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        z_f=z_f,
        z_c=z_c,
        dz_f=dz_f,
        dz_c=dz_c,
    )
    return adv, grid


def node_coords(comp, grid):
    """Node coordinate arrays (x (nx,), y (ny,), z (NZ,)) per component."""
    nx, ny, dx, dy = grid["nx"], grid["ny"], grid["dx"], grid["dy"]
    xs = torch.arange(nx, dtype=torch.float64)
    ys = torch.arange(ny, dtype=torch.float64)
    x = xs * dx if comp == "u" else (xs + 0.5) * dx
    y = ys * dy if comp == "v" else (ys + 0.5) * dy
    z = grid["z_f"] if comp == "w" else grid["z_c"]
    return x, y, z.to(torch.float64)


def analytic_nodes(comp, grid, fn):
    """Global node array of fn(x, y, z) on the component's nodes."""
    x, y, z = node_coords(comp, grid)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    return fn(X, Y, Z)


def default_fields(grid, seed=0):
    """Smooth periodic-in-x/y analytic velocity nodes for the three comps
    (w vanishing at the walls so the wall pinning matches)."""
    Lz = grid["Lz"]
    k1 = TWO_PI / grid["Lx"]
    k2 = TWO_PI / grid["Ly"]
    s = 0.15 + 0.05 * seed

    def fu(X, Y, Z):
        return 1.0 + s * torch.sin(k1 * X) * torch.cos(2 * k2 * Y) * torch.cos(TWO_PI * Z / Lz)

    def fv(X, Y, Z):
        return s * torch.cos(2 * k1 * X) * torch.sin(k2 * Y) * (1.0 + 0.3 * Z / Lz)

    def fw(X, Y, Z):
        zz = Z.clamp(0.0, Lz)
        return s * torch.sin(k1 * X + k2 * Y) * torch.sin(TWO_PI * zz / Lz) * 0.5

    return {
        "u": analytic_nodes("u", grid, fu),
        "v": analytic_nodes("v", grid, fv),
        "w": analytic_nodes("w", grid, fw),
    }


def mono_advect_nodes(adv, grid, nodes, mid_nodes, dt_t):
    """Run the monolithic eager advect on node arrays; return the results AS
    NODE ARRAYS (mapping the production arrival slots u[1:nx+1] etc. back to
    node numbering: x-roll for u, y-roll for v)."""
    nx, ny = grid["nx"], grid["ny"]
    U = nodes_to_mono(nodes["u"], "u", nx, ny)
    V = nodes_to_mono(nodes["v"], "v", nx, ny)
    W = nodes_to_mono(nodes["w"], "w", nx, ny)
    MU = nodes_to_mono(mid_nodes["u"], "u", nx, ny)
    MV = nodes_to_mono(mid_nodes["v"], "v", nx, ny)
    MW = nodes_to_mono(mid_nodes["w"], "w", nx, ny)
    us, vs, ws = adv.advect(U, V, W, MU, MV, MW, dt_t)
    nz = grid["nz"]
    out = {}
    out["u"] = torch.zeros(nx, ny, node_z_len("u", nz), device=us.device)
    out["u"][:, :, 1 : nz + 1] = torch.roll(us[1 : nx + 1, 1 : ny + 1, 1 : nz + 1], 1, dims=0)
    out["v"] = torch.zeros(nx, ny, node_z_len("v", nz), device=us.device)
    out["v"][:, :, 1 : nz + 1] = torch.roll(vs[1 : nx + 1, 1 : ny + 1, 1 : nz + 1], 1, dims=1)
    out["w"] = torch.zeros(nx, ny, node_z_len("w", nz), device=us.device)
    out["w"][:, :, 1:nz] = ws[1 : nx + 1, 1 : ny + 1, 1:nz]
    return out
