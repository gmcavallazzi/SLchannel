"""Shared helpers for the semi-Lagrangian tests: build ghost-shaped staggered
fields from analytic functions evaluated at every entry's physical position
(ghosts included), so interpolation tests need no BC machinery."""

import torch


def full_positions(comp, nx, ny, nz, dx, dy, z_f, z_c):
    """Broadcastable physical coordinates (X, Y, Z) of every entry of the
    ghost-shaped array of component `comp` ('u'|'v'|'w'|'p')."""
    if comp == 'u':
        X = (torch.arange(0, nx + 1, dtype=torch.float64) * dx).view(-1, 1, 1)
        Y = ((torch.arange(0, ny + 2, dtype=torch.float64) - 0.5) * dy).view(1, -1, 1)
        Z = z_c.view(1, 1, -1)
    elif comp == 'v':
        X = ((torch.arange(0, nx + 2, dtype=torch.float64) - 0.5) * dx).view(-1, 1, 1)
        Y = (torch.arange(0, ny + 1, dtype=torch.float64) * dy).view(1, -1, 1)
        Z = z_c.view(1, 1, -1)
    elif comp == 'w':
        X = ((torch.arange(0, nx + 2, dtype=torch.float64) - 0.5) * dx).view(-1, 1, 1)
        Y = ((torch.arange(0, ny + 2, dtype=torch.float64) - 0.5) * dy).view(1, -1, 1)
        Z = z_f.view(1, 1, -1)
    else:
        raise ValueError(comp)
    return X, Y, Z


def make_field(comp, fn, nx, ny, nz, dx, dy, z_f, z_c):
    """Ghost-shaped tensor for `comp` filled with fn(X, Y, Z) everywhere."""
    X, Y, Z = full_positions(comp, nx, ny, nz, dx, dy, z_f, z_c)
    shape = {'u': (nx + 1, ny + 2, nz + 2),
             'v': (nx + 2, ny + 1, nz + 2),
             'w': (nx + 2, ny + 2, nz + 1)}[comp]
    return (fn(X, Y, Z) * torch.ones(shape, dtype=torch.float64))


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}  {detail}")
    return ok
