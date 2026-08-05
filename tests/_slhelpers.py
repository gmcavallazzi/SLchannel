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


def make_config_file(tmpdir, *, nx=16, ny=16, nz=32,
                     Lx=6.283185307179586, Ly=6.283185307179586, Lz=2.0,
                     Re=1000.0, Re_tau=180.0, U_bulk=1.0, gamma=1.5,
                     dt=0.01, t_max=1e9, n_steps=10**9,
                     scheme='sl', init_type='vortices', pert=0.05,
                     extra=None, name='config.yaml'):
    """Write a minimal slChannel YAML config for solver-level tests (CPU)."""
    import yaml, os
    cfg = {
        'grid': {'nx': nx, 'ny': ny, 'nz': nz},
        'domain': {'Lx': Lx, 'Ly': Ly, 'Lz': Lz},
        'flow': {'Re': Re, 'Re_tau': Re_tau, 'U_bulk': U_bulk, 'gamma': gamma},
        'initialization': {'type': init_type, 'perturbation_intensity': pert,
                           'n_vortices': 2},
        'solver': {'type': 'fft'},
        'time': {'dt': dt, 'n_steps': n_steps, 't_max': t_max,
                 'CFL_target': 3.0, 'dt_update_interval': 0,
                 'dt_max': 10 * dt, 'dt_min': dt / 100, 'scheme': 'IMEX'},
        'compute': {'device': 'cpu'},
        'output': {'results_folder': os.path.join(tmpdir, 'results'),
                   'n_out': 10**8, 'n_save': 10**8},
        'advection': {'scheme': scheme},
        'sl': {'interp_order': 4, 'n_traj_iters': 2, 'interp_dtype': 'fp64'},
        'statistics': {'n_stats': 0},
    }
    if extra:
        for section, values in extra.items():
            cfg.setdefault(section, {}).update(values)
    path = os.path.join(tmpdir, name)
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)
    return path
