"""M0 microbenchmark — semi-Lagrangian advection cost on the GPU at the
Re_tau=550 production size (768 x 768 x 180, fp64).

Measures, per full 3-component advect() call (departure points + high-order
interpolation), for tricubic/triquintic and fp64/fp32_accum64:
  - T_SL: total advection time
  - departure-only time (trajectory sampling dominates it)
plus the effect of n_traj_iters and traj_interp_order.

Context: torChannel's whole IMEX step at this size runs at ~49 ms/step
(TORCHANNEL_COMPILE=1 + Poisson CUDA graph). The SL solver replaces the
advection kernels (a small share of those 49 ms) with T_SL and runs at
trajectory CFL 2-5 instead of 0.28. Break-even: T_SL + ~49 ms per step must
beat (CFL_ratio) x 49 ms per unit simulated time.

Run:  PYTORCH_JIT=0 python bench/bench_interp.py [nx ny nz]
"""

import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils import generate_grid
from semilag import SLAdvector

torch.set_default_dtype(torch.float64)

NX, NY, NZ = 768, 768, 180
if len(sys.argv) == 4:
    NX, NY, NZ = map(int, sys.argv[1:4])

Lx, Ly, Lz = 15.707, 6.282, 2.0
GAMMA = 2.6
N_WARMUP, N_ITER = 3, 10


def make_fields(nx, ny, nz, z_c, z_f, dx, dy, device):
    """Smooth synthetic velocity fields with realistic magnitudes (u ~ 1)."""
    def mesh(comp):
        if comp == 'u':
            X = (torch.arange(0, nx + 1, dtype=torch.float64, device=device) * dx).view(-1, 1, 1)
            Y = ((torch.arange(0, ny + 2, dtype=torch.float64, device=device) - 0.5) * dy).view(1, -1, 1)
            Z = z_c.view(1, 1, -1)
        elif comp == 'v':
            X = ((torch.arange(0, nx + 2, dtype=torch.float64, device=device) - 0.5) * dx).view(-1, 1, 1)
            Y = (torch.arange(0, ny + 1, dtype=torch.float64, device=device) * dy).view(1, -1, 1)
            Z = z_c.view(1, 1, -1)
        else:
            X = ((torch.arange(0, nx + 2, dtype=torch.float64, device=device) - 0.5) * dx).view(-1, 1, 1)
            Y = ((torch.arange(0, ny + 2, dtype=torch.float64, device=device) - 0.5) * dy).view(1, -1, 1)
            Z = z_f.view(1, 1, -1)
        return X, Y, Z

    X, Y, Z = mesh('u')
    u = 1.0 + 0.3 * torch.sin(2 * math.pi * X / Lx) * torch.cos(4 * math.pi * Y / Ly) \
        * torch.sin(math.pi * Z / Lz)
    X, Y, Z = mesh('v')
    v = 0.2 * torch.sin(4 * math.pi * X / Lx) * torch.cos(2 * math.pi * Y / Ly) \
        * torch.sin(math.pi * Z / Lz) * torch.ones_like(X + Y + Z)
    X, Y, Z = mesh('w')
    w = 0.1 * torch.cos(2 * math.pi * X / Lx) * torch.sin(2 * math.pi * Y / Ly) \
        * torch.sin(math.pi * Z / Lz) ** 2 * torch.ones_like(X + Y + Z)
    return u, v, w


def timeit(fn, device):
    for _ in range(N_WARMUP):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            fn()
        torch.cuda.synchronize()
    else:
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            fn()
    return (time.perf_counter() - t0) / N_ITER * 1e3  # ms


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ''))
    print(f"Grid: {NX} x {NY} x {NZ}  ({NX * NY * NZ / 1e6:.1f} M points/component)\n")

    dx, dy = Lx / NX, Ly / NY
    z_f, z_c, _, _ = generate_grid(GAMMA, NZ, Lz, device=device, stretching_type='symmetric')
    u, v, w = make_fields(NX, NY, NZ, z_c, z_f, dx, dy, device)
    dt_t = torch.tensor(0.05, device=device)  # ~ CFL 3 displacement

    header = f"{'config':<44} {'T_SL(ms)':>10} {'depart(ms)':>11}"
    print(header)
    print("-" * len(header))

    for order in (4, 6):
        for interp_dtype in ('fp64', 'fp32_accum64'):
            for traj_order, n_iters in [(2, 2), (4, 2)]:
                adv = SLAdvector(NX, NY, NZ, dx, dy, Lx, Ly, Lz, z_f, z_c, GAMMA,
                                 order=order, traj_order=traj_order,
                                 n_traj_iters=n_iters,
                                 interp_dtype=interp_dtype, device=device)

                t_total = timeit(lambda: adv.advect(u, v, w, u, v, w, dt_t), device)

                adv._fill(adv.mbuf['u'], 'u', u)
                adv._fill(adv.mbuf['v'], 'v', v)
                adv._fill(adv.mbuf['w'], 'w', w)
                t_dep = timeit(lambda: [adv.compute_departure(c, dt_t) for c in 'uvw'],
                               device)

                label = (f"order={order} traj={traj_order} iters={n_iters} "
                         f"{interp_dtype}")
                print(f"{label:<44} {t_total:>10.1f} {t_dep:>11.1f}", flush=True)
                del adv
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    print("\nBreak-even guide (Re550 grid): torChannel step ~49 ms at CFL 0.28.")
    print("SL wall-clock win ~ (CFL_target/0.28) * 49 / (49 + T_SL) per unit time")
    print("(the ~49 ms Eulerian step keeps its Poisson+diffusion but drops its")
    print("advection kernels, so this slightly understates the win).")


if __name__ == "__main__":
    main()
