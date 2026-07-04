"""M0 microbenchmark — semi-Lagrangian advection cost on the GPU.

Benchmarks the full 3-component advect() (departure points + high-order
interpolation) across the implementation tiers:

  triton    : hand-written Triton kernels (fp32 pipeline) — the production
              fast path; stencil weights/indices are register-resident.
  inductor  : torch.compile fused graphs (fp32 or fp64) — portable fallback.
  eager     : the bit-exact reference (fp64) — DO NOT run at production size
              (~40 s per call on the GB10; fp64 flops are 1/64 of fp32).

Run:  TORCHANNEL_COMPILE=1 CC=gcc PYTORCH_JIT=0 python bench/bench_interp.py [nx ny nz]

Context (GB10, 768x768x180): triton tricubic advect ~126 ms, quintic
~143 ms. NOTE: torChannel's "~49 ms/step" (commit a10e8e8) was measured at
128^3 Re180 — NOT at this grid. The honest end-to-end numbers at this size
live in bench/bench_step.py (2026-07-04: Eulerian IMEX step ~1.29 s with
the Triton RHS, SL v1 ~1.69 s, SL v2 ~3.53 s; at the Re550 operating dts
0.005 vs 0.013 SL v1 is ~2.0x faster per simulated time unit).
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
N_WARMUP, N_ITER = 3, 5


def timeit(fn, device):
    for _ in range(N_WARMUP):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITER * 1e3


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ''))
    print(f"Grid: {NX} x {NY} x {NZ} ({NX * NY * NZ / 1e6:.1f} M pts/component); "
          f"TORCHANNEL_COMPILE={os.environ.get('TORCHANNEL_COMPILE', '0')}\n")

    dx, dy = Lx / NX, Ly / NY
    z_f, z_c, _, _ = generate_grid(GAMMA, NZ, Lz, device=device, stretching_type='symmetric')
    u = 1.0 + 0.1 * torch.rand(NX + 1, NY + 2, NZ + 2, device=device)
    v = 0.2 * torch.rand(NX + 2, NY + 1, NZ + 2, device=device)
    w = 0.1 * torch.rand(NX + 2, NY + 2, NZ + 1, device=device)
    dt_t = torch.tensor(0.05, device=device)  # ~ trajectory CFL 3 displacement

    header = f"{'tier':<10} {'order':>5} {'dtype':<14} {'T_SL (ms)':>10}"
    print(header)
    print("-" * len(header))

    cases = [('triton', 4, 'fp32_accum64', '1'),
             ('triton', 6, 'fp32_accum64', '1'),
             ('inductor', 4, 'fp32_accum64', '0'),
             ('inductor', 4, 'fp64', '0')]
    for tier, order, interp_dtype, triton_flag in cases:
        os.environ['SLCHANNEL_TRITON'] = triton_flag
        adv = SLAdvector(NX, NY, NZ, dx, dy, Lx, Ly, Lz, z_f, z_c, GAMMA,
                         order=order, interp_dtype=interp_dtype, device=device)
        if tier == 'triton' and adv._triton is None:
            print(f"{tier:<10} {order:>5} {interp_dtype:<14} {'unavailable':>10}")
            continue
        t = timeit(lambda: adv.advect(u, v, w, u, v, w, dt_t), device)
        print(f"{tier:<10} {order:>5} {interp_dtype:<14} {t:>10.1f}", flush=True)
        del adv
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    os.environ.pop('SLCHANNEL_TRITON', None)


if __name__ == "__main__":
    main()
