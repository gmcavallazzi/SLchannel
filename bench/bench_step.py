"""End-to-end step benchmark — Eulerian IMEX vs semi-Lagrangian (M4 prep).

Times, at the production Re_tau=550 grid (768 x 768 x 180):

  1. the Eulerian explicit RHS alone: eager fused kernel vs the Triton
     fast path (eulerian_triton) — the fair-comparison requirement is that
     the baseline is engineered as hard as the SL advector;
  2. full solver steps: step_imex (Eulerian) and step_sl (v1 and v2,
     tricubic fp32_accum64 Triton pipeline);
  3. the wall-clock-per-simulated-time ratio at each scheme's operating
     dt (Re550: Eulerian CFL 0.28 -> dt ~ 0.005; SL dt+ <= 0.4 cap ->
     dt = 0.013).

Run:  TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 CC=gcc \
      PYTORCH_JIT=0 python bench/bench_step.py [nx ny nz]

Timing protocol: 3 warmup + 10 timed steps, cuda-synchronized. Field
content is a smooth vortices init — step cost is content-independent.
"""

import sys, os, time, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml

torch.set_default_dtype(torch.float64)

NX, NY, NZ = 768, 768, 180
if len(sys.argv) == 4:
    NX, NY, NZ = map(int, sys.argv[1:4])

DT_EULERIAN = 0.005   # Re550 advective CFL ~ 0.28
DT_SL = 0.013         # Re550 dt+ ~ 0.4 physical-accuracy cap
N_WARMUP, N_ITER = 3, 10


def make_config(tmpdir, scheme, sl_time_scheme='v1'):
    cfg = {
        'grid': {'nx': NX, 'ny': NY, 'nz': NZ},
        'domain': {'Lx': 15.707, 'Ly': 6.282, 'Lz': 2.0},
        'flow': {'Re': 10000.0, 'Re_tau': 550.0, 'U_bulk': 1.0, 'gamma': 2.6},
        'advection': {'scheme': scheme},
        'sl': {'interp_order': 4, 'traj_interp_order': 2, 'n_traj_iters': 2,
               'time_scheme': sl_time_scheme, 'interp_dtype': 'fp32_accum64'},
        'initialization': {'type': 'vortices', 'perturbation_intensity': 0.05,
                           'n_vortices': 4},
        'solver': {'type': 'fft'},
        'time': {'dt': 0.005, 'n_steps': 10**9, 't_max': 1e9,
                 'CFL_target': 3.0, 'dt_update_interval': 0,
                 'dt_max': 0.013, 'dt_min': 1e-4, 'scheme': 'IMEX'},
        'compute': {'device': 'auto'},
        'output': {'results_folder': os.path.join(tmpdir, 'results'),
                   'n_out': 10**8, 'n_save': 10**8},
        'statistics': {'n_stats': 0},
    }
    path = os.path.join(tmpdir, f'bench_{scheme}_{sl_time_scheme}.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)
    return path


def timeit(fn):
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITER * 1e3


def main():
    if not torch.cuda.is_available():
        print("CUDA required for this benchmark.")
        return
    from solver import SLChannelFlow

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Grid: {NX} x {NY} x {NZ}; "
          f"TORCHANNEL_COMPILE={os.environ.get('TORCHANNEL_COMPILE', '0')}, "
          f"POISSON_CUDAGRAPH={os.environ.get('TORCHANNEL_POISSON_CUDAGRAPH', '0')}\n")

    tmpdir = tempfile.mkdtemp(prefix='slbench_')
    results = {}

    # ---- Eulerian: RHS microbench + full step --------------------------
    flow = SLChannelFlow(make_config(tmpdir, 'eulerian'))
    assert flow._triton_eul is not None, "Triton Eulerian RHS did not enable"

    t_rhs_triton = timeit(lambda: flow.compute_momentum_rhs_explicit_imex())
    tri = flow._triton_eul
    flow._triton_eul = None
    t_rhs_eager = timeit(lambda: flow.compute_momentum_rhs_explicit_imex())
    flow._triton_eul = tri

    print(f"Eulerian explicit RHS (advection + xy-diffusion, fp64):")
    print(f"  eager fused kernel : {t_rhs_eager:9.2f} ms")
    print(f"  Triton kernel      : {t_rhs_triton:9.2f} ms   "
          f"({t_rhs_eager / t_rhs_triton:.1f}x)\n")

    results['imex_triton'] = timeit(lambda: flow.step_imex(DT_EULERIAN))
    flow._triton_eul = None
    results['imex_eager'] = timeit(lambda: flow.step_imex(DT_EULERIAN))
    flow._triton_eul = tri
    del flow
    torch.cuda.empty_cache()

    # ---- SL steps -------------------------------------------------------
    for ts in ('v1', 'v2'):
        flow = SLChannelFlow(make_config(tmpdir, 'sl', ts))
        results[f'sl_{ts}'] = timeit(lambda: flow.step_sl(DT_SL))
        del flow
        torch.cuda.empty_cache()

    # ---- summary ---------------------------------------------------------
    print("Full solver step (ms):")
    print(f"  Eulerian IMEX, eager RHS  : {results['imex_eager']:9.2f}")
    print(f"  Eulerian IMEX, Triton RHS : {results['imex_triton']:9.2f}")
    print(f"  SL v1 (tricubic, Triton)  : {results['sl_v1']:9.2f}")
    print(f"  SL v2 (tricubic, Triton)  : {results['sl_v2']:9.2f}\n")

    cost_e = results['imex_triton'] / DT_EULERIAN
    print(f"Wall-clock per simulated time unit (Re550 operating points: "
          f"dt_E={DT_EULERIAN}, dt_SL={DT_SL}):")
    print(f"  Eulerian IMEX (Triton RHS): {cost_e / 1000:9.2f} s")
    for ts in ('v1', 'v2'):
        cost_sl = results[f'sl_{ts}'] / DT_SL
        print(f"  SL {ts}                     : {cost_sl / 1000:9.2f} s   "
              f"(speedup vs Eulerian: {cost_e / cost_sl:.2f}x)")


if __name__ == "__main__":
    main()
