"""End-to-end step benchmark: Eulerian IMEX vs semi-Lagrangian BDF2.

This is the script behind the headline speedup. It measures three things:

  1. the Eulerian explicit RHS alone, eager fused kernel vs the Triton fast
     path -- the fair-comparison requirement being that the *baseline* is
     engineered as hard as the SL advector, so the comparison is of schemes
     and not of implementation effort;
  2. the cost of a full solver step under each scheme;
  3. the wall-clock per simulated time unit at each scheme's own operating
     dt, which is the only comparison that means anything: the SL step is
     more expensive, and wins by being allowed a much larger dt.

Run (CUDA required):

    SLCHANNEL_COMPILE=1 SLCHANNEL_POISSON_CUDAGRAPH=1 CC=gcc PYTORCH_JIT=0 \
        python benchmarks/bench_step.py --json bench.json

Timing protocol: 3 warm-up plus 10 timed steps, CUDA-synchronised. Step cost
is independent of field content, so a smooth vortex initial field is fine.
"""

import argparse
import json
import os
import tempfile
import time

import torch
import yaml

# Operating points. The Eulerian dt is set by its advective CFL (~0.28); the SL
# dt by physical accuracy and the two-foot stability limit (dt+ <= 0.25).
DEFAULTS = {
    "re180": dict(
        nx=256,
        ny=256,
        nz=256,
        Lx=12.566370614359172,
        Ly=6.283185307179586,
        Re=2792.8,
        Re_tau=180.0,
        gamma=1.6,
        dt_eulerian=0.0057,
        dt_sl=0.0215,
    ),
    "re550": dict(
        nx=768,
        ny=768,
        nz=180,
        Lx=15.707,
        Ly=6.282,
        Re=10000.0,
        Re_tau=550.0,
        gamma=2.6,
        dt_eulerian=0.005,
        dt_sl=0.013,
    ),
}
N_WARMUP, N_ITER = 3, 10


def make_config(tmpdir, case, scheme, interp_order):
    cfg = {
        "grid": {"nx": case["nx"], "ny": case["ny"], "nz": case["nz"]},
        "domain": {"Lx": case["Lx"], "Ly": case["Ly"], "Lz": 2.0},
        "flow": {"Re": case["Re"], "Re_tau": case["Re_tau"], "U_bulk": 1.0, "gamma": case["gamma"]},
        "advection": {"scheme": scheme},
        "sl": {
            "interp_order": interp_order,
            "traj_interp_order": 2,
            "n_traj_iters": 2,
            "interp_dtype": "fp32_accum64",
        },
        "initialization": {"type": "vortices", "perturbation_intensity": 0.05, "n_vortices": 4},
        "solver": {"type": "fft"},
        "time": {
            "dt": case["dt_eulerian"],
            "n_steps": 10**9,
            "t_max": 1e9,
            "CFL_target": 3.0,
            "dt_update_interval": 0,
            "dt_max": case["dt_sl"],
            "dt_min": 1e-4,
            "scheme": "IMEX",
        },
        "compute": {"device": "auto"},
        "output": {
            "results_folder": os.path.join(tmpdir, "results"),
            "n_out": 10**8,
            "n_save": 10**8,
        },
        "statistics": {"enabled": False, "n_stats": 0},
    }
    path = os.path.join(tmpdir, f"bench_{scheme}_{interp_order}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh)
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", choices=sorted(DEFAULTS), default="re180")
    ap.add_argument(
        "--grid",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        help="override the grid of the selected case",
    )
    ap.add_argument("--json", metavar="PATH", help="also write the measurements as JSON")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    from slchannel import env
    from slchannel.solver import SLChannelFlow

    case = dict(DEFAULTS[args.case])
    if args.grid:
        case["nx"], case["ny"], case["nz"] = args.grid

    dt_e, dt_sl = case["dt_eulerian"], case["dt_sl"]
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Case: {args.case}, grid {case['nx']} x {case['ny']} x {case['nz']}")
    print(f"Performance layers: {env.summary()}\n")

    tmpdir = tempfile.mkdtemp(prefix="slbench_")
    ms = {}

    # ---- Eulerian: RHS microbenchmark, then the full step ------------------
    flow = SLChannelFlow(make_config(tmpdir, case, "eulerian", 4))
    if flow._triton_eul is None:
        raise SystemExit(
            "the Triton Eulerian RHS did not enable; the comparison would be unfair to the baseline"
        )

    ms["rhs_triton"] = timeit(flow.compute_momentum_rhs_explicit_imex)
    triton_rhs = flow._triton_eul
    flow._triton_eul = None
    ms["rhs_eager"] = timeit(flow.compute_momentum_rhs_explicit_imex)

    print("Eulerian explicit RHS (advection + xy-diffusion, fp64):")
    print(f"  eager fused kernel : {ms['rhs_eager']:9.2f} ms")
    print(
        f"  Triton kernel      : {ms['rhs_triton']:9.2f} ms   "
        f"({ms['rhs_eager'] / ms['rhs_triton']:.1f}x)\n"
    )

    ms["step_eulerian_eager"] = timeit(lambda: flow.step_imex(dt_e))
    flow._triton_eul = triton_rhs
    ms["step_eulerian"] = timeit(lambda: flow.step_imex(dt_e))
    flow = None
    torch.cuda.empty_cache()

    # ---- SL steps, both interpolation orders -------------------------------
    for order, name in ((4, "cubic"), (6, "quintic")):
        flow = SLChannelFlow(make_config(tmpdir, case, "sl", order))
        ms[f"step_sl_{name}"] = timeit(lambda f=flow: f.step_sl_bdf2(dt_sl))
        flow = None
        torch.cuda.empty_cache()

    # ---- summary -----------------------------------------------------------
    print("Full solver step (ms):")
    print(f"  Eulerian IMEX, eager RHS  : {ms['step_eulerian_eager']:9.2f}")
    print(f"  Eulerian IMEX, Triton RHS : {ms['step_eulerian']:9.2f}")
    print(f"  SL bdf2, tricubic         : {ms['step_sl_cubic']:9.2f}")
    print(f"  SL bdf2, triquintic       : {ms['step_sl_quintic']:9.2f}\n")

    cost_e = ms["step_eulerian"] / dt_e / 1000.0
    print(f"Wall-clock per simulated time unit (dt_eulerian={dt_e}, dt_sl={dt_sl}):")
    print(f"  Eulerian IMEX             : {cost_e:9.2f} s")
    speedups = {}
    for name in ("cubic", "quintic"):
        cost = ms[f"step_sl_{name}"] / dt_sl / 1000.0
        speedups[name] = cost_e / cost
        print(f"  SL bdf2, tri{name:<9s}    : {cost:9.2f} s   ({speedups[name]:.2f}x vs Eulerian)")

    if args.json:
        payload = dict(
            case=args.case,
            grid=[case["nx"], case["ny"], case["nz"]],
            device=torch.cuda.get_device_name(0),
            dt_eulerian=dt_e,
            dt_sl=dt_sl,
            step_ms=ms,
            speedup=speedups,
            seconds_per_time_unit_eulerian=cost_e,
        )
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
