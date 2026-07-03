# slChannel — Semi-Lagrangian DNS of Turbulent Channel Flow

GPU-oriented research DNS solver for incompressible channel flow using
**unconditionally stable high-order semi-Lagrangian advection** + FFT pressure
projection, forked from [torChannel]. Instead of an advective CFL limit
(AB2 at CFL ~ 0.28), the timestep is set by physical accuracy (trajectory
CFL 2-5, dt+ < 0.4), targeting a multi-x wall-clock reduction per simulated
time unit on a single GPU. Research question: do near-wall turbulence
statistics survive the interpolation dissipation?

## Quick start

    python main.py configs/config180_sl.yaml          # SL run, Re_tau = 180
    python main.py configs/config180_ref.yaml         # Eulerian reference
    for t in tests/test_*.py; do python "$t"; done    # test suite (CPU ok)
    PYTORCH_JIT=0 python bench/bench_interp.py        # GPU microbenchmark

## Scheme (one step)

1. Trajectory velocity V^{n+1/2} = 1.5 V^n - 0.5 V^{n-1}
2. Departure points per staggered component: iterated midpoint rule,
   analytic inverse tanh map for the stretched-z stencil location
3. Tricubic/triquintic Lagrange interpolation of V^n at departure points
   (nonuniform-node weights in z, periodic modulo in x,y, one-sided at walls)
4. Explicit xy-diffusion + bulk forcing; Crank-Nicolson implicit z-diffusion
5. FFT-based pressure projection (exactly divergence-free each step)

`sl.time_scheme: v2` upgrades to a characteristic-consistent 2nd-order step
(pressure-in-predictor + trajectory-CN + increment projection); with
`sl.traj_interp_order: 4` the measured temporal self-convergence ratio is
3.93 (ideal 4).

See CLAUDE.md for architecture notes and conventions.
