# Architecture

slChannel is a research DNS solver for incompressible plane channel flow. It
replaces the CFL-limited explicit advection of a conventional staggered-grid
solver with **unconditionally stable second-order semi-Lagrangian advection**
along characteristics (Boukir et al. 1997, BDF2), so the timestep is limited by
physical accuracy — trajectory CFL 2–5, dt⁺ ≲ 0.25 — instead of by an advective
CFL of about 0.28.

Everything is `torch.float64` except the interpolation arithmetic, which is
deliberately fp32-with-fp64-accumulation on GPUs where fp64 throughput is a
small fraction of fp32.

## Modules

| Module | Role |
|---|---|
| `semilag.py` | **The novel physics.** `SLAdvector`: departure points, interpolation weights, the gather. |
| `semilag_triton.py` | Hand-written Triton gather kernels for the fp32 fast path. |
| `solver.py` | `SLChannelFlow`: config, time loop, both step functions, I/O, restart. |
| `operators.py` | Staggered-MAC stencils, fused IMEX RHS, implicit z-diffusion. |
| `projection.py` | FFT + tridiagonal pressure Poisson solve and the velocity correction. |
| `tridiag.py` | Batched tridiagonal solve by parallel cyclic reduction. |
| `eulerian_triton.py` | Triton kernel for the Eulerian RHS — so the *baseline* is optimised as hard as the SL path and the comparison is fair. |
| `initflow.py` | Initial conditions and restart. |
| `turbstats.py` | On-the-fly statistics. |
| `utils.py` | Grid, diagnostics, checkpoint I/O. |
| `env.py` | The one place environment flags are read. |
| `cli.py` | `slchannel <config.yaml>`. |

Six of these are inherited verbatim from the parent Eulerian solver; see
`PROVENANCE.md` for why that matters and what was changed.

## Grid and boundary conventions

Staggered MAC with one ghost layer per side: `u` at x-faces
`(nx+1, ny+2, nz+2)`, `v` at y-faces, `w` at z-faces, pressure at cell centres.
The interior is `[1:n+1]`. x and y are periodic; z is wall-bounded and
tanh-stretched (`symmetric` or `bottom`). The bottom wall is always no-slip;
the top is no-slip or free-slip.

## The semi-Lagrangian advector

Each velocity component is advanced by tracing the characteristic back from its
own arrival face and interpolating the old field at the departure point.

**Departure points.** Iterated midpoint rule
`x_m ← x_a − ½ dt V(x_m)`, then `x_d = 2 x_m − x_a`, with `V` the frozen
time-extrapolated trajectory velocity supplied by the solver. Two iterations
suffice at trajectory CFL 2–5. Trajectory sampling is trilinear by default; it
is cheap but its C⁰ interpolant caps overall convergence at O(dt), so the
convergence tests use `traj_interp_order: 4`.

**Interpolation.** Tensor-product Lagrange, tricubic or triquintic. x and y are
uniform, so the weights are closed-form polynomials of the fractional offset.

> **z is the subtle one.** The weights must be built against the *actual* node
> coordinates. `z_c` are arithmetic means of the faces, **not** the image of
> uniform computational points under the tanh map, so uniform-ξ weights in z
> would silently lose an order — the code would still run and still look
> convergent in x and y.

The z stencil is *located* by the analytic inverse of the tanh map plus one
comparison against the actual node — exact regardless of the centre-vs-face
subtlety, and no search. Periodic wrap in x and y is a modulo; stencils go
one-sided at the walls; departure z is clamped just inside the wall and the
number of clamped points is reported every step as a diagnostic.

`IndexWeights` (flat gather indices plus weights) is reusable across every
field living on the same component grid, which is what makes the three-component
advect affordable.

## One BDF2 step

1. Freeze the trajectory velocity `U* = 2Vⁿ − Vⁿ⁻¹`.
2. Trace **one** characteristic over `[t^{n−1}, t^{n+1}]`, giving two feet at
   depths `dt` and `2dt`.

   > Each foot is integrated independently from the arrival point. Continuing
   > the far foot from the near one drops the scheme to first order (Boukir et
   > al., Remark 4i).

3. Interpolate `Vⁿ` and `Vⁿ⁻¹` at their respective feet.
4. Update `(3u^{n+1} − 4ū + ū̄)/2dt`, explicit xy-diffusion, θ=1 implicit
   z-solve at `dt_eff = 2dt/3`.
5. Project at the same `dt_eff`; then shift uniformly and divergence-freely to
   pin the bulk flux exactly.

Constant dt is assumed — any change re-bootstraps with one BDF1 step. The full
algebra, substep by substep, is section 3.2 of the report.

## Why dt is capped despite unconditional stability

Two distinct effects, both absent from the deterministic literature analyses:

* Above dt⁺ ≈ 0.22 the streamwise spectrum at z⁺ ≈ 15 develops a high-k floor.
  The injection is the *trajectory-velocity extrapolation*, not interpolation
  error — better interpolants raise the floor, because interpolation error is
  what damps it.
* The two independent feet combine as `⁴⁄₃ū − ⅓ū̄`, and decorrelated phases add
  in energy rather than amplitude, giving a per-step tail gain of
  `(4/3)² + (1/3)² = 17/9 ≈ 1.89`. Once that outruns the per-step damping the
  floor stops being a floor. In practice: clean at dt⁺ = 0.25, monotonic
  blow-up at dt⁺ ≥ 0.30.

## Performance shape

The interpolation is flop-dense and the rest is bandwidth-bound, which on the
development GPU (fp64 at 1/64 of fp32) dictates the split: interpolation in
fp32-accumulate-64 through Triton kernels that keep the `3(P+1)` weights
register-resident, everything else in fp64. A compiler will otherwise
materialise the multi-use weight tensors and spend ~10 GB of traffic on them.

At 256³ the SL step costs about 1.3–2.1× an Eulerian step but is allowed ~4×
the timestep, which is where the 2.65× per-simulated-time-unit win comes from.
The remaining wall-clock is shared fp64 infrastructure — FFT-Poisson and the
implicit z-solves — not the SL machinery.

## Known limitations

See the "Known limitations" section of `../CHANGELOG.md`.
