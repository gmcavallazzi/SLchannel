# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this
project uses [semantic versioning](https://semver.org/).

## [1.0.0] — 2026-08-25

First public release: the version the published results were produced with.

### Features

- Second-order semi-Lagrangian advection along characteristics (Boukir et al.
  1997 BDF2), unconditionally stable in advection, so the timestep is set by
  accuracy (trajectory CFL 2–5, dt⁺ ≤ 0.25) rather than by CFL ≈ 0.28.
- Tensor-product Lagrange interpolation, tricubic or triquintic, with
  nonuniform-node weights against the actual stretched grid and an analytic
  inverse tanh map for stencil location.
- Hand-written Triton gather kernels keeping the interpolation weights
  register-resident; automatic fallback to torch.compile/eager.
- Eulerian IMEX scheme retained as a like-for-like validation reference.
- FFT + tridiagonal pressure projection with the singular Neumann–Neumann mode
  pinned; exact bulk-flux constraint by a uniform divergence-free shift.
- On-the-fly statistics about the time mean at each component's own staggered
  nodes.

### Validation

Closed KMM channel at the exact bulk Reynolds number Re_b = 2792.8, 256²×256,
70 washouts: Re_τ within +0.4% and peak u′rms⁺ within −0.2% of Moser, Kim &
Mansour (1999), and 2.65× faster per simulated time unit than the Eulerian
solver on the same GPU.

Test suite: 16 tests, 70 measured quantities, all passing. Last full GPU run on
NVIDIA GB10 (sm_121), 2026-08-25.

### Known limitations

- Single GPU; no domain decomposition.
- The interpolation stencil is one-sided and symmetry-blind at a boundary. At a
  free-slip plane the correct extension is a mirror, which is not implemented,
  so open-channel near-surface v,w statistics are affected. Prefer the closed
  channel.
- Advective-form SL conserves neither momentum nor energy. The bulk flux is
  pinned exactly, but second moments drift at the interpolation-dissipation
  level.
- dt⁺ ≥ 0.30 is unstable by construction: the two independent BDF2 feet give a
  per-step tail-energy gain of 17/9 that outruns the damping.

### Pre-history

Developed as a fork of the unpublished Eulerian solver torChannel; see
`docs/PROVENANCE.md`. Six infrastructure modules were imported verbatim in
commit `05a1b30`. Commit `85cb224` removed the experimental time schemes
(`v1`, `v2`, `pc`) and the C² spline interpolation path after the campaigns
settled on BDF2 — configs still carrying those keys raise an explanatory error.

[1.0.0]: https://github.com/gmcavallazzi/SLchannel/releases/tag/v1.0.0
