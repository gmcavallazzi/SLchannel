# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

slChannel is a research DNS solver for incompressible turbulent channel flow that replaces the CFL-limited explicit advection of its parent code (torChannel, at `/home/giorgio/torChannel`) with **unconditionally stable high-order semi-Lagrangian advection**. dt is then limited only by physical accuracy (trajectory CFL 2–5, dt⁺ ≲ 0.4) instead of CFL ≈ 0.28, targeting a net wall-clock win per simulated time unit on a single GPU. The open research question: do near-wall turbulence statistics survive the SL interpolation dissipation?

Python/PyTorch, `torch.float64` throughout. Infrastructure modules (`operators.py`, `projection_fft.py`, `projection.py`, `tridiag.py`, `utils.py`, `initflow.py`, `turbstats.py`) are copies from torChannel @ a10e8e8 — keep them in sync conceptually; the new physics lives in `semilag.py` and `solver.py`. Local divergences from torChannel: `projection_fft.py` pins the singular (kx=0, ky=0) Neumann–Neumann pressure mode (torChannel still has the latent NaN — port when convenient), and `eulerian_triton.py` adds a Triton kernel for the Eulerian explicit RHS (fair-comparison baseline; auto-enabled on CUDA for `advection.scheme: eulerian`, disable with `SLCHANNEL_TRITON=0`).

## Commands

```bash
# Run a simulation
python main.py configs/config180_sl.yaml

# Full test suite (standalone scripts, no pytest; each prints [PASS]/[FAIL])
for t in tests/test_*.py; do python "$t"; done

# GPU interpolation microbenchmark (M0)
PYTORCH_JIT=0 python bench/bench_interp.py

# Compare statistics across runs
python scripts/compare_stats.py resultsA/turbulence_stats.npz resultsB/turbulence_stats.npz --labels A B
```

On the GB10 GPU always run with `CC=gcc PYTORCH_JIT=0` (TorchScript fuser and Triton's launcher build both break otherwise on sm_121). Opt-in perf layers (same env vars as torChannel): `TORCHANNEL_COMPILE=1` and `TORCHANNEL_POISSON_CUDAGRAPH=1`.

**Performance tiers of the SL advection** (measured at 768×768×180 on the GB10): hand-written Triton kernels (`semilag_triton.py`, auto-enabled for `interp_dtype: fp32_accum64` + `traj_interp_order: 2` on CUDA, disable with `SLCHANNEL_TRITON=0`) — 126 ms tricubic / 143 ms quintic for the full 3-component advect; torch.compile fused graphs (`TORCHANNEL_COMPILE=1`) — ~420 ms fp32; eager fp64 reference — ~40 s (do not use on GPU; it's the CPU-test path). Key GB10 facts baked into this design: fp64 flops are 1/64 of fp32 (flop-dense interpolation must be fp32; bandwidth-bound stencils are fine in fp64), and Inductor materializes multi-use (N,order) weight tensors (~10 GB traffic) unless `realize_*_threshold` are raised — the Triton kernels keep everything register-resident instead.

**End-to-end step cost** (768×768×180, GB10, `bench/bench_step.py`, 2026-07-04): Eulerian IMEX 1.29 s/step (Triton RHS 57 ms; eager fused RHS is 132 ms), SL v1 1.69 s/step, SL v2 3.53 s/step. Per simulated time unit at the Re550 operating dts (0.005 Eulerian / 0.013 SL): SL v1 ≈ 2.0× faster, SL v2 ≈ 0.95× (not yet a win). Both schemes are dominated by shared fp64 infrastructure: FFT-Poisson 408 ms (CUDA-graphed serial Thomas — PCR would cut it), implicit z-diffusion 3×190 ms, projection 101 ms. The v2 gap over v1 is ~0.9 s of eager `diffusion_* - diffusion_xy_*` z-RHS stencils plus ~0.25 s extra Triton gathers and ~0.1 s pressure-gradient work — a fused z-Laplacian kernel is the next target.

## Architecture

**`semilag.py` — the new core.** `SLAdvector` advances each staggered velocity component by tracing the characteristic back from the arrival face (iterated midpoint rule, trajectory velocity = AB2 extrapolation to t^{n+1/2}) and interpolating the old field at the departure point (tensor-product Lagrange: tricubic `order=4` or triquintic `order=6`). Key internals:
- x,y uniform → closed-form weights; z stretched → **nonuniform-node weights against the actual `z_c`/`z_f` nodes** (uniform-ξ weights would silently lose order: `z_c` are face midpoints, not the tanh-map image).
- z stencil located by the **analytic inverse tanh map** + one node compare (no searchsorted). Periodic x,y via modulo; one-sided stencils at walls; departure z clamped (counted in `n_clamped_last`, printed as the `clamped` diagnostic column).
- `IndexWeights` (flat gather indices + weights) is reusable across fields on the same component grid — v2 interpolates the RHS with the same stencils via `advect(..., extra_rhs=[...])`.
- `traj_order=2` (trilinear) trajectory sampling is the fast default but its C⁰ interpolant caps overall convergence at O(dt) with a small h² coefficient; `traj_order=4` restores clean O(dt²) (verified: self-convergence ratio 3.93).
- `sl.field_interp: "spline"` — C² field remap replacing the Lagrange gather: prefiltered cubic B-spline in x,y (FFT symbol division over the buffer's exact period) + nonuniform C² cubic spline in z (batched tridiagonal for node derivatives, Lagrange-cubic-clamped ends → cubic-exact, no natural-BC wall loss; Hermite evaluation). Coefficients stored interleaved (`qbuf[..., 2k]=c_k, 2k+1=m_k`) so the 4-point z gather is contiguous and reuses `_gather_interp` unchanged. Motivation: Lagrange interpolants of any order are C⁰ across faces; under repeated remap the kinks scatter energy into a high-k spectral floor ~25× the Eulerian tail (M3 finding, precision-independent). No Triton path yet (compiled/eager only).

**`solver.py` — `SLChannelFlow`.** `advection.scheme: "sl"` selects `step_sl`; `"eulerian"` selects the torChannel-identical IMEX reference (`step_imex`) for like-for-like comparisons. SL step: BCs → AB2 mid-velocity → SL advection → explicit xy-diffusion + forcing → CN implicit z-diffusion (PCR) → FFT projection → bulk-forcing relaxation controller.
- `sl.time_scheme: "v1"` — simple/robust, globally O(dt) (like the parent code's splitting).
- `sl.time_scheme: "v2"` — 2nd order: explicit terms **and the AB-extrapolated pressure gradient** averaged between departure and arrival, xy-RHS time-centered via AB2, z-diffusion as trajectory-CN (departure half explicit + arrival half implicit via `theta=1, dt/2`), projection solves a pressure **increment** on the extrapolated pressure (half-level history `_P_curr/_P_prev`). Pitfall (verified analytically and numerically): subtracting the old-pressure gradient AFTER the diffusion solve is algebraically identical to non-incremental — the pressure must enter the predictor before the implicit solve.
- `CFL_target` means **trajectory CFL** (2–5) under SL; dt additionally capped by `dt_max` (set to the dt⁺≲0.4 physics limit) and the explicit xy-diffusion stability bound (`time.diff_stability_C`, default 0.2 — non-binding at DNS resolutions).

**Grid/BC conventions** are torChannel's exactly: staggered MAC, one ghost layer per side, u:(nx+1,ny+2,nz+2) etc., interior `[1:n+1]`, periodic x,y, walls in z (bottom always no-slip; top dirichlet/neumann), tanh-stretched z (`symmetric`/`bottom` only — no hybrid/double grids here).

**Known accuracy subtleties** (documented in tests): restart re-bootstraps the AB2 trajectory extrapolation (one O(dt²) step, not bit-exact); SL self-convergence at fixed grid saturates at O(dt) unless `traj_order=4` (trilinear flow-map effect); the bulk-forcing controller has its own O(dt) dynamics — freeze `solver.forcing` when measuring temporal order.

## Milestones / configs

- `configs/config180_sl.yaml` — Re_τ=180 research run (SL); `config180_ref.yaml` — Eulerian baseline, same grid; sweep CFL {1,2,3,4} × {cubic,quintic} and compare statistics (`scripts/compare_stats.py`).
- `configs/config550_sl_ab.yaml` — Re_τ=550 A/B restart from the torChannel checkpoint (same 768×768×180 grid) for the wall-clock benchmark.
- `bench/bench_interp.py` — M0 GPU cost of the SL machinery; break-even math in its docstring.
