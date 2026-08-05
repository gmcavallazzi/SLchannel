# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

slChannel is a research DNS solver for incompressible turbulent channel flow that replaces the CFL-limited explicit advection of its parent code (torChannel, at `/home/giorgio/torChannel`) with **unconditionally stable second-order semi-Lagrangian advection** (Boukir et al. 1997 BDF2 characteristics). dt is then limited only by physical accuracy (trajectory CFL 2–5, dt⁺ ≲ 0.25) instead of CFL ≈ 0.28. **Verdict (2026-08-05, closed KMM channel at exact Re_b = 2792.8, 256²×256, same GPU):** 2.65× faster than torChannel per simulated time unit, with quintic interpolation matching MKM statistics to within sampling noise (Re_τ +0.4%, peak u′rms⁺ −0.2%) — better than the Eulerian twin. Cubic at the same grid loses ~1% u_τ and +3.4% on the u′ peak; at Δx⁺ = 11.8 it loses ~3% — interpolation dissipation is the entire SL statistics bias and it is removable by order/resolution. The remaining intrinsic limit is temporal: the AB2/extrapolation spectral floor above dt⁺ ≈ 0.22 and the two-foot stochastic gain 17/9 that blows bdf2 up at dt⁺ ≥ 0.30 (production point: dt⁺ = 0.25).

Python/PyTorch, `torch.float64` throughout. Infrastructure modules (`operators.py`, `projection_fft.py`, `projection.py`, `tridiag.py`, `utils.py`, `initflow.py`, `turbstats.py`) are copies from torChannel @ a10e8e8 — keep them in sync conceptually; the new physics lives in `semilag.py` and `solver.py`. Local divergences from torChannel: `projection_fft.py` pins the singular (kx=0, ky=0) Neumann–Neumann pressure mode (torChannel still has the latent NaN — port when convenient), `eulerian_triton.py` adds a Triton kernel for the Eulerian explicit RHS, statistics accumulate about the TIME mean at each component's own staggered nodes, and the bulk flux is imposed exactly by a uniform divergence-free shift (CaNS convention; the forcing is diagnosed, not steered).

**2026-08-05 cleanup:** the experimental SL time schemes (v1, v2, pc/none trajectory extrapolation) and the C² spline field-interpolation path were removed after the campaigns settled the design: **bdf2 is the only SL scheme**; interpolation is tensor-product Lagrange, tricubic (`interp_order: 4`) or triquintic (`interp_order: 6`, the production choice). The Eulerian IMEX scheme is kept as the like-for-like testing reference. Configs with `sl.time_scheme`/`sl.traj_extrapolation`/`sl.field_interp` now raise. History (mechanism studies, the M3/fix campaign analysis, the removed schemes) is in git before commit series of 2026-08-05 and in `report/sl_dns_report.tex`.

## Commands

```bash
# Run a simulation
python main.py configs/config_kmm180_tc256q_sl_bdf2.yaml

# Full test suite (standalone scripts, no pytest; each prints [PASS]/[FAIL])
for t in tests/test_*.py; do python "$t"; done

# GPU interpolation microbenchmark (M0)
PYTORCH_JIT=0 python bench/bench_interp.py

# Compare statistics across runs
python scripts/compare_stats.py resultsA/turbulence_stats.npz resultsB/turbulence_stats.npz --labels A B
```

On the GB10 GPU always run with `CC=gcc PYTORCH_JIT=0` (TorchScript fuser and Triton's launcher build both break otherwise on sm_121). Opt-in perf layers (same env vars as torChannel): `TORCHANNEL_COMPILE=1` and `TORCHANNEL_POISSON_CUDAGRAPH=1`.

**Performance:** hand-written Triton gather kernels (`semilag_triton.py`, auto-enabled for `interp_dtype: fp32_accum64` + `traj_interp_order: 2` on CUDA, disable with `SLCHANNEL_TRITON=0`) keep the 3(P+1) interpolation weights register-resident — 126 ms tricubic / 143 ms triquintic for the full 3-component advect at 768²×180. Key GB10 facts baked into this design: fp64 flops are 1/64 of fp32 (flop-dense interpolation must be fp32; bandwidth-bound stencils are fine in fp64), and Inductor materializes multi-use (N,order) weight tensors (~10 GB traffic) unless kept in registers. **End-to-end at 256³ (measured on production runs, 2026-08-05):** SL bdf2 quintic 336 ms/step at dt⁺ = 0.25 vs torChannel Eulerian 203 ms/step at dt⁺ ≈ 0.057 → 15.6 vs 41.4 s per simulated t.u. = **2.65×**; quintic costs only +2.6%/step over cubic (the step is dominated by shared fp64 infrastructure: FFT-Poisson, implicit z-solves, projection — a fused z-Laplacian kernel and PCR Poisson are the remaining wall-clock targets).

## Architecture

**`semilag.py` — the SL advector.** `SLAdvector` advances each staggered velocity component by tracing the characteristic back from the arrival face (iterated midpoint rule, K=2 iterations) and interpolating the old field at the departure point (tensor-product Lagrange: tricubic `order=4` or triquintic `order=6`). Key internals:
- x,y uniform → closed-form weights; z stretched → **nonuniform-node weights against the actual `z_c`/`z_f` nodes** (uniform-ξ weights would silently lose order: `z_c` are face midpoints, not the tanh-map image).
- z stencil located by the **analytic inverse tanh map** + one node compare (no searchsorted). Periodic x,y via modulo; one-sided stencils at walls; departure z clamped (counted in `n_clamped_last`, printed as the `clamped` diagnostic column).
- `IndexWeights` (flat gather indices + weights) is reusable across fields on the same component grid.
- `traj_order=2` (trilinear) trajectory sampling is the fast default but its C⁰ interpolant caps overall convergence at O(dt) with a small h² coefficient; `traj_order=4` restores clean O(dt²) (used by the convergence tests).

**`solver.py` — `SLChannelFlow`.** `advection.scheme: "sl"` selects the BDF2-characteristics step (`step_sl_bdf2`); `"eulerian"` selects the torChannel-identical IMEX reference (`step_imex`) for like-for-like comparisons and testing. The bdf2 step (Boukir et al. 1997): ONE characteristic over [t^{n−1},t^{n+1}] traced with frozen U* = 2Vⁿ−Vⁿ⁻¹, two independent feet at depths dt and 2dt (never continue the far foot from the near foot — order drops to 1, paper Remark 4i), update (3u^{n+1}−4ū+ū̄)/2dt with θ=1 z-solve and projection at dt_eff = 2dt/3, then the exact bulk-flux shift. Options: `bdf2_pressure: noninc|inc` (inc = O(dt²), self-convergence ratio 4.00; noninc robust, ratio 2.0 — production default), `bdf2_xy_rhs: extrap|lagged`. Constant dt assumed — BDF1 re-bootstrap on first step/restart/dt change; **pin dt** (`dt_update_interval: 0`) in bdf2 configs. `CFL_target` means **trajectory CFL** (2–5) under SL; dt additionally capped by `dt_max` (set it to the dt⁺ = 0.25 physics/stability limit) and the explicit xy-diffusion stability bound (`time.diff_stability_C`, default 0.2 — non-binding at DNS resolutions). The full discrete algorithm, substep by substep with equations, is §3.2 of `report/sl_dns_report.tex`.

**Grid/BC conventions** are torChannel's exactly: staggered MAC, one ghost layer per side, u:(nx+1,ny+2,nz+2) etc., interior `[1:n+1]`, periodic x,y, walls in z (bottom always no-slip; top dirichlet/neumann), tanh-stretched z (`symmetric`/`bottom` only — no hybrid/double grids here).

**Known accuracy subtleties** (documented in tests): restart re-bootstraps with one BDF1 step (one O(dt²) step, not bit-exact); SL self-convergence at fixed grid saturates at O(dt) unless `traj_order=4` (trilinear flow-map effect); the exact bulk-flux shift is applied inside the step — tests measuring temporal order monkeypatch `solver._apply_bulk_forcing = lambda dt: (0.0, 0.0)`.

## Configs

- `configs/config_kmm180_tc256q_sl_bdf2.yaml` — **the production reference**: closed KMM channel, exact Re_b = 2792.8, 256²×256 (Δx⁺ = 8.8), quintic, dt⁺ = 0.25, restart from the torChannel 256³ field. Its cubic twin `config_kmm180_tc256_sl_bdf2.yaml` and the coarser `config_kmm180_sl_bdf2.yaml` / `config_kmm180_tc192_sl_bdf2.yaml` (192², Δx⁺ = 11.8) quantify the interpolation-dissipation bias. Launch scripts `run_kmm_*.sh` (sbatch, with automatic MKM-comparison analysis into `figures_fix/`).
- `configs/config180_ref.yaml` — Eulerian baseline (testing reference).
- `configs/config550_sl_ab.yaml` — Re_τ=550 A/B restart from the torChannel checkpoint (768×768×180) for the at-scale benchmark. NOTE: its torChannel restart path is currently stale.
- `bench/bench_interp.py` — M0 GPU cost of the SL machinery; break-even math in its docstring.
- MKM reference data: `data/mkm_chandata/chan180/` (KMM87/MKM99, exact Re_b computed from chan180.means); comparison drivers `scripts/plot_stats_torstyle.py` (torChannel house style, **requires `module load texlive`**) and `scripts/compare_mkm.py`.

## Data layout

- **No `results_*` dirs are kept in the repo root.** Archived stats from the M3 campaign live in `data/m3_stats/<run>.npz` (analysis scripts point there; `ref.npz` is the Eulerian reference). New runs write under `results_fix/<run>/`. Field npz files are always disposable: authoritative restart sources are the CaNS checkpoints (`/home/giorgio/CaNS_DRL/run*/data`, converted via `scripts/cans_to_npz.py`) and the torChannel closed-channel fields (`/home/giorgio/torChannel/results_re180_closed*`).
- `output.t_snapshot` (sim time units) writes `fields_t*.npz` at fixed simulation-time thresholds — uniform in t⁺ across runs with different dt (step-based `n_snapshot` is not).
- GPU is slurm-managed: check `squeue` before running; queue long runs with `sbatch`; report quantities to the user in wall (+) units (KMM closed case: t⁺ = 11.6·t, washout = Lx/U_b = 12.566 t.u.).
