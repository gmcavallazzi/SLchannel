# Configuration reference

A run is fully described by one YAML file:

```bash
slchannel configs/demo_sl_re180.yaml
```

Every key slChannel reads is listed below. Keys are grouped by section; a
missing key takes its default. Keys that do not appear here are **ignored
silently** — check spelling against this page.

Wall units referred to throughout: `dt+ = dt u_tau^2 / nu`,
`dx+ = dx u_tau / nu`, `t+ = t u_tau^2 / nu`.

---

## `grid` — resolution

| Key | Type | Default | Meaning |
|---|---|---|---|
| `nx` | int | *required* | Streamwise cells (periodic). |
| `ny` | int | *required* | Spanwise cells (periodic). |
| `nz` | int | *required* | Wall-normal cells (walls). Must exceed the interpolation stencil width. |

## `domain` — box and wall-normal stretching

| Key | Type | Default | Meaning |
|---|---|---|---|
| `Lx`, `Ly`, `Lz` | float | *required* | Box extents. `Lz` is the full channel height (2 for a half-height of 1). |
| `stretching_type` | `symmetric` \| `bottom` | `symmetric` | tanh grid clustered at both walls, or at the bottom only. |

## `flow` — physical parameters

| Key | Type | Default | Meaning |
|---|---|---|---|
| `Re` | float | *required* | Bulk Reynolds number; `nu = 1/Re`. |
| `Re_tau` | float | *required* | Target friction Reynolds number. Used for the wall-unit diagnostics and the statistics sampling height, not to drive the flow. |
| `U_bulk` | float | *required* | Bulk velocity held exactly by a divergence-free uniform shift each step. |
| `gamma` | float | *required* | tanh stretching strength. Larger clusters more points at the wall. |

## `boundary_conditions`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `top_wall.type` | `dirichlet` \| `neumann` | `dirichlet` | `dirichlet` = no-slip (closed channel). `neumann` = free-slip symmetry plane (open channel). The bottom wall is always no-slip. |

> The SL interpolation stencil is one-sided at a boundary and symmetry-blind. At
> a free-slip plane the correct extension is a mirror (`u,v` even, `w` odd),
> which is not implemented; near-surface `v,w` statistics in the open-channel
> configuration are affected. Prefer `dirichlet` for production.

## `advection` — which scheme

| Key | Type | Default | Meaning |
|---|---|---|---|
| `scheme` | `sl` \| `eulerian` | `sl` | `sl` is the production semi-Lagrangian BDF2 scheme. `eulerian` is the IMEX (AB2) reference, kept for like-for-like validation and benchmarking — not a production path. |

## `sl` — semi-Lagrangian parameters (used when `advection.scheme: sl`)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `interp_order` | `4` \| `6` | `4` | Field interpolation: tricubic or triquintic. This is the dominant control on interpolation dissipation and hence on statistics fidelity. `6` is the production choice. |
| `traj_interp_order` | `2` \| `4` | `2` | Order used to sample velocity *along the trajectory*. `2` (trilinear) is much faster but its C0 interpolant caps temporal convergence at O(dt); `4` restores clean O(dt²) and is what the convergence tests use. |
| `n_traj_iters` | int | `2` | Fixed-point iterations of the midpoint rule for the departure point. 2 suffices at trajectory CFL 2–5. |
| `interp_dtype` | `fp64` \| `fp32_accum64` | `fp64` | `fp32_accum64` enables the Triton fast path on CUDA (with `traj_interp_order: 2`) and is the production choice: interpolation is flop-dense and fp64 flops are heavily rate-limited on the target GPUs. |
| `bdf2_pressure` | `noninc` \| `inc` | `noninc` | `inc` (incremental) is formally O(dt²) — self-convergence ratio 4.00 — but more fragile. `noninc` is robust and caps velocity self-convergence near first order in the projection. `noninc` is the production default. |
| `bdf2_xy_rhs` | `extrap` \| `lagged` | `extrap` | `extrap` uses `2Rⁿ − Rⁿ⁻¹` at arrival (consistent at t^{n+1}); `lagged` uses `Rⁿ` (diagnostic). |

Keys removed in the bdf2-only cleanup — `sl.time_scheme`, `sl.traj_extrapolation`,
`sl.field_interp` — now raise with an explanatory message rather than being
ignored.

## `time` — stepping

| Key | Type | Default | Meaning |
|---|---|---|---|
| `dt` | float | *required* | Timestep. **BDF2 assumes constant dt**: any change re-bootstraps with one BDF1 step. Pin it with `dt_update_interval: 0`. |
| `dt_max` | float | `0.01` | Upper cap. Set it to the dt+ = 0.25 physics/stability limit for SL runs. |
| `dt_min` | float | `0.0001` | Lower cap. |
| `dt_update_interval` | int | `0` | Steps between adaptive-dt updates. **`0` disables adaptation — required for bdf2.** |
| `CFL_target` | float | *required* | Under `sl` this is the **trajectory** CFL (2–5 typical), not an advective CFL. Under `eulerian` it is the usual advective CFL (≤ 0.28). |
| `n_steps` | int | *required* | Step budget; the run stops at whichever of this and `t_max` comes first. |
| `stop_on_blowup` | bool | `true` | Stop when u_tau exceeds `blowup_u_tau_factor` x the nominal u_tau (Re_tau nu/delta) on three consecutive diagnostics. A blown SL run saturates at finite amplitude, so the NaN check never fires; this is the guard that ends it. The last state is saved to `fields_blowup.npz`. |
| `blowup_u_tau_factor` | float | `2.0` | Threshold factor for `stop_on_blowup`. Healthy u_tau fluctuations are a few percent; a blown run doubles. |
| `t_max` | float | `1000.0` | Simulation-time budget. |
| `scheme` | `IMEX` | `IMEX` | Only meaningful under `advection.scheme: eulerian`. |
| `diff_stability_C` | float | `0.2` | Safety factor on the explicit xy-diffusion stability bound. Non-binding at DNS resolutions. |

> **Why dt is capped even though SL is unconditionally stable.** Above
> dt+ ≈ 0.22 the streamwise spectrum develops a high-k floor, and at
> dt+ ≥ 0.30 the two independent BDF2 feet give a per-step tail-energy gain of
> 17/9 ≈ 1.89 that outruns the damping and diverges. dt+ = 0.25 is the
> production point.

## `initialization`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | `vortices` \| `parabolic` \| `file` \| `interpolate` | `parabolic` | How the initial field is built. |
| `field_file` | path | `None` | Restart field (`type: file`) or source field to interpolate (`type: interpolate`). |
| `reset_time` | bool | `false` | Restart from the stored field but reset `t` and the step counter to zero. |
| `perturbation_intensity` | float | `0.0` | Amplitude of the initial perturbation (`type: vortices`). |
| `n_vortices` | int | `4` | Number of seeded vortex pairs (`type: vortices`). |
| `source_half` | `lower` \| `upper` | `lower` | Which half of the source field to use when interpolating. |

> Setting `field_file` together with `type: vortices` restarts from the file and
> silently ignores the `vortices` setting. Use `type: file` when you mean a
> restart.

## `solver`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | `fft` | `fft` | FFT in the periodic directions plus a tridiagonal solve in z. The dense direct Poisson matrix was removed: it does not scale to DNS grids and no production case used it. |

## `output`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `results_folder` | path | `results` | Where everything is written. Created if absent and marked with a `.slchannel_results` sentinel. |
| `n_out` | int | `10` | Steps between progress lines. |
| `n_save` | int | `100` | Steps between checkpoint writes (`fields.npz`). |
| `n_snapshot` | int | `0` | Steps between numbered field snapshots. `0` disables. |
| `t_snapshot` | float | `0.0` | **Simulation-time** interval between snapshots. Preferred over `n_snapshot`: it is uniform in t+ across runs with different dt. `0` disables. |
| `clean_results_on_fresh_start` | bool | `false` | Empty `results_folder` on a non-restart start. Refuses when the folder is the working directory, an ancestor of it, the home directory, or lacks the `.slchannel_results` sentinel. |

## `statistics`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `true` | Explicit off switch. `n_stats: 0` also disables. |
| `n_stats` | int | `0` | Steps between statistics samples. `0` disables. |
| `t_stats` | float | `10.0` | Simulation time before sampling starts — the warm-up window. |
| `z_plus_target` | float | `15.0` | Wall-unit height at which spectra are taken (the near-wall peak). |
| `output_file` | str | `turbulence_stats.npz` | Finalised statistics, written into `results_folder`. |
| `state_file` | str | `turbulence_stats_state.npz` | Raw accumulators, so a restart can continue the same averaging window. |
| `restart_state_file` | path | `None` | Accumulator state to resume from. |

> Reynolds stresses are accumulated about the **time** mean at each component's
> own staggered nodes, not about the instantaneous plane mean.

## `compute`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `device` | `auto` \| `cuda` \| `cpu` | `auto` | `auto` picks CUDA when available. `cuda` errors if it is not. |

---

## Environment variables

These are read **once, when `slchannel` is imported**, so set them in the shell
before launching — setting them from Python afterwards has no effect.

| Variable | Default | Effect |
|---|---|---|
| `SLCHANNEL_TRITON` | `1` | Hand-written Triton gather kernels. Needs CUDA and `triton`; falls back to torch.compile/eager with a printed message. |
| `SLCHANNEL_COMPILE` | `0` | `torch.compile` (Inductor) on the launch-bound hot functions. |
| `SLCHANNEL_POISSON_CUDAGRAPH` | `0` | Capture and replay the FFT Poisson solve as a CUDA graph. |

The legacy `TORCHANNEL_COMPILE` / `TORCHANNEL_POISSON_CUDAGRAPH` spellings still
work but emit a `DeprecationWarning`.

On NVIDIA GB10 (sm_121) also export `CC=gcc PYTORCH_JIT=0`: the TorchScript
fuser cannot NVRTC-compile there, and Inductor's host compiler must not be
`nvc`, which rejects `-Wno-psabi`.

The resolved set is printed in the run banner as `Performance layers: ...`.
