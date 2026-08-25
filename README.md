# slChannel — Semi-Lagrangian DNS of turbulent channel flow

[![CI](https://github.com/gmcavallazzi/SLchannel/actions/workflows/ci.yml/badge.svg)](https://github.com/gmcavallazzi/SLchannel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

GPU DNS solver for incompressible turbulent channel flow that replaces
CFL-limited explicit advection with **unconditionally stable second-order
semi-Lagrangian advection along characteristics** (Boukir et al. 1997, BDF2),
combined with implicit wall-normal diffusion and an FFT pressure projection.

The timestep is then set by physical accuracy — trajectory CFL 2–5,
`dt+ <= 0.25` — instead of by an advective CFL of about 0.28. On a closed
KMM channel at the exact bulk Reynolds number `Re_b = 2792.8` (256x256x256),
this is **2.65x faster per simulated time unit** than the equivalent Eulerian
solver on the same GPU, with triquintic interpolation matching the
Moser–Kim–Mansour reference statistics to within sampling noise
(`Re_tau` +0.4%, peak `u'_rms+` -0.2%).

## Install

```bash
git clone https://github.com/gmcavallazzi/SLchannel && cd SLchannel
pip install -e .
```

The Triton fast paths use the `triton` that PyTorch's Linux CUDA wheels already
bundle — normally there is nothing extra to install. On a CPU-only PyTorch they
are skipped automatically.

Requires Python >= 3.10 and PyTorch >= 2.1. The solver runs in
`torch.float64` throughout; a GPU with usable fp64 throughput is strongly
recommended for production runs (the code runs on CPU, fine for the tests and
the demo).

## Quick start

```bash
slchannel configs/demo_sl_re180.yaml     # self-contained demo, ~2 min on CPU
pytest -m "not slow"                     # test suite, ~10 s, CPU
```

The demo needs no external data. It runs a coarse channel with the production
scheme and should hold `max(div) < 1e-11` at every step with the bulk velocity
pinned at 1.0. It relaminarises — it demonstrates that the solver works, not
turbulence.

The validation case (`configs/kmm180_sl_bdf2_quintic.yaml`) restarts from a
stored field; fetch it first with `python tools/fetch_data.py kmm180_seed`, and
see `docs/REPRODUCING.md`.

## The scheme, one step

The SL step is the Boukir et al. (1997) BDF2-characteristics scheme:

1. Freeze the trajectory velocity `U* = 2 V^n - V^(n-1)`.
2. Trace **one** backward characteristic per staggered face by the iterated
   midpoint rule, giving **two independent feet** at depths `dt` and `2 dt`.
   The far foot is integrated from the arrival point, never continued from the
   near foot — continuing it drops the scheme to first order.
3. Interpolate `V^n` and `V^(n-1)` at their respective feet by tensor-product
   Lagrange interpolation — tricubic (`interp_order: 4`) or triquintic
   (`interp_order: 6`, the production choice). `x, y` are uniform and periodic;
   `z` is tanh-stretched, so the weights are built against the actual grid
   nodes and the stencil is located by the analytic inverse of the tanh map.
4. Update `(3 u^(n+1) - 4 ubar + ubarbar) / (2 dt)`, with explicit `x,y`
   diffusion and a Crank–Nicolson implicit `z`-diffusion solve at
   `dt_eff = 2 dt / 3`.
5. FFT pressure projection at the same `dt_eff`, then an exact
   divergence-free shift that pins the bulk flux (CaNS convention).

Constant `dt` is assumed: the scheme re-bootstraps with one BDF1 step on the
first step, on restart, and on any `dt` change. Pin `dt` in production configs
with `dt_update_interval: 0`.

The Eulerian IMEX scheme (`advection.scheme: eulerian`) is kept as a
like-for-like reference for testing and benchmarking.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `SLCHANNEL_TRITON` | `1` | Hand-written Triton gather kernels. Needs CUDA + `triton`; falls back to torch.compile/eager automatically. |
| `SLCHANNEL_COMPILE` | `0` | `torch.compile` (Inductor) layer. Needs `CC=gcc` on some systems. |
| `SLCHANNEL_POISSON_CUDAGRAPH` | `0` | CUDA-graph capture of the FFT Poisson solve. |

They are read **once, at import**, so set them in the shell before launching.
The resolved set is echoed in the run banner as `Performance layers: ...`.
The legacy `TORCHANNEL_*` spellings still work, with a deprecation warning.

On NVIDIA GB10 (sm_121) also export `CC=gcc PYTORCH_JIT=0` — the TorchScript
fuser and Triton's launcher build both fail otherwise.

## Documentation

| | |
|---|---|
| `docs/CONFIG.md` | Every YAML key: type, default, meaning. |
| `docs/ARCHITECTURE.md` | Module map, grid conventions, the scheme, and the subtleties that break silently. |
| `docs/TESTING.md` | What each test establishes; how to add one. |
| `docs/REPRODUCING.md` | From a clone to the published numbers. |
| `docs/PROVENANCE.md` | What was inherited from the parent solver, and what changed. |
| `report/sl_dns_report.pdf` | Theory, the literature, and the full discrete algorithm substep by substep (§3.2). |

## License and citation

MIT (see `LICENSE`).

If you use slChannel in published work, please cite it via `CITATION.cff` —
GitHub renders it as a "Cite this repository" button with BibTeX and APA forms.
The software DOI is minted on the first tagged release.

If you use the validation dataset, cite it separately:

> G. M. Cavallazzi, *seed field SLchannel Re_tau = 180*. Zenodo (CC-BY-4.0).
> https://doi.org/10.5281/zenodo.22099568

Contributions are welcome — see `CONTRIBUTING.md`.

Reference data used for validation is from Kim, Moin & Moser (1987) and
Moser, Kim & Mansour (1999), distributed by the Oden Institute, UT Austin.
It is downloaded, not redistributed here.
