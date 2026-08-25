# Provenance

slChannel began as a fork of **torChannel**, an Eulerian staggered-grid channel
DNS solver by the same author. torChannel is unpublished; slChannel is
standalone and needs nothing from it.

## What was inherited

Six modules were imported verbatim in commit `05a1b30`
("Import verbatim torChannel modules") and carry a provenance header saying so:

| Module | Role |
|---|---|
| `slchannel/operators.py` | Staggered-MAC advection/diffusion stencils, fused IMEX momentum RHS, implicit z-diffusion solves |
| `slchannel/projection.py` | FFT + tridiagonal pressure Poisson solver and the velocity correction |
| `slchannel/tridiag.py` | Batched tridiagonal solve by parallel cyclic reduction |
| `slchannel/utils.py` | Grid construction, diagnostics, checkpoint I/O |
| `slchannel/initflow.py` | Initial conditions |
| `slchannel/turbstats.py` | On-the-fly turbulence statistics |

They are shared infrastructure: the semi-Lagrangian scheme changes *advection*
only, so keeping the rest identical to the Eulerian parent is what makes a
comparison between the two schemes a comparison of advection alone.

They are therefore **never reformatted** — `pyproject.toml` excludes them from
`ruff format` and from style-only lint rules, while correctness rules still
apply. Diffed against the parent solver, the only changes are:

* whole functions deleted, where slChannel does not use that code path;
* `import x` → `from . import x` for the package layout;
* the provenance header itself;
* the local divergences listed below.

Nothing is rewrapped or restyled, so the diff stays readable and the claim on
this page is checkable rather than asserted.

## What is new here

`slchannel/semilag.py`, `slchannel/semilag_triton.py` and the SL half of
`slchannel/solver.py` are original to slChannel: the departure-point solve,
the nonuniform-node Lagrange interpolation, the Triton gather kernels, and the
BDF2-characteristics step.

## Local divergences from the parent

* the singular `(kx=0, ky=0)` Neumann–Neumann pressure mode is pinned
  explicitly in `projection.py`;
* Reynolds stresses accumulate about the **time** mean at each component's own
  staggered nodes, not the instantaneous plane mean;
* the bulk flux is imposed exactly by a uniform divergence-free shift (CaNS
  convention); the driving force is diagnosed, not steered;
* the dense direct Poisson path, the hybrid/double-stretched grids, and other
  code paths slChannel does not exercise have been removed.

## Reference data

Validation uses the Moser–Kim–Mansour channel DNS dataset, distributed by the
Oden Institute, UT Austin. It is **downloaded, not redistributed** — see
`tools/fetch_data.py` and `data/README.md`. Please cite:

* J. Kim, P. Moin & R. Moser, *Turbulence statistics in fully developed channel
  flow at low Reynolds number*, J. Fluid Mech. **177**, 133–166 (1987).
* R. D. Moser, J. Kim & N. N. Mansour, *DNS of turbulent channel flow up to
  Re_tau = 590*, Phys. Fluids **11**, 943–945 (1999).
