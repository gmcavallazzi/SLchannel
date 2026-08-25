# Testing

```bash
pip install -e ".[dev]"

pytest                    # everything (~2.5 min)
pytest -m "not slow"      # inner loop (~10 s)
pytest -m gpu             # the CUDA-only tests, on a GPU box
pytest tests/test_sl_shear.py -q
```

Every test reports the **quantities it measured**, not just pass/fail. They are
reprinted in a table at the end of the run:

```
tests/test_interp_convergence.py
  [PASS] interp order=4  errors=[...] ratios=16.0,15.9 (ideal 16)
  [PASS] interp order=6  errors=[...] ratios=63.7,63.0 (ideal 64)
```

Those numbers are the point: a convergence ratio drifting from 16 toward 8 is a
regression even while the assertion still holds. CI archives them as JUnit XML
artifacts so they can be diffed across commits.

## What each test establishes

| Test | Establishes |
|---|---|
| `test_zmap_inverse` | The analytic inverse of the tanh z-map recovers face indices exactly, over both stretchings, three γ and two resolutions. |
| `test_interp_convergence` | The tensor-product Lagrange remap converges at O(h⁴) (tricubic) and O(h⁶) (triquintic) on the stretched grid, including near-wall one-sided stencils. |
| `test_departure_points` | Departure points are exact under uniform translation and linear shear, and O(dt³) locally for solid-body rotation. |
| `test_sl_shear` | Under an analytic shear the only error is interpolation, and it converges at the expected order. |
| `test_sl_solid_body` | The full `advect()` path is exact for a cubic-in-z field and converges on a smooth 3-D field, with no clamped feet. |
| `test_sl_wall_clamp` | Feet pushed through the walls are clamped and counted; fields stay bounded. |
| `test_sl_divergence` | Every SL BDF2 step returns a discretely divergence-free field (both interpolation orders) with no NaN. |
| `test_stokes_decay` | The whole step reproduces the analytic Stokes decay rate to 1%. |
| `test_poiseuille_sl` | *(slow)* The solver converges to the exact laminar parabola, holds it steady, and pins the bulk velocity. |
| `test_sl_vs_eulerian` | *(slow)* SL and the Eulerian reference agree at small dt, and SL self-converges in dt at the expected Richardson ratio. |
| `test_restart_sl` | A checkpoint-and-restart matches an uninterrupted run to the one-step BDF1 re-bootstrap allowance. |
| `test_stats_plane_mean` | Reynolds stresses are taken about the time mean, at each component's own nodes — with a built-in check that the test is not vacuous. |
| `test_eulerian_triton` | *(gpu)* The Triton Eulerian kernel reproduces the eager reference to rounding. |
| `test_semilag_triton` | *(gpu)* The Triton SL gather kernels reproduce the eager `advect()`, **and are actually enabled** — the fast path fails open, so a broken one costs an order of magnitude silently. |
| `test_results_guard` | `clean_results_on_fresh_start` refuses to empty any folder slChannel did not create, the working directory, an ancestor of it, or a home directory. |

## GPU coverage

Hosted CI runners have no GPU, so `gpu`-marked tests are **skipped** there —
they are not silently passed. The Triton paths (`semilag_triton.py`,
`eulerian_triton.py`) are therefore covered only by manual runs.

Last full GPU run: NVIDIA GB10 (sm_121), 2026-08-25, `pytest` all green:
16 tests, 70 measured quantities. Record the date and commit here with each
release.

## Adding a test

Take the `check` fixture and call it once per measured quantity — it collects
soft assertions, so every check runs and the test fails at the end if any was
false:

```python
def test_something(check):
    err = measure()
    check("what this shows", err < 1e-8, f"err={err:.2e}")
```

Mark it `slow` if it runs more than ~20 s, `gpu` if it needs CUDA. Solver-level
tests should take the `config_file` fixture, which writes a minimal CPU config
into a temporary directory; interpolation tests should take `advector`.
