# Reproducing the results

Two routes: a five-minute check that the code is sound, and the full
validation run that produced the published numbers.

## The cheap route (~5 minutes, laptop)

```bash
pip install -e ".[dev]"
pytest                                    # 16 tests, 70 measured quantities
slchannel configs/demo_sl_re180.yaml      # ~2 min on CPU, seconds on a GPU
```

The test suite is the real evidence that the scheme is implemented correctly:
it checks the interpolation converges at O(h⁴) and O(h⁶) on the stretched grid,
that departure points are exact for flows whose characteristics are known
analytically, that a whole step reproduces the analytic Stokes decay rate, that
the SL and Eulerian schemes agree at small dt, and that SL self-converges in dt
at the expected Richardson ratio. `docs/TESTING.md` lists what each establishes.

Two supporting experiments, both self-contained and on CPU:

```bash
python examples/remap_gain.py     # the remap is strictly contractive: gain 0.86-0.91
python examples/floor_model_1d.py # the 17/9 two-foot gain, in a 1-D model
```

## The full validation (GPU, day-scale)

Closed channel at the exact KMM bulk Reynolds number Re_b = 2792.8, 256²×256
(Δx⁺ = 8.8), triquintic interpolation, dt⁺ = 0.25, one washout of warm-up then
70 washouts of statistics.

```bash
python tools/fetch_data.py mkm180                 # reference profiles
python tools/fetch_data.py kmm180_seed            # 548 MB restart field
slchannel configs/kmm180_sl_bdf2_quintic.yaml
python tools/compare_mkm.py results/kmm180_quintic/turbulence_stats.npz \
    --labels "SL bdf2 quintic" --output figures/kmm180_quintic
```

On a slurm cluster use `examples/slurm/run_case.sbatch` (edit the `#SBATCH`
placeholders for your site first).

Expected, against Moser, Kim & Mansour (1999):

| Quantity | Expected |
|---|---|
| Re_τ | within +0.4% of 178.12 |
| peak u′rms⁺ | within −0.2% |
| cost | ~336 ms/step at 256³ on a GB10 |

Both agreement figures are inside the sampling noise of a 70-washout window,
and are better than the Eulerian solver achieves on the same grid.

The seed is hosted on Zenodo (CC-BY-4.0), pinned by version DOI so the fetch
gets the exact bytes the published run started from; its SHA-256 is verified on
download. Cite it separately from the software:
<https://doi.org/10.5281/zenodo.22099568>

You can also generate your own: run any developed closed-channel case at
Re_b = 2792.8 on a 256³ grid to statistical steady state and point
`initialization.field_file` at its `fields.npz`, or convert a CaNS checkpoint
with `tools/cans_to_npz.py`. Statistics converge to the same answer; only the
warm-up cost differs.

## The speedup number

```bash
SLCHANNEL_COMPILE=1 SLCHANNEL_POISSON_CUDAGRAPH=1 CC=gcc PYTORCH_JIT=0 \
    python benchmarks/bench_step.py --case re180 --json bench.json
```

This times both schemes at their own operating timesteps and reports the
wall-clock per simulated time unit, which is the only comparison that means
anything — the SL step is more expensive and wins by being allowed a larger dt.
It also times the Eulerian RHS with and without its own Triton kernel, so the
baseline is demonstrably optimised as hard as the SL path.

Published figure: **2.65×** at 256³ (336 ms/step at dt⁺ = 0.25, against
203 ms/step at dt⁺ ≈ 0.057).

> The benchmark asserts that the Triton fast paths actually engaged. If they
> silently fall back to eager, the SL numbers are roughly an order of magnitude
> worse and the comparison is meaningless.

## Resolution and order dependence

The interpolation-dissipation bias, from the same case at other settings:

| Δx⁺ | interpolation | Re_τ vs MKM | peak u′rms⁺ |
|---|---|---|---|
| 11.8 | tricubic | −2.4% | +5.6% |
| 8.8 | tricubic | −1.0% | +3.4% |
| 8.8 | triquintic | +0.4% | −0.2% |

The bias is interpolation dissipation, and it is removable by order and
resolution rather than intrinsic to the scheme. What is intrinsic is temporal:
the spectral floor above dt⁺ ≈ 0.22 and the 17/9 two-foot gain that ends the
usable range near dt⁺ = 0.30.
