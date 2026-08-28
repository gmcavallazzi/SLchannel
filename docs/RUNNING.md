# Running slChannel

A self-contained guide for setting up and running the solver on a new
machine, single-GPU and multi-GPU. For the scheme itself see
`docs/ARCHITECTURE.md`; for every config key see `docs/CONFIG.md`; for the
published validation numbers see `docs/REPRODUCING.md`.

## Requirements

| | Requirement | Notes |
|---|---|---|
| Python | >= 3.10 | |
| PyTorch | >= 2.1 | CUDA build for GPU runs; the CPU build works for tests and the demo. |
| GPU | CUDA, usable fp64 throughput | The solver runs in `torch.float64` throughout. Data-centre GPUs (A100/H100/GH200/GB10) are fine; consumer GPUs run but fp64 is ~1/32 rate. |
| Triton | bundled with the PyTorch CUDA wheels | Nothing to install; used for the fast gather kernels, falls back to eager automatically. |
| Disk | ~5 GB per stored 3D field at 768x640x320; ~550 MB at 256^3 | Checkpoints are single `.npz` files. |

No MPI, no compiled extensions, no CMake: `pip install` is the whole build.

```bash
git clone https://github.com/gmcavallazzi/SLchannel && cd SLchannel
pip install -e ".[dev]"
pytest -m "not slow"     # ~10 s sanity check, CPU only
```

Machine-specific quirks:

- **NVIDIA GB10 (sm_121):** export `CC=gcc PYTORCH_JIT=0` — the TorchScript
  fuser and Triton's launcher build both fail otherwise.
- Performance env flags (`SLCHANNEL_TRITON`, `SLCHANNEL_COMPILE`,
  `SLCHANNEL_POISSON_CUDAGRAPH`, see the README table) are read **once at
  import**: set them in the shell before launching. The run banner echoes the
  resolved set as `Performance layers: ...` — check it, because a broken
  Triton path falls back silently to a ~10x slower eager path.

## Single GPU (the production path)

```bash
slchannel configs/demo_sl_re180.yaml        # self-contained demo, no data needed
```

A production case is one YAML file. The Re_tau = 180 validation case,
end to end:

```bash
python tools/fetch_data.py mkm180           # reference profiles
python tools/fetch_data.py kmm180_seed      # 548 MB restart field (Zenodo, sha256-verified)
slchannel configs/kmm180_sl_bdf2_quintic.yaml
python tools/compare_mkm.py results/kmm180_quintic/turbulence_stats.npz \
    --labels "SL bdf2 quintic" --output figures/kmm180_quintic
```

Higher-Re campaign configs (`configs/re395full_*.yaml`,
`configs/m950_*.yaml`) follow the same pattern and document their grids and
timesteps inline.

Operational features, all driven from the run directory:

- **Outputs** land in `results/<run_name>/`: `timeseries.npz` (bulk
  velocity, u_tau, live summed CFL per step), `turbulence_stats.npz`
  (finalized profiles and spectra), `fields.npz` (restart checkpoint),
  `grid.csv`.
- **Restart** is config-driven: point `initial_condition.field_file` at a
  checkpoint (and `statistics.restart_state_file` at the accumulator state
  to continue the same averaging window). BDF2 re-bootstraps with one BDF1
  step. `configs/m950_sl_dt020_cont.yaml` is a working example.
- **Graceful pause:** `touch results/<run_name>/STOP` — the solver
  checkpoints fields and statistics state, flushes the timeseries, and
  exits cleanly at the end of the current step. Remove the STOP file and
  relaunch to resume.
- **Slurm:** `examples/slurm/run_case.sbatch` is the single-job template
  (edit the `#SBATCH` placeholders for your site).
  `examples/slurm/run_m950_chain.sbatch` shows the long-campaign pattern:
  a singleton chain of jobs plus the STOP-file pause, a snapshot janitor
  that keeps a rotating buffer of restart fields, and a statistics ladder
  for retroactive averaging windows.

## Multi GPU (validated prototype)

`parallel/` contains a z-aligned pencil decomposition of the full solver
step: the domain is split `px x py` in the periodic directions, each rank
keeps all of z, so the implicit z-diffusion and tridiagonal Poisson solves
are rank-local and the semi-Lagrangian trajectories need only a fixed halo
(width from the CFL bound; overflow raises instead of wrapping). It is
**validated, not yet a production driver**:

- The decomposed step matches the single-GPU solver **bitwise (0 ulp)** on
  CPU for every tested rank grid, and bitwise on GPU per step with the
  localized Triton kernels. Over long chaotic horizons GPU runs decorrelate
  through last-bit FFT differences (cuFFT `rfft2` vs the pencil
  `rfft`/`fft` composition); a 150-time-unit Re_tau = 180 replication
  agrees with the single-GPU twin in all statistics (peaks within 0.5%,
  octave-band spectra within a few percent — i.e. within sampling noise).
- Communication backends: an in-process emulator (all ranks in one
  process, no launcher needed) and `torch.distributed` with gloo
  (CPU-staged). NCCL on a single shared GPU needs CUDA MPS and is opt-in
  via `SLC_DIST_BACKEND=nccl`; multi-node NCCL is future work, as is the
  production driver integration.

To verify the decomposition on a new machine:

```bash
pytest parallel/tests                              # emulated backend, CPU
torchrun --nproc_per_node=2 parallel/run_dist_test.py   # real multi-process (gloo)
```

To run the full-physics replication case (4 ranks emulated, one GPU):

```bash
python parallel/run_re180_4rank.py --mode 4rank --px 2 --py 2 \
    --config configs/kmm180_sl_bdf2_quintic.yaml --out results/re180_4rank
python parallel/run_re180_4rank.py --mode mono --out results/re180_mono
python parallel/compare_re180.py results/re180_4rank results/re180_mono
```

For production science today, use the single-GPU path.

## Troubleshooting

- `pytest tests/test_semilag_triton.py` asserts the Triton fast path is
  genuinely enabled on your GPU — run it after install on any CUDA machine.
- If you mutate `os.environ` inside Python, call `slchannel.env.refresh()`;
  the flags are cached at import.
- `dt` must be constant for clean BDF2: pin it with
  `dt_update_interval: 0` in production configs.
