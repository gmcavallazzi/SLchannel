# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

The user-facing documentation is the source of truth; do not duplicate it here:

- `README.md` — what this is, install, quick start
- `docs/ARCHITECTURE.md` — module map, grid conventions, the SL scheme, the
  numerical subtleties that are easy to break silently
- `docs/CONFIG.md` — every YAML key, its type, default and meaning
- `docs/TESTING.md` — what each test establishes, markers, how to add one
- `docs/PROVENANCE.md` — the six modules inherited from torChannel
- `docs/REPRODUCING.md` — from clone to the published numbers
- `report/sl_dns_report.tex` — theory, literature, and §3.2, the full discrete
  algorithm substep by substep

## Working here

```bash
pip install -e ".[dev]"
pytest -m "not slow"                     # ~10 s
pytest                                   # ~2 min
slchannel configs/demo_sl_re180.yaml     # self-contained demo
```

On the GB10 GPU always export `CC=gcc PYTORCH_JIT=0` — the TorchScript fuser
and Triton's launcher build both fail otherwise on sm_121.

The GPU is slurm-managed: check `squeue` and `nvidia-smi` before running
anything heavy, and queue long runs with `sbatch`
(`examples/slurm/run_case.sbatch`).

## Things that bite

- **Report wall units.** Give the user dt⁺, z⁺, t⁺ — scaled with u_τ and ν —
  not raw code units. For the KMM closed case t⁺ = 11.6·t and one washout is
  Lx/U_b = 12.566 time units.
- **The Triton fast paths fail open.** They are selected inside `try/except`
  and fall back to eager with a printed message, so a broken Triton path costs
  an order of magnitude in speed and fails nothing. `tests/test_semilag_triton.py`
  asserts the path is actually enabled; keep it that way.
- **Env flags are read at import.** `slchannel/env.py` caches them. Mutating
  `os.environ` afterwards is a no-op unless you call `env.refresh()`.
- **The inherited modules are not to be reformatted.** `operators.py`,
  `projection.py`, `tridiag.py`, `utils.py`, `initflow.py`, `turbstats.py` are
  kept diffable against the parent solver; ruff is configured to skip them.
- **z-interpolation weights** must be built against the actual `z_c`/`z_f`
  nodes. `z_c` are face midpoints, not the tanh-map image of uniform points, so
  uniform-ξ weights would silently lose an order while still looking convergent.
- **Constant dt.** BDF2 re-bootstraps with one BDF1 step on any dt change; pin
  it with `dt_update_interval: 0`.
- **Field npz files are disposable.** Authoritative sources are the fetchable
  seed (`tools/fetch_data.py`) and CaNS checkpoints via `tools/cans_to_npz.py`.
- **Keep the repo root clean.** Run outputs go under `results/`, figures under
  `figures/`; both are gitignored. Archive small artifacts rather than leaving
  run directories around.
