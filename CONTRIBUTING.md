# Contributing

## Development install

```bash
git clone https://github.com/gmcavallazzi/SLchannel
cd SLchannel
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
```

## Tests

```bash
pytest -m "not slow"      # inner loop, ~10 s
pytest                    # everything, ~2 min
pytest -m gpu             # CUDA-only tests, on a GPU box
```

Every test reports the quantities it measured, not just pass/fail — see
`docs/TESTING.md` for how to write one and what each existing test establishes.
A convergence ratio drifting away from its ideal is a regression even when the
assertion still holds, so please check the printed table, not only the exit
code.

Anything touching the numerics should keep the measured values in the summary
table unchanged; say so in the PR if they move, and why.

## Style

```bash
ruff check .
ruff format .
```

**The six modules inherited from torChannel are exempt** from style rules and
from formatting: `operators.py`, `projection.py`, `tridiag.py`, `utils.py`,
`initflow.py`, `turbstats.py`. They are kept diffable against the parent
solver, which is what makes "imported verbatim" in `docs/PROVENANCE.md` a
checkable claim rather than a story. Reformatting them would destroy that.

Correctness rules still apply to them — `F821` undefined name, `F811`
redefinition — so a real bug is still caught. Only the stylistic pyflakes rules
(`F401` unused import, `F841` unused local, `F541` placeholder-free f-string)
are waived alongside `E`/`W`/`I`, because acting on those means editing lines
that should stay identical to upstream.

Everything else — `solver.py`, `semilag.py`, the Triton kernels, `cli.py`,
`env.py`, `tests/`, `tools/` — is formatted normally.

## Adding a configuration key

Read it in `SLChannelFlow.__init__`, validate it there with a clear `raise`,
and document it in `docs/CONFIG.md`. That file is the authoritative reference;
a key that is not in it is effectively undiscoverable.

## Scope

slChannel is a research solver for one problem: incompressible plane channel
flow with semi-Lagrangian advection. Contributions that sharpen that — better
interpolation, boundary treatment, parallelisation, performance — are very
welcome. Generalisations to other geometries are probably a different code.
