"""Shared pytest fixtures for the slChannel suite.

The suite's value is in the *numbers* it measures — convergence ratios, clamp
counts, divergence magnitudes — not merely in whether an assertion held. The
`check` fixture below therefore records every measured quantity and prints the
familiar ``[PASS] name  detail`` table in the terminal summary, on success as
well as on failure, while still failing the test if any check is false.

Markers (declared in pyproject.toml):

``slow``
    Runs thousands of solver steps: minutes, not seconds.
``gpu``
    Needs CUDA. Auto-skipped elsewhere by :func:`pytest_collection_modifyitems`
    rather than passing vacuously.
"""

import os
import sys

import pytest
import torch

# the repo root, so tests can import the `parallel` package regardless of
# how pytest was invoked (the bare `pytest` binary does not put cwd on the
# path, unlike `python -m pytest`)
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from slchannel.semilag import SLAdvector  # noqa: E402  (path hook above)
from slchannel.utils import generate_grid  # noqa: E402

# Collected across the whole session so the summary can reprint everything.
_ALL_CHECKS = []


@pytest.fixture(scope="session", autouse=True)
def _double_precision():
    """The operators, grid and statistics all assume float64."""
    torch.set_default_dtype(torch.float64)


class Checker:
    """Soft assertions that keep their diagnostic string.

    Call it once per measured quantity; the test fails at teardown if any call
    was false, and every call is echoed in the terminal summary either way.
    """

    def __init__(self, nodeid):
        self.nodeid = nodeid
        self.results = []

    def __call__(self, name, ok, detail=""):
        ok = bool(ok)
        self.results.append((name, ok, detail))
        return ok

    @property
    def failures(self):
        return [(n, d) for n, ok, d in self.results if not ok]


@pytest.fixture
def check(request):
    checker = Checker(request.node.nodeid)
    yield checker
    _ALL_CHECKS.extend((checker.nodeid, n, ok, d) for n, ok, d in checker.results)
    if not checker.results:
        pytest.fail("test recorded no checks")
    if checker.failures:
        pytest.fail(
            "failed checks:\n  "
            + "\n  ".join(f"{name}  {detail}" for name, detail in checker.failures)
        )


@pytest.fixture
def advector():
    """Factory for an SLAdvector plus the grid it was built on.

    Returns a callable; every keyword mirrors the SLAdvector argument of the
    same name. The five interpolation tests all build essentially this object.
    """

    def build(
        n=32,
        nz=None,
        order=4,
        gamma=1.5,
        Lx=6.283185307179586,
        Ly=6.283185307179586,
        Lz=2.0,
        stretching_type="symmetric",
        **kwargs,
    ):
        nx = ny = n
        nz = n if nz is None else nz
        dx, dy = Lx / nx, Ly / ny
        z_f, z_c, _, _ = generate_grid(gamma, nz, Lz, stretching_type=stretching_type)
        adv = SLAdvector(
            nx,
            ny,
            nz,
            dx,
            dy,
            Lx,
            Ly,
            Lz,
            z_f,
            z_c,
            gamma,
            stretching_type=stretching_type,
            order=order,
            **kwargs,
        )
        return adv, dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, Lx=Lx, Ly=Ly, Lz=Lz, z_f=z_f, z_c=z_c)

    return build


@pytest.fixture
def config_file(tmp_path):
    """`make_config_file` bound to this test's temporary directory."""
    import functools

    from helpers import make_config_file

    return functools.partial(make_config_file, str(tmp_path))


def pytest_collection_modifyitems(config, items):
    """Skip `gpu`-marked tests when there is no CUDA device, rather than
    letting them report a vacuous pass."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs CUDA (run locally on a GPU box)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Reprint every measured quantity, in the suite's historical format."""
    if not _ALL_CHECKS:
        return
    tr = terminalreporter
    tr.write_sep("=", "measured quantities")
    last = None
    for nodeid, name, ok, detail in _ALL_CHECKS:
        head = nodeid.split("::")[0]
        if head != last:
            tr.write_line(head)
            last = head
        tr.write_line(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", green=ok, red=not ok)
