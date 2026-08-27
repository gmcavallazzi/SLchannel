"""Fixtures for the decomposition prototype suite.

`check`/`Checker` and `_double_precision` are copied from tests/conftest.py
(tests/ is not a package; a local copy is the least fragile option for this
self-contained prototype layer). The `dist` marker gates the real
torch.distributed tests: they only run with `-m dist` or SLC_RUN_DIST=1.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_ALL_CHECKS = []


@pytest.fixture(scope="session", autouse=True)
def _double_precision():
    torch.set_default_dtype(torch.float64)


class Checker:
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


def pytest_configure(config):
    config.addinivalue_line("markers", "dist: real multi-process torch.distributed test (opt-in)")


def pytest_collection_modifyitems(config, items):
    run_dist = os.environ.get("SLC_RUN_DIST") == "1" or "dist" in config.getoption("-m", "")
    skip_dist = pytest.mark.skip(reason="dist tests are opt-in (-m dist or SLC_RUN_DIST=1)")
    skip_gpu = pytest.mark.skip(reason="requires CUDA")
    for item in items:
        if "dist" in item.keywords and not run_dist:
            item.add_marker(skip_dist)
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


def pytest_terminal_summary(terminalreporter):
    if not _ALL_CHECKS:
        return
    terminalreporter.section("recorded checks (parallel)")
    for nodeid, name, ok, detail in _ALL_CHECKS:
        status = "ok  " if ok else "FAIL"
        terminalreporter.write_line(f"{status} {nodeid} :: {name}  {detail}")
