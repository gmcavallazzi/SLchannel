"""env.py — the one place slChannel reads environment flags.

Three optional performance layers are controlled by environment variables. They
are read **once, at package import**, and cached here, so that:

* a flag cannot mean one thing to a module that read it at import time and
  another to a module that read it when an object was constructed (that
  inconsistency was a real footgun in earlier versions), and
* the resolved values can be printed in the run banner, making every log
  self-describing.

Because they are read at import, setting one from inside Python after
``import slchannel`` has no effect — set them in the shell before launching, or
call :func:`refresh` explicitly (intended for benchmarks and tests).

Flags
-----
``SLCHANNEL_TRITON`` (default ``1``)
    Use the hand-written Triton gather kernels for the semi-Lagrangian
    interpolation and the Eulerian RHS. Requires CUDA and a working ``triton``;
    if either is missing the solver falls back to torch.compile/eager and says
    so.
``SLCHANNEL_COMPILE`` (default ``0``)
    Wrap the launch-bound hot functions in ``torch.compile`` (Inductor).
``SLCHANNEL_POISSON_CUDAGRAPH`` (default ``0``)
    Capture and replay the FFT Poisson solve as a CUDA graph.

The legacy ``TORCHANNEL_*`` spellings of the latter two are still accepted, with
a DeprecationWarning, and will be removed in a future release.
"""

import os
import warnings

_LEGACY_PREFIX = "TORCHANNEL_"
_PREFIX = "SLCHANNEL_"


def _read(name, default):
    """Read SLCHANNEL_<name>, falling back to the deprecated TORCHANNEL_<name>."""
    value = os.environ.get(_PREFIX + name)
    if value is None:
        legacy = os.environ.get(_LEGACY_PREFIX + name)
        if legacy is not None:
            warnings.warn(
                f"{_LEGACY_PREFIX}{name} is deprecated; use {_PREFIX}{name}.",
                DeprecationWarning,
                stacklevel=2,
            )
            value = legacy
    return default if value is None else value


def refresh():
    """Re-read every flag from the environment. Benchmarks and tests only."""
    global USE_TRITON, USE_COMPILE, USE_POISSON_CUDAGRAPH
    USE_TRITON = _read("TRITON", "1") == "1"
    USE_COMPILE = _read("COMPILE", "0") == "1"
    USE_POISSON_CUDAGRAPH = _read("POISSON_CUDAGRAPH", "0") == "1"


def summary():
    """One-line description of the active performance layers, for the banner."""
    on = [
        name
        for name, active in (
            ("triton", USE_TRITON),
            ("torch.compile", USE_COMPILE),
            ("poisson-cudagraph", USE_POISSON_CUDAGRAPH),
        )
        if active
    ]
    return ", ".join(on) if on else "none (plain eager)"


refresh()
