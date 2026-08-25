"""slChannel — semi-Lagrangian DNS of turbulent channel flow.

Incompressible channel-flow DNS on a staggered MAC grid, advancing advection
by second-order semi-Lagrangian characteristics (Boukir et al. 1997, BDF2)
instead of a CFL-limited explicit scheme, with implicit wall-normal diffusion
and an FFT pressure projection.

The whole solver runs in ``torch.float64``; importing this package sets it as
the default dtype, because the operators, grid construction and statistics all
assume it.

Typical use::

    from slchannel import SLChannelFlow
    SLChannelFlow(config_file="configs/demo_sl_re180.yaml").run_simulation()

or from the shell::

    slchannel configs/demo_sl_re180.yaml
"""

import torch

# The solver, its operators and the statistics all assume float64 tensors.
# Set it at import so library users get the same numerics as the CLI.
torch.set_default_dtype(torch.float64)

__version__ = "1.0.1"

# Imported after the dtype is set: the submodules build tensors at import time
# and must see float64.
from .semilag import SLAdvector  # noqa: E402
from .solver import SLChannelFlow  # noqa: E402

__all__ = ["SLChannelFlow", "SLAdvector", "__version__"]
