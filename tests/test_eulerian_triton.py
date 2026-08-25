"""Triton Eulerian RHS vs the eager fused reference.

The fair-comparison baseline kernel (eulerian_triton.TritonEulerianRHS) must
reproduce operators.compute_momentum_rhs_fused_imex to reassociation-level
rounding on random ghost-filled fields, on both a symmetric and a bottom
stretched grid. Marked `gpu`: skipped, not silently passed, without CUDA.
"""

import pytest
import torch

from slchannel import operators
from slchannel.utils import generate_grid

# compares the Triton kernel against the eager reference
pytestmark = pytest.mark.gpu


def test_eulerian_triton(check):
    pytest.importorskip("triton", reason="Triton fast path needs triton")
    from slchannel.eulerian_triton import TritonEulerianRHS

    device = torch.device("cuda")
    torch.manual_seed(7)

    for nx, ny, nz, stretching in [
        (24, 20, 36, "symmetric"),
        (17, 19, 23, "bottom"),
        (32, 32, 48, "symmetric"),
    ]:
        Lx, Ly, Lz = 4.0, 2.0, 2.0
        dx, dy = Lx / nx, Ly / ny
        nu = 1.0 / 180.0
        z_f, z_c, dz_f, dz_c = generate_grid(2.0, nz, Lz, device=device, stretching_type=stretching)

        u = torch.randn(nx + 1, ny + 2, nz + 2, device=device)
        v = torch.randn(nx + 2, ny + 1, nz + 2, device=device)
        w = torch.randn(nx + 2, ny + 2, nz + 1, device=device)

        ref = operators.compute_momentum_rhs_fused_imex(u, v, w, nx, ny, nz, dx, dy, dz_c, dz_f, nu)
        tri = TritonEulerianRHS(nx, ny, nz, dx, dy, dz_c, dz_f, nu, device)(u, v, w)

        for comp, r, t in zip("uvw", ref, tri):
            scale = r.abs().max().item()
            err = (t - r).abs().max().item() / max(scale, 1e-30)
            check(
                f"rhs_{comp} {nx}x{ny}x{nz} {stretching}", err < 1e-13, f"rel max err = {err:.2e}"
            )
