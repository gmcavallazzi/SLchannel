"""Triton Eulerian RHS vs the eager fused reference.

The fair-comparison baseline kernel (eulerian_triton.TritonEulerianRHS) must
reproduce operators.compute_momentum_rhs_fused_imex to reassociation-level
rounding on random ghost-filled fields, on both a symmetric and a bottom
stretched grid. CUDA-only (Triton); prints [SKIP] and exits 0 on CPU.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils import generate_grid
import operators

torch.set_default_dtype(torch.float64)


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}  {detail}")
    return ok


def run():
    if not torch.cuda.is_available():
        print("[SKIP] test_eulerian_triton: CUDA not available")
        return True
    from eulerian_triton import TritonEulerianRHS

    device = torch.device('cuda')
    ok = True
    torch.manual_seed(7)

    for nx, ny, nz, stretching in [(24, 20, 36, 'symmetric'),
                                   (17, 19, 23, 'bottom'),
                                   (32, 32, 48, 'symmetric')]:
        Lx, Ly, Lz = 4.0, 2.0, 2.0
        dx, dy = Lx / nx, Ly / ny
        nu = 1.0 / 180.0
        z_f, z_c, dz_f, dz_c = generate_grid(2.0, nz, Lz, device=device,
                                             stretching_type=stretching)

        u = torch.randn(nx + 1, ny + 2, nz + 2, device=device)
        v = torch.randn(nx + 2, ny + 1, nz + 2, device=device)
        w = torch.randn(nx + 2, ny + 2, nz + 1, device=device)

        ref = operators.compute_momentum_rhs_fused_imex(
            u, v, w, nx, ny, nz, dx, dy, dz_c, dz_f, nu)
        tri = TritonEulerianRHS(nx, ny, nz, dx, dy, dz_c, dz_f, nu, device)(u, v, w)

        for comp, r, t in zip('uvw', ref, tri):
            scale = r.abs().max().item()
            err = (t - r).abs().max().item() / max(scale, 1e-30)
            ok &= report(f"rhs_{comp} {nx}x{ny}x{nz} {stretching}",
                         err < 1e-13, f"rel max err = {err:.2e}")

    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
