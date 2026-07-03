"""Batched tridiagonal solve via parallel cyclic reduction (PCR).

Pure PyTorch, dtype-generic (real or complex RHS), and launch-light: PCR needs
~ceil(log2(n)) vectorized steps instead of the ~2n sequential steps of a serial
Thomas sweep. This matters on GPUs where the JIT fuser is unavailable and each
sequential step is its own kernel launch (e.g. GB10/sm_121 with PYTORCH_JIT=0).

Solves  a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i  (a_0, c_{n-1} ignored).

Diagonals a, b, c may be 1-D (shape (n,), shared across the batch) or 2-D
(shape (batch, n)); the RHS d is (batch, n). PCR's reduction of a, b, c does not
depend on d, so 1-D diagonals stay 1-D throughout (cheaper).

PCR is stable for the diagonally dominant systems here (implicit diffusion and
the FFT-Poisson z-operator).
"""
import torch


def _shift(x: torch.Tensor, s: int, down: bool, fill: float):
    """Shift along the last axis by s. down=True: y[i]=x[i-s] (low end filled);
    down=False: y[i]=x[i+s] (high end filled). Out-of-range entries set to fill."""
    n = x.shape[-1]
    y = torch.full_like(x, fill)
    if s < n:
        if down:
            y[..., s:] = x[..., :n - s]
        else:
            y[..., :n - s] = x[..., s:]
    return y


def pcr_solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
              d: torch.Tensor) -> torch.Tensor:
    """Solve batched tridiagonal systems. Returns x with the same shape as d."""
    n = d.shape[-1]
    # Work copies; broadcast 1-D diagonals to d's batch lazily via ops.
    a = a.clone(); b = b.clone(); c = c.clone(); d = d.clone()

    s = 1
    while s < n:
        # neighbour rows i-s and i+s; b padded with 1 (avoid /0 at boundaries),
        # a/c/d padded with 0 so absent neighbours contribute nothing.
        a_dn = _shift(a, s, True, 0.0); b_dn = _shift(b, s, True, 1.0)
        c_dn = _shift(c, s, True, 0.0)
        a_up = _shift(a, s, False, 0.0); b_up = _shift(b, s, False, 1.0)
        c_up = _shift(c, s, False, 0.0)
        d_dn = _shift(d, s, True, 0.0); d_up = _shift(d, s, False, 0.0)

        alpha = -a / b_dn
        gamma = -c / b_up
        b = b + alpha * c_dn + gamma * a_up
        d = d + alpha * d_dn + gamma * d_up
        a = alpha * a_dn
        c = gamma * c_up
        s *= 2

    return d / b
