"""Compare the 4-rank and monolithic Re180 replication runs.

    python parallel/compare_re180.py results/re180_4rank results/re180_mono

Reports max absolute differences of the final fields and of every array in
the accumulated statistics. The decomposed step is bit-identical to the
production step, so the expectation is EXACT equality (0.0)."""

import sys

import numpy as np
import torch


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    a = torch.load(f"{a_dir}/fields_final.pt", map_location="cpu", weights_only=True)
    b = torch.load(f"{b_dir}/fields_final.pt", map_location="cpu", weights_only=True)
    print(
        f"fields at t = {a['time']:.4f} (A) vs {b['time']:.4f} (B), "
        f"steps {a['step']} vs {b['step']}"
    )
    exact = True
    for c in "uvw":
        err = float((a[c] - b[c]).abs().max())
        exact &= err == 0.0
        print(f"  field {c}: max|A-B| = {err:.3e}")

    sa = np.load(f"{a_dir}/turbulence_stats.npz")
    sb = np.load(f"{b_dir}/turbulence_stats.npz")
    keys = sorted(set(sa.files) & set(sb.files))
    for k in keys:
        xa, xb = np.asarray(sa[k], dtype=np.float64), np.asarray(sb[k], dtype=np.float64)
        if xa.shape != xb.shape:
            print(f"  stats {k}: SHAPE MISMATCH {xa.shape} vs {xb.shape}")
            exact = False
            continue
        err = float(np.abs(xa - xb).max())
        exact &= err == 0.0
        flag = "" if err == 0.0 else "   <-- nonzero"
        print(f"  stats {k}: max|A-B| = {err:.3e}{flag}")
    print("VERDICT:", "EXACT MATCH (bitwise)" if exact else "not bitwise -- inspect above")
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
