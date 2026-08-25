"""Reynolds stresses must be about the TIME mean, not the instantaneous
plane mean. Constructed case with a known answer: a fixed fluctuation
pattern with zero plane mean plus a plane-uniform offset alternating
+a, -a over successive samples (so var_t(U) = a^2 exactly). The recovered
u'u' must equal pattern_variance + a^2; the RAW fluctuation accumulator
alone must NOT satisfy this (otherwise the test would pass whether or not
the correction is wired in). Also checks <u'w'> stays exactly zero."""

import numpy as np
import torch

from slchannel.turbstats import TurbulenceStats
from slchannel.utils import generate_grid


def test_stats_plane_mean(check):
    nx, ny, nz = 16, 12, 24
    Lx, Ly, Lz = 4.0, 2.0, 2.0
    z_f, z_c, dz_f, dz_c = generate_grid(1.5, nz, Lz, device="cpu", stretching_type="symmetric")
    ts = TurbulenceStats(
        nx,
        ny,
        nz,
        Lx,
        Ly,
        Lz,
        z_c,
        z_f,
        dz_c,
        dz_f,
        Lx / nx,
        Ly / ny,
        1e-3,
        180.0,
        z_plus_target=15.0,
        device="cpu",
    )

    # fixed pattern on u faces, zero plane mean by construction (full cosine
    # period over the periodic faces); v, w identically zero
    i = torch.arange(nx + 1, dtype=torch.float64)
    face_pattern = torch.cos(2 * np.pi * i / nx)  # face nx == face 0
    u = torch.zeros(nx + 1, ny + 2, nz + 2)
    v = torch.zeros(nx + 2, ny + 1, nz + 2)
    w = torch.zeros(nx + 2, ny + 2, nz + 1)

    # The accumulator now takes u'u' at the u NODES (x-faces), unfiltered, so the
    # reference variance is that of the face pattern itself. It used to average
    # adjacent faces to the cell centre first, which is a low-pass filter with
    # transfer function cos(k*dx/2) -- at this wavenumber it removes
    # 1 - cos^2(pi/nx) = 3.8% of the variance. Keeping both numbers here lets the
    # test guard against a regression to the filtered form as well.
    pattern_var = float((face_pattern[1:] ** 2).mean())  # nx distinct faces
    cc = 0.5 * (face_pattern[:-1] + face_pattern[1:])
    pattern_var_filtered = float((cc**2).mean())

    a = 0.37
    n_samp = 8  # even: <U> = 0 exactly
    for n in range(n_samp):
        offset = a if n % 2 == 0 else -a
        u[:] = (face_pattern + offset).view(-1, 1, 1)
        ts.accumulate_statistics(u, v, w, 0.05)

    raw_uu = float((ts.uu_sum / ts.n_samples)[nz // 2])
    stats = ts.finalize_statistics()
    fixed_uu = float(stats["uu_mean"][nz // 2])
    expected = pattern_var + a * a

    check(
        "corrected u'u' = pattern_var + a^2",
        abs(fixed_uu - expected) < 1e-12,
        f"got {fixed_uu:.6f} expected {expected:.6f} (a^2={a * a:.4f})",
    )
    check(
        "raw accumulator alone is biased (sanity)",
        abs(raw_uu - pattern_var) < 1e-12 and abs(raw_uu - expected) > 0.9 * a * a,
        f"raw {raw_uu:.6f} = pattern_var (missing a^2)",
    )
    check(
        "u'u' taken at the u nodes, not cell centres",
        abs(raw_uu - pattern_var_filtered) > 0.5 * (pattern_var - pattern_var_filtered),
        f"raw {raw_uu:.6f} vs filtered {pattern_var_filtered:.6f} "
        f"(cell-centre averaging would remove "
        f"{100 * (1 - pattern_var_filtered / pattern_var):.1f}% here)",
    )
    check(
        "<u'w'> unaffected",
        abs(float(stats["uw_mean"][nz // 2])) < 1e-14,
        f"uw={float(stats['uw_mean'][nz // 2]):.2e}",
    )
    check(
        "w'w' zero for zero w",
        abs(float(stats["ww_mean"][nz // 2])) < 1e-14,
        f"ww={float(stats['ww_mean'][nz // 2]):.2e}",
    )
