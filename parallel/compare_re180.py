"""Compare the 4-rank and monolithic Re180 replication runs on the
statistics: mean profile, fluctuation profiles, Reynolds shear stress and
2D spectra, each in wall units of its own run.

    python parallel/compare_re180.py results/re180_4rank results/re180_mono

If the two runs remained bit-identical the differences are exactly zero
(reported as a bonus); the PASS/FAIL verdict uses sampling-aware statistical
tolerances so it stays meaningful even if a hidden nondeterminism let the
trajectories decorrelate chaotically over the 150 t.u. window."""

import sys

import numpy as np
import torch

TOL = {
    "u_tau": 0.003,  # relative
    "U_plus": 0.015,  # max relative; decorrelated 150-tu windows wander ~1% in the log region (each run deviates more from the 880-tu archive than from the other)
    "rms_peak": 0.015,  # relative at each component's peak
    "rms_prof": 0.04,  # max relative where the signal is >20% of its peak
    "uw": 0.05,  # same masking
    "spec_band": 0.15,  # max |ratio-1| of E_uu(kx), lowest 2/3 of kx, band-avg'd
}


def _rel(a, b, mask=None):
    d = np.abs(a - b)
    ref = np.abs(b)
    if mask is None:
        mask = ref > 0.2 * ref.max()
    return float((d[mask] / ref[mask]).max()) if mask.any() else 0.0


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    ok = True

    def check(name, val, tol, extra=""):
        nonlocal ok
        good = val <= tol
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: {val:.3e} (tol {tol:g}) {extra}")

    # informational: are the runs still bit-identical?
    try:
        a = torch.load(f"{a_dir}/fields_final.pt", map_location="cpu", weights_only=True)
        b = torch.load(f"{b_dir}/fields_final.pt", map_location="cpu", weights_only=True)
        fmax = max(float((a[c] - b[c]).abs().max()) for c in "uvw")
        print(
            f"final fields at t = {a['time']:.4f}: max|A-B| = {fmax:.3e}"
            + ("  (bit-identical)" if fmax == 0.0 else "  (decorrelated -- judging on statistics)")
        )
    except FileNotFoundError:
        print("(final field files missing; statistics only)")

    sa = np.load(f"{a_dir}/turbulence_stats.npz")
    sb = np.load(f"{b_dir}/turbulence_stats.npz")
    uta, utb = float(sa["u_tau"]), float(sb["u_tau"])
    print(
        f"samples: {int(sa['n_samples'])} (A) vs {int(sb['n_samples'])} (B); "
        f"u_tau {uta:.6f} vs {utb:.6f}"
    )
    check("u_tau rel diff", abs(uta / utb - 1.0), TOL["u_tau"])

    Ua, Ub = sa["U_mean"] / uta, sb["U_mean"] / utb
    m = np.abs(Ub) > 0.2 * np.abs(Ub).max()
    check("U+ profile", _rel(Ua, Ub, m), TOL["U_plus"])

    for key, lab in [("uu_mean", "u'"), ("vv_mean", "v'"), ("ww_mean", "w'")]:
        ra = np.sqrt(np.maximum(sa[key], 0)) / uta
        rb = np.sqrt(np.maximum(sb[key], 0)) / utb
        check(f"{lab} rms peak", abs(ra.max() / rb.max() - 1.0), TOL["rms_peak"])
        check(f"{lab} rms profile", _rel(ra, rb), TOL["rms_prof"])

    uwa, uwb = -sa["uw_mean"] / uta**2, -sb["uw_mean"] / utb**2
    m = np.abs(uwb) > 0.2 * np.abs(uwb).max()
    check("-u'w'+ profile", _rel(uwa, uwb, m), TOL["uw"])

    # spectra at the stats plane: band-averaged E_uu(kx) ratio, low 2/3 of kx
    for key in ("E_uu_2d", "E_vv_2d", "E_ww_2d"):
        ea = np.asarray(sa[key]).sum(axis=1)
        eb = np.asarray(sb[key]).sum(axis=1)
        n = len(ea)
        lo = ea[: 2 * n // 3]
        lob = eb[: 2 * n // 3]
        # average in octave bands to suppress per-mode sampling noise
        nb = 8
        edges = np.unique(np.geomspace(1, len(lo), nb + 1).astype(int))
        ra = np.array([lo[e0:e1].mean() for e0, e1 in zip(edges[:-1], edges[1:])])
        rb = np.array([lob[e0:e1].mean() for e0, e1 in zip(edges[:-1], edges[1:])])
        check(f"{key} band ratio", float(np.abs(ra / rb - 1.0).max()), TOL["spec_band"])

    print("VERDICT:", "PASS -- statistics equivalent" if ok else "FAIL -- see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
