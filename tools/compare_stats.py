"""Overlay turbulence statistics from several runs (SL sweeps vs Eulerian
reference vs literature): mean profile in wall units, rms profiles, and the
Reynolds shear stress.

Usage:
    python tools/compare_stats.py run1.npz [run2.npz ...] \
        --labels "SL CFL3" "Eulerian" --out compare.png [--delta 1.0]

Each npz is a turbulence_stats.npz written by turbstats.TurbulenceStats
(keys: z_c, U_mean, uu_mean, vv_mean, ww_mean, uw_mean, u_tau, nu, ...).
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="compare_stats.png")
    ap.add_argument("--delta", type=float, default=1.0, help="channel half height")
    args = ap.parse_args()

    labels = args.labels if args.labels else [f.split("/")[-1] for f in args.files]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_u, ax_uu, ax_vw, ax_uw = axes.ravel()

    for path, lab in zip(args.files, labels):
        s = load(path)
        z, U = s["z_c"], s["U_mean"]
        u_tau, nu = float(s["u_tau"]), float(s["nu"])
        half = z <= args.delta
        zp = z[half] * u_tau / nu
        ax_u.semilogx(zp, U[half] / u_tau, label=lab)
        ax_uu.plot(zp, np.sqrt(np.maximum(s["uu_mean"][half], 0)) / u_tau, label=lab)
        ax_vw.plot(zp, np.sqrt(np.maximum(s["vv_mean"][half], 0)) / u_tau, "-", label=f"{lab} v'")
        ax_vw.plot(zp, np.sqrt(np.maximum(s["ww_mean"][half], 0)) / u_tau, "--", label=f"{lab} w'")
        ax_uw.plot(zp, -s["uw_mean"][half] / u_tau**2, label=lab)

    zp_ref = np.logspace(0, 2.8, 100)
    ax_u.semilogx(zp_ref[zp_ref < 12], zp_ref[zp_ref < 12], "k:", lw=1)
    ax_u.semilogx(
        zp_ref[zp_ref > 25], np.log(zp_ref[zp_ref > 25]) / 0.41 + 5.2, "k:", lw=1, label="log law"
    )

    ax_u.set_xlabel(r"$z^+$")
    ax_u.set_ylabel(r"$U^+$")
    ax_uu.set_xlabel(r"$z^+$")
    ax_uu.set_ylabel(r"$u'_{rms}/u_\tau$")
    ax_vw.set_xlabel(r"$z^+$")
    ax_vw.set_ylabel(r"$v'_{rms}, w'_{rms}/u_\tau$")
    ax_uw.set_xlabel(r"$z^+$")
    ax_uw.set_ylabel(r"$-\overline{u'w'}/u_\tau^2$")
    for ax in axes.ravel():
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
