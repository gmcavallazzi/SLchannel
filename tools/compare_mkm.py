"""Compare slChannel statistics against the MKM (Moser, Kim & Mansour 1999)
Re_tau=178 reference DNS.

Reads the axis-remapped CSV written by tools/fetch_data.py (z wall-normal,
w wall-normal velocity, stresses in u_tau^2 units); fetch it first with

    python tools/fetch_data.py mkm180

Overlays U+, u'/v'/w' rms+ and -u'w'+ profiles in wall units, normalized by
each dataset's OWN u_tau.

MKM is a CLOSED channel. Comparing an OPEN-channel run (free-slip top,
boundary_conditions.top_wall.type: neumann) against it, expect agreement near
the wall and a real departure toward the centreline (z+ >~ 100), where a
symmetry plane suppresses the centreline-crossing large scales — that is
physics, not an error. Pass --open-channel to note this on the figure.

Usage:
    python tools/compare_mkm.py <stats.npz> [more.npz ...] [--output PREFIX]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "stix", "font.family": "STIXGeneral"})

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MKM_CSV = os.path.join(_REPO, "data", "reference", "mkm180.csv")


def load_mkm(path):
    header, rows = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = [c.strip() for c in line.split(",")]
                continue
            rows.append([float(v) for v in line.split(",")])
    data = np.asarray(rows)
    return {h: data[:, i] for i, h in enumerate(header)}


def load_run(path):
    d = np.load(path)
    u_tau, nu = float(d["u_tau"]), float(d["nu"])
    z_plus = d["z_c"] * u_tau / nu
    return {
        "z_plus": z_plus,
        "U_plus": d["U_mean"] / u_tau,
        "u_rms": np.sqrt(np.maximum(d["uu_mean"], 0.0)) / u_tau,
        "v_rms": np.sqrt(np.maximum(d["vv_mean"], 0.0)) / u_tau,
        "w_rms": np.sqrt(np.maximum(d["ww_mean"], 0.0)) / u_tau,
        "uw_plus": -d["uw_mean"] / u_tau**2,
        "u_tau": u_tau,
        "n_samples": int(d["n_samples"]),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stats", nargs="+", help="turbulence_stats.npz files to overlay")
    ap.add_argument("--labels", nargs="*", default=None, help="one label per file")
    ap.add_argument(
        "--reference", default=MKM_CSV, help="MKM CSV (default: the one fetch_data.py writes)"
    )
    ap.add_argument(
        "--output", default="mkm_comparison", help="output path prefix (default: mkm_comparison)"
    )
    ap.add_argument(
        "--open-channel",
        action="store_true",
        help="note on the figure that the run has a free-slip top",
    )
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(p)) or p for p in args.stats]
    runs = [(p, lab, f"C{i + 1}") for i, (p, lab) in enumerate(zip(args.stats, labels))]

    if not os.path.exists(args.reference):
        sys.exit(
            f"reference data not found: {args.reference}\n"
            f"Fetch it with:  python tools/fetch_data.py mkm180"
        )
    mkm = load_mkm(args.reference)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))

    ax = axes[0, 0]
    ax.semilogx(mkm["z_plus"], mkm["U_plus"], "k-", lw=1.8, label="MKM99 $Re_\\tau$=178 (closed)")
    ax = axes[0, 1]
    ax.plot(mkm["z_plus"], np.sqrt(mkm["uu_plus"]), "k-", lw=1.8)
    ax = axes[1, 0]
    ax.plot(mkm["z_plus"], np.sqrt(mkm["vv_plus"]), "k-", lw=1.8, label="MKM99 v'")
    ax.plot(mkm["z_plus"], np.sqrt(mkm["ww_plus"]), "k--", lw=1.8, label="MKM99 w'")
    ax = axes[1, 1]
    ax.plot(mkm["z_plus"], -mkm["uw_plus"], "k-", lw=1.8)
    # total stress tau/tau_w = dU+/dz+ + (-u'w'+); converged flow -> 1 - z/h
    ax = axes[0, 2]
    mkm_tot = np.gradient(mkm["U_plus"], mkm["z_plus"]) - mkm["uw_plus"]
    ax.plot(mkm["z_plus"], mkm_tot, "k-", lw=1.8, label="MKM99 total")
    ax.plot(
        mkm["z_plus"], 1.0 - mkm["z_plus"] / 178.12, "k:", lw=1.0, label=r"$1 - z/h$ (converged)"
    )

    for path, label, color in runs:
        if not os.path.exists(path):
            print(f"[skip] {path}")
            continue
        r = load_run(path)
        print(
            f"{label}: u_tau={r['u_tau']:.5f}, {r['n_samples']} samples, "
            f"z+ range {r['z_plus'][0]:.2f}-{r['z_plus'][-1]:.1f}"
        )
        axes[0, 0].semilogx(r["z_plus"], r["U_plus"], color=color, lw=1.3, label=label)
        axes[0, 1].plot(r["z_plus"], r["u_rms"], color=color, lw=1.3, label=label)
        axes[1, 0].plot(r["z_plus"], r["v_rms"], color=color, lw=1.3, label=f"{label} v'")
        axes[1, 0].plot(r["z_plus"], r["w_rms"], color=color, lw=1.3, ls="--", label=f"{label} w'")
        axes[1, 1].plot(r["z_plus"], r["uw_plus"], color=color, lw=1.3, label=label)
        tot = np.gradient(r["U_plus"], r["z_plus"]) + r["uw_plus"]
        axes[0, 2].plot(r["z_plus"], tot, color=color, lw=1.3, label=f"{label} total")

    for ax, ylab in [
        (axes[0, 0], "$U^+$"),
        (axes[0, 1], "$u'_{rms}{}^+$"),
        (axes[0, 2], r"$\tau_{tot}/\tau_w = dU^+\!/dz^+ - \overline{u'w'}^+$"),
        (axes[1, 0], "$v'_{rms}{}^+,\\ w'_{rms}{}^+$"),
        (axes[1, 1], "$-\\overline{u'w'}^+$"),
    ]:
        ax.set_xlabel("$z^+$")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, which="both")
        ax.set_xlim(left=0.0 if ax is not axes[0, 0] else None)
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    axes[1, 2].axis("off")
    title = "slChannel vs MKM99 ($Re_\\tau$=178, closed channel)"
    if args.open_channel:
        title += (
            " — open-channel run: near-wall agreement expected, centreline departure is physical"
        )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = args.output if args.output.endswith(".png") else args.output + ".png"
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
