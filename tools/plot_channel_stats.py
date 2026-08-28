"""Standard channel-statistics figures from a turbulence_stats.npz.

Produces the campaign's validation figures — mean profile, rms
fluctuations, Reynolds/total stress — folded over both channel halves,
wall-scaled with a no-slip-anchored wall gradient, overlaid on a reference
DNS, plus printed peak and outer-tail deviations.

    python tools/plot_channel_stats.py results/case_dir \
        [--ref torroja950|mkm180|lm1000|lm2000|none] [--out figures/case]

References: torroja950 needs data/torroja_re950/Re950.prof (download it
from https://torroja.dmt.upm.es, channel statistics Re950 — the file is
not redistributed here); mkm180/lm1000/lm2000 come from
`python tools/fetch_data.py <name>`.
"""

import argparse
import os
import shutil
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {"text.usetex": shutil.which("latex") is not None, "font.family": "serif", "font.size": 10}
)
BLUE, GREY = "#0072B2", "#707070"
REF_STYLE = dict(color="k", marker="o", ms=2.8, lw=0, mfc="none", mew=0.8)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def fold_half(q, sign=+1):
    """Average the two channel halves: 0.5*(q[k] + sign*q[nz-1-k])."""
    return 0.5 * (q + sign * q[::-1])[: (len(q) + 1) // 2]


def load_case(path):
    npz = path if path.endswith(".npz") else os.path.join(path, "turbulence_stats.npz")
    d = np.load(npz)
    z_c, nu = d["z_c"], float(d["nu"])
    U = fold_half(d["U_mean"])
    uu, vv, ww = (fold_half(d[k]) for k in ("uu_mean", "vv_mean", "ww_mean"))
    uw = fold_half(d["uw_mean"], sign=-1)
    z = z_c[: len(U)]
    # no-slip-anchored wall gradient: U(0)=0 at z=0, first sample at z_c[0]
    u_tau = float(np.sqrt(nu * U[0] / z[0]))
    Re_tau = u_tau / nu  # h = 1
    return dict(
        z_plus=z * Re_tau,
        z=z,
        U_plus=U / u_tau,
        u_rms=np.sqrt(np.maximum(uu, 0)) / u_tau,
        v_rms=np.sqrt(np.maximum(vv, 0)) / u_tau,
        w_rms=np.sqrt(np.maximum(ww, 0)) / u_tau,
        uw_plus=uw / u_tau**2,
        u_tau=u_tau,
        Re_tau=Re_tau,
        nu=nu,
        n_samples=int(d["n_samples"]),
    )


def load_reference(name):
    if name == "none":
        return None
    if name == "torroja950":
        path = os.path.join(REPO, "data", "torroja_re950", "Re950.prof")
        if not os.path.exists(path):
            sys.exit(
                f"missing {path}: download Re950.prof from https://torroja.dmt.upm.es "
                f"(channel statistics, Re950) and place it there"
            )
        m = np.loadtxt(path, comments="%")
        # columns: y/h y+ U+ u'+ v'+ w'+ ... uv'+ (10). Torroja's v is
        # wall-normal and w spanwise -> swap into slChannel's convention.
        return dict(
            label=r"Torroja, $Re_\tau = 934$",
            z_plus=m[:, 1],
            U_plus=m[:, 2],
            u_rms=m[:, 3],
            v_rms=m[:, 5],
            w_rms=m[:, 4],
            uw_plus=-m[:, 10],
        )
    if name == "mkm180":
        path = os.path.join(REPO, "data", "reference", "mkm180.csv")
        if not os.path.exists(path):
            sys.exit(f"missing {path}: run `python tools/fetch_data.py mkm180`")
        m = np.genfromtxt(path, delimiter=",", names=True, skip_header=4)
        return dict(
            label=r"MKM, $Re_\tau = 178$",
            z_plus=m["z_plus"],
            U_plus=m["U_plus"],
            u_rms=np.sqrt(np.maximum(m["uu_plus"], 0)),
            v_rms=np.sqrt(np.maximum(m["vv_plus"], 0)),
            w_rms=np.sqrt(np.maximum(m["ww_plus"], 0)),
            uw_plus=m["uw_plus"],
        )
    if name in ("lm1000", "lm2000"):
        base = os.path.join(REPO, "data", "reference", name)
        if not os.path.exists(base + "_mean.csv"):
            sys.exit(f"missing {base}_mean.csv: run `python tools/fetch_data.py {name}`")
        mean = np.genfromtxt(base + "_mean.csv", delimiter=",", names=True, skip_header=3)
        fluc = np.genfromtxt(base + "_fluc.csv", delimiter=",", names=True, skip_header=4)
        retau = {"lm1000": 1000.5, "lm2000": 1994.8}[name]
        return dict(
            label=rf"Lee--Moser, $Re_\tau = {retau:g}$",
            z_plus=mean["z_plus"],
            U_plus=mean["U_plus"],
            fluc_z_plus=fluc["z_plus"],
            u_rms=np.sqrt(np.maximum(fluc["uu_plus"], 0)),
            v_rms=np.sqrt(np.maximum(fluc["vv_plus"], 0)),
            w_rms=np.sqrt(np.maximum(fluc["ww_plus"], 0)),
            uw_plus=fluc["uw_plus"],
        )
    sys.exit(f"unknown reference {name!r}")


def metrics(sl, ref):
    """Peak and outer-tail (0.3 < z/h < 0.7 in wall units of the case)
    deviations against the reference, in percent."""
    out = {}
    rz = ref.get("fluc_z_plus", ref["z_plus"])
    for q in ("u_rms", "v_rms", "w_rms"):
        out[f"peak_{q}"] = 100 * (
            sl[q].max() / np.interp(sl["z_plus"][np.argmax(sl[q])], rz, ref[q]) - 1
        )
        m = (sl["z_plus"] > 0.3 * sl["Re_tau"]) & (sl["z_plus"] < 0.7 * sl["Re_tau"])
        out[f"tail_{q}"] = 100 * np.mean(sl[q][m] / np.interp(sl["z_plus"][m], rz, ref[q]) - 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="standard channel statistics figures")
    ap.add_argument("case", help="results dir or turbulence_stats.npz")
    ap.add_argument(
        "--ref", default="torroja950", choices=["torroja950", "mkm180", "lm1000", "lm2000", "none"]
    )
    ap.add_argument("--out", default=None, help="output prefix (default figures/<case name>)")
    args = ap.parse_args()

    sl = load_case(args.case)
    ref = load_reference(args.ref)
    name = os.path.basename(os.path.normpath(args.case)).replace(".npz", "")
    out = args.out or os.path.join(REPO, "figures", name)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    print(
        f"{name}: Re_tau = {sl['Re_tau']:.1f}, u_tau = {sl['u_tau']:.5f}, {sl['n_samples']} samples"
    )

    # profiles figure: U+ (log) + the three rms panels
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), constrained_layout=True)
    label = rf"slChannel, $Re_\tau = {sl['Re_tau']:.0f}$"
    ax = axes[0, 0]
    if ref is not None:
        ax.semilogx(ref["z_plus"], ref["U_plus"], **REF_STYLE, label=ref["label"])
    ax.semilogx(sl["z_plus"], sl["U_plus"], color=BLUE, label=label)
    ax.set_xlabel(r"$z^+$"), ax.set_ylabel(r"$U^+$"), ax.legend(frameon=False)
    for ax, q, lab in (
        (axes[0, 1], "u_rms", r"$u'^+_{\rm rms}$"),
        (axes[1, 0], "v_rms", r"$v'^+_{\rm rms}$"),
        (axes[1, 1], "w_rms", r"$w'^+_{\rm rms}$"),
    ):
        if ref is not None:
            ax.plot(ref.get("fluc_z_plus", ref["z_plus"]), ref[q], **REF_STYLE)
        ax.plot(sl["z_plus"], sl[q], color=BLUE)
        ax.set_xlabel(r"$z^+$"), ax.set_ylabel(lab)
    fig.savefig(out + "_profiles.png", dpi=200)
    fig.savefig(out + "_profiles.pdf")
    plt.close(fig)
    print(f"wrote {out}_profiles.{{png,pdf}}")

    # stress figure: -u'w'+ and the total-stress line 1 - z/h
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    zh = sl["z"]
    visc = sl["nu"] * np.gradient(sl["U_plus"] * sl["u_tau"], zh) / sl["u_tau"] ** 2
    if ref is not None:
        rz = ref.get("fluc_z_plus", ref["z_plus"])
        ax.plot(rz / rz.max(), -ref["uw_plus"], **REF_STYLE, label=ref["label"])
    ax.plot(zh, -sl["uw_plus"], color=BLUE, label=r"$-\overline{u'w'}^+$")
    ax.plot(zh, -sl["uw_plus"] + visc, color=GREY, lw=1.0, label="total")
    ax.plot([0, 1], [1, 0], "k--", lw=0.8, label=r"$1 - z/h$")
    ax.set_xlabel(r"$z/h$"), ax.set_ylabel(r"stress$^+$"), ax.legend(frameon=False)
    fig.savefig(out + "_stress.png", dpi=200)
    fig.savefig(out + "_stress.pdf")
    plt.close(fig)
    print(f"wrote {out}_stress.{{png,pdf}}")

    if ref is not None:
        m = metrics(sl, ref)
        print("vs", ref["label"].replace("$", "").replace("\\", ""))
        for q in ("u_rms", "v_rms", "w_rms"):
            print(f"  {q}: peak {m[f'peak_{q}']:+.2f}%   tail(0.3-0.7 h) {m[f'tail_{q}']:+.2f}%")


if __name__ == "__main__":
    main()
