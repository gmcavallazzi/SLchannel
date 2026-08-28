"""Rescale a converged field's wall layer to a nearby target Re_tau.

Builds a restart file for a re-run of the SAME channel at a slightly
different bulk Reynolds number without repeating the long transient: the
inner layer (below --zp-inner, in the OLD run's wall units) is multiplied
by the ratio of target to current wall shear,

    s_w = (u_tau_new^2 Re_new) / (u_tau_old^2 Re_old)
        = (Re_tau_new^2 / Re_new) / (Re_tau_old^2 / Re_old),

the outer fluctuations (above --zp-outer) are left untouched — at a small
Re_tau difference they are statistically the same — and a smoothstep
blends the factor in between (no sharp seam). Both walls are treated
symmetrically. The scaling's slight divergence error is removed by the
restart projection at solver startup.

    python tools/retau_rescale.py results/old_run/fields.npz \
        --config configs/new_case.yaml --re-tau-old 911.4 --re-old 18000 \
        --out results/new_case/seed_rescaled.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from slchannel.utils import save_flow_fields  # noqa: E402


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def blend_factor(z, Lz, re_tau_old, s_w, zp_inner, zp_outer):
    """Per-height multiplicative factor: s_w at the walls, 1 in the core,
    smoothstepped between zp_inner and zp_outer (old wall units)."""
    d = np.minimum(z, Lz - z)  # distance to the nearest wall
    zp = d * re_tau_old * 2.0 / Lz  # half-height h = Lz/2 scales wall units
    return s_w + (1.0 - s_w) * smoothstep((zp - zp_inner) / (zp_outer - zp_inner))


def main():
    ap = argparse.ArgumentParser(description="wall-layer Re_tau rescale of a restart field")
    ap.add_argument("field", help="source fields npz (converged run)")
    ap.add_argument("--config", required=True, help="target-case YAML (new Re, Re_tau)")
    ap.add_argument("--re-tau-old", type=float, required=True, help="measured Re_tau of the source")
    ap.add_argument("--re-old", type=float, required=True, help="bulk Re of the source run")
    ap.add_argument("--zp-inner", type=float, default=150.0)
    ap.add_argument("--zp-outer", type=float, default=350.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    re_new, re_tau_new = cfg["flow"]["Re"], cfg["flow"]["Re_tau"]
    Lz = cfg["domain"]["Lz"]

    shear_old = args.re_tau_old**2 / args.re_old
    shear_new = re_tau_new**2 / re_new
    s_w = shear_new / shear_old
    print(
        f"wall shear {shear_old:.4f} -> {shear_new:.4f}  (Re_tau {args.re_tau_old:g} @ "
        f"Re {args.re_old:g}  ->  {re_tau_new:g} @ {re_new:g}):  s_w = {s_w:.5f}, "
        f"blend z+ [{args.zp_inner:g}, {args.zp_outer:g}]"
    )

    d = np.load(args.field)
    z_c, z_f = d["z_c"], d["z_f"]
    out = {}
    for comp, znodes in (("u", z_c), ("v", z_c), ("w", z_f)):
        f = blend_factor(znodes, Lz, args.re_tau_old, s_w, args.zp_inner, args.zp_outer)
        out[comp] = torch.from_numpy(d[comp] * f[None, None, :])
    out["p"] = torch.from_numpy(np.array(d["p"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_flow_fields(
        out["u"],
        out["v"],
        out["w"],
        out["p"],
        torch.from_numpy(np.array(z_c)),
        torch.from_numpy(np.array(z_f)),
        float(d["Lx"]),
        float(d["Ly"]),
        0,
        0.0,
        re_tau_new / re_new,
        float(d["forcing"]),
        os.path.dirname(os.path.abspath(args.out)),
        os.path.basename(args.out),
    )
    print(f"wrote {args.out} (time and step reset; forcing state carried over)")


if __name__ == "__main__":
    main()
