"""M3 decision analysis: Eulerian vs SL-fp32 vs SL-fp64 (Re180 CaNS case).

Produces the two decision numbers Giorgio's criteria need:
  1. fp32-vs-fp64 indistinguishability (profiles + spectra) -> is the fp32
     pipeline admissible?
  2. the high-k spectral floor of BOTH SL runs vs Eulerian -> is the floor
     interpolation-induced (identical floors) or precision-induced (fp64
     floor much lower)?

Usage: python scripts/m3_threeway.py  (from the repo root, after all three
runs finished and were re-finalized with the BC-aware u_tau)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral'})

RUNS = [('data/m3_stats/ref.npz', 'Eulerian', 'C0'),
        ('data/m3_stats/sl_fp32.npz', 'SL fp32', 'C1'),
        ('data/m3_stats/sl_fp64.npz', 'SL fp64', 'C2')]

data = [(np.load(p), lab, c) for p, lab, c in RUNS]

# ---------- decision numbers -------------------------------------------------
d32 = data[1][0]
d64 = data[2][0]
print("=== fp32 vs fp64 indistinguishability (profiles) ===")
for key in ['U_mean', 'uu_mean', 'vv_mean', 'ww_mean', 'uw_mean']:
    a, b = d32[key], d64[key]
    scale = np.abs(b).max()
    print(f"  {key:8s}: max|diff|/max|fp64| = {np.abs(a - b).max() / scale:.3e}")
print(f"  u_tau   : fp32 {float(d32['u_tau']):.5f}  fp64 {float(d64['u_tau']):.5f}  "
      f"rel diff {abs(float(d32['u_tau']) - float(d64['u_tau'])) / float(d64['u_tau']):.3e}")

print("\n=== high-kx spectral floor (mean of last 10 kx modes, ky-integrated) ===")
for key in ['E_uu_2d', 'E_vv_2d', 'E_ww_2d']:
    tails = {lab: d[key].sum(axis=1)[-10:].mean() for d, lab, _ in data}
    e, f32, f64 = tails['Eulerian'], tails['SL fp32'], tails['SL fp64']
    print(f"  {key}: Eulerian {e:.3e}  fp32 {f32:.3e}  fp64 {f64:.3e}  "
          f"| fp32/Eul {f32 / e:5.1f}x  fp64/Eul {f64 / e:5.1f}x  fp32/fp64 {f32 / f64:5.2f}x")

# ---------- figures ----------------------------------------------------------
os.makedirs('figures_m3', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
for ax, key, name in [(axes[0], 'E_uu_2d', r'$E_{uu}$'),
                      (axes[1], 'E_ww_2d', r'$E_{ww}$'),
                      (axes[2], 'E_vv_2d', r'$E_{vv}$')]:
    for d, lab, c in data:
        E1 = d[key].sum(axis=1) / float(d['u_tau'])**2
        ax.loglog(d['kx'], E1, c, label=lab, lw=1.3,
                  ls='--' if lab == 'SL fp64' else '-')
    ax.set_xlabel(r'$k_x$')
    ax.set_title(name + r'$(k_x)/u_\tau^2$ at $z^+\!\approx\!15$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which='both')
fig.tight_layout()
fig.savefig('figures_m3/spectra_threeway.png', dpi=150)
print("\nsaved figures_m3/spectra_threeway.png")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
nu = float(data[0][0]['nu'])
for d, lab, c in data:
    ut = float(d['u_tau'])
    zp = d['z_c'] * ut / nu
    axes[0].semilogx(zp, d['U_mean'] / ut, c, label=lab, lw=1.2,
                     ls='--' if lab == 'SL fp64' else '-')
    axes[1].plot(zp, np.sqrt(np.maximum(d['uu_mean'], 0)) / ut, c, lw=1.2,
                 ls='--' if lab == 'SL fp64' else '-', label=lab)
    axes[2].plot(zp, -d['uw_mean'] / ut**2, c, lw=1.2,
                 ls='--' if lab == 'SL fp64' else '-', label=lab)
axes[0].set_ylabel(r'$U^+$'); axes[1].set_ylabel(r"$u'_{rms}/u_\tau$")
axes[2].set_ylabel(r"$-\overline{u'w'}/u_\tau^2$")
for ax in axes:
    ax.set_xlabel(r'$z^+$'); ax.legend(fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('figures_m3/profiles_threeway.png', dpi=150)
print("saved figures_m3/profiles_threeway.png")
