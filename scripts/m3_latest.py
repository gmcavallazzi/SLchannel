"""Eulerian reference vs the LATEST SL configuration (M3 Re180 case).

Two-curve fidelity figures (spectra at z+~15, mean/rms/stress profiles)
against the Eulerian full-window reference. Picks the most advanced SL
full-window run that exists, in this order:
  1. results_180cans_sl_pc_full      (v2 + predictor-corrector trajectories)
  2. results_180cans_sl_noextrap_full (v2, traj_extrapolation none;
     NOTE: its averaged spectra carry one intermittent burst at t~43)
  3. results_180cans_sl_fp32          (v2 AB2 baseline — sustained floor)

Usage: python scripts/m3_latest.py  (from the repo root)
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

REF = 'data/m3_stats/ref.npz'
CANDIDATES = [('results_fix/pc_full/turbulence_stats.npz',
               'SL v2 pc'),
              ('data/m3_stats/sl_noextrap_full.npz',
               'SL v2 no-extrap'),
              ('data/m3_stats/sl_fp32.npz',
               'SL v2 ab2')]

sl_path, sl_lab = next((p, l) for p, l in CANDIDATES if os.path.exists(p))
print(f"latest SL run: {sl_lab} ({sl_path})")

ref = np.load(REF)
sl = np.load(sl_path)
nu = float(ref['nu'])

print(f"u_tau: {sl_lab} {float(sl['u_tau']):.5f}  Eulerian {float(ref['u_tau']):.5f}  "
      f"rel {abs(float(sl['u_tau']) - float(ref['u_tau'])) / float(ref['u_tau']) * 100:.2f}%")

# ---------- spectra ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
for ax, key, name in [(axes[0], 'E_uu_2d', r'$E_{uu}$'),
                      (axes[1], 'E_ww_2d', r'$E_{ww}$'),
                      (axes[2], 'E_vv_2d', r'$E_{vv}$')]:
    for d, lab, c in [(ref, 'Eulerian', 'C0'), (sl, sl_lab, 'C3')]:
        E1 = d[key].sum(axis=1) / float(d['u_tau'])**2
        ax.loglog(d['kx'], E1, c, label=lab, lw=1.4)
    ax.set_xlabel(r'$k_x$')
    ax.set_title(name + r'$(k_x)/u_\tau^2$ at $z^+\!\approx\!15$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which='both')
fig.tight_layout()
os.makedirs('figures_m3', exist_ok=True)
fig.savefig('figures_m3/latest_spectra.png', dpi=150)
print("saved figures_m3/latest_spectra.png")

# ---------- profiles ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
for d, lab, c in [(ref, 'Eulerian', 'C0'), (sl, sl_lab, 'C3')]:
    ut = float(d['u_tau'])
    zp = d['z_c'] * ut / nu
    axes[0].semilogx(zp, d['U_mean'] / ut, c, label=lab, lw=1.3)
    axes[1].plot(zp, np.sqrt(np.maximum(d['uu_mean'], 0)) / ut, c, lw=1.3, label=lab)
    axes[2].plot(zp, -d['uw_mean'] / ut**2, c, lw=1.3, label=lab)
axes[0].set_ylabel(r'$U^+$')
axes[1].set_ylabel(r"$u'_{rms}/u_\tau$")
axes[2].set_ylabel(r"$-\overline{u'w'}/u_\tau^2$")
for ax in axes:
    ax.set_xlabel(r'$z^+$')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('figures_m3/latest_profiles.png', dpi=150)
print("saved figures_m3/latest_profiles.png")
