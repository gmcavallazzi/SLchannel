"""M3 interpolant-scaling analysis: does a better interpolant kill the SL
high-k spectral floor?

Order-scaling data point (tricubic -> triquintic, both fp32 Triton — valid
for floor physics, the floor is precision-independent per m3_threeway) plus
the C^2 spline run (fp64) when it exists. Eulerian is the reference tail;
SL tricubic fp64 is the production baseline.

Prints the floor ratios (mean of last 10 kx modes, ky-integrated, z+~15)
and saves figures_m3/spectra_interp_scaling.png. Runs whose stats file is
missing are skipped, so this can be re-run as results come in.

Usage: python scripts/m3_interp_scaling.py  (from the repo root)
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

RUNS = [('results_180cans_ref/turbulence_stats.npz', 'Eulerian', 'C0', '-'),
        ('results_180cans_sl_fp32/turbulence_stats.npz', 'SL cubic fp32', 'C1', '-'),
        ('results_180cans_sl_fp64/turbulence_stats.npz', 'SL cubic fp64', 'C2', '--'),
        ('results_180cans_sl_o6/turbulence_stats.npz', 'SL quintic fp32', 'C3', '-'),
        ('results_180cans_sl_spline/turbulence_stats.npz', 'SL spline fp32', 'C4', '-'),
        ('results_180cans_sl_traj4/turbulence_stats.npz', 'SL cubic traj4 fp32', 'C5', '-')]

data = []
for p, lab, c, ls in RUNS:
    if os.path.exists(p):
        data.append((np.load(p), lab, c, ls))
    else:
        print(f"[skip] {lab}: {p} not found")

# ---------- floor numbers ----------------------------------------------------
print("\n=== high-kx spectral floor (mean of last 10 kx modes, ky-integrated, z+~15) ===")
ref = data[0][0]
for key in ['E_uu_2d', 'E_vv_2d', 'E_ww_2d']:
    e = ref[key].sum(axis=1)[-10:].mean()
    line = f"  {key}: Eulerian {e:.3e}"
    for d, lab, _, _ in data[1:]:
        t = d[key].sum(axis=1)[-10:].mean()
        line += f"  | {lab} {t / e:5.1f}x"
    print(line)

print("\n=== u_tau ===")
for d, lab, _, _ in data:
    print(f"  {lab:16s}: {float(d['u_tau']):.5f}")

# ---------- figures ----------------------------------------------------------
os.makedirs('figures_m3', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
for ax, key, name in [(axes[0], 'E_uu_2d', r'$E_{uu}$'),
                      (axes[1], 'E_ww_2d', r'$E_{ww}$'),
                      (axes[2], 'E_vv_2d', r'$E_{vv}$')]:
    for d, lab, c, ls in data:
        E1 = d[key].sum(axis=1) / float(d['u_tau'])**2
        ax.loglog(d['kx'], E1, c, label=lab, lw=1.3, ls=ls)
    ax.set_xlabel(r'$k_x$')
    ax.set_title(name + r'$(k_x)/u_\tau^2$ at $z^+\!\approx\!15$', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which='both')
fig.tight_layout()
fig.savefig('figures_m3/spectra_interp_scaling.png', dpi=150)
print("\nsaved figures_m3/spectra_interp_scaling.png")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
nu = float(data[0][0]['nu'])
for d, lab, c, ls in data:
    ut = float(d['u_tau'])
    zp = d['z_c'] * ut / nu
    axes[0].semilogx(zp, d['U_mean'] / ut, c, label=lab, lw=1.2, ls=ls)
    axes[1].plot(zp, np.sqrt(np.maximum(d['uu_mean'], 0)) / ut, c, lw=1.2,
                 ls=ls, label=lab)
    axes[2].plot(zp, -d['uw_mean'] / ut**2, c, lw=1.2, ls=ls, label=lab)
axes[0].set_ylabel(r'$U^+$'); axes[1].set_ylabel(r"$u'_{rms}/u_\tau$")
axes[2].set_ylabel(r"$-\overline{u'w'}/u_\tau^2$")
for ax in axes:
    ax.set_xlabel(r'$z^+$'); ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('figures_m3/profiles_interp_scaling.png', dpi=150)
print("saved figures_m3/profiles_interp_scaling.png")
