"""Floor-vs-dt scaling (M3 Re180 case): does the SL high-k floor drop as dt
approaches the Eulerian operating point?

Context (2026-07-04): precision, Lagrange order, C^2 field smoothness and
trajectory order are ALL refuted as the floor mechanism — every accuracy
improvement RAISED the floor, so interpolation error is the floor's
dissipation, not its injection. The remaining regime parameter vs the
Eulerian reference is dt itself (remap-cadence aliasing and/or SL divergence
errors projected into solenoidal high-k noise). This script plots the floor
ratio against dt_max for the cubic fp32 runs.

Short-window runs (t_stats=5, t_max=20) answer ONLY the floor question
(Giorgio's rule); the dt_max=0.034 point is the full-window cubic run.

Usage: python scripts/m3_dt_scaling.py  (from the repo root)
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

REF = 'results_180cans_ref/turbulence_stats.npz'
SWEEP = [(0.034, 'results_180cans_sl_fp32/turbulence_stats.npz'),
         (0.017, 'results_180cans_sl_dt17/turbulence_stats.npz'),
         (0.0085, 'results_180cans_sl_dt85/turbulence_stats.npz'),
         (0.00425, 'results_180cans_sl_dt425/turbulence_stats.npz')]

ref = np.load(REF)
tail = {k: ref[k].sum(axis=1)[-10:].mean() for k in ('E_uu_2d', 'E_vv_2d', 'E_ww_2d')}

dts, ratios = [], {k: [] for k in tail}
print("=== floor/Eulerian tail vs dt_max (cubic fp32, z+~15) ===")
for dt, path in SWEEP:
    if not os.path.exists(path):
        print(f"[skip] dt_max={dt}: {path} not found")
        continue
    d = np.load(path)
    dts.append(dt)
    line = f"  dt_max={dt:8.5f}:"
    for k in tail:
        r = d[k].sum(axis=1)[-10:].mean() / tail[k]
        ratios[k].append(r)
        line += f"  {k.split('_')[1]} {r:6.1f}x"
    line += f"  | u_tau {float(d['u_tau']):.5f}"
    print(line)

fig, ax = plt.subplots(figsize=(5.2, 4.0))
for k, lab, c in [('E_uu_2d', r'$E_{uu}$', 'C0'),
                  ('E_vv_2d', r'$E_{vv}$', 'C1'),
                  ('E_ww_2d', r'$E_{ww}$', 'C2')]:
    ax.loglog(dts, ratios[k], c + 'o-', label=lab, lw=1.3)
ax.axhline(1.0, color='k', lw=0.8, ls=':')
if dts:
    d0 = np.array(sorted(dts))
    ax.loglog(d0, ratios['E_ww_2d'][dts.index(max(dts))] * (d0 / max(dts))**2,
              'k--', lw=0.8, label=r'$\propto \Delta t^2$')
ax.set_xlabel(r'$\Delta t_{max}$')
ax.set_ylabel(r'floor / Eulerian tail')
ax.set_title(r'SL floor vs $\Delta t$ at $z^+\!\approx\!15$', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.25, which='both')
fig.tight_layout()
os.makedirs('figures_m3', exist_ok=True)
fig.savefig('figures_m3/floor_vs_dt.png', dpi=150)
print("\nsaved figures_m3/floor_vs_dt.png")
