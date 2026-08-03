"""Accuracy-limited dt search (M3 Re180): tail and mid-k fidelity vs dt for
both trajectory-velocity treatments (ab2 sweep + traj_extrapolation none).

Defines the honest SL operating point per Giorgio's reframe (2026-07-04):
dt+ = 0.4 was the Choi-Moin-era Eulerian target, not a measured SL limit;
the data locate the spectra-faithful dt instead. CAVEAT: short-window runs —
for near-marginal points confirm with the burst detector (snapshots), since
averaged stats hide intermittency (the no-extrap full-window lesson).

Usage: python scripts/m3_accuracy_dt.py  (from the repo root)
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

# 2026-08-03: the results_180cans_* dirs were removed; their stats live in
# data/m3_stats/<name>.npz. Entries are full stats-file paths.
REF = 'data/m3_stats/ref.npz'
SERIES = {
    'ab2': [(0.00425, 'data/m3_stats/sl_dt425.npz'),
            (0.0085, 'data/m3_stats/sl_dt85.npz'),
            (0.017, 'data/m3_stats/sl_dt17.npz'),
            (0.021, 'data/m3_stats/sl_dt21.npz'),
            (0.0255, 'data/m3_stats/sl_dt255.npz'),
            (0.030, 'data/m3_stats/sl_dt30.npz'),
            (0.034, 'data/m3_stats/sl_fp32.npz')],
    'none': [(0.013, 'data/m3_stats/sl_ne_dt13.npz'),
             (0.017, 'data/m3_stats/sl_ne_dt17.npz'),
             (0.021, 'data/m3_stats/sl_ne_dt21.npz'),
             (0.0255, 'data/m3_stats/sl_ne_dt255.npz'),
             (0.034, 'data/m3_stats/sl_noextrap.npz')],
    # 2026-08-03 fix tracks: projected-predictor pc (v2) and Boukir BDF2
    'pc': [(0.021, 'results_fix/pc_dt21/turbulence_stats.npz'),
           (0.0255, 'results_fix/pc_dt255/turbulence_stats.npz'),
           (0.034, 'results_fix/pc_dt34/turbulence_stats.npz')],
    'bdf2': [(0.021, 'results_fix/bdf2_dt21/turbulence_stats.npz'),
             (0.0255, 'results_fix/bdf2_dt255/turbulence_stats.npz'),
             (0.034, 'results_fix/bdf2_dt34/turbulence_stats.npz')],
}

ref = np.load(REF)
UTAU2NU = float(ref['u_tau'])**2 / float(ref['nu'])   # dt -> dt+
KEYS = ('E_uu_2d', 'E_vv_2d', 'E_ww_2d')

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
for lab, runs, marker in [('ab2', SERIES['ab2'], 'o'), ('none', SERIES['none'], 's'),
                          ('pc', SERIES['pc'], '^'), ('bdf2', SERIES['bdf2'], 'D')]:
    dts, tails, mids = [], [], []
    print(f"--- {lab} ---")
    for dt, p in runs:
        if not os.path.exists(p):
            print(f"[skip] {p}")
            continue
        d = np.load(p)
        t = np.mean([d[k].sum(axis=1)[-10:].mean() / ref[k].sum(axis=1)[-10:].mean()
                     for k in KEYS])
        m = np.mean([d[k].sum(axis=1)[30] / ref[k].sum(axis=1)[30] for k in KEYS])
        dts.append(dt); tails.append(t); mids.append(m)
        print(f"  dt={dt:8.5f} (dt+={dt*UTAU2NU:5.2f}): tail {t:9.2e}  "
              f"mid-k {m:5.2f}  u_tau {float(d['u_tau']):.5f}")
    axes[0].loglog(np.array(dts) * UTAU2NU, tails, marker + '-', label=lab, lw=1.3)
    axes[1].semilogx(np.array(dts) * UTAU2NU, mids, marker + '-', label=lab, lw=1.3)

for ax, name in [(axes[0], 'tail (last 10 $k_x$)'), (axes[1], 'mid-$k$ ($k_x$ idx 30)')]:
    ax.axhline(1.0, color='k', lw=0.8, ls=':')
    ax.set_xlabel(r'$\Delta t^+$')
    ax.set_ylabel('E / Eulerian')
    ax.set_title(name + r' at $z^+\!\approx\!15$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which='both')
fig.tight_layout()
os.makedirs('figures_m3', exist_ok=True)
fig.savefig('figures_m3/accuracy_vs_dt.png', dpi=150)
print("\nsaved figures_m3/accuracy_vs_dt.png")
