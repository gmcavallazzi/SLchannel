"""Fix-track sweep report (2026-08-03): projected-predictor 'pc' vs Boukir
'bdf2', against the Eulerian reference. All axes in wall (+) units.

Outputs (figures_fix/):
  tail_vs_t.png       instantaneous z+~15 high-k tail ratio vs t+ per run,
                      from the uniform-dt+ snapshots (burst detection: a time
                      average can hide intermittent spikes — the noextrap
                      full-window lesson)
  spectra_dt40.png    time-averaged 1D spectra E(kx+) at z+~15, dt+ = 0.40
  profiles_dt40.png   U+/rms+ profile comparison at dt+ = 0.40 (compare_stats)
  (+ figures_m3/accuracy_vs_dt.png from m3_accuracy_dt.py: tail/mid-k vs dt+)

Also verifies every run started from the byte-identical initial field
(md5 of fields_init.npz) and prints a wall-unit summary table.

Usage: python scripts/fix_sweep_report.py   (from the repo root)
"""

import glob
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import operators
from utils import generate_grid
from turbstats import TurbulenceStats

torch.set_default_dtype(torch.float64)
plt.rcParams.update({'font.size': 11, 'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral'})

REF = 'data/m3_stats/ref.npz'
# fixed family colors (identity, never cycled); dt+ encoded by linestyle
FAMILIES = {
    'pc':   ('C2', [(0.021, 'results_fix/pc_dt21'),
                    (0.0255, 'results_fix/pc_dt255'),
                    (0.034, 'results_fix/pc_dt34')]),
    'bdf2': ('C3', [(0.021, 'results_fix/bdf2_dt21'),
                    (0.0255, 'results_fix/bdf2_dt255'),
                    (0.034, 'results_fix/bdf2_dt34')]),
}
DT_STYLE = {0.021: ':', 0.0255: '--', 0.034: '-'}
KEYS = ('E_uu_2d', 'E_vv_2d', 'E_ww_2d')

# case constants (config180cans_*): open channel, bottom stretching
NX, NY, NZ = 256, 256, 100
LX, LY, LZ = 10.68, 3.2, 1.0
GAMMA, RE_TAU, NU = 1.8, 180.0, 1.0 / 2870.0

ref = np.load(REF)
U_TAU = float(ref['u_tau'])
T2PLUS = U_TAU**2 / NU           # t   -> t+
K2PLUS = NU / U_TAU              # kx  -> kx+
ref_tail = {k: ref[k].sum(axis=1)[-10:].mean() for k in KEYS}

os.makedirs('figures_fix', exist_ok=True)


GRID = generate_grid(GAMMA, NZ, LZ, device='cpu', stretching_type='bottom')
Z_F, Z_C, DZ_F, DZ_C = GRID


def make_ts():
    return TurbulenceStats(NX, NY, NZ, LX, LY, LZ, Z_C, Z_F, DZ_C, DZ_F,
                           LX / NX, LY / NY, NU, RE_TAU,
                           z_plus_target=15.0, device='cpu')


def snapshot_tail(ts, path):
    """Instantaneous z+~15 tail (mean over uu/vv/ww of tail/ref_tail) and
    the max advective frequency (solver's compute_cfl_fused convention;
    multiply by dt for the max CFL)."""
    d = np.load(path)
    u = torch.from_numpy(d['u']).to(torch.float64)
    v = torch.from_numpy(d['v']).to(torch.float64)
    w = torch.from_numpy(d['w']).to(torch.float64)
    finv = operators.compute_cfl_fused(u, v, w, NX, NY, NZ,
                                       LX / NX, LY / NY, DZ_F, DZ_C)
    finv = finv.item() if torch.is_tensor(finv) else float(finv)
    for a in ('U_sum', 'uu_sum', 'vv_sum', 'ww_sum', 'uw_sum',
              'E_uu_2d_sum', 'E_vv_2d_sum', 'E_ww_2d_sum', 'E_uw_2d_sum'):
        getattr(ts, a).zero_()
    ts.n_samples = 0
    ts.accumulate_statistics(u, v, w, float(d['u_tau']))
    tails = [ts.E_uu_2d_sum.sum(dim=1)[-10:].mean().item() / ref_tail['E_uu_2d'],
             ts.E_vv_2d_sum.sum(dim=1)[-10:].mean().item() / ref_tail['E_vv_2d'],
             ts.E_ww_2d_sum.sum(dim=1)[-10:].mean().item() / ref_tail['E_ww_2d']]
    return float(d['time']), float(np.mean(tails)), finv


# ---- initial-field identity check --------------------------------------
md5s = {}
for _, runs in FAMILIES.values():
    for _, folder in runs:
        p = os.path.join(folder, 'fields_init.npz')
        if os.path.exists(p):
            md5s[folder] = hashlib.md5(open(p, 'rb').read()).hexdigest()
if md5s:
    uniq = set(md5s.values())
    print(f"initial-field md5s: {len(uniq)} distinct over {len(md5s)} runs "
          f"{'[OK: identical]' if len(uniq) == 1 else '[MISMATCH!]'}")
    for f, h in md5s.items():
        print(f"  {h[:12]}  {f}")

# ---- (a) instantaneous tail vs t+, and max CFL vs t+ -------------------
ts = make_ts()
fig, ax = plt.subplots(figsize=(7.0, 4.2))
figc, axc = plt.subplots(figsize=(7.0, 4.2))
for fam, (color, runs) in FAMILIES.items():
    for dt, folder in runs:
        snaps = sorted(glob.glob(os.path.join(folder, 'fields_t*.npz')))
        init = os.path.join(folder, 'fields_init.npz')
        if os.path.exists(init):
            snaps = [init] + snaps
        if not snaps:
            print(f"[skip] no snapshots in {folder}")
            continue
        tt, rr, cc = [], [], []
        for s in snaps:
            t, r, finv = snapshot_tail(ts, s)
            tt.append(t * T2PLUS)
            rr.append(r)
            cc.append(dt * finv)               # max CFL at this run's dt
        ax.semilogy(tt, rr, DT_STYLE[dt], color=color, lw=1.4,
                    label=f"{fam}  $\\Delta t^+={dt * T2PLUS:.2f}$")
        axc.plot(tt, cc, DT_STYLE[dt], color=color, lw=1.4,
                 label=f"{fam}  $\\Delta t^+={dt * T2PLUS:.2f}$")
        print(f"  {fam} dt+={dt * T2PLUS:.2f}: {len(snaps)} snapshots, "
              f"tail median {np.median(rr):.2e} max {np.max(rr):.2e}, "
              f"max CFL {np.max(cc):.2f} (median {np.median(cc):.2f})")
ax.axhline(1.0, color='k', lw=0.8, ls=':')
ax.set_xlabel(r'$t^+$')
ax.set_ylabel('instantaneous tail / Eulerian mean tail')
ax.set_title(r'high-$k_x$ tail at $z^+\!\approx\!15$ (last 10 modes, uu/vv/ww mean)',
             fontsize=11)
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.25, which='both')
fig.tight_layout()
fig.savefig('figures_fix/tail_vs_t.png', dpi=150)
print("saved figures_fix/tail_vs_t.png")

axc.axhline(1.0, color='k', lw=0.8, ls=':')
axc.set_xlabel(r'$t^+$')
axc.set_ylabel(r'max CFL  ($\Delta t\cdot\max_i \sum |u_j|/\Delta x_j$)')
axc.set_title('max advective CFL per snapshot', fontsize=11)
axc.legend(fontsize=8, ncol=2)
axc.grid(alpha=0.25)
figc.tight_layout()
figc.savefig('figures_fix/cfl_vs_t.png', dpi=150)
print("saved figures_fix/cfl_vs_t.png")

# ---- (b) time-averaged spectra at dt+ = 0.40 ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=False)
kxp = ref['kx'] * K2PLUS
for axi, key, name in zip(axes, KEYS, ('E_{uu}', 'E_{vv}', 'E_{ww}')):
    axi.loglog(kxp, ref[key].sum(axis=1), 'k-', lw=1.6, label='Eulerian')
    for fam, (color, runs) in FAMILIES.items():
        for dt, folder in runs:
            if abs(dt - 0.034) > 1e-9:
                continue
            p = os.path.join(folder, 'turbulence_stats.npz')
            if not os.path.exists(p):
                continue
            d = np.load(p)
            axi.loglog(kxp, d[key].sum(axis=1), color=color, lw=1.3, label=fam)
    axi.set_xlabel(r'$k_x^+$')
    axi.set_ylabel(rf'${name}(k_x)$')
    axi.set_title(rf'${name}$ at $z^+\!\approx\!15$, $\Delta t^+=0.40$', fontsize=10)
    axi.grid(alpha=0.25, which='both')
axes[0].legend(fontsize=9)
fig.tight_layout()
fig.savefig('figures_fix/spectra_dt40.png', dpi=150)
print("saved figures_fix/spectra_dt40.png")

# ---- (c) profiles at dt+ = 0.40 (compare_stats) ------------------------
files = [REF]
labels = ['eulerian']
for fam, (_, runs) in FAMILIES.items():
    p = os.path.join(runs[-1][1], 'turbulence_stats.npz')
    if os.path.exists(p):
        files.append(p)
        labels.append(fam)
if len(files) > 1:
    subprocess.run([sys.executable, 'scripts/compare_stats.py', *files,
                    '--labels', *labels, '--out', 'figures_fix/profiles_dt40.png'],
                   check=False)
    print("saved figures_fix/profiles_dt40.png")

# ---- summary table (wall units) ----------------------------------------
print("\n=== summary (z+~15 tail ratio vs Eulerian; u_tau %dev) ===")
print(f"{'run':>34} {'dt+':>6} {'tail uu':>10} {'vv':>10} {'ww':>10} "
      f"{'mid-k uu':>9} {'u_tau %':>8}")
for fam, (_, runs) in FAMILIES.items():
    for dt, folder in runs:
        p = os.path.join(folder, 'turbulence_stats.npz')
        if not os.path.exists(p):
            continue
        d = np.load(p)
        r = [d[k].sum(axis=1)[-10:].mean() / ref_tail[k] for k in KEYS]
        mid = d['E_uu_2d'].sum(axis=1)[30] / ref['E_uu_2d'].sum(axis=1)[30]
        du = 100.0 * (float(d['u_tau']) - U_TAU) / U_TAU
        print(f"{folder:>34} {dt * T2PLUS:6.2f} {r[0]:10.2e} {r[1]:10.2e} "
              f"{r[2]:10.2e} {mid:9.2f} {du:8.2f}")
