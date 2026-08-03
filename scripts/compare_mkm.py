"""Compare slChannel statistics against the MKM (Moser, Kim & Mansour 1999)
Re_tau=178 reference DNS, using torChannel's bundled, axis-remapped CSV
(z wall-normal, w wall-normal velocity, stresses in u_tau^2 units).

Overlays U+, u'/v'/w' rms+ and -u'w'+ profiles in wall units, normalized by
each dataset's OWN u_tau. Caveat baked into the title: MKM is a CLOSED
channel; this case is an open channel (free-slip top) — agreement is
expected near the wall, and a departure toward the centreline (z+ >~ 100)
is physical (suppressed centreline-crossing large scales), not an error.

Usage: python scripts/compare_mkm.py [stats.npz ...]
  default stats: results_fix/bdf2_full/turbulence_stats.npz + data/m3_stats/ref.npz
Output: figures_fix/mkm_comparison.png
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

MKM_CSV = '/home/giorgio/torChannel/torchannel/data/reference/mkm180.csv'

DEFAULT_RUNS = [('results_fix/bdf2_full/turbulence_stats.npz',
                 'SL bdf2 $\\Delta t^+$=0.25 (open ch.)', 'C1')]


def load_mkm(path):
    header, rows = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if header is None:
                header = [c.strip() for c in line.split(',')]
                continue
            rows.append([float(v) for v in line.split(',')])
    data = np.asarray(rows)
    return {h: data[:, i] for i, h in enumerate(header)}


def load_run(path):
    d = np.load(path)
    u_tau, nu = float(d['u_tau']), float(d['nu'])
    z_plus = d['z_c'] * u_tau / nu
    return {'z_plus': z_plus,
            'U_plus': d['U_mean'] / u_tau,
            'u_rms': np.sqrt(np.maximum(d['uu_mean'], 0.0)) / u_tau,
            'v_rms': np.sqrt(np.maximum(d['vv_mean'], 0.0)) / u_tau,
            'w_rms': np.sqrt(np.maximum(d['ww_mean'], 0.0)) / u_tau,
            'uw_plus': -d['uw_mean'] / u_tau**2,
            'u_tau': u_tau, 'n_samples': int(d['n_samples'])}


def main():
    runs = DEFAULT_RUNS
    if len(sys.argv) > 1:
        runs = [(p, os.path.basename(os.path.dirname(p)) or p, f'C{i}')
                for i, p in enumerate(sys.argv[1:])]
    mkm = load_mkm(MKM_CSV)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))

    ax = axes[0, 0]
    ax.semilogx(mkm['z_plus'], mkm['U_plus'], 'k-', lw=1.8,
                label='MKM99 $Re_\\tau$=178 (closed)')
    ax = axes[0, 1]
    ax.plot(mkm['z_plus'], np.sqrt(mkm['uu_plus']), 'k-', lw=1.8)
    ax = axes[1, 0]
    ax.plot(mkm['z_plus'], np.sqrt(mkm['vv_plus']), 'k-', lw=1.8, label="MKM99 v'")
    ax.plot(mkm['z_plus'], np.sqrt(mkm['ww_plus']), 'k--', lw=1.8, label="MKM99 w'")
    ax = axes[1, 1]
    ax.plot(mkm['z_plus'], -mkm['uw_plus'], 'k-', lw=1.8)
    # total stress tau/tau_w = dU+/dz+ + (-u'w'+); converged flow -> 1 - z/h
    ax = axes[0, 2]
    mkm_tot = np.gradient(mkm['U_plus'], mkm['z_plus']) - mkm['uw_plus']
    ax.plot(mkm['z_plus'], mkm_tot, 'k-', lw=1.8, label='MKM99 total')
    ax.plot(mkm['z_plus'], 1.0 - mkm['z_plus'] / 178.12, 'k:', lw=1.0,
            label=r'$1 - z/h$ (converged)')

    for path, label, color in runs:
        if not os.path.exists(path):
            print(f"[skip] {path}")
            continue
        r = load_run(path)
        print(f"{label}: u_tau={r['u_tau']:.5f}, {r['n_samples']} samples, "
              f"z+ range {r['z_plus'][0]:.2f}-{r['z_plus'][-1]:.1f}")
        axes[0, 0].semilogx(r['z_plus'], r['U_plus'], color=color, lw=1.3, label=label)
        axes[0, 1].plot(r['z_plus'], r['u_rms'], color=color, lw=1.3, label=label)
        axes[1, 0].plot(r['z_plus'], r['v_rms'], color=color, lw=1.3, label=f"{label} v'")
        axes[1, 0].plot(r['z_plus'], r['w_rms'], color=color, lw=1.3, ls='--',
                        label=f"{label} w'")
        axes[1, 1].plot(r['z_plus'], r['uw_plus'], color=color, lw=1.3, label=label)
        tot = np.gradient(r['U_plus'], r['z_plus']) + r['uw_plus']
        axes[0, 2].plot(r['z_plus'], tot, color=color, lw=1.3, label=f'{label} total')

    for ax, ylab in [(axes[0, 0], '$U^+$'), (axes[0, 1], "$u'_{rms}{}^+$"),
                     (axes[0, 2], r"$\tau_{tot}/\tau_w = dU^+\!/dz^+ - \overline{u'w'}^+$"),
                     (axes[1, 0], "$v'_{rms}{}^+,\\ w'_{rms}{}^+$"),
                     (axes[1, 1], "$-\\overline{u'w'}^+$")]:
        ax.set_xlabel('$z^+$')
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, which='both')
        ax.set_xlim(left=0.0 if ax is not axes[0, 0] else None)
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    axes[1, 2].axis('off')
    fig.suptitle('MKM99 closed channel vs slChannel open channel — '
                 'near-wall agreement expected; centreline departure is physical',
                 fontsize=11)
    fig.tight_layout()
    os.makedirs('figures_fix', exist_ok=True)
    fig.savefig('figures_fix/mkm_comparison.png', dpi=150)
    print("saved figures_fix/mkm_comparison.png")


if __name__ == '__main__':
    main()
