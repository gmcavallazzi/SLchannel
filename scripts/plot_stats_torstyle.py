"""Render slChannel statistics in torChannel's plot_statistics.py house style
(the figures_local/re180_closed_partial look): serif/Computer-Modern text,
reference DNS as open circles, simulation as thick colored lines, one figure
per quantity (velocity / normal stresses / shear+vorticity / total stress).

Thin driver around /home/giorgio/torChannel/plot_statistics.py: imports it
and forwards the CLI. The original style uses text.usetex — load TeX first
(`module load texlive`, i.e. run via
  source /etc/profile.d/modules.sh && module load texlive && python ...);
if latex is not on PATH, falls back to matplotlib's 'cm' mathtext, which is
visually close. 2-D spectra figures are produced by the underlying script
but not part of the report set.

Usage (from the repo root):
  python scripts/plot_stats_torstyle.py <stats.npz> --reference mkm180 \
      --open-channel --format png --dpi 150 --output figures_fix/<prefix>
"""

import shutil
import sys

sys.path.insert(0, '/home/giorgio/torChannel')

import matplotlib.pyplot as plt

import plot_statistics as ps

SIM_LABEL = 'slChannel (SL BDF2)'

# plot_statistics hardcodes label='torChannel' (lines ~304, ~368); wrap the
# two functions to relabel the simulation curve and redraw the legend.
_orig_mv = ps.plot_mean_velocity
_orig_sv = ps.plot_shear_vorticity


def _mv(z_c, U_mean, u_tau, nu, ax_outer, ax_inner, ref=None):
    _orig_mv(z_c, U_mean, u_tau, nu, ax_outer, ax_inner, ref)
    for ln in ax_inner.get_lines():
        if ln.get_label() == 'torChannel':
            ln.set_label(SIM_LABEL)
    ax_inner.legend(fontsize=7, loc='upper left')


def _sv(z_c, uw, dUdz, u_tau, nu, ax_uw, ax_omega, ref=None, delta=None):
    _orig_sv(z_c, uw, dUdz, u_tau, nu, ax_uw, ax_omega, ref=ref, delta=delta)
    for ln in ax_uw.get_lines():
        if ln.get_label() == 'torChannel':
            ln.set_label(SIM_LABEL)
    if ax_uw.get_legend() is not None:
        ax_uw.legend(fontsize=7)


ps.plot_mean_velocity = _mv
ps.plot_shear_vorticity = _sv

# Freshness stamp (bottom-right of every saved figure): sample count of the
# stats file being plotted + wall-clock time. Makes live-refreshed figures
# visibly distinguishable from stale ones.
try:
    import time as _time
    import numpy as _np
    _stats_path = next(a for a in sys.argv[1:] if a.endswith('.npz'))
    _d = _np.load(_stats_path)
    _ns = int(_d['n_samples'])
    _stamp = f"{_ns} samples | plotted {_time.strftime('%H:%M:%S')}"
    _orig_savefig = plt.Figure.savefig

    def _savefig(self, *a, **k):
        self.text(0.995, 0.005, _stamp, ha='right', va='bottom',
                  fontsize=6, color='0.55')
        return _orig_savefig(self, *a, **k)

    plt.Figure.savefig = _savefig
except (StopIteration, Exception):
    pass

if shutil.which('latex') is None:
    print("[torstyle] latex not on PATH (module load texlive?) — "
          "falling back to cm mathtext")
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
    })

if __name__ == '__main__':
    ps.main()
