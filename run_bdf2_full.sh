#!/bin/bash
#SBATCH --job-name=bdf2_full
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=slurm-bdf2_full-%j.out

# Full-window validation of the sweep winner: bdf2 at dt+ = 0.25 (its clean
# operating point), t+ = 0..1296, stats window t+ = 118-1296.
export TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 CC=gcc PYTORCH_JIT=0
cd /home/giorgio/slChannel
echo "=== START config180cans_sl_bdf2_full $(date '+%H:%M:%S')"
python main.py configs/config180cans_sl_bdf2_full.yaml > logs/config180cans_sl_bdf2_full.log 2>&1
echo "=== DONE rc=$? $(date '+%H:%M:%S')"
python - <<'EOF' > logs/bdf2_full_analysis.log 2>&1
import numpy as np, subprocess, sys
ref = np.load('data/m3_stats/ref.npz')
d = np.load('results_fix/bdf2_full/turbulence_stats.npz')
for k in ('E_uu_2d', 'E_vv_2d', 'E_ww_2d'):
    print(f"{k} tail/Eul: {d[k].sum(axis=1)[-10:].mean()/ref[k].sum(axis=1)[-10:].mean():.3e}")
ut, utr = float(d['u_tau']), float(ref['u_tau'])
print(f"u_tau {ut:.5f} vs {utr:.5f} ({100*(ut-utr)/utr:+.2f}%)")
subprocess.run([sys.executable, 'scripts/compare_stats.py',
                'data/m3_stats/ref.npz', 'results_fix/bdf2_full/turbulence_stats.npz',
                '--labels', 'eulerian', 'bdf2 dt+=0.25',
                '--out', 'figures_fix/profiles_bdf2_full.png'], check=False)
EOF
echo "=== ANALYSIS DONE"
