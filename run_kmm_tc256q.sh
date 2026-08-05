#!/bin/bash
#SBATCH --job-name=kmm_tc256q
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=slurm-kmm_tc256q-%j.out

# Quintic-interpolation probe of the SL near-wall stress excess.
# Same grid/box/Re/dt+ = 0.25 as run_kmm_tc256, but sl.interp_order: 6 and
# seeded from the torChannel Eulerian run on the IDENTICAL mesh
# (results_re180_closed_256) -- same-grid restart, so 1 washout of warmup
# (t_stats = 12.6) then 70 washouts of statistics.
# Submit with --dependency=afterany:<torChannel jobid> if that run is still going.
export TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 CC=gcc PYTORCH_JIT=0
source /etc/profile.d/modules.sh 2>/dev/null && module load texlive
cd /home/giorgio/slChannel

# Snapshot the latest torChannel checkpoint as the seed.
cp /home/giorgio/torChannel/results_re180_closed_256/fields.npz data/tor180c256_seed.npz
python - <<'EOF'
import numpy as np
d = np.load('data/tor180c256_seed.npz')
print(f"seed: t = {float(d['time']):.2f}, u_tau = {float(d['u_tau']):.5f}, shape = {d['u'].shape}", flush=True)
EOF

echo "=== START config_kmm180_tc256q_sl_bdf2 $(date '+%H:%M:%S')"
python main.py configs/config_kmm180_tc256q_sl_bdf2.yaml > logs/config_kmm180_tc256q_sl_bdf2.log 2>&1
echo "=== DONE rc=$? $(date '+%H:%M:%S')"
python scripts/plot_stats_torstyle.py results_fix/kmm_tc256q/turbulence_stats.npz \
    --reference mkm180 --format png --dpi 150 \
    --output figures_fix/kmm_tc256q_mkm > logs/kmm_tc256q_analysis.log 2>&1
echo "=== ANALYSIS DONE"
