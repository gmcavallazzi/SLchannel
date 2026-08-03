#!/bin/bash
#SBATCH --job-name=kmm_bdf2
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=slurm-kmm_bdf2-%j.out

# KMM/MKM full (closed) channel, exact Re_b = 2792.8 (computed from the
# chan180 dataset itself), SL bdf2 at dt+ = 0.25. Seeded from torChannel's
# mirrored full-channel field; t_stats = 100 (centreline structures),
# stats window t = 100-480 = 30 washouts.
export TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 CC=gcc PYTORCH_JIT=0
source /etc/profile.d/modules.sh 2>/dev/null && module load texlive
cd /home/giorgio/slChannel
echo "=== START config_kmm180_sl_bdf2 $(date '+%H:%M:%S')"
python main.py configs/config_kmm180_sl_bdf2.yaml > logs/config_kmm180_sl_bdf2.log 2>&1
echo "=== DONE rc=$? $(date '+%H:%M:%S')"
python scripts/plot_stats_torstyle.py results_fix/kmm_bdf2/turbulence_stats.npz \
    --reference mkm180 --format png --dpi 150 \
    --output figures_fix/kmm_mkm > logs/kmm_analysis.log 2>&1
echo "=== ANALYSIS DONE"
