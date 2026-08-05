#!/bin/bash
#SBATCH --job-name=kmm_tc192
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=slurm-kmm_tc192-%j.out

# KMM closed channel from the CaNS run_re180_tc192 equilibrated
# checkpoint (t=150), exact Re_b = 2792.8, SL bdf2 at dt+ = 0.25,
# 70 washouts of node-sampled statistics after a 30 t.u. transient.
export TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 CC=gcc PYTORCH_JIT=0
source /etc/profile.d/modules.sh 2>/dev/null && module load texlive
cd /home/giorgio/slChannel
echo "=== START config_kmm180_tc192_sl_bdf2 $(date '+%H:%M:%S')"
python main.py configs/config_kmm180_tc192_sl_bdf2.yaml > logs/config_kmm180_tc192_sl_bdf2.log 2>&1
echo "=== DONE rc=$? $(date '+%H:%M:%S')"
python scripts/plot_stats_torstyle.py results_fix/kmm_tc192/turbulence_stats.npz \
    --reference mkm180 --format png --dpi 150 \
    --output figures_fix/kmm_tc192_mkm > logs/kmm_tc192_analysis.log 2>&1
echo "=== ANALYSIS DONE"
