#!/bin/bash
#SBATCH --job-name=bdf2_ext
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=slurm-bdf2_ext-%j.out

# Converged-statistics extension of the bdf2 dt+=0.25 run: restart from
# results_fix/bdf2_full/fields_final.npz (t=110), resume the stats
# accumulators, run to t=600 (stats window t+ ~ 118-7070, ~44 ETT).
export TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 CC=gcc PYTORCH_JIT=0
source /etc/profile.d/modules.sh 2>/dev/null && module load texlive   # usetex figures
cd /home/giorgio/slChannel
echo "=== START config180cans_sl_bdf2_ext $(date '+%H:%M:%S')"
python main.py configs/config180cans_sl_bdf2_ext.yaml > logs/config180cans_sl_bdf2_ext.log 2>&1
echo "=== DONE rc=$? $(date '+%H:%M:%S')"
python scripts/compare_mkm.py > logs/bdf2_ext_analysis.log 2>&1
# torChannel house-style figure set (velocity / normal stresses /
# shear+vorticity / total stress; the 2d-spectra figure is not reported)
python scripts/plot_stats_torstyle.py results_fix/bdf2_full/turbulence_stats.npz \
    --reference mkm180 --open-channel --format png --dpi 150 \
    --output figures_fix/bdf2_mkm >> logs/bdf2_ext_analysis.log 2>&1
echo "=== ANALYSIS DONE"
