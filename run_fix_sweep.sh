#!/bin/bash
#SBATCH --job-name=sl_fix_sweep
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=slurm-sl_fix_sweep-%j.out

# Fix-track floor sweep (2026-08-03): projected-predictor pc (v2) and Boukir
# BDF2 characteristics, short stats window (t_stats=5, t_max=20) at
# dt+ = 0.25 / 0.30 / 0.40. Answers ONLY the floor-vs-Eulerian-tail question.
# Queued via slurm so it starts when the GPU frees (busy-GPU policy).
export TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 CC=gcc PYTORCH_JIT=0
cd /home/giorgio/slChannel
for cfg in config180cans_sl_pc_dt21 config180cans_sl_pc_dt255 config180cans_sl_pc_dt34 \
           config180cans_sl_bdf2_dt21 config180cans_sl_bdf2_dt255 config180cans_sl_bdf2_dt34; do
  echo "=== START $cfg $(date '+%H:%M:%S')"
  python main.py configs/$cfg.yaml > logs/$cfg.log 2>&1
  rc=$?
  echo "=== DONE $cfg rc=$rc $(date '+%H:%M:%S')"
  # do NOT abort on failure: a diverged run is itself a data point (the 1-D
  # model predicts bdf2 may blow up at dt+ >= 0.30); analysis must still run
done
echo "=== SWEEP DONE"
python scripts/m3_accuracy_dt.py > logs/fix_sweep_analysis.log 2>&1
python scripts/fix_sweep_report.py >> logs/fix_sweep_analysis.log 2>&1
echo "=== ANALYSIS DONE"
