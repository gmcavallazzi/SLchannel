#!/bin/bash
# CPU-only watcher for the M950 transient (no GPU). Every 10 minutes it
# applies the dpdx criterion to results/m950_sl_dt020/timeseries.npz; when
# it fires it writes t0_transient.txt and cancels the running transient
# segment, so the queued singleton restarts within minutes with statistics
# on (the chain script reads the marker). Exits after firing, or once the
# run is complete.
#
#SBATCH --job-name=m950_watch
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=compute

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
RES=results/m950_sl_dt020

while true; do
    [ -f "$RES/fields_final.npz" ] && { echo "run complete, watcher done"; exit 0; }
    [ -f "$RES/t0_transient.txt" ] && { echo "T0 already set, watcher done"; exit 0; }
    if [ -f "$RES/timeseries.npz" ]; then
        python - <<'PY'
import os
import numpy as np

RES = "results/m950_sl_dt020"
U_TAU_EXPECTED = 912.0 / 18000.0  # realized target for Re_b=18000 (log-law, user kept 18000)
BAND, MEAN_W, HOLD_W, T_MIN = 0.02, 20.0, 30.0, 20.0
d = np.load(f"{RES}/timeseries.npz")
t, ut = d["time"], d["u_tau"]
rm_last = ut[t > t[-1] - MEAN_W].mean() if t[-1] > MEAN_W else float("nan")
print(f"watch: t = {t[-1]:.1f}, 20-tu mean u_tau = {rm_last:.5f} "
      f"(target {U_TAU_EXPECTED:.5f}, {100*(rm_last/U_TAU_EXPECTED-1):+.1f}%)", flush=True)
for i in range(len(t)):
    if t[i] < T_MIN or t[-1] - t[i] < HOLD_W:
        continue
    m = (t >= t[i]) & (t <= t[i] + HOLD_W)
    rm = np.array([ut[(t >= tj - MEAN_W) & (t <= tj)].mean() for tj in t[m]])
    if np.all(np.abs(rm / U_TAU_EXPECTED - 1) < BAND) and \
       abs(rm[-3:].mean() - rm[:3].mean()) / rm.mean() < 0.01:
        open(f"{RES}/t0_transient.txt", "w").write(f"{float(t[i]):.4f}\n")
        print(f"TRANSIENT OVER: T0 = {t[i]:.1f}", flush=True)
        break
PY
        if [ -f "$RES/t0_transient.txt" ]; then
            RUNNING=$(squeue -h -n m950_sl -t R -o %i | head -1)
            if [ -n "$RUNNING" ]; then
                echo "criterion fired: cancelling transient segment $RUNNING"
                scancel "$RUNNING"
            fi
            echo "watcher done: T0 = $(cat $RES/t0_transient.txt)"
            exit 0
        fi
    fi
    sleep 600
done
