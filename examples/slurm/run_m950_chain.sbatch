#!/bin/bash
# One 8-hour segment of the M950-replica run (Re_tau ~ 935, 2pi x pi x 2).
# Submit N copies with --dependency=singleton and the same job name.
#
# Transient handling (dpdx criterion): statistics stay off until the 20-tu
# running mean of u_tau holds a trailing 30-tu stretch inside +-2% of the
# expected u_tau = Re_tau/Re_b with no residual trend. The first segment that
# sees this writes T0 to $RES/t0_transient.txt; later segments accumulate
# statistics from T0 and integrate to t_max = T0 + 385 (20 eddy turnovers).
#
#SBATCH --job-name=m950_sl
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
RES=results/m950_sl_dt020
if [ -f "$RES/STOP" ]; then
    echo "STOP file present -- run held (rm $RES/STOP to resume)"
    exit 0
fi
if [ -f "$RES/fields_final.npz" ]; then
    echo "run already complete, nothing to do"
    exit 0
fi

export SLCHANNEL_TRITON=1
export SLCHANNEL_COMPILE=1
export SLCHANNEL_POISSON_CUDAGRAPH=1
export CC=gcc PYTORCH_JIT=0

CFG=$(mktemp --suffix=.yaml)
python - "$CFG" <<'PY'
import os, sys, yaml
import numpy as np

RES = "results/m950_sl_dt020"
U_TAU_EXPECTED = 912.0 / 18000.0  # realized target for Re_b=18000 (log-law, user kept 18000)
BAND, MEAN_W, HOLD_W, T_MIN = 0.02, 20.0, 30.0, 20.0
STATS_TU = 385.0        # 20 eddy turnovers at u_tau ~ 0.052

restart = os.path.exists(f"{RES}/fields.npz")
base = "configs/m950_sl_dt020_cont.yaml" if restart else "configs/m950_sl_dt020.yaml"
c = yaml.safe_load(open(base))

t0 = None
t0_file = f"{RES}/t0_transient.txt"
if os.path.exists(t0_file):
    t0 = float(open(t0_file).read())
elif os.path.exists(f"{RES}/timeseries.npz"):
    d = np.load(f"{RES}/timeseries.npz")
    t, ut = d["time"], d["u_tau"]
    # 20-tu running mean, then find the earliest start of a 30-tu stretch
    # that stays in the band with no residual trend
    for i in range(len(t)):
        if t[i] < T_MIN or t[-1] - t[i] < HOLD_W:
            continue
        m = (t >= t[i]) & (t <= t[i] + HOLD_W)
        rm = np.array([ut[(t >= tj - MEAN_W) & (t <= tj)].mean() for tj in t[m]])
        inband = np.all(np.abs(rm / U_TAU_EXPECTED - 1) < BAND)
        trend = abs(rm[-3:].mean() - rm[:3].mean()) / rm.mean() < 0.01
        if inband and trend:
            t0 = float(t[i])
            open(t0_file, "w").write(f"{t0:.4f}\n")
            print(f"TRANSIENT OVER: T0 = {t0:.1f}", flush=True)
            break

if t0 is None:
    c["statistics"]["enabled"] = False
    c["statistics"].pop("restart_state_file", None)
    c["time"]["t_max"] = 999.0
else:
    c["statistics"]["t_stats"] = t0
    c["time"]["t_max"] = t0 + STATS_TU
    if not os.path.exists(f"{RES}/turbulence_stats_state.npz"):
        c["statistics"].pop("restart_state_file", None)
print(f"segment config: restart={restart}, T0={t0}, "
      f"t_stats={c['statistics'].get('t_stats')}, t_max={c['time']['t_max']}, "
      f"stats_enabled={c['statistics'].get('enabled', True)}", flush=True)
yaml.safe_dump(c, open(sys.argv[1], "w"))
PY

echo "=== segment start $(date '+%F %T')"
slchannel "$CFG"
echo "=== segment end rc=$? $(date '+%F %T')"
