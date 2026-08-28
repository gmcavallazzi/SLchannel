#!/bin/bash
# One segment of the Re934 M950 rerun (corrected Re_b = 18518; seeded from
# the wall-layer-rescaled t = 191 field of the first campaign, so no long
# transient: statistics start at t+ = 100). Submit N copies with
# --dependency=singleton and the same job name; STOP file pauses the chain.
#
#SBATCH --job-name=m950_re934
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=UNLIMITED

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
RES=results/m950_re934_sl_dt020
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

if [ -f "$RES/fields.npz" ]; then
    CFG=configs/m950_re934_sl_dt020_cont.yaml
else
    CFG=configs/m950_re934_sl_dt020.yaml
fi
echo "=== segment start $(date '+%F %T')  config=$CFG"
slchannel "$CFG"
echo "=== segment end rc=$? $(date '+%F %T')"
