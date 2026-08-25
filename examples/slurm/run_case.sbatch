#!/bin/bash
# Generic slChannel batch job: submit one configuration.
#
#   sbatch examples/slurm/run_case.sbatch configs/config_kmm180_tc256q_sl_bdf2.yaml
#
# SITE CONFIGURATION -- the #SBATCH lines below are placeholders. Set the
# partition, account, wall time and GPU request your cluster expects before
# using this; they cannot be guessed portably.
#
#SBATCH --job-name=slchannel
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --partition=<your-gpu-partition>
##SBATCH --account=<your-account>
##SBATCH --gres=gpu:1
##SBATCH --time=24:00:00

set -euo pipefail

CONFIG="${1:?usage: sbatch run_case.sbatch <config.yaml>}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Optional performance layers; see docs/CONFIG.md and slchannel/env.py.
export SLCHANNEL_TRITON=1
export SLCHANNEL_COMPILE=1
export SLCHANNEL_POISSON_CUDAGRAPH=1

# NVIDIA GB10 (sm_121) only: the TorchScript fuser and Triton's launcher build
# both fail without these. Harmless elsewhere; drop them if your GPU differs.
export CC=gcc PYTORCH_JIT=0

NAME="$(basename "${CONFIG%.yaml}")"

echo "=== START ${NAME} $(date '+%F %H:%M:%S')"
slchannel "${CONFIG}" 2>&1 | tee "logs/${NAME}.log"
echo "=== DONE rc=$? $(date '+%F %H:%M:%S')"

# Statistics vs the MKM reference (fetch it first: tools/fetch_data.py mkm180).
RESULTS="$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['output']['results_folder'])
PY
)"
if [ -f "${RESULTS}/turbulence_stats.npz" ]; then
    mkdir -p figures
    python tools/compare_mkm.py "${RESULTS}/turbulence_stats.npz" \
        --output "figures/${NAME}" || echo "(analysis skipped)"
fi
echo "=== ANALYSIS DONE"
