#!/bin/bash
# Re180 KMM triquintic replication pair: 4 emulated ranks (2x2 pencils,
# Triton-local SL, pencil Poisson) followed by the monolithic production
# twin over the IDENTICAL protocol, then the bitwise comparison. On exit
# (success OR failure) the M950 STOP file is removed so the dependent M950
# chain resumes.
#
#SBATCH --job-name=re180_4rank
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export SLCHANNEL_TRITON=1
export SLCHANNEL_COMPILE=1
export SLCHANNEL_POISSON_CUDAGRAPH=1
export CC=gcc PYTORCH_JIT=0

cleanup() { rm -f results/m950_sl_dt020/STOP; }
trap cleanup EXIT

echo "=== START re180 4rank $(date '+%F %T')"
python parallel/run_re180_4rank.py --mode 4rank --out results/re180_4rank
echo "=== 4rank rc=$? $(date '+%F %T')"
echo "=== START re180 mono $(date '+%F %T')"
python parallel/run_re180_4rank.py --mode mono --out results/re180_mono
echo "=== mono rc=$? $(date '+%F %T')"
echo "=== COMPARISON"
python parallel/compare_re180.py results/re180_4rank results/re180_mono
echo "=== DONE rc=$? $(date '+%F %T')"
