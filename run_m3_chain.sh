#!/bin/bash
# M3 Re180 CaNS-reference chain: ref -> SL fp32 -> SL fp64 (sequential, one GPU)
export TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 CC=gcc PYTORCH_JIT=0
cd /home/giorgio/slChannel
for cfg in config180cans_ref config180cans_sl_fp32 config180cans_sl_fp64; do
  echo "=== START $cfg $(date '+%H:%M:%S')"
  python main.py configs/$cfg.yaml > logs/$cfg.log 2>&1
  rc=$?
  echo "=== DONE $cfg rc=$rc $(date '+%H:%M:%S')"
  [ $rc -ne 0 ] && echo "=== ABORT chain" && exit $rc
done
echo "=== ALL DONE"
