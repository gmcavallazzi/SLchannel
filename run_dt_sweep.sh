#!/bin/bash
# dt/CFL floor sweep: cubic fp32 Triton, short stats window (t_stats=5, t_max=20)
# per Giorgio's rule: these runs answer ONLY the floor-vs-Eulerian-tail question.
export TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 CC=gcc PYTORCH_JIT=0
cd /home/giorgio/slChannel
for cfg in config180cans_sl_ne_dt13 config180cans_sl_ne_dt17 config180cans_sl_ne_dt21 config180cans_sl_ne_dt255; do
  echo "=== START $cfg $(date '+%H:%M:%S')"
  python main.py configs/$cfg.yaml > logs/$cfg.log 2>&1
  rc=$?
  echo "=== DONE $cfg rc=$rc $(date '+%H:%M:%S')"
  [ $rc -ne 0 ] && echo "=== ABORT sweep" && exit $rc
done
echo "=== SWEEP DONE"
