#!/bin/bash
# Archive every save of the M950 statistics state into stats_ladder/, named
# by sample count. Differences of any two entries give the exact statistics
# of the interval between them (all accumulators are linear sums).
cd "$(dirname "$0")/../.."
RES=results/m950_sl_dt020
mkdir -p "$RES/stats_ladder"
LAST=-1
while true; do
    [ -f "$RES/fields_final.npz" ] && { echo "run complete, ladder done"; exit 0; }
    if [ -f "$RES/turbulence_stats_state.npz" ]; then
        N=$(python3 -c "
import numpy as np
try: print(int(np.load('$RES/turbulence_stats_state.npz')['n_samples']))
except Exception: print(-1)")
        if [ "$N" -gt "$LAST" ] 2>/dev/null && [ "$N" -gt 0 ]; then
            cp "$RES/turbulence_stats_state.npz" "$RES/stats_ladder/state_n$(printf %06d $N).npz"
            echo "archived state at n_samples = $N"
            LAST=$N
        fi
    fi
    sleep 300
done
