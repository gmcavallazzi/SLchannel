#!/bin/bash
# Rotating snapshot buffer for the M950 run: keep the newest KEEP
# fields_t*.npz files, delete older ones. Never touches fields.npz,
# fields_init.npz or fields_final.npz. Exits when the run completes.
RES=~/slChannel/results/m950_sl_dt020
KEEP=15
while true; do
    ls "$RES"/fields_t*.npz >/dev/null 2>&1 && \
    ls -t "$RES"/fields_t*.npz | tail -n +$((KEEP + 1)) | while read f; do
        echo "$(date '+%F %T') removing old snapshot $(basename "$f")"
        rm -f "$f"
    done
    [ -f "$RES/fields_final.npz" ] && { echo "run complete, janitor done"; exit 0; }
    sleep 1800
done
