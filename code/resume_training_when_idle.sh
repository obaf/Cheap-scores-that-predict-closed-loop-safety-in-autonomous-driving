#!/bin/bash
# Resume the paused raster-baseline training (SIGCONT) once fewer than $1 simulations are running, i.e. in the tail of
# the full pass. Under the full pass the training's 50 OpenMP threads per run stalled (assumption A24), so it was paused
# at 18:01 UTC and continues here with free cores. Usage on the host: setsid nohup bash resume_training_when_idle.sh 100 &
TH=${1:-100}
LOG=/data/exp2/bigrun.log
while :; do
  n=$(pgrep -f "^python .*run_simulation\.py" | wc -l)
  [ "$n" -lt "$TH" ] && break
  sleep 30
done
P=$(pgrep -f "run_training.py py_func=train")
kill -CONT $P
echo "$(date -u +%FT%TZ) raster training resumed (SIGCONT) with $n simulations still running" >> "$LOG"
