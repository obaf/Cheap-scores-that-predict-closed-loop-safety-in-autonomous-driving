#!/bin/bash
# Second half of bigrun.sh, used on 28 Oct 2026 to start the full pass while the last raster baselines were still training
# (one of them was slowed by a CPU block spanning two NUMA nodes). Jobs whose checkpoint did not exist yet fail at start-up
# and are retried after the main loop; a final runner call after training catches anything still missing. Then the
# tidy run table is extracted and NAVSIM PDMS/EPDMS are scored.
set -u
ROOT=${EXP2_ROOT:-/data/exp2}
cd "$ROOT"
LOG=$ROOT/bigrun.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
N=$(nproc)  # before env.sh, which sets OMP_NUM_THREADS=1 (GNU nproc honours it)
source /data/code/envs/env.sh nuplan
export PYTHONPATH=$ROOT/code:${PYTHONPATH:-}
python code/runner.py make --registry code/planners_core.yaml,code/planners_learned.yaml,code/planners_baselines.yaml >> "$LOG" 2>&1
log "full pass start with $((N - 4)) workers (raster training still running; jobs without a checkpoint yet are retried)"
python code/runner.py run --workers $((N - 4)) --retries 2 > "$ROOT/pass.log" 2>&1
log "full pass main loop exit=$?"
while pgrep -f "[t]rain_baselines.sh" > /dev/null; do sleep 30; done
log "baseline training finished; final sweep for any job still not done"
python code/runner.py run --workers $((N - 4)) --retries 1 >> "$ROOT/pass.log" 2>&1
log "full pass exit=$?"
python code/extract_metrics.py >> "$LOG" 2>&1
log "extract exit=$?"
source /data/code/envs/env.sh navsim
export PYTHONPATH=$ROOT/code:${PYTHONPATH:-}
python code/navsim_score.py --workers $((N - 2)) >> "$LOG" 2>&1
log "navsim exit=$?"
log "BIGRUN_DONE"
