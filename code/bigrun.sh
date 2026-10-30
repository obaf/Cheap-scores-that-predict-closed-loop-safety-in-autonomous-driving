#!/bin/bash
# Experiment 2 full pass on the big host (AWS m8a.48xlarge, 192 vCPU). Runs unattended; resumable (finished jobs are
# skipped). Usage:  bash code/bigrun.sh [--skip-training]
#   1. arms a self-shutdown cost cap (the instance stops itself after CAP_MIN minutes; shutdown behaviour = stop)
#   2. trains the small baselines (code/train_baselines.sh)
#   3. builds every job from the planner registries and runs the full pass with one worker per vCPU
#   4. extracts the tidy run table and scores NAVSIM PDMS/EPDMS on the captured first-frame trajectories
# Progress: /data/exp2/progress.json (read by tools/eta.py); log: /data/exp2/bigrun.log
set -u
ROOT=${EXP2_ROOT:-/data/exp2}
CAP_MIN=${CAP_MIN:-480}
REGISTRY=${REGISTRY:-code/planners_core.yaml,code/planners_learned.yaml,code/planners_baselines.yaml}
cd "$ROOT"
LOG=$ROOT/bigrun.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
N=$(nproc)
sudo shutdown -h +"$CAP_MIN" "exp2 cost cap" && log "self-shutdown armed: instance stops in $CAP_MIN min (cancel: sudo shutdown -c)"
source /data/code/envs/env.sh nuplan
export PYTHONPATH=$ROOT/code:${PYTHONPATH:-}
# Pre-warm the page cache with every database the pass and the training read (sequential reads are fast; 188 workers
# starting together on cold SQLite files would otherwise be limited by the volume's random-read IOPS).
log "pre-warming the page cache"
python -c "import pandas as pd; [print('/data/nuplan/dataset/nuplan-v1.1/splits/val/%s.db' % x) for x in sorted(pd.read_csv('$ROOT/scenarios/val700.csv').log_name.unique())]" > "$ROOT/prewarm.txt"
ls /data/nuplan/dataset/nuplan-v1.1/splits/test/*.db /data/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh/*.db >> "$ROOT/prewarm.txt"
xargs -a "$ROOT/prewarm.txt" -P 16 -n 20 cat > /dev/null
log "pre-warm done: $(wc -l < "$ROOT/prewarm.txt") files; $(free -g | awk '/Mem/ {print $6}') GB in cache"
if [ "${1:-}" != "--skip-training" ]; then
  log "training the small baselines on $N vCPUs"
  # env.sh sets OMP_NUM_THREADS=1, which GNU nproc honours: pass the real vCPU count explicitly
  NPROC=$N bash code/train_baselines.sh > "$ROOT/train.log" 2>&1
  log "training exit=$?"
  python code/sim_check_baselines.py --parallel 12 >> "$LOG" 2>&1
  log "baseline simulation check exit=$?"
fi
python code/runner.py make --registry "$REGISTRY" >> "$LOG" 2>&1 || { log "job build failed"; exit 1; }
log "full pass start with $((N - 4)) workers"
python code/runner.py run --workers $((N - 4)) > "$ROOT/pass.log" 2>&1
log "full pass exit=$?"
python code/extract_metrics.py >> "$LOG" 2>&1
log "extract exit=$?"
source /data/code/envs/env.sh navsim
export PYTHONPATH=$ROOT/code:${PYTHONPATH:-}
python code/navsim_score.py --workers $((N - 2)) >> "$LOG" 2>&1
log "navsim exit=$?"
log "BIGRUN_DONE"
