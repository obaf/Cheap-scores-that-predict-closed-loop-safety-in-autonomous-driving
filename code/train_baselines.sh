#!/bin/bash
# =====================================================================================================================
# Experiment 2 "small baselines": three learned planners trained with nuPlan's training framework on train_pittsburgh
#   mlp    = ego-status MLP (exp2_nuplan/models/ego_mlp.py): ego past poses only -> 16 poses at 0.5 s
#   vector = nuPlan simple_vector_model           raster = nuPlan raster_model with a timm resnet18 backbone
# One command:
#   1. caches features once per model type: nuPlan run_training.py py_func=cache on the same N_SAMPLES samples
#      (frames of train_pittsburgh thinned to >= MIN_GAP_S apart per log, then nuPlan's limit_total_scenarios);
#   2. keeps the samples present in all three caches and splits them by LOG (hashed 5 % of logs -> validation);
#   3. trains 3 models x SEEDS (default 0 1 2) = 9 runs in parallel, threads per run sized from nproc; every run saves
#      EARLY (after 15 % of its optimizer steps) and FINAL (end of training) checkpoints;
#   4. writes $OUT/manifest.json (samples, epochs, steps, seed, wall times, train/val losses per run).
#
#   bash /data/exp2/code/train_baselines.sh             # full run, sized for the 192-vCPU m8a.48xlarge
#   SMOKE=1 bash /data/exp2/code/train_baselines.sh     # pipeline check: 500 samples, 1 epoch, seed 0, 4 threads
#   DRYRUN=1 bash /data/exp2/code/train_baselines.sh    # print the thread/CPU plan and the steps only
#
# Outputs ($OUT = /data/exp2/baselines; smoke: /data/exp2/train_smoke/baselines):
#   $OUT/<mlp|vector|raster>/seed<k>/{early,final}.ckpt + train_manifest.json   (planners_baselines.yaml uses these)
#   $OUT/cache/<model>/        nuPlan feature caches        $OUT/train_tokens.txt, data_split.json   samples + split
#   $OUT/manifest.json, manifest_runs.csv, run_config.json  $OUT/logs/   one log per step + nuPlan/Hydra outputs
# Re-running resumes: caches with a DONE marker and runs with final.ckpt + train_manifest.json are skipped
# (delete them to redo). Settings can be overridden from the environment:
#   N_SAMPLES EPOCHS SEEDS MODELS BATCH LR LR_MILESTONES LR_GAMMA MIN_GAP_S DATA_SEED EARLY_FRACTION NPROC
#   CACHE_WORKERS PIN PREWARM DRYRUN OUT DB_FILES
# Notes: /data/code/envs/README-training.md. Never point DB_FILES at val/test logs (checked below).
# =====================================================================================================================
set -o pipefail

SMOKE=${SMOKE:-0}
CODE=${CODE:-/data/exp2/code}
ENV_SH=/data/code/envs/env.sh
SPLITS=/data/nuplan/dataset/nuplan-v1.1/splits
TRAIN_DB=${DB_FILES:-$SPLITS/train_pittsburgh}
if [ "$SMOKE" = 1 ]; then
  OUT=${OUT:-/data/exp2/train_smoke/baselines}
  N_SAMPLES=${N_SAMPLES:-500}; EPOCHS=${EPOCHS:-1}; SEEDS=${SEEDS:-0}; BATCH=${BATCH:-16}; NPROC=${NPROC:-4}
else
  OUT=${OUT:-/data/exp2/baselines}
  N_SAMPLES=${N_SAMPLES:-50000}; EPOCHS=${EPOCHS:-10}; SEEDS=${SEEDS:-"0 1 2"}; BATCH=${BATCH:-64}
  NPROC=${NPROC:-$(nproc)}
fi
MODELS=${MODELS:-"mlp vector raster"}
LR=${LR:-1e-3}   # Adam's default; 1e-4 converged 3-5x slower on a 4 k-sample check (README-training.md)
# step decay (nuPlan's multistep_lr): lr x LR_GAMMA at 70 % and 90 % of the epochs (epochs 7 and 9 of 10)
LR_MILESTONES=${LR_MILESTONES:-$(( (EPOCHS * 7 + 5) / 10 )),$(( (EPOCHS * 9 + 5) / 10 ))}
LR_GAMMA=${LR_GAMMA:-0.1}
MIN_GAP_S=${MIN_GAP_S:-2.0}
DATA_SEED=${DATA_SEED:-0}
EARLY_FRACTION=${EARLY_FRACTION:-0.15}
PIN=${PIN:-auto}
PREWARM=${PREWARM:-auto}
DRYRUN=${DRYRUN:-0}   # 1 = only print the thread/CPU plan and what each step would run (no files written)
SEARCHPATH="hydra.searchpath=[pkg://exp2_nuplan.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"

if [ "$DRYRUN" != 1 ]; then
  mkdir -p "$OUT/logs" "$OUT/cache"
  MAIN_LOG=$OUT/logs/train_baselines_$(date -u +%Y%m%dT%H%M%SZ).log
  exec > >(tee -a "$MAIN_LOG") 2>&1
fi

say() { echo "$(date -u +%H:%M:%S) $*"; }
die() { say "ERROR: $*"; exit 1; }
phase_start() { PH_NAME=$1; PH_T0=$(date +%s); say "=== phase $1"; }
phase_end() {
  local t1; t1=$(date +%s)
  [ "$DRYRUN" = 1 ] || printf '%s\t%s\t%s\n' "$PH_NAME" "$PH_T0" "$t1" >> "$OUT/logs/phases.tsv"
  say "=== phase $PH_NAME done in $((t1 - PH_T0)) s"
}
has() { case " $MODELS " in *" $1 "*) return 0;; *) return 1;; esac; }
exp_of() {
  case $1 in mlp) echo training_ego_mlp_model;; vector) echo training_simple_vector_model;;
             raster) echo training_raster_model;; *) return 1;; esac
}
feats_of() {
  case $1 in mlp) echo agents,trajectory;; vector) echo vector_map,agents,trajectory;;
             raster) echo raster,trajectory;; esac
}
model_ov() { if [ "$1" = raster ]; then echo "model.model_name=resnet18 model.pretrained=$2"; fi; }
# nuplan conda env + exp2 code on PYTHONPATH (env.sh honours a preset OMP_NUM_THREADS)
nuplan_env() { set +u; source "$ENV_SH" nuplan || return 1; export PYTHONPATH=$CODE:$PYTHONPATH; }

for m in $MODELS; do exp_of "$m" > /dev/null || die "unknown model '$m' (allowed: mlp vector raster)"; done
NSEEDS=$(echo $SEEDS | wc -w)
NMODELS=$(echo $MODELS | wc -w)

# ---------------------------------------------------------------------------------------------------- thread plan
# parallel mode (>= 24 vCPUs): all caches at once (NPROC/NMODELS processes each), then all runs at once with
#   mlp    1 torch thread + 3 DataLoader workers   (loading-bound, ~1.2 ms/sample/worker)
#   vector 2 torch threads + 6 workers             (nuPlan's kinematic augmentation costs ~5 ms/sample/worker)
#   raster the remaining vCPUs: 2 workers + the rest as torch threads (compute-bound, ~51 ms/sample/thread)
# sequential mode (smoke test or small hosts): one step after another, each with all NPROC threads.
if [ "$NPROC" -ge 24 ]; then
  MODE=parallel
  T_mlp=1; D_mlp=3; T_vector=2; D_vector=6; D_raster=2
  used=0
  has mlp && used=$((used + NSEEDS * (T_mlp + D_mlp)))
  has vector && used=$((used + NSEEDS * (T_vector + D_vector)))
  T_raster=$(( (NPROC - used) / NSEEDS - D_raster ))
  [ "$T_raster" -ge 1 ] || die "not enough vCPUs ($NPROC) for the raster runs"
  CACHE_WORKERS=${CACHE_WORKERS:-$(( NPROC / NMODELS ))}
else
  MODE=sequential
  T_all=$(( NPROC > 1 ? NPROC / 2 : 1 )); D_all=$(( NPROC - T_all > 0 ? NPROC - T_all : 1 ))
  T_mlp=$T_all; D_mlp=$D_all; T_vector=$T_all; D_vector=$D_all; T_raster=$T_all; D_raster=$D_all
  CACHE_WORKERS=${CACHE_WORKERS:-$NPROC}
fi
PIN_OK=0   # PIN=auto: pin each run to its own vCPU block if CPUs 0..NPROC-1 are online; PIN=1 force; PIN=0 off
if [ "$MODE" = parallel ] && command -v taskset > /dev/null; then
  if [ "$PIN" = 1 ] || { [ "$PIN" = auto ] && [ "$(cat /sys/devices/system/cpu/online 2>/dev/null)" = "0-$((NPROC - 1))" ]; }
  then PIN_OK=1; fi
fi
NEXT_CPU=0
alloc_cpus() {  # sets CPUS to the next block of $1 vCPUs (empty when not pinning)
  CPUS=""
  if [ "$PIN_OK" = 1 ]; then CPUS="$NEXT_CPU-$((NEXT_CPU + $1 - 1))"; NEXT_CPU=$((NEXT_CPU + $1)); fi
}

say "train_baselines: SMOKE=$SMOKE MODE=$MODE NPROC=$NPROC OUT=$OUT"
say "data: $TRAIN_DB  N_SAMPLES=$N_SAMPLES MIN_GAP_S=$MIN_GAP_S DATA_SEED=$DATA_SEED"
say "training: MODELS='$MODELS' SEEDS='$SEEDS' EPOCHS=$EPOCHS BATCH=$BATCH LR=$LR (x$LR_GAMMA at epochs" \
    "$LR_MILESTONES) EARLY_FRACTION=$EARLY_FRACTION"
say "threads (torch+workers): mlp $T_mlp+$D_mlp vector $T_vector+$D_vector raster $T_raster+$D_raster;" \
    "cache workers $CACHE_WORKERS; pinning $PIN_OK"

# ---------------------------------------------------------------------------------------------------- preflight
phase_start preflight
ls "$TRAIN_DB"/*.db > /dev/null 2>&1 || die "no .db files in $TRAIN_DB"
for d in "$SPLITS/val" "$SPLITS/test"; do
  [ -d "$d" ] || continue
  n=$(comm -12 <(ls "$TRAIN_DB" | grep '\.db$' | sort) <(ls "$d" | grep '\.db$' | sort) | wc -l)
  [ "$n" = 0 ] || die "$n logs of $d are also in $TRAIN_DB: training data must not include val/test logs"
done
case "$(cd "$TRAIN_DB" && pwd)" in */splits/val*|*/splits/test*) die "DB_FILES points at a val/test folder";; esac
df_dir=$OUT; while [ ! -d "$df_dir" ]; do df_dir=$(dirname "$df_dir"); done
free_gb=$(df -BG --output=avail "$df_dir" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 20 ] || die "only ${free_gb} GB free under $OUT"
( nuplan_env && python -c "import nuplan, exp2_nuplan.models.ego_mlp, exp2_nuplan.training.pl2_compat" ) \
  || die "nuplan env / exp2 code not importable"
PRETRAINED=false; HF_OFFLINE=0
if [ "$DRYRUN" = 1 ]; then
  PRETRAINED=true; HF_OFFLINE=1
  say "DRYRUN: would fetch timm resnet18 ImageNet weights and write $OUT/run_config.json"
elif has raster; then
  # timm ImageNet weights for resnet18 (raster_model's default pretrained: true), fetched once before training
  if ( nuplan_env && python -c "import timm; timm.create_model('resnet18', pretrained=True, num_classes=0, in_chans=4)" ) \
      > "$OUT/logs/pretrained_download.log" 2>&1; then
    PRETRAINED=true; HF_OFFLINE=1
  else
    say "WARNING: could not fetch timm resnet18 ImageNet weights (see logs/pretrained_download.log):" \
        "raster runs start from random weights"
  fi
fi
DEVKIT_COMMIT=$(git -C /data/code/nuplan-devkit rev-parse --short HEAD 2>/dev/null)
[ "$DRYRUN" = 1 ] || ( nuplan_env && python - "$OUT/run_config.json" <<EOF
import json, sys, socket, time, torch, pytorch_lightning as pl, timm
cfg = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=socket.gethostname(),
           nproc=$NPROC, mode="$MODE", smoke=$SMOKE, out="$OUT", train_db="$TRAIN_DB",
           models="$MODELS".split(), seeds=[int(s) for s in "$SEEDS".split()], n_samples=$N_SAMPLES,
           min_gap_s=$MIN_GAP_S, data_seed=$DATA_SEED, epochs=$EPOCHS, batch_size=$BATCH, lr=$LR,
           lr_schedule=dict(scheduler="torch.optim.lr_scheduler.MultiStepLR (nuPlan multistep_lr)",
                            milestones_epochs=[$LR_MILESTONES], gamma=$LR_GAMMA),
           weight_decay=5e-4, optimizer="Adam (nuPlan optimizer=adam)", early_fraction=$EARLY_FRACTION,
           raster_pretrained_imagenet=("$PRETRAINED" == "true"), raster_channels_last=True,
           threads=dict(mlp=[$T_mlp, $D_mlp], vector=[$T_vector, $D_vector], raster=[$T_raster, $D_raster],
                        cache_workers=$CACHE_WORKERS, pinned=bool($PIN_OK), note="[torch threads, loader workers]"),
           nuplan_devkit_commit="$DEVKIT_COMMIT", torch=torch.__version__, pytorch_lightning=pl.__version__,
           timm=timm.__version__)
json.dump(cfg, open(sys.argv[1], "w"), indent=1)
EOF
) || die "could not write run_config.json"
phase_end

# ---------------------------------------------------------------------------------------------------- 0. prewarm
# Caching issues small random SQLite reads; on a cold page cache (e.g. right after an instance resize) they are
# IOPS-bound on EBS. Reading every training DB once sequentially (~52 GB, a few minutes at the volume's throughput)
# turns them into page-cache hits. Full mode only (PREWARM=auto) and only if the DBs fit in half the free RAM.
need_cache=0
for m in $MODELS; do [ -f "$OUT/cache/$m/DONE" ] || need_cache=1; done
if [ "$need_cache" = 1 ] && { [ "$PREWARM" = 1 ] || { [ "$PREWARM" = auto ] && [ "$SMOKE" != 1 ]; }; }; then
  db_gb=$(( $(du -sb "$TRAIN_DB" | cut -f1) / 1000000000 ))
  avail_gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1000000 ))
  if [ "$DRYRUN" = 1 ]; then
    say "DRYRUN prewarm: would read ${db_gb} GB of DBs if 2 x ${db_gb} GB < available RAM ${avail_gb} GB"
  elif [ $((2 * db_gb)) -lt "$avail_gb" ]; then
    phase_start prewarm
    find "$TRAIN_DB" -maxdepth 1 -name '*.db' -print0 | xargs -0 -P 16 -n 16 cat > /dev/null
    phase_end
  else
    say "prewarm skipped: DBs ${db_gb} GB, available RAM ${avail_gb} GB"
  fi
fi

# ---------------------------------------------------------------------------------------------------- 1. caching
run_cache() {  # <model> <workers>
  local m=$1 W=$2 log=$OUT/logs/cache_$1.log t0 rc
  if [ -f "$OUT/cache/$m/DONE" ]; then say "cache $m: done before, skipped"; return 0; fi
  if [ "$DRYRUN" = 1 ]; then
    say "DRYRUN cache $m: py_func=cache +training=$(exp_of "$m") $W worker processes, $N_SAMPLES samples," \
        "min_gap_s $MIN_GAP_S, seed $DATA_SEED -> $OUT/cache/$m"
    return 0
  fi
  say "cache $m: start ($W worker processes) -> $log"
  t0=$(date +%s)
  ( export OMP_NUM_THREADS=1; nuplan_env && cd "$OUT/logs" && \
    python "$CODE/exp2_nuplan/training/run_training.py" py_func=cache +training="$(exp_of "$m")" "$SEARCHPATH" \
      scenario_builder=exp2_thinned_nuplan scenario_builder.db_files="$TRAIN_DB" \
      scenario_builder.min_gap_s="$MIN_GAP_S" scenario_filter.limit_total_scenarios="$N_SAMPLES" \
      cache.cache_path="$OUT/cache/$m" worker=single_machine_thread_pool worker.use_process_pool=true \
      worker.max_workers="$W" seed="$DATA_SEED" gpu=false $(model_ov "$m" false) \
      group="$OUT/logs" experiment_name=cache job_name="$m" output_dir="$OUT/logs/cache_$m" ) > "$log" 2>&1
  rc=$?
  printf '{"model": "%s", "workers": %d, "wall_seconds": %d, "returncode": %d}\n' \
    "$m" "$W" "$(( $(date +%s) - t0 ))" "$rc" > "$OUT/cache/$m.timing.json"
  if [ "$rc" = 0 ] && grep -q "Completed dataset caching" "$log"; then
    touch "$OUT/cache/$m/DONE"; say "cache $m: ok in $(( $(date +%s) - t0 )) s"
  else
    say "cache $m: FAILED (rc=$rc), see $log"; return 1
  fi
}
phase_start cache
pids=()
for m in $MODELS; do
  if [ "$MODE" = parallel ]; then run_cache "$m" "$CACHE_WORKERS" & pids+=($!); else run_cache "$m" "$CACHE_WORKERS"; fi
done
for p in "${pids[@]}"; do wait "$p"; done
phase_end
if [ "$DRYRUN" != 1 ]; then
  for m in $MODELS; do [ -f "$OUT/cache/$m/DONE" ] || die "cache $m incomplete; fix and re-run (finished caches are kept)"; done
fi

# ---------------------------------------------------------------------------------------------------- 2. split
phase_start split
cache_args=()
for m in $MODELS; do cache_args+=(--cache "$m=$OUT/cache/$m:$(feats_of "$m")"); done
if [ "$DRYRUN" = 1 ]; then
  say "DRYRUN split: make_allowlist.py ${cache_args[*]} -> $OUT/train_tokens.txt, $OUT/data_split.json"
else
  ( nuplan_env && python "$CODE/exp2_nuplan/training/make_allowlist.py" --out "$OUT/train_tokens.txt" \
      --summary "$OUT/data_split.json" "${cache_args[@]}" \
      --forbidden-logs-dir "$SPLITS/val" --forbidden-logs-dir "$SPLITS/test" ) || die "make_allowlist failed"
fi
phase_end

# ---------------------------------------------------------------------------------------------------- 3. training
run_train() {  # <model> <seed> <torch threads> <loader workers> <cpu list or "">
  local m=$1 s=$2 T=$3 D=$4 cpus=$5 dir=$OUT/$1/seed$2 log=$OUT/logs/train_$1_seed$2.log rc pre="" cl=0
  if [ -f "$dir/final.ckpt" ] && [ -f "$dir/train_manifest.json" ]; then say "train $m seed$s: done before, skipped"; return 0; fi
  [ -n "$cpus" ] && pre="taskset -c $cpus"
  [ "$m" = raster ] && cl=1
  if [ "$DRYRUN" = 1 ]; then
    say "DRYRUN train $m seed$s: torch threads $T, loader workers $D${cpus:+, cpus $cpus}, channels_last $cl," \
        "$EPOCHS epochs, batch $BATCH, lr $LR (x$LR_GAMMA at epochs $LR_MILESTONES), pretrained $PRETRAINED -> $dir"
    return 0
  fi
  say "train $m seed$s: start (torch threads $T, loader workers $D${cpus:+, cpus $cpus}) -> $log"
  local t0; t0=$(date +%s)
  # file_system sharing: nuPlan's VectorMap batches hold thousands of small tensors (see run_training.py)
  # all thread variables: with MKL, PyTorch sizes its pool from MKL_NUM_THREADS, which env.sh may have preset to 1
  ( export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T NUMBA_NUM_THREADS=$T            EXP2_CHANNELS_LAST=$cl HF_HUB_OFFLINE=$HF_OFFLINE EXP2_SHARING_STRATEGY=file_system
    ulimit -n "$(ulimit -Hn)" 2> /dev/null; nuplan_env && cd "$OUT/logs" && \
    $pre python "$CODE/exp2_nuplan/training/run_training.py" py_func=train +training="$(exp_of "$m")" "$SEARCHPATH" \
      cache.cache_path="$OUT/cache/$m" cache.use_cache_without_dataset=true \
      splitter=exp2_log_hash splitter.allowlist="$OUT/train_tokens.txt" \
      callbacks=exp2_baseline_callbacks callbacks.exp2_checkpoint_callback.dirpath="$dir" \
      callbacks.exp2_checkpoint_callback.early_fraction="$EARLY_FRACTION" \
      lightning.trainer.params.max_epochs="$EPOCHS" +lightning.trainer.params.log_every_n_steps=10 \
      data_loader.params.batch_size="$BATCH" data_loader.params.num_workers="$D" optimizer.lr="$LR" \
      lr_scheduler=multistep_lr "lr_scheduler.milestones=[$LR_MILESTONES]" lr_scheduler.gamma="$LR_GAMMA" \
      worker=sequential seed="$s" gpu=false $(model_ov "$m" "$PRETRAINED") \
      group="$OUT/logs" experiment_name=train job_name="${m}_seed$s" output_dir="$OUT/logs/train_${m}_seed$s" \
  ) > "$log" 2>&1
  rc=$?
  printf '{"model": "%s", "seed": %d, "process_wall_seconds": %d, "returncode": %d, "torch_threads": %d, "loader_workers": %d, "cpus": "%s"}\n' \
    "$m" "$s" "$(( $(date +%s) - t0 ))" "$rc" "$T" "$D" "$cpus" > "$OUT/logs/train_${m}_seed$s.timing.json"
  if [ "$rc" = 0 ] && [ -f "$dir/final.ckpt" ] && [ -f "$dir/train_manifest.json" ]; then
    say "train $m seed$s: ok in $(( $(date +%s) - t0 )) s"
  else
    say "train $m seed$s: FAILED (rc=$rc), see $log"; return 1
  fi
}
phase_start train
pids=(); names=()
for m in raster vector mlp; do
  has "$m" || continue
  eval "T=\$T_$m; D=\$D_$m"
  for s in $SEEDS; do
    alloc_cpus $((T + D))
    if [ "$MODE" = parallel ]; then
      run_train "$m" "$s" "$T" "$D" "$CPUS" & pids+=($!); names+=("$m/seed$s")
    else
      run_train "$m" "$s" "$T" "$D" "" || FAILED="$FAILED $m/seed$s"
    fi
  done
done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || FAILED="$FAILED ${names[$i]}"; done
phase_end

# ---------------------------------------------------------------------------------------------------- 4. manifest
if [ "$DRYRUN" = 1 ]; then say "DRYRUN: would write $OUT/manifest.json"; exit 0; fi
( nuplan_env && python "$CODE/exp2_nuplan/training/write_manifest.py" --out "$OUT" ) || say "WARNING: write_manifest failed"
if [ -n "${FAILED:-}" ]; then
  say "FAILED runs:$FAILED (re-run this script to retry them; finished runs are kept)"; exit 1
fi
say "all runs finished; checkpoints under $OUT/<model>/seed<k>/, manifest $OUT/manifest.json"
