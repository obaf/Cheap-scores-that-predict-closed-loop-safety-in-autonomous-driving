# Experiment 2 - training the "small baselines" (README-training.md)

Written 28 Oct 2026 (UTC) on the m8a.2xlarge, where the pipeline was only VALIDATED (tiny subset, <= 4 threads).
The full training runs later on the m8a.48xlarge (192 vCPUs) with one command. Host copy:
/data/code/envs/README-training.md; laptop copy: experiment2/env/README-training.md.

## 1. What is trained

| name | model | input | nuPlan config |
|---|---|---|---|
| mlp | ego-status MLP (ours, `exp2_nuplan/models/ego_mlp.py`, 3 x 512 ReLU, 0.56 M params) | ego only: present + 4 past rear-axle poses over 1.5 s (ego part of nuPlan's AgentsFeatureBuilder); no map, no agents | `+training=training_ego_mlp_model`, `model=ego_mlp_model` |
| vector | nuPlan `simple_vector_model` (VectorMapSimpleMLP, 0.11 M params) | lanes within 20 m + ego and vehicle history (4 poses / 1.5 s) | `+training=training_simple_vector_model` |
| raster | nuPlan `raster_model` with timm `resnet18` (11.2 M params), ImageNet initialisation (raster_model's default `pretrained: true`) | nuPlan's default rasterisation: 224 x 224 x 4 channels, 0.5 m/pixel | `+training=training_raster_model model.model_name=resnet18` |

All three predict 16 poses at 0.5 s (8 s) in the ego frame (nuPlan EgoTrajectoryTargetBuilder), loss = nuPlan
`imitation_objective` (MSE on x, y + L1 on heading), metrics ADE/AHE/FDE/FHE. Data augmentation = the experiment
configs' defaults: vector uses nuPlan's `kinematic_agent_augmentation`; raster and mlp none (the ego-status baselines
of AD-MLP and NAVSIM use none).

Checkpoints: `/data/exp2/baselines/<mlp|vector|raster>/seed<k>/{early,final}.ckpt`, k = 0, 1, 2.
EARLY = after 15 % of the run's optimizer steps (saved explicitly by a callback at step round(0.15 x total));
FINAL = after the last step (not the best-validation model). Both are Lightning checkpoints of nuPlan's
LightningModuleWrapper (weights + hyper-parameters, no optimizer state) that nuPlan's MLPlanner loads directly.

## 2. Commands

```bash
# full run on the m8a.48xlarge (sized from nproc; see section 7 for the time estimate)
DRYRUN=1 bash /data/exp2/code/train_baselines.sh          # optional: print the thread/CPU plan first
nohup bash /data/exp2/code/train_baselines.sh > /data/exp2/baselines_run.out 2>&1 < /dev/null &
grep "exp2 epoch" /data/exp2/baselines/logs/train_raster_seed0.log   # progress, one line per epoch
# pipeline check on any host (500 samples, 1 epoch, batch 16, <= 4 threads, about 3 min;
# outputs under /data/exp2/train_smoke/baselines; add SEEDS="0 1 2" for all nine runs)
SMOKE=1 bash /data/exp2/code/train_baselines.sh
# after training: drive every registry entry on one val scenario (closed_loop_reactive_agents)
source /data/code/envs/env.sh nuplan && python /data/exp2/code/sim_check_baselines.py --parallel 12
# ... or check the smoke checkpoints instead
python /data/exp2/code/sim_check_baselines.py --replace /data/exp2/baselines/=/data/exp2/train_smoke/baselines/
# experiment jobs: a runner that merges comma-separated registries (runner.py.new on 28 Oct) takes
#   --registry /data/exp2/code/planners_core.yaml,/data/exp2/code/planners_baselines.yaml
# the 28 Oct runner.py takes one file: --registry /data/exp2/code/planners_baselines.yaml
```

`train_baselines.sh` steps (each logged to `$OUT/logs/`, phase times in `$OUT/logs/phases.tsv`):
0. preflight (train folder has no val/test log, free disk, imports, timm resnet18 weights fetched once, `run_config.json`);
   full mode also reads every train_pittsburgh DB once sequentially ("prewarm", 52 GB) so that the caching workers'
   random SQLite reads hit the page cache instead of EBS IOPS;
1. cache: nuPlan `run_training.py py_func=cache` once per model type (the three in parallel, nproc/3 worker processes
   each, `worker=single_machine_thread_pool worker.use_process_pool=true`), into `$OUT/cache/<model>/`;
2. split: `make_allowlist.py` keeps the samples present in all three caches (`train_tokens.txt`) and writes
   `data_split.json` (counts per split, scenario types; fails if any sample comes from a val/test log);
3. train: nuPlan `run_training.py py_func=train` (via the PL-2 wrapper, section 5) for 3 models x 3 seeds, all in
   parallel, each pinned to its own block of vCPUs (`taskset`);
4. manifest: `write_manifest.py` -> `$OUT/manifest.json` (+ `manifest_runs.csv`): per run train/val samples, epochs,
   steps, seed, batch, threads, wall time, samples/s, train/val loss at EARLY and FINAL, checkpoint paths; plus run
   configuration, data split, caching statistics and phase times.

Re-running resumes: caches with a `DONE` marker and runs with `final.ckpt` + `train_manifest.json` are skipped.
Environment overrides: `N_SAMPLES EPOCHS SEEDS MODELS BATCH LR LR_MILESTONES LR_GAMMA MIN_GAP_S DATA_SEED
EARLY_FRACTION NPROC CACHE_WORKERS PIN PREWARM DRYRUN OUT DB_FILES`. `DRYRUN=1` prints the thread/CPU plan and the
steps without running anything. Progress of a run: `grep exp2 $OUT/logs/train_<model>_seed<k>.log` (start, EARLY,
one line per epoch, FINAL).

## 3. Data

* Source: nuPlan v1.1 `train_pittsburgh` only (1,560 logs, 107.8 h; 5.89 M lidar frames in nuPlan's "valid" scenes).
  `scenario_builder.db_files=/data/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh`. nuPlan's val and test logs are
  never read for training: the preflight checks that no train_pittsburgh log name is in `splits/val` or `splits/test`,
  and `make_allowlist.py` re-checks every selected sample.
* Candidates: nuPlan's `training_scenarios` filter (expand_scenarios, remove_invalid_goals; every frame is a 1-frame
  scenario). Because enumerating all 5.9 M frames would build millions of NuPlanScenario objects in the main process,
  `ThinnedNuPlanScenarioBuilder` (= nuPlan's builder) keeps, per log and inside the worker processes, at most one
  frame every `MIN_GAP_S` = 2.0 s (first frame of each gap, any scenario type): 75,737 candidates.
* Selection: nuPlan's `scenario_filter.limit_total_scenarios=N_SAMPLES` (random.sample with `seed=DATA_SEED=0`):
  N_SAMPLES = 50,000. The three cache jobs use the same builder, filter and seed, so they select the same samples
  (verified: identical token lists); the allowlist (intersection) guards against per-model feature failures.
* Split: by LOG, never by frame: a log is validation if sha1("exp2-train-pittsburgh:" + log name) falls in the lowest
  5 % (`LogHashSplitter`, `splitter=exp2_log_hash`). Expected about 47.5 k train / 2.5 k val samples. Validation is
  for monitoring and the manifest only (FINAL is not chosen on it).
* The natural frame distribution is kept (no per-type balancing). In a 500-sample draw the commonest nuPlan tags were
  stationary 31 %, medium_magnitude_speed 12 %, high_magnitude_speed 11 %, traversing_intersection 11 %.

## 4. Training settings (all three models)

Adam (nuPlan `optimizer=adam`) with lr 1e-3 (Adam's default), x0.1 after epochs 7 and 9 (nuPlan
`lr_scheduler=multistep_lr`), weight decay 5e-4 (nuPlan default), batch 64, 10 epochs (about 7.4 k optimizer steps;
EARLY at about 1.1 k, i.e. 1.5 epochs, still at lr 1e-3), fp32 on CPU, seeds 0, 1, 2 (`seed=k` ->
`pl.seed_everything`: initialisation, data order, augmentation). The same samples, batch, epochs, optimizer and
schedule for all models. Raster runs use PyTorch's channels_last memory layout (`EXP2_CHANNELS_LAST=1`): performance
only, 1.35x faster on this CPU, same computation; checkpoints load into nuPlan's unchanged RasterModel.

Why lr 1e-3 and not nuPlan's default 5e-5 or 1e-4: on a 4,000-sample check (3,827 train / 173 val, batch 64, 4 epochs
= 236 steps, constant lr) the validation loss after 236 steps was, at lr 1e-4 / 1e-3: mlp 50.4 / 39.4, vector
49.4 / 42.5; raster after 118 steps 230.6 / 58.6 (149.1 after 236 steps at 1e-4). The imitation loss is in m^2
(MSE of x, y over 16 poses) and regressing 8 s trajectories from normalised features needs large output weights, which
Adam reaches slowly at small learning rates. No run diverged at 1e-3. The step decay gives a converged FINAL model.

## 5. nuPlan's training framework under pytorch-lightning 2.0.1

nuplan-devkit e924167 targets pytorch-lightning 1.3.8, the shared `nuplan` env has 2.0.1 (needed by planTF, pluto and
Diffusion-Planner) and was NOT changed. Caching runs unchanged. `exp2_nuplan/training/run_training.py` = nuPlan's
`run_training.py` main() with run-time patches (`pl2_compat.py`; nuplan-devkit files untouched):
1. `build_trainer` -> PL-2 Trainer (CPU, fp32) from the PL-2-valid keys of `lightning.trainer.params` and the
   callbacks of `cfg.callbacks`; TensorBoard + CSV loggers under the run's output_dir;
2. `DataModule.transfer_batch_to_device` gets PL-2's `dataloader_idx` argument;
3. `LightningModuleWrapper._step` logs with `batch_size=len(scenarios)` (PL 2 otherwise guesses it, e.g. the number
   of lanes of the first sample) and epoch means for training too; `forward` adds the optional channels_last layout.
nuPlan's `default_callbacks` use the removed `on_epoch_end` hook, so training uses `callbacks=exp2_baseline_callbacks`
(LearningRateMonitor + `BaselineCheckpointCallback`: early/final checkpoints, per-run `train_manifest.json`, and the
mean validation loss at both checkpoints computed the same way).

## 6. Simulation (planner registry `/data/exp2/code/planners_baselines.yaml`)

Entries: mlp_early, mlp_final, vector_early, vector_final, raster_early, raster_final (seed 0) and the seed variants
mlp_final_s1/_s2, vector_final_s1/_s2, raster_final_s1/_s2 (`seed_of`, `seed`). Example (what runner.build_command
produces for raster_final):
```
... run_simulation.py +simulation=closed_loop_reactive_agents ... planner=ml_planner model=raster_model
    model.model_name=resnet18 model.pretrained=false 'planner.ml_planner.model_config=${model}'
    planner.ml_planner.checkpoint_path=/data/exp2/baselines/raster/seed0/final.ckpt
```
mlp entries add `searchpath: [pkg://exp2_nuplan.config]` (for `model=ego_mlp_model`). `model.pretrained=false` in
simulation only avoids timm's ImageNet download at model construction; the checkpoint provides all weights.

## 7. Measured throughput (m8a.2xlarge, AMD EPYC 9R45, other agents' simulations running) and full-run estimate

Caching (nuPlan `py_func=cache`; one worker process = one thread; 4,000 samples with 4 workers; the mlp job ran
first with partly cold DB pages, raster last with warm ones):

| | mlp | vector | raster |
|---|---|---|---|
| ms per sample per worker | 47 | 45 | 21 |
| samples/s per worker thread | 21 | 22 | 48 |
| cache size per 1,000 samples | 8.5 MB | 20 MB | 18 MB |

Enumerating and thinning the 1,560 DBs (75,737 candidates at 2 s): 9 s with 4 workers on warm pages, about 80 s cold.
Cold sequential read rate of /data: 264 MB/s (36.8 GB, 16 readers), so the 52 GB prewarm takes about 3.3 min.

Training (`exp2_nuplan/training/bench_train.py` on cached samples, batch 64, plus the 4,000-sample nuPlan runs):

| | mlp | vector | raster (resnet18) |
|---|---|---|---|
| load + collate, ms/sample per DataLoader worker | 1.16 | 4.86 (0.26 without nuPlan's kinematic augmentation) | 0.92 |
| fwd + loss + bwd + Adam, ms/sample, 1 torch thread | 0.05 | 1.06-1.47 (Python loop over samples) | 51.0 channels_last (70.1 default NCHW) |
| same, 2 / 4 torch threads | 0.04 / 0.03 | 0.66 / 0.62 | 27.7 / 15.7 (efficiency 92 % / 81 %) |
| nuPlan training end to end, 2 torch threads + 2 workers | 2,400-2,950 samples/s | 190-310 samples/s | 37.3 samples/s (18.7 per torch thread) |

Full run on 192 vCPUs (`DRYRUN=1 bash train_baselines.sh` prints this plan): caches 3 jobs at once with 64 worker
processes each; training runs raster seeds 0-2 with 50 torch threads + 2 loader workers each (vCPUs 0-51, 52-103,
104-155), vector 2 + 6 (156-179), mlp 1 + 3 (180-191). N_SAMPLES = 50,000 -> about 47.5 k training samples; 10
epochs = 475 k samples per run.

| phase | estimate |
|---|---|
| preflight (imports, resnet18 weights) | < 1 min |
| prewarm (only if the page cache is cold, e.g. after the resize) | 3-4 min |
| caching, 3 models in parallel | 1-2 min (50 k x <= 50 ms / 64 workers = 40 s per model, plus start-up) |
| split (allowlist) | < 1 min |
| training (raster-bound) | 475 k x 51 ms / (50 threads x eta): eta 0.45 -> 18 min (eta 0.7: 12 min, eta 0.25: 32 min); vector about 8 min and mlp 3-4 min run alongside |
| total | about 24 min expected (17-38 min) -> within the 45 min target |

eta = parallel efficiency of one PyTorch process at 50 threads (0.81 measured at 4 threads; 50 threads cannot be
measured on the m8a.2xlarge, 0.45 assumed). Check early: `grep "exp2 epoch" $OUT/logs/train_raster_seed0.log` prints
one line per epoch; if the first raster epoch takes more than about 3.5 min the run would pass about 40 min, and
EPOCHS=8 (or a smaller N_SAMPLES) brings it back under 45 min. Disk: caches about 2.5 GB, checkpoints about 0.3 GB.

## 8. Validation done on the m8a.2xlarge (28 Oct)

All on the m8a.2xlarge with <= 4 threads, outputs under /data/exp2/train_smoke/ (other agents' simulations were
running at the same time, so the timings are conservative).
1. `SMOKE=1 SEEDS="0 1 2" bash train_baselines.sh` from scratch (500 samples, 1 epoch, batch 16, 2 torch threads +
   2 loader workers per run): preflight 6 s, three caches 105 s (75,737 candidates, 500 selected, 0 failed
   features, identical token lists in the three caches), split 1 s (484 train samples from 264 logs, 16 val from 13
   logs, no val/test log), nine trainings 77 s, manifest written; 3.2 min in total. Outputs:
   /data/exp2/train_smoke/baselines/ (manifest.json, <model>/seed<k>/{early,final}.ckpt).
2. `sim_check_baselines.py --replace /data/exp2/baselines/=/data/exp2/train_smoke/baselines/`: all 12 registry
   entries, built with runner.build_command, drove val scenario ac83bb7914fa5c49 (behind_long_vehicle, log
   2021.<ip>.02_veh-12_01831_03390) in closed_loop_reactive_agents and wrote nuPlan metrics
   (/data/exp2/train_smoke/sim/sim_check.json). MLPlanner loads each checkpoint with a strict state-dict match, and
   the entries drove differently (e.g. mlp_early no collision, mlp_final an at-fault collision), so each uses its
   own weights. Planner time per 0.1 s step: mlp 14 ms, vector 26-32 ms, raster 42 ms (IDM 16 ms); simulated scenario
   15-20 s, hence the registry `speed` values 1.0 / 1.15 / 1.3. The same check with `--protocol ols`
   (open_loop_boxes plus the runner's trajectory-capture callback) for mlp_final, vector_final and raster_final also
   passed and wrote the captured trajectories (/data/exp2/train_smoke/sim_ols/). The interplan protocol was not run.
3. `runner.py make --registry planners_baselines.yaml` (to a scratch jobs file): 492 jobs (6 base entries x ols /
   clsnr / clsr on the 688 scenarios + interplan, 6 seed variants x ols / clsnr / clsr on seed140).
4. A 4,000-sample run (4 epochs, batch 64) for throughput and the learning-rate check (section 4); the parallel code
   paths of train_baselines.sh (three caches at once, runs in the background with `taskset`) with NPROC=24 and
   CACHE_WORKERS=1; `DRYRUN=1 NPROC=192 PIN=1` for the full plan (section 7).
Problems met and fixed on the way: nuPlan training under PL 2 (section 5); "Too many open files" in the vector
DataLoader at batch 64 (nuPlan's VectorMap batches hold thousands of tensors) -> `EXP2_SHARING_STRATEGY=file_system`
(also 33 % faster than raising the descriptor limit) plus a raised `ulimit -n`; Hydra needs
`+lightning.trainer.params.log_every_n_steps` (append, not override); nuPlan's val loader drops the last partial
batch, so smoke runs use batch 16 to keep a validation batch.

## 9. Decisions made without the user (candidates for experiment2/assumptions.md)

T1. nuPlan's training framework runs under the env's pytorch-lightning 2.0.1 through run-time shims (section 5)
    instead of installing nuPlan's PL 1.3.8, which would break planTF, pluto and Diffusion-Planner in the shared env.
T2. Training pool: train_pittsburgh only, nuPlan's `training_scenarios` filter (valid mission goals, every frame of a
    valid scene a 1-frame scenario), frames thinned to >= 2 s apart within each log (75,737 candidates), then 50,000
    drawn at random (seed 0), the same for all three models; scenario types left at their natural frequencies.
T3. Validation: a salted hash of the log name puts 5 % of the logs (about 2.5 k samples) in validation; whole logs,
    so no validation frame is near a training frame. Validation is reported, not used to pick checkpoints.
T4. EARLY = weights after round(0.15 x total optimizer steps), about 1.5 of 10 epochs; FINAL = weights after the
    last step (not the best validation loss).
T5. Ego-status MLP: input = present + 4 past ego poses over 1.5 s (the ego branch of simple_vector_model's input),
    3 hidden layers of 512 (NAVSIM EgoStatusMLP width), no data augmentation (as AD-MLP/NAVSIM). nuPlan has no
    driving-command feature, so unlike NAVSIM's EgoStatusMLP it has no command input; velocity and acceleration are
    implicit in the pose history.
T6. Raster: timm resnet18 with ImageNet initialisation (nuPlan raster_model's default `pretrained: true`), default
    rasterisation; channels_last layout during training (speed only). Vector: nuPlan's `training_simple_vector_model`
    experiment as is, including its kinematic data augmentation.
T7. Optimisation, identical for all models: Adam, lr 1e-3 (chosen on the 4 k-sample check, section 4), x0.1 after
    epochs 7 and 9, weight decay 5e-4, batch 64, 10 epochs, fp32, seeds 0, 1, 2.
T8. Budget: 50,000 samples x 10 epochs, sized so that caching + all nine runs finish in about 24 min on the
    m8a.48xlarge (17-38 min range for the unknown 50-thread efficiency), within the 45-minute target.
T9. Simulation of the raster entries uses `model.pretrained=false` (all weights come from the checkpoint).

## 10. Packages and downloads

No package was installed, upgraded or downgraded. Downloaded once: timm ImageNet weights `timm/resnet18.a1_in1k`
(Hugging Face hub, into `~/.cache/huggingface/hub`; timm adapts conv1 from 3 to 4 input channels). Training runs set
`HF_HUB_OFFLINE=1` after this download. To be listed in experiment2/data-used.md.

## 11. Files (all under /data/exp2/code; laptop mirror experiment2/code/, same relative paths)

| file | role |
|---|---|
| train_baselines.sh | the one command (sections 2-4) |
| planners_baselines.yaml | planner registry entries (section 6) |
| sim_check_baselines.py | drives registry entries on one scenario with runner.build_command's command |
| exp2_nuplan/models/ego_mlp.py | EgoStatusMLP (TorchModuleWrapper) |
| exp2_nuplan/config/model/ego_mlp_model.yaml | its Hydra model config |
| exp2_nuplan/config/training/training_ego_mlp_model.yaml | its experiment config (as nuPlan's training_raster_model) |
| exp2_nuplan/config/scenario_builder/exp2_thinned_nuplan.yaml | nuPlan builder + per-log thinning |
| exp2_nuplan/config/splitter/exp2_log_hash.yaml | log-hash train/val split + allowlist |
| exp2_nuplan/config/callbacks/exp2_baseline_callbacks.yaml | PL-2 training callbacks |
| exp2_nuplan/training/run_training.py, pl2_compat.py | nuPlan run_training.py under PL 2.0.1 |
| exp2_nuplan/training/scenario_builder.py, splitter.py, checkpointing.py | builder, splitter, EARLY/FINAL callback |
| exp2_nuplan/training/make_allowlist.py, write_manifest.py | common samples + split; manifest |
| exp2_nuplan/training/bench_train.py | per-sample load/compute benchmark used for the sizing |
