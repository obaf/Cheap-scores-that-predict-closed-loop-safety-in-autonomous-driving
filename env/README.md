# Experiment 2 software environments (EC2 host)

Set up 2026-10-28 on the AWS host `ip-<host>` (Ubuntu 22.04, m8a.2xlarge: 8 vCPU AMD EPYC 9R45, 32 GB RAM, no GPU;
root disk 100 GB, data disk `/data` 500 GB). Everything here was run on that host. Nothing under `/data/nuplan`,
`/data/checkpoints`, `/data/.fetch`, `/data/downloads` or `/data/tmp_extract` was created, moved or changed.

## 1. Summary

Two conda environments (Miniforge3 at `/home/ubuntu/miniforge3`, both Python 3.9.23, CPU-only torch 2.0.1 wheels):

| env | runs | why separate |
|---|---|---|
| `nuplan` (main, 2.8 GB) | nuplan-devkit e924167 simulation (open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents) and its training framework (through the PL-2 entry point, section 8); tuplan_garage PDM-Closed/Open/Hybrid, GC-PGP, PlanCNN, Urban Driver; interPlan benchmark + its IDM-MOBIL planner; Diffusion Planner; PlanTF and PLUTO (model, training, planner); IDM, log replay, constant velocity | - |
| `navsim` (2.9 GB) | NAVSIM v2.2 (359c7f7) scorer: metric cache, PDM simulator/scorer, PDMS and EPDMS | NAVSIM pins hydra-core 1.2.0, pytorch-lightning 2.2.1, opencv 4.9 and unpinned newer pandas/scipy; nuplan-devkit pins hydra 1.1.0rc1 etc. The NAVSIM metric cache also fails under the main env's scipy 1.9.1 (section 10). |

Activation (env vars of the brief are in `env.sh`; the optional argument activates the env):

```bash
source /data/code/envs/env.sh nuplan     # or: source /data/code/envs/env.sh navsim
```

`env.sh` exports NUPLAN_DATA_ROOT=/data/nuplan/dataset, NUPLAN_MAPS_ROOT=/data/nuplan/dataset/maps,
NUPLAN_MAP_VERSION=nuplan-maps-v1.0, NUPLAN_EXP_ROOT=/data/nuplan/exp, OPENSCENE_DATA_ROOT=/data/navsim/dataset,
NAVSIM_EXP_ROOT=/data/navsim/exp, NAVSIM_DEVKIT_ROOT=/data/code/navsim, plus convenience paths (NUPLAN_DEVKIT_ROOT,
TUPLAN_GARAGE_ROOT, INTERPLAN_ROOT, DIFFUSION_PLANNER_ROOT, PLANTF_ROOT, PLUTO_ROOT, NUPLAN_DB_VAL, NUPLAN_DB_TEST,
NUPLAN_DB_TRAIN_PIT), `PYTHONPATH=/data/code/envs/tools` (for `exp2tools`), HYDRA_FULL_ERROR=1, WANDB_MODE=disabled and
OMP/MKL/OPENBLAS/NUMEXPR/NUMBA thread counts defaulting to 1 (override by exporting OMP_NUM_THREADS before sourcing).
`/data/nuplan/exp` and `/data/navsim/*` do not exist yet (not created by this setup).

## 2. Code checkouts (`/data/code/<name>`)

| repo | commit | install in env | local changes |
|---|---|---|---|
| nuplan-devkit | e9241677 (nuplan-devkit-v1.2-13; `nuplan/` package identical to tag v1.2) | nuplan: `pip install -e` | none |
| tuplan_garage | b51d5d04 | nuplan: `pip install -e --no-deps` | none |
| interPlan | a85024a4 | nuplan: `pip install -e --no-deps` | none |
| Diffusion-Planner | a3a621f0 | nuplan: `pip install -e --no-deps` | none |
| planTF | f5406085 | not installable; run from repo root with `PYTHONPATH+=/data/code/planTF` | none |
| pluto | b9964b64 | not installable; run from repo root with `PYTHONPATH+=/data/code/pluto` | added `src/custom_training/__init__.py` (upstream run_training.py imports from a package that has no `__init__.py`) |
| navsim | 359c7f72 (v2.2) | navsim: `pip install -e` | none |

planTF and pluto both use the top-level package name `src`: give each planner entry its own PYTHONPATH and cwd, never both.

## 3. How the envs were built (recipes: `build_env_nuplan.sh`, `build_env_navsim.sh`)

`nuplan`: conda-forge python=3.9 pip=23.3.1; `torch==2.0.1+cpu torchvision==0.15.2+cpu` (download.pytorch.org/whl/cpu);
`pip install -r nuplan-devkit/requirements.txt` constrained by nuplan-devkit's own `requirements_lock.txt`
(`constraints/nuplan_lock_constraints.txt`, 186 pins: numpy 1.23.4, pandas 1.5.0, shapely 2.0.1, geopandas 0.12.2,
scipy 1.9.1, ray 2.0.0, hydra-core 1.1.0rc1, omegaconf 2.1.0rc1, opencv-python <ip>, SQLAlchemy 1.4.27, protobuf
3.20.2, setuptools 59.5.0, ...); then pytorch-lightning 2.0.1, torchmetrics 0.10.2 (planTF/pluto pins), timm 1.0.10 and
mmengine (Diffusion Planner), tensorboard 2.11.2, wandb 0.14.2, rich 13.3.4, numba 0.60.0, imageio, scikit-learn 1.2.2 and
positional-encodings 6.0.1 (tuplan_garage); natten 0.14.6 compiled from the PyPI sdist against torch 2.0.1 CPU
(`--no-build-isolation`, 1 min; the shi-labs wheel index is offline). torch 1.9/PL 1.3.8 of nuplan's lock were NOT used:
planTF/pluto/Diffusion Planner need PL 2.0.1 / torch 2.0.

`navsim`: exactly NAVSIM's `environment.yml` + `requirements.txt` + `pip install -e .`, except torch/torchvision are the CPU
wheels of the pinned versions. nuplan-devkit comes from git tag nuplan-devkit-v1.2 (ce3c323, non-editable; the yaml configs
of nuplan are not packaged in that install). Key versions: hydra-core 1.2.0, pytorch-lightning 2.2.1, pandas 2.3.3, scipy
1.13.1, shapely 2.0.7, geopandas 1.0.1, opencv-python <ip>, ray 2.51.2, timm 1.0.30, setuptools 65.5.1.

Exports: `nuplan.yml`, `nuplan-pip-freeze.txt`, `navsim.yml`, `navsim-pip-freeze.txt` (`conda env export --no-builds`
and `pip freeze`). The yml files list the editable repos as plain versions; rebuild with the scripts instead.
System packages added with apt: libgl1 libsm6 libxrender1 libxext6 (opencv-python needs libGL), cmake, ninja-build.
`conda clean -a` was run after the installs. Do not upgrade pip in these envs past 24.0: pip >= 24.1 rejects omegaconf
2.1.0rc1's metadata (`PyYAML>=5.1.*`).

## 4. DB layout decision: command-line `scenario_builder.db_files`, no symlinks

The nuplan builders default to `$NUPLAN_DATA_ROOT/nuplan-v1.1/trainval` (does not exist; interPlan's builder too).
Every command here passes `scenario_builder.db_files=<x>` where `<x>` is a `.db` file, a folder, or a list
`[a.db,b.db,folder]`; when db_files is set, data_root is not read. Measured cost of
`scenario_builder.db_files=$NUPLAN_DB_VAL` (all 1,381 val DBs) with one `scenario_tokens` entry: 4 s of scenario
extraction (whole one-scenario run 21 s), so passing a whole split folder is fine; a shard can also pass only its DBs.
`tools/find_token_db.py <db_dir> <token...>` maps scenario tokens to DB files (sqlite, read-only, 8 s for 3 tokens over val).
DB folders: `$NUPLAN_DB_VAL` (1,381 DBs), `$NUPLAN_DB_TEST` (the 7 interPlan logs), `$NUPLAN_DB_TRAIN_PIT`.

## 5. Common nuPlan simulation command (env `nuplan`)

```bash
source /data/code/envs/env.sh nuplan
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
  +simulation=closed_loop_reactive_agents \
  scenario_builder=nuplan scenario_builder.db_files=$NUPLAN_DB_VAL/2021.<ip>.11_veh-38_03639_04063.db \
  scenario_filter=all_scenarios 'scenario_filter.scenario_tokens=[cb468419f191575a]' \
  worker=sequential '~callback.simulation_log_callback' \
  group=<output root> experiment_name=<planner id> \
  <planner overrides, section 6>
```

- `+simulation=` closed_loop_reactive_agents | closed_loop_nonreactive_agents | open_loop_boxes (all three checked).
- Simulation-log saving off: `'~callback.simulation_log_callback'`. (`callback=[]` alone crashes: "Key 'callback' is not in
  struct"; it only works if another callback is added.)
- One scenario: `scenario_filter.scenario_tokens=[...]`, or `scenario_filter.num_scenarios_per_type=1
  scenario_filter.limit_total_scenarios=1`. nuPlan starts the simulation at token time + extraction offset (-3 s for
  most types; `nuplan/planning/script/config/common/scenario_builder/scenario_mapping/nuplan_scenario_mapping.yaml`),
  15 s, 0.1 s steps -> 149 planner calls.
- Per-step timing: add `+callback.step_timer._target_=exp2tools.step_timer_callback.StepTimerCallback
  +callback.step_timer.output_file=<file.jsonl> +callback.step_timer.tag=<tag>` (appends one JSON line per simulation:
  init_s, n_steps, step mean/median/p95/max, planner mean/median).
- Skip the PDF summary (about 2 s per run): `'main_callback=[time_callback,metric_file_callback,metric_aggregator_callback]'`.
- Parallel workers verified: `worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=N` and
  `worker=ray_distributed worker.threads_per_node=N` (4 scenarios each, all succeeded).
- Outputs: `<group>/<experiment_name>/<job_name>/<timestamp>/`: `metrics/<metric>.parquet` (per scenario),
  `aggregator_metric/<challenge>_weighted_average_metrics_<timestamp>.parquet` (one row per scenario, one per scenario type,
  and `scenario == 'final_score'`; column `score` = CLS-R / CLS-NR / OLS), `runner_report.parquet`
  (duration, compute_trajectory_runtimes_mean/median/std), `summary/summary.pdf`, `log.txt`.
  Planner name in the outputs is the class name, so GC-PGP, PlanCNN and Urban Driver all appear as `MLPlanner`:
  identify runs by `experiment_name`.

## 6. Planner integration

Hydra search paths (append to the command when the planner row says so):

```
TG='hydra.searchpath=[pkg://tuplan_garage.planning.script.config.common,pkg://tuplan_garage.planning.script.config.simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]'
IP='hydra.searchpath=[pkg://interplan.planning.script.config.common,pkg://interplan.planning.script.config.simulation,pkg://interplan.planning.script.experiments,pkg://tuplan_garage.planning.script.config.common,pkg://tuplan_garage.planning.script.config.simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.config.simulation,pkg://nuplan.planning.script.experiments]'
DP='hydra.searchpath=[pkg://diffusion_planner.config.scenario_filter,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]'
CK=/data/checkpoints/tuplan_garage ; DPCK=/data/checkpoints/diffusion_planner
```

| id | planner overrides (append to the section 5 command) |
|---|---|
| idm | `planner=idm_planner` |
| idm_mobil | `planner=idm_mobil_planner "$IP"` |
| log_replay | `planner=log_future_planner` |
| constant_velocity | `planner=simple_planner planner.simple_planner.max_velocity=100.0 planner.simple_planner.horizon_seconds=8.0` |
| pdm_closed | `planner=pdm_closed_planner "$TG"` |
| pdm_open | `planner=pdm_open_planner planner.pdm_open_planner.checkpoint_path=$CK/pdm_open_checkpoint.ckpt "$TG"` |
| pdm_hybrid | `planner=pdm_hybrid_planner planner.pdm_hybrid_planner.checkpoint_path=$CK/pdm_offset_checkpoint.ckpt "$TG"` |
| gc_pgp | `planner=ml_planner 'planner.ml_planner.model_config=${model}' planner.ml_planner.checkpoint_path=$CK/gc_pgp_checkpoint.ckpt model=gc_pgp_model model.aggregator.pre_train=false "$TG"` |
| plancnn | `planner=ml_planner 'planner.ml_planner.model_config=${model}' planner.ml_planner.checkpoint_path=$CK/plancnn_checkpoint.ckpt model=raster_model model.pretrained=false "$TG"` |
| urban_driver | `planner=ml_planner 'planner.ml_planner.model_config=${model}' planner.ml_planner.checkpoint_path=$CK/urbandriver_checkpoint.ckpt model=urban_driver_open_loop_model "$TG"` |
| diffusion_planner | `planner=diffusion_planner planner.diffusion_planner.config.args_file=$DPCK/args.json planner.diffusion_planner.ckpt_path=$DPCK/model.pth planner.diffusion_planner.device=cpu "$DP"` |
| diffusion_planner_guidance | same as above with `planner=diffusion_planner_guidance` |
| diffusion_planner_noema | diffusion_planner + `+planner.diffusion_planner.enable_ema=false` |

PlanTF and PLUTO use their own repo's `run_simulation.py` (own `config/default_simulation.yaml`, own planner config and
search path, nuPlan's configs via `pkg://`), run from the repo root with the repo on PYTHONPATH. Registry entries:

| id | cwd | extra PYTHONPATH | script | planner overrides |
|---|---|---|---|---|
| plantf | /data/code/planTF | /data/code/planTF | /data/code/planTF/run_simulation.py | `planner=planTF planner.imitation_planner.planner_ckpt=/data/checkpoints/plantf/planTF.ckpt enable_simulation_progress_bar=false number_of_gpus_allocated_per_simulation=0` |
| pluto | /data/code/pluto | /data/code/pluto | /data/code/pluto/run_simulation.py | `planner=pluto_planner planner.pluto_planner.planner_ckpt=/data/checkpoints/pluto/pluto_1M_aux_cil.ckpt enable_simulation_progress_bar=false number_of_gpus_allocated_per_simulation=0` |
| pluto_nopost | /data/code/pluto | /data/code/pluto | /data/code/pluto/run_simulation.py | pluto + `planner.pluto_planner._target_=exp2tools.pluto_no_post.PlutoPlannerNoPost` |

The rest of the section 5 command (`+simulation=..., scenario_builder..., scenario_filter..., worker...,
'~callback.simulation_log_callback', group, experiment_name`) is identical. planTF/pluto's default_simulation.yaml sets
`number_of_gpus_allocated_per_simulation: 1` and `enable_simulation_progress_bar: true`; override both as shown (with
ray, 1 GPU per task would never be scheduled on a GPU-less host).

### Timing (1 thread, worker=sequential, simulation log off)

Scenario `cb468419f191575a` (Val14, near_multiple_vehicles, log 2021.<ip>.11_veh-38_03639_04063, 149 steps),
host m8a.2xlarge (AMD EPYC 9R45), OMP_NUM_THREADS=1, one simulation at a time (other processes of the experiment were
occasionally running on the 8 vCPUs; repeated runs of the same planner agreed within about 10 %).
`step_ms_*` = mean wall time of one full simulation step (planner + observation update, i.e. the reactive IDM agents in
R + controller + propagation), R = closed_loop_reactive_agents, NR = closed_loop_nonreactive_agents, OL = open_loop_boxes;
`planner_ms_R` = `compute_trajectory` only; `process_wall_R` = whole command incl. Python start-up (about 5 s), scenario
building, metric computation and the PDF summary; scores are this one scenario only (sanity check, not a result).
Sources: `timing/final_timing.jsonl`, `timing/other_challenges.jsonl`, tables `timing/compact_table.md`,
`timing/final_timing_t1.md`, `timing/other_challenges.md`.

| planner | step_ms_R | p95_step_ms_R | planner_ms_R | max_rss_MB_R | process_wall_R | step_ms_NR | step_ms_OL | CLS_R | CLS_NR | OLS |
|---|---|---|---|---|---|---|---|---|---|---|
| idm | 50.2 | 50.8 | 12.7 | 865 | 0:18.58 | 24.9 | 21.7 | 0.0 | 0.991 | 0.0 |
| idm_mobil | 53.0 | 52.9 | 17.5 | 875 | 0:19.02 | 29.6 | 26.6 | 0.0 | 1.0 | 0.0 |
| log_replay | 43.0 | 42.9 | 5.5 | 858 | 0:16.96 | 18.5 | 15.8 | 0.999 | 0.999 | 1.0 |
| constant_velocity | 39.1 | 39.1 | 1.3 | 870 | 0:16.14 | 12.9 | 12.7 | 0.0 | 0.0 | 0.914 |
| pdm_closed | 116.5 | 121.8 | 83.0 | 884 | 0:27.45 | 93.6 | 99.3 | 0.975 | 0.992 | 0.95 |
| pdm_open | 43.2 | 55.3 | 9.7 | 876 | 0:16.77 | 19.8 | 15.7 | 0.95 | 0.95 | 0.882 |
| pdm_hybrid | 124.3 | 135.4 | 90.5 | 886 | 0:28.91 | 104.5 | 97.2 | 0.975 | 0.992 | 0.899 |
| gc_pgp | 135.1 | 138.5 | 98.4 | 904 | 0:30.86 | 108.6 | 108.4 | 0.867 | 0.861 | 0.928 |
| plancnn | 128.7 | 130.0 | 89.5 | 1076 | 0:32.05 | 104.1 | 105.0 | 0.984 | 0.992 | 0.85 |
| urban_driver | 113.3 | 397.2 | 80.6 | 908 | 0:28.29 | 91.3 | 90.8 | 0.0 | 0.0 | 0.894 |
| diffusion_planner | 272.2 | 570.0 | 237.6 | 1076 | 0:51.80 | 240.5 | 250.9 | 0.838 | 0.994 | 0.83 |
| diffusion_planner_guidance | 350.9 | 657.4 | 315.2 | 1076 | 1:03.30 | 307.3 | 316.3 | 0.843 | 1.0 | 0.0 |
| diffusion_planner_noema | 278.0 | 582.7 | 238.1 | 1073 | 0:52.55 | | | 0.813 | | |
| plantf | 128.2 | 128.3 | 89.8 | 1039 | 0:30.24 | 114.4 | 107.4 | 0.959 | 0.978 | 0.951 |
| pluto | 271.5 | 282.4 | 232.1 | 1064 | 0:51.08 | 243.6 | 246.1 | 0.952 | 0.981 | 0.947 |
| pluto_nopost | 172.8 | 169.0 | 137.3 | 1061 | 0:35.78 | 154.4 | 151.6 | 0.952 | 0.981 | 0.947 |

The plancnn rows were run without `model.pretrained=false`; with it the R run gave the same CLS-R (0.9843), 132.7 ms/step,
988 MB. Init (observation + planner initialisation, not included in step_ms) is at most 1.4 s per scenario.
interPlan scenarios are 30 s (300 steps): PDM-Closed 76-83 ms/step there.

With 4 threads per process (`timing/final_timing_t4.md`, R) only PlanCNN gains much:
pdm_closed 127.8 ms/step, plancnn 79.3, diffusion_planner 231.8, plantf 112.6, pluto 252.3, pluto_nopost 157.7.
So for throughput run one simulation per vCPU with 1 thread. Peak RSS is 0.7-1.1 GB per simulation process
(192 processes on m8a.48xlarge = about 210 GB). Results can change in the last digits with the thread count (PlanCNN CLS-R
0.9843 at 1 thread vs 0.9844 at 4): keep OMP_NUM_THREADS fixed across the study.

Registry-style check (`smoke/registry_check.sh`): plantf, pluto and pluto_nopost each with own cwd/PYTHONPATH/script,
`worker=ray_distributed worker.threads_per_node=2`, 2 scenarios, `number_of_gpus_allocated_per_simulation=0`: all
simulations succeeded.

### Planner notes

- **IDM** (`nuplan.planning.simulation.planner.idm_planner.IDMPlanner`, base class `abstract_idm_planner.py`).
  Parameters: `nuplan-devkit/nuplan/planning/script/config/simulation/planner/idm_planner.yaml` (target_velocity 10,
  min_gap_to_lead_agent 1.0, headway_time 1.5, accel_max 1.0, decel_max 3.0, planned_trajectory_samples 16,
  planned_trajectory_sample_interval 0.5, occupancy_map_radius 40); vary with `planner.idm_planner.<param>=<value>`.
  Do not confuse with the reactive background agents' IDM (`.../config/simulation/observation/idm_agents_observation.yaml`).
- **IDM + MOBIL**: the only MOBIL / lane-changing IDM in these repos is interPlan's
  `interplan.planning.simulation.planner.idm_mobil_planner.IDMMobilPlanner`
  (`interPlan/interplan/planning/simulation/planner/idm_mobil_planner.py`; config
  `interPlan/interplan/planning/script/config/simulation/planner/idm_mobil_planner.yaml`: the IDM parameters above plus
  politeness_factor 0.5, changing_threshold 0.2, push_lane_change true). It runs on plain nuPlan scenarios with nuPlan's
  run_simulation.py and the `$IP` search path (row idm_mobil). nuPlan's IDMPlanner and IDM agents, NAVSIM's NavsimIDMAgents
  (background agents) and PDM-Closed (±1 m lateral offsets) do not change lanes.
- **PDM-Closed parameters**: Hydra-overridable in
  `tuplan_garage/tuplan_garage/planning/script/config/simulation/planner/pdm_closed_planner.yaml`
  (`planner.pdm_closed_planner.idm_policies.speed_limit_fraction=[0.2,0.4,0.6,0.8,1.0]`, `...idm_policies.fallback_target_velocity=15.0`,
  `...idm_policies.min_gap_to_lead_agent=1.0`, `...idm_policies.headway_time=1.5`, `...idm_policies.accel_max=1.5`,
  `...idm_policies.decel_max=3.0`, `planner.pdm_closed_planner.lateral_offsets=[-1.0,1.0]` (or null), `map_radius=50`,
  `trajectory_sampling` 80 x 0.1 s, `proposal_sampling` 40 x 0.1 s). The proposal scorer's weights and thresholds are
  module constants in `tuplan_garage/tuplan_garage/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py` lines 47-56
  (progress 5, TTC 5, comfort 2; driving-direction 2 m / 6 m; stopped speed 5e-3 m/s; progress threshold 0.1 m), not config.
  PDM-Hybrid uses the same keys under `planner.pdm_hybrid_planner` plus correction_horizon 2.0.
- **GC-PGP** needs `model.aggregator.pre_train=false` (as in tuplan_garage's script).
- **PlanCNN**: nuPlan's `raster_model` has `pretrained: true`, which downloads timm resnet50 ImageNet weights from Hugging
  Face at model construction. `model.pretrained=false` gives the identical result (checkpoint overwrites all weights;
  CLS-R 0.9843 both ways) and avoids 192 concurrent downloads.
- **Diffusion Planner**: the yaml default `device: cuda` must be overridden with `planner.diffusion_planner.device=cpu`.
  Options exposed by its nuPlan planner (`diffusion_planner/planner/planner.py`, configs `diffusion_planner/config/planner/`):
  (1) `planner=diffusion_planner`: no guidance (`config.guidance_fn: null`, DPM-Solver++ "uncond");
  (2) `planner=diffusion_planner_guidance`: classifier guidance with `GuidanceWrapper`, which hard-codes one energy, the
  collision guidance `diffusion_planner/model/guidance/collision.py` (active for diffusion time 0.005 < t < 0.1), guidance
  scale 0.5 hard-coded in `diffusion_planner/model/module/decoder.py` (`"guidance_scale": 0.5`);
  (3) `enable_ema` (default true: `ema_state_dict`; false: `model` weights; both present in model.pth).
  Not exposed: number of DPM-Solver steps (10, `diffusion_utils/sampling.py`), guidance scale, other guidance energies
  (add to `GuidanceWrapper._guidance_fns`). There is **no refinement option**: the paper's "w/ refine" variant is not in
  the released code (the only "refine" code is training-time augmentation in `utils/data_augmentation.py`).
  Sampling is stochastic; nuPlan seeds once per process (`seed: 0` -> `pl.seed_everything`), so with several scenarios per
  process the result depends on scenario order; run one scenario per process or reseed for reproducibility.
- **Log replay**: nuPlan's `LogFuturePlanner` (`planner=log_future_planner`, num_poses 16, horizon 8 s).
- **Constant velocity**: nuPlan has no dedicated CV planner. `SimplePlanner` (`nuplan/planning/simulation/planner/simple_planner.py`)
  propagates a kinematic bicycle with steering 0 and constant acceleration `acceleration=[0,0]` from the current state,
  i.e. constant speed along the current heading (straight line, ignores lanes). Its default `max_velocity: 5.0` makes it
  brake to 5 m/s, hence `max_velocity=100.0`; horizon set to 8 s (default 10 s).
- **PlanTF** (released ckpt): planner `src.planners.imitation_planner.ImitationPlanner` with
  `planTF/config/planner/planTF.yaml` (replan_interval 1). The checkpoint contains NATTEN `attn.rpb` tensors: keep natten
  0.14.6. Loaded strictly (`load_state_dict` default strict=True) without error.
- **PLUTO** official setting (`planner=pluto_planner`, `pluto/config/planner/pluto_planner.yaml`) = PlutoFeatureBuilder +
  network + post-processing in `src/planners/pluto_planner.py::_run_planning_once`: (1) `_trim_candidates`: top-20 candidates by learned probability,
  softmax, plus the reference-free trajectory with fixed probability 0.25; (2) `TrajectoryEvaluator`
  (`src/post_processing/trajectory_evaluator.py`): LQR/bicycle forward simulation of every candidate and PDM-style
  rule-based scoring against the network's agent predictions (multiplicative: no at-fault collision, drivable area, driving
  direction; weighted: progress 5, TTC 5, speed limit 2, comfort 2); (3) final score = rule score +
  `learning_based_score_weight` (0.3 in `config/planner/pluto_planner.yaml`) x learned probability, argmax;
  (4) `EmergencyBrake` (`src/post_processing/emergency_brake.py`): if the chosen candidate's time to at-fault collision
  <= max(v/2.4 + 0.5, 3.0) s and v <= 8 m/s, a braking trajectory replaces it.
  **There is no config switch for post-processing in pluto@b9964b6.** `pluto_nopost` swaps the planner class to
  `exp2tools.pluto_no_post.PlutoPlannerNoPost` (`/data/code/envs/tools/exp2tools/pluto_no_post.py`, a 50-line subclass;
  the pluto repo is untouched): same features, model and checkpoint, but it returns the network's own `output_trajectory`
  (the highest-probability candidate over reference lines x modes, padded reference lines masked; the reference-free
  trajectory when no reference line exists), i.e. steps (1)-(4) are all off. On the timing scenario the two settings
  differ slightly (ego progress 0.8475 vs 0.8469) and no emergency brake fired.

## 7. interPlan benchmark (env `nuplan`)

```bash
source /data/code/envs/env.sh nuplan
python $INTERPLAN_ROOT/interplan/planning/script/run_simulation.py \
  +simulation=default_interplan_benchmark scenario_filter=interplan10 \
  'scenario_filter.scenario_tokens=[2d62c3139aa95007-s0]' \
  scenario_builder.db_files=$NUPLAN_DB_TEST \
  planner=pdm_closed_planner worker=sequential '~callback.simulation_log_callback' \
  group=<output root> experiment_name=interplan_pdm_closed "$IP"
```

- Drop the `scenario_tokens` override to run the official 80-scenario split (`interplan10`: 10 per type, tokens carry
  modification suffixes such as `-s0` or `-dlgloa`; logs restricted to the 7 test DBs). `scenario_filter=benchmark_scenarios`
  is the full 335-scenario set. interPlan's `run_simulation.py` rewrites the suffixed tokens into scenario modifications.
- `default_interplan_benchmark` sets observation `idm_modified_agents_observation`, two-stage controller, metric set
  `simulation_benchmark` (adds `lane_changes_to_goal`, `ego_sorts_*` scenario-specific metrics) and the aggregator
  `InterplanWeightedAverageMetricAggregator` (`config/simulation/metric_aggregator/benchmark_weighted_average.yaml`).
- Verified with PDM-Closed on `2d62c3139aa95007-s0` (overtake_parked_vehicle, 300 steps of 0.1 s, 76-83 ms/step,
  32-35 s wall, run twice):
  outputs in `<group>/<experiment_name>/default_interplan_benchmark/<timestamp>/`; the interPlan score is column `score`
  of the row `scenario == 'final_score'` in `aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_<timestamp>.parquet`
  (the aggregator keeps nuPlan's file name; the paper's "interPlan score" = 100 x this over interplan10); per-scenario rows in
  the same file, per-metric values in `metrics/*.parquet` (incl. `lane_changes_to_goal.parquet`,
  `ego_sorts_stopped_vehicle.parquet`). PDM-Closed scored 0 on that scenario (stuck behind the parked vehicle:
  ego_is_making_progress 0, ego_sorts_stopped_vehicle 0), consistent with the paper's PDM-Closed weakness.

## 8. Training framework (env `nuplan`)

- nuplan-devkit's own `nuplan/planning/script/run_training.py` is PL-1.x code and fails under PL 2.0.1:
  `AttributeError: module 'pytorch_lightning.plugins' has no attribute 'DDPPlugin'` (training_builder.py line 112; its
  `default_lightning.yaml` also passes Trainer args removed in PL 2: gpus, amp_level, auto_lr_find, terminate_on_nan, ...).
- Use planTF's (or pluto's) `run_training.py`: the same nuPlan framework (feature caching via
  `nuplan.planning.training.experiments.caching.cache_data`, scenario builder/filter, splitter, datamodule) with a PL-2
  trainer builder. Verified on CPU (2 DBs: one train_pittsburgh + one val, 48 scenarios):
  - planTF: cache (22 s, 96 files) -> train 1 epoch x 3 batches (7 s, `last.ckpt`) -> closed-loop simulation with it. Commands
    in `smoke/plantf_pipeline.sh`; CPU trainer overrides: `lightning.trainer.params.accelerator=cpu
    lightning.trainer.params.devices=1 lightning.trainer.params.strategy=auto lightning.trainer.params.sync_batchnorm=false`.
  - pluto: same (after the `__init__.py` fix). Note pluto's training model config (`config/model/pluto_model.yaml`) differs
    from its planner config (`use_hidden_proj`, `cat_x`, `ref_free_traj`); train with `model.use_hidden_proj=true
    +model.cat_x=true +model.ref_free_traj=true` to get checkpoints the planner config can load.
  - stock nuPlan/tuplan models through nuplan's `LightningModuleWrapper`: e.g. from /data/code/planTF,
    `python run_training.py py_func=train +training=training_urban_driver_open_loop_model ...` (`smoke/nuplan_train_via_plantf.sh`);
    the checkpoint loads in nuPlan's MLPlanner and simulates (verified).

## 9. Checkpoints (loaded only on this host)

| model | file | weights_only=True | Lightning | epoch | tensors | numel (excl. BN buffers) |
|---|---|---|---|---|---|---|
| PDM-Open | tuplan_garage/pdm_open_checkpoint.ckpt (15.6 MB) | fails (pickled hparams/callbacks); full unpickle OK | 1.3.8 | 83 | 10 | 1,048,112 |
| PDM-Offset (PDM-Hybrid) | tuplan_garage/pdm_offset_checkpoint.ckpt (19.0 MB) | fails; unpickle OK | 1.3.8 | 84 | 12 | 1,335,344 |
| GC-PGP | tuplan_garage/gc_pgp_checkpoint.ckpt (4.9 MB) | fails; unpickle OK | 1.3.8 | 89 | 77 | 150,546 |
| PlanCNN | tuplan_garage/plancnn_checkpoint.ckpt (286.7 MB) | fails; unpickle OK | 1.3.8 | 100 | 320 | 23,662,693 (23,609,520) |
| Urban Driver | tuplan_garage/urbandriver_checkpoint.ckpt (34.8 MB) | fails; unpickle OK | 1.3.8 | 101 | 29 | 3,502,384 |
| Diffusion Planner | diffusion_planner/model.pth (48.5 MB) | OK | - | - | 276 + 276 | 6,042,628 (`ema_state_dict`; `model` same size) |
| PlanTF | plantf/planTF.ckpt (8.2 MB, git blob 091a5d65 verified) | OK | (state_dict only) | - | 218 | 2,024,047 (2,023,277) |
| PLUTO | pluto/pluto_1M_aux_cil.ckpt (51.4 MB, sha256 ce60d7c8... verified) | OK | 2.1.0 | 24 | 438 | 4,242,129 (4,240,589) |

(`tools/inspect_checkpoints.py`, raw output `checkpoint_inspection*.jsonl`.) The planners themselves load with
`torch.load` defaults (torch 2.0.1: full unpickle) via `LightningModuleWrapper.load_from_checkpoint` (tuplan/nuPlan) or
their own loaders.

## 10. NAVSIM v2.2 scorer on nuPlan DB scenarios (env `navsim`)

Working example: `tools/navsim_score_nuplan.py --db <file.db> --token <token>` (about 8 s for one scenario incl. imports:
metric cache 1.5-1.7 s, then 0.085 s per trajectory for both scores). Output on cb468419f191575a: log (human) trajectory
EPDMS 0.9997 / PDMS 0.9997; constant velocity 1.0 / 1.0; standing still 0.246 / 0.120; PDM-Closed reference 1.0 / 1.0.

1. Scenario: `nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.NuPlanScenarioBuilder(data_root, map_root,
   sensor_root, db_files, map_version, include_cameras=False, max_workers=None, verbose=False, scenario_mapping=<nuplan
   ScenarioMapping, e.g. instantiate(OmegaConf.load(".../nuplan_scenario_mapping.yaml"))>, vehicle_parameters=get_pacifica_parameters())`
   `.get_scenarios(ScenarioFilter(...scenario_tokens=[token]...), Sequential())` -> `NuPlanScenario` with
   database_interval 0.1 s. Same scenario window as the nuPlan simulation when the same mapping is used. Any frame:
   `NuPlanScenario(data_root, log_file_load_path, initial_lidar_token, initial_lidar_timestamp, scenario_type, map_root,
   map_version, map_name, scenario_extraction_info, ego_vehicle_parameters, sensor_root=None)`.
2. Metric cache: `navsim.planning.metric_caching.metric_cache_processor.MetricCacheProcessor(cache_path: Optional[str],
   force_feature_computation: bool, proposal_sampling: TrajectorySampling)` with
   `proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)`;
   `.compute_metric_cache(scenario) -> navsim.planning.metric_caching.metric_cache.MetricCache` (iteration 0 of the
   scenario: runs NAVSIM's PDM-Closed reference, builds PDMObservation from GT tracks interpolated 2 Hz -> 10 Hz, drivable
   area map, centerline, route lanes, human future/past). `.compute_and_save_metric_cache(scenario)` writes
   `<cache_path>/<log>/<type>/<token>/metric_cache.pkl` (lzma pickle). A NuPlanScenario works directly: the processor only
   uses AbstractScenario methods, `database_interval` must equal 0.1 s (default subsample 0.5), and the 16-hex nuPlan token
   is treated as an original (non-synthetic) scene.
3. Score one 4 s trajectory: `navsim.evaluate.pdm_score.pdm_score(metric_cache, model_trajectory, future_sampling,
   simulator, scorer, traffic_agents_policy) -> (pd.DataFrame one row, simulated ego states np.ndarray[41, 11])` with
   - `model_trajectory = navsim.common.dataclasses.Trajectory(poses=np.ndarray[N, 3] (x, y, heading) in the ego rear-axle
     frame at t0, trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5))` (N=8; any sampling works,
     e.g. 40 x 0.1 s); velocities are ignored (LQR + bicycle re-simulation);
   - `future_sampling = proposal_sampling`; `simulator = navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator.PDMSimulator(proposal_sampling)`;
   - `scorer = navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer.PDMScorer(proposal_sampling,
     PDMScorerConfig(human_penalty_filter=True))` (weights progress 5, TTC 5, lane keeping 2, history comfort 2, two-frame
     extended comfort 2; `human_penalty_filter=True` as in the official `config/pdm_scoring/scorer/pdm_scorer.yaml`;
     the dataclass default is None);
   - `traffic_agents_policy = navsim.traffic_agents_policies.log_replay_traffic_agents.LogReplayTrafficAgents(proposal_sampling)`
     (non-reactive, the one-stage default; reactive: `navsim_IDM_traffic_agents.NavsimIDMTrafficAgents`).
   - A global-frame nuPlan `InterpolatedTrajectory` (e.g. a planner's output at iteration 0) can be scored without
     conversion: `navsim.evaluate.pdm_score.pdm_score_from_interpolated_trajectory(metric_cache, pred_trajectory, future_sampling, simulator, scorer, traffic_agents_policy)`.
4. Scores from the returned row (columns no_at_fault_collisions, drivable_area_compliance, driving_direction_compliance,
   traffic_light_compliance, ego_progress, time_to_collision_within_bound, lane_keeping, history_comfort,
   multiplicative_metrics_prod, weighted_metrics, weighted_metrics_array, pdm_score):
   - **EPDMS** = NAVSIM's own `navsim.planning.script.run_pdm_score_one_stage.compute_final_scores(df)` (column `score`)
     = multiplicative_metrics_prod (NC x DAC x DDC x TLC) x weighted mean of (EP 5, TTC 5, LK 2, HC 2, EC 2). Do **not**
     read the row's `pdm_score` column: `pdm_score()` applies the human penalty filter after scoring and updates
     `multiplicative_metrics_prod`/`weighted_metrics` but not `pdm_score`. Single frame: set
     `df["two_frame_extended_comfort"] = np.nan` before `compute_final_scores` (EC is then dropped from the weights).
     Two-frame EC needs the same planner scored at frames 0.5 s apart in one log: add `start_time`, `log_name`,
     `frame_type`, `token` and `ego_simulated_states` to the rows and run `infer_start_adjacent_mapping` and
     `create_scene_aggregators` from the same module (`SceneAggregator` in `.../scoring/scene_aggregator.py`) before
     `compute_final_scores`, as `run_pdm_score_one_stage.main` does.
   - **PDMS** (NAVSIM v1, docs/metrics.md) is not produced by v2.2 directly: PDMS = NC x DAC x (5 TTC + 5 EP + 2 C) / 12
     from a second scoring pass with `PDMScorerConfig(human_penalty_filter=False)` (v1 had no human filter), with
     C = `ego_is_comfortable(states[None], t).all()` (`navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics`)
     on that pass's simulated states (t = 0, 0.1, ..., 4.0 s), i.e. v1 comfort without history padding. EP/TTC/NC/DAC
     are v2.2's implementations (same definitions as v1 per the NAVSIM docs). `NavsimScorer.score()` in the example
     script does both passes (0.085 s per trajectory).
- Do not run the NAVSIM scorer in the `nuplan` env: `MetricCacheProcessor` fails there (`StateInterpolator` ->
  scipy 1.9.1 `interp1d` raises "x and y arrays must have at least 2 entries" for objects seen in a single frame;
  scipy 1.13.1 in `navsim` is fine). Planner trajectories therefore have to be passed between envs as arrays
  (e.g. global x, y, heading at 0.1 s), not as live objects.

## 11. Fixes applied and gotchas

1. `pluto/src/custom_training/__init__.py` added (re-exports TrainingEngine, build_training_engine,
   update_config_for_training) so pluto's run_training.py imports.
2. natten 0.14.6 built from source (CPU) because the shi-labs wheel index is offline.
3. apt: libgl1 libsm6 libxrender1 libxext6 for opencv-python <ip>; cmake/ninja-build for builds.
4. pip pinned at 23.3.1 (see section 3).
5. Simulation log off with `~callback.simulation_log_callback`, not `callback=[]`.
6. Diffusion Planner `device=cpu`; PlanCNN `model.pretrained=false`; planTF/pluto `number_of_gpus_allocated_per_simulation=0`.
7. `exp2tools` (in `/data/code/envs/tools`, on PYTHONPATH via env.sh): `step_timer_callback.StepTimerCallback`,
   `pluto_no_post.PlutoPlannerNoPost`.

## 12. Open issues / workarounds

1. nuplan-devkit's own run_training.py cannot run in `nuplan` (PL 2.0.1). Workaround: planTF's run_training.py (section 8).
   If the original PL-1.3.8 trainer is ever needed, build a third env from nuplan's lock files (torch 1.9.0+cpu,
   pytorch-lightning 1.3.8, torchmetrics 0.7.2); it would not run planTF/pluto/Diffusion Planner.
2. PLUTO: `PlutoPlanner._trim_candidates` raises `ValueError: zero-size array to reduction operation maximum` (scipy softmax
   of an empty array) when the feature builder finds no reference line; seen with an untrained smoke checkpoint, may also
   hit the released checkpoint in off-route states. The harness should run with `exit_on_failure=false` (default) and record
   failed simulations; `pluto_nopost` falls back to the reference-free trajectory instead.
3. Diffusion Planner: no refinement variant in the released code. Three settings verified without code changes:
   `diffusion_planner` (no guidance, EMA weights), `diffusion_planner_guidance` (collision guidance, EMA),
   `diffusion_planner_noema` (no guidance, raw weights). Anything else (guidance scale, DPM steps, other energies) needs a
   patch of `decoder.py` / `sampling.py` / `guidance_wrapper.py`, to be recorded.
4. Diffusion Planner sampling is random: reproducibility needs one scenario per process (seed 0) or explicit reseeding.
5. Floating-point results depend slightly on the thread count; fix OMP_NUM_THREADS for the whole study.
6. GC-PGP, PlanCNN, Urban Driver all report planner_name `MLPlanner`.
7. The `navsim` env's nuplan-devkit has no yaml configs (non-editable git install), so nuPlan Hydra configs are not
   available there; the scorer path above does not need them.
8. planTF/pluto do not ship an installable package; registry entries must set cwd and PYTHONPATH per planner.

## 13. Files in this folder

- `env.sh`; `build_env_nuplan.sh`, `build_env_navsim.sh`; `constraints/nuplan_lock_constraints.txt`
- `nuplan.yml`, `nuplan-pip-freeze.txt`, `navsim.yml`, `navsim-pip-freeze.txt`
- `tools/`: `exp2tools/` (StepTimerCallback, PlutoPlannerNoPost), `find_token_db.py`, `inspect_checkpoints.py`,
  `navsim_score_nuplan.py` (NAVSIM example + `NavsimScorer`), `summarize_smoke.py`, `timing_table.py`, `compact_table.py`
- `smoke/`: `smoke_nuplan.sh` (generic one-scenario runner; env SMOKE_DB, SMOKE_TOKEN, SMOKE_SCRIPT, SMOKE_CWD,
  SMOKE_PYTHONPATH, SMOKE_TIMING_FILE), `smoke_interplan.sh`, `final_timing.sh`, `batch_plantf_pluto.sh`,
  `registry_check.sh`, `final_extra.sh`, `batch_ml.sh`, `batch_dp.sh`, `plantf_pipeline.sh`, `nuplan_train_via_plantf.sh`,
  `misc_checks*.sh`
- `timing/`: `compact_table.md`, `final_timing.jsonl` (+ `_t4`), `final_timing_t1.md`, `final_timing_t4.md`,
  `other_challenges.*`, `misc_checks*`, `final_extra.out`
- `logs/`: build logs; `checkpoint_inspection*.jsonl`
- Smoke-run outputs (not copied to the laptop): `/data/code/smoke_runs/` (`exp/` nuPlan outputs, `logs/` per-run logs and
  GNU time files).
