# Which cheap scores predict closed-loop safety? A same-scenario study of 27 nuPlan planners

## Introduction

Self-driving planners are usually compared with cheap scores: how far a planned path is from what a human drove
(open-loop error), or short simulations in which the other road users only replay a recording. Running every planner
in a full closed-loop simulation where other vehicles react is far more expensive, so the cheap scores are what most
papers report. The open question is what a cheap score is evidence of. If one planner has a better open-loop score than
another, is it also safer once other road users react to it?

This experiment answers that for 27 planners from 7 families (IDM (5), PDM (5), Released imitation (4), PLUTO (2), Diffusion Planner (3), Small baselines trained here (6), Anchors (2)). Every planner
runs the **same 688 nuPlan scenarios** under every protocol, so that differences between scores come from
the protocol and not from different scenario sets. Each cheap score (open-loop L2 error, collision with logged agents,
nuPlan's open-loop score, NAVSIM's PDMS and single-stage EPDMS, and nuPlan's non-reactive closed-loop score) is compared
with **safety outcomes** in nuPlan's reactive closed loop (at-fault collisions, time-to-collision, drivable-area and
driving-direction violations, severity-weighted, and with a progress control), and with interPlan's long-tail benchmark.

**Why it is innovative.** Earlier studies related cheap scores to an aggregate driving score, mostly over a few planner
families and their hyper-parameter variants. This study targets disaggregated safety outcomes, uses the same scenarios
for every score, includes modern learned planners (PlanTF, PLUTO, Diffusion Planner, GC-PGP), reports intervals that
resample both scenarios and planners, and tests whether a composite cheap score generalises to a planner family it was
not fitted on.

## Aims and objectives

Aim: measure how well cheap planning scores predict the closed-loop safety ranking of driving planners, and where they fail.

1. RQ1 - Agreement: Kendall tau (with two-way bootstrap intervals), pairwise ranking inversions and the Mean Maximum
   Rank Violation between each cheap score and each reactive closed-loop safety outcome.
2. RQ2 - Where it breaks: per scenario type, reactive versus non-reactive traffic, interPlan's long tail, and the effect of
   the scenario set itself (same-half versus cross-half analysis).
3. RQ3 - Composite: a non-negative ridge combination of cheap scores, validated by leaving one planner family out.
4. RQ4 - Sensitivity: invalid outputs, severity weighting, seeds, scenario count, without the anchor planners, and on the
   Val14 scenarios alone.

## Datasets and models (download links)

| Dataset / model | Link | Description |
|---|---|---|
| nuPlan v1.1 maps (map version nuplan-maps-v1.0) | https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip | HD maps of Las Vegas, Pittsburgh, Boston and Singapore used by every simulation and score (0.97 GB). |
| nuPlan v1.1 validation split | https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_val.zip | 1,381 driving logs (SQLite, no sensor data); the 688 study scenarios are drawn from them (97 GB zip). |
| nuPlan v1.1 test split, 7 logs | https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip | Only the 7 logs interPlan builds its 80 long-tail scenarios from (extracted member by member, 1.1 GB). |
| nuPlan v1.1 train_pittsburgh split | https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_pittsburgh.zip | 1,560 training logs from Pittsburgh; 50,000 samples train the three small baselines (30.6 GB zip). |
| tuplan_garage checkpoints (PDM-Open, PDM-Offset, GC-PGP, PlanCNN, Urban Driver) | https://github.com/autonomousvision/tuplan_garage (Google Drive links in its README) | Released learned planners evaluated in the study. |
| Diffusion Planner checkpoint | https://huggingface.co/ZhengYinan2001/Diffusion-Planner | Released diffusion-based planner (model.pth, args.json). |
| PlanTF checkpoint (third-party copy) | https://raw.githubusercontent.com/richie-live/SAH-Drive/master/SAH_Diffusion_Model/planTF.ckpt | The authors' OneDrive link is dead; this copy is used (git blob 091a5d65d0d50aa56725256c9057e55fc8054b06). |
| PLUTO checkpoint (third-party copy) | https://huggingface.co/rxinhkust/PlanScope_model_zoo/resolve/main/pluto_1M_aux_cil_m12_ori.ckpt | The authors' OneDrive link is dead; sha256 ce60d7c854d10b310c4e1705099d799a6ea008baa4288e668626f4a2f22943e1. |
| timm ResNet-18 ImageNet weights | https://huggingface.co/timm/resnet18.a1_in1k | Initialisation of the small raster baseline trained here. |
| interPlan benchmark definition | https://github.com/mh0797/interPlan | The 80-scenario long-tail split and its modifications (code and configuration, no extra data). |

Code repositories used (pinned commits):

| Repository | URL | Commit |
|---|---|---|
| nuplan-devkit | https://github.com/motional/nuplan-devkit | `e9241677997d` |
| tuplan_garage | https://github.com/autonomousvision/tuplan_garage | `b51d5d04fac1` |
| interPlan | https://github.com/mh0797/interPlan | `a85024a48935` |
| planTF | https://github.com/jchengai/planTF | `f54060850649` |
| pluto | https://github.com/jchengai/pluto | `b9964b649c66` |
| Diffusion-Planner | https://github.com/ZhengYinan-AIR/Diffusion-Planner | `a3a621f0b724` |
| navsim (v2.2) | https://github.com/autonomousvision/navsim | `359c7f72304b` |

Licences: the nuPlan dataset is for non-commercial research use (CC BY-NC-SA 4.0 and the nuPlan terms; register at
nuscenes.org/nuplan). PlanTF's and PLUTO's original repositories carry no licence file; their checkpoints are used for
research evaluation only and are not redistributed. See docs/data-used.md and docs/assumptions.md.
None of the files above are included here; the repository does include the weights of the three small baselines trained for the study, in models/ (see Using the trained models).

## Steps to reproduce

The simulations need a Linux host (the study used one AWS EC2 m8a.48xlarge, 192 vCPU, for the training of the small
baselines and the full pass, 3.3 h on 188 parallel processes; development ran on an m8a.2xlarge). About 220 GB of disk for the data.
The scripts expect the nuPlan data under /data/nuplan, the checkpoints under /data/checkpoints, the planner
repositories under /data/code and this repository at /data/exp2 (set EXP2_ROOT to use another folder; the small
baselines' checkpoint paths in code/planners_baselines.yaml point to /data/exp2/baselines, and the runner uses the released copies in models/ when those files are absent).

1. Clone the seven repositories above at the pinned commits into /data/code/ and build the two conda environments:
   `bash env/build_env_nuplan.sh` and `bash env/build_env_navsim.sh` (Python 3.9, CPU PyTorch 2.0.1; exact versions in
   env/*.yml and env/*-pip-freeze.txt). `source env/env.sh nuplan` sets the paths.
2. Download the data (public, about 130 GB, CRC-checked and resumable): `bash code/data/download_all.sh`.
3. Build the scenario lists (identical to scenarios/ in this repository): `python code/select_scenarios.py`.
   Check the installation with a five-minute run of nine planners on two scenarios under every protocol:
   `EXP2_ROOT=$(pwd) bash tools/repro_smoke.sh` (prints REPRO SMOKE PASS).
4. Run everything on the host: `bash code/bigrun.sh` (small-baseline training, 61,248 simulations of
   27 planners plus seed variants under open loop, closed loop non-reactive, closed loop reactive and
   interPlan, metric extraction, NAVSIM scoring). It is resumable; finished jobs are skipped.
5. Copy the host's results/ folder, then on any machine: `pip install -r requirements.txt` and
   `python code/run_all.py analysis` (merge, statistics, figures, summary.json).

The statistics alone can be re-run from the per-run table shipped here:
`python code/stats.py --runs results/runs_merged.csv --out results/stats` (decompress results/runs_merged.csv.gz first).

## Using the trained models

`models/` holds the weights of the three small baselines trained for this study on 50,000 samples of nuPlan's Pittsburgh training
logs. These are the 12 checkpoints the study evaluated: seed 0 early (after 15% of the optimizer steps) and final (end of training), and seeds 1 and 2 final for the seed-variance check; models/README.md lists them with their checksums. Each model plans the
ego vehicle's next 8 s as 16 poses (x, y and heading every 0.5 s) and runs on a CPU inside nuPlan's ML planner.

| Model | Planners (keys in code/planners_baselines.yaml) | Input | Checkpoint |
|---|---|---|---|
| Ego-status MLP (`ego_mlp_model`, code/exp2_nuplan/models/ego_mlp.py) | `mlp_early`, `mlp_final`, `mlp_final_s1`, `mlp_final_s2` | the ego vehicle's own past 1.5 s only (no map, no other road users) | 2.3 MB |
| nuPlan's `simple_vector_model` | `vector_early`, `vector_final`, `vector_final_s1`, `vector_final_s2` | lanes within 20 m as vectors, and the past 1.5 s of the ego vehicle and nearby agents | 0.5 MB |
| nuPlan's `raster_model` with a ResNet-18 backbone | `raster_early`, `raster_final`, `raster_final_s1`, `raster_final_s2` | a 224 x 224 bird's-eye image (0.5 m per pixel) of the map, the agents and the ego vehicle | 45.0 MB |

They need the simulation host of "Steps to reproduce": the nuplan environment (step 1) and the nuPlan maps and validation
logs (step 2). Nothing has to be trained or moved: when /data/exp2/baselines does not hold the training output,
code/runner.py and tools/infer_baseline.py load the checkpoints from models/.

**One plan per scenario.** `tools/infer_baseline.py` loads a model exactly as nuPlan's simulator does, gives it the first
2 s of a study scenario and prints the 8 s plan with its distance to the trajectory the human driver actually drove:

```bash
source env/env.sh nuplan && export PYTHONPATH=$(pwd)/code:$PYTHONPATH
python tools/infer_baseline.py --planner raster_final                  # the first study scenario
python tools/infer_baseline.py --planner mlp_final --n 5 --out plans.json
python tools/infer_baseline.py --planner vector_early --tokens 63ef5aab300c5f09,470cb336d24754b4
```

Example output (`raster_final`, run on the study's host):

```text
raster_final: models/raster/seed0/final.ckpt
63ef5aab300c5f09 (behind_long_vehicle, us-nv-las-vegas-strip): planned in 0.59 s; distance to the human trajectory 8.26 m on average, 15.48 m at 7.6 s
    t [s]       x [m]       y [m]   heading [rad]
      0.0   664432.47  3998281.03    -1.530
      1.0   664432.62  3998278.35    -1.518
      2.0   664432.82  3998275.84    -1.507
      3.0   664433.01  3998273.51    -1.520
      4.0   664433.21  3998271.26    -1.518
      5.0   664433.30  3998269.08    -1.549
      6.0   664433.14  3998266.95    -1.547
      7.0   664433.00  3998264.88    -1.542
      8.0   664432.82  3998262.77    -1.558
```

**Closed-loop simulation.** To let a model drive in nuPlan's simulator and score it, run its planner through the
harness. Here the three final models drive two scenarios in the non-reactive closed loop (`--protocols ols,clsnr,clsr`
runs all three protocols; leave out `--limit` for all 688 study scenarios):

```bash
source env/env.sh nuplan && export PYTHONPATH=$(pwd)/code:$PYTHONPATH EXP2_ROOT=$(pwd)
python code/runner.py make --registry code/planners_baselines.yaml --protocols clsnr --limit 2 --tag demo --jobs demo_jobs.jsonl
python code/runner.py run --jobs demo_jobs.jsonl --only __mlp_final__,__vector_final__,__raster_final__ --workers 3
grep -E '"planner_entry": "(mlp_final|vector_final|raster_final)"' demo_jobs.jsonl > demo_selected.jsonl
python code/extract_metrics.py --jobs demo_selected.jsonl --out demo_runs.csv
```

demo_runs.csv has one row per scenario: validity, nuPlan's closed-loop score and its parts (at-fault collisions, drivable
area, driving direction, progress, time to collision, comfort) and the planner's run time per step.

**In your own nuPlan code.** The models are ordinary nuPlan ML planners. With nuPlan's `run_simulation.py`, the raster
model is `planner=ml_planner model=raster_model model.model_name=resnet18 model.pretrained=false
'planner.ml_planner.model_config=${model}' planner.ml_planner.checkpoint_path=models/raster/seed0/final.ckpt`; every
planner's exact overrides are in code/planners_baselines.yaml (the MLP also needs `pkg://exp2_nuplan.config` on
`hydra.searchpath` and code/ on `PYTHONPATH`). In Python, nuPlan's `build_planners(cfg.planner, scenario)` loads the
checkpoint into an `MLPlanner`, and `planner.initialize(...)` followed by `planner.compute_trajectory(planner_input)`
plans, as in tools/infer_baseline.py.

The checkpoints are PyTorch Lightning files, which are Python pickles: loading one can run code, so check them against
models/SHA256SUMS first. The weights were trained on nuPlan data and are shared for non-commercial research use under the nuPlan dataset terms (CC BY-NC-SA 4.0); the MIT licence covers the code only.

## Results

- Ranking planners by **at-fault collisions in reactive closed loop**: the best cheap score is **CLS-NR** (Kendall tau = 0.82, 95% CI 0.58 to 0.94); the weakest is L2 error (tau = 0.05).
- Ranking planners by **any safety violation in reactive closed loop**: the best cheap score is **CLS-NR** (Kendall tau = 0.86, 95% CI 0.67 to 0.96); the weakest is L2 error (tau = 0.09).
- Ranking planners by **a safety violation or no progress in reactive closed loop**: the best cheap score is **CLS-NR** (Kendall tau = 0.90, 95% CI 0.73 to 0.98); the weakest is L2 error (tau = 0.06).
- Ranking planners by **the interPlan long-tail score**: the best cheap score is **CLS-NR** (Kendall tau = 0.60, 95% CI 0.36 to 0.82); the weakest is L2 error (tau = -0.01).

### Why these results matter

- **Which cheap numbers count as safety evidence.** Papers and leaderboards report cheap scores as if a better score meant a safer planner. Here the non-reactive closed-loop score (CLS-NR) reproduced the reactive safety ranking best (tau = 0.86, 95% CI [0.67, 0.96]). The open-loop scores (L2 error, logged-agent collision, OLS) barely agree with the safety ranking (tau up to 0.29): an open-loop number is evidence about imitating the logged driver, not about safety. NAVSIM's first-frame scores, a 4-second non-reactive simulation of each plan, sit in between (tau up to 0.60).
- **Averages hide where it fails.** For the best cheap score agreement differs somewhat between scenario types (tau 0.58 for stationary in traffic to 0.85 for changing lane), and it drops on interPlan's long tail (tau 0.60 against 0.86). A benchmark-wide correlation therefore overstates how well a cheap score protects against the situations that matter most.
- **The scenario sample is not the bottleneck.** Changing the scenarios alone moves the agreement little (0.00) when the protocol is fixed, so with several hundred scenarios the sample does not limit the agreement; benchmarks built from different scenario distributions can still disagree for that reason.
- **Safety cases.** Standards such as UL 4600 and the UN regulation on automated driving systems ask for evidence that the measures used actually track safety; per-score agreement with uncertainty intervals is that kind of evidence, for planners nuPlan can simulate.

### What is new

- To our knowledge, the first measurement of how well cheap scores rank planners by **reactive closed-loop safety outcomes** (not an aggregate driving score), across 27 planners in 7 families including PlanTF, PLUTO and Diffusion Planner, all on the same scenarios.
- Agreement measured **within scenario types** (tau 0.58 to 0.85 for CLS-NR) and on **interPlan's long tail** (best cheap score tau = 0.60), plus a split-half test of the scenario-set effect.
- A composite cheap score **tested on planner families it was not fitted on**: tau = 0.86 against 0.83 for the best single score, so combining scores did not clearly help out of family.
- Not claimed: the largest study of its kind (NAVSIM compared 151 planners), the first open-loop versus closed-loop correlation, or the first use of Kendall's tau here.

Figures are in figures/, every table in results/stats/, and every number in results/summary.json.

## Repository layout

- `code/` harness (runner, scenario selection, planner wrappers and registries, trajectory capture, NAVSIM adapter),
  training of the small baselines, statistics, figures, data download scripts.
- `env/` conda recipes, pinned versions and the setup notes for both environments.
- `scenarios/` the 688 study scenarios, the seed subset and the selection record.
- `results/` summary.json, statistics tables, the per-run table, the baseline training manifest and the compute log.
- `models/` the trained weights of the three small baselines, with their checksums.
- `figures/`, `docs/` (assumptions and data list), `tools/` (ETA helper, installation smoke test, inference with
  the released models).

## Licence

Code: MIT (see LICENSE). Data and third-party models keep their own terms (see above). The small baselines' weights in models/ were trained on nuPlan data and are shared for non-commercial research use under the nuPlan dataset terms (CC BY-NC-SA 4.0).
