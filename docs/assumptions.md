# Experiment 2 - assumptions and decisions made without the user

Each entry: date/time (UTC), decision, why, effect on the study.

A1 (28 Oct 13:30). Compute host: one AWS EC2 instance in us-east-1b, on demand (the user chose on demand over spot),
launched as m8a.2xlarge for set-up, downloads, development and CPU training, and resized to m8a.48xlarge only for the timed
pilot and the full pass. Why: the m8a.48xlarge costs US$11.69/h, so idle or set-up time is spent on the US$0.49/h size.
Effect: none on results; it changes wall-clock time only.

A2 (28 Oct 13:45). Small learned baselines are trained on the AWS host's CPU instead of the laptop GPU named in Appendix H.
Why: the nuPlan toolchain is Linux-only and the laptop has no WSL2 (installing it needs administrator rights and a
reboot, which an unattended job cannot do); the baselines are small enough for CPU training. Effect: none on the design.

A3 (28 Oct 13:45). Training data for the small baselines = the full nuPlan v1.1 train_pittsburgh split (Appendix H: "one
city's training logs"); Pittsburgh is the smallest city archive (30.6 GB compressed). Validation and test logs are never used
for training.

A4 (28 Oct 13:45). NAVSIM PDMS and EPDMS are computed with NAVSIM v2.2's scorer on the same nuPlan Val14 scenarios, with the
scorer's inputs built from the nuPlan databases at each scenario's first frame, instead of from OpenScene metadata. Why:
OpenScene metadata covers only 225 of the 1,381 val logs, so the Val14 scenarios could not all be scored from it; the nuPlan
databases hold the same logs. Effect: PDMS/EPDMS are NAVSIM's metrics on nuPlan scenarios, not NAVSIM's navtest numbers.

A5 (28 Oct 13:45). Released checkpoints are Python pickle files; they are loaded only on the isolated EC2 host (no IAM role, no
AWS credentials on it, SSH open to the researcher's IP only), never on the laptop. The user's instruction to run the
experiment end to end is taken as the go-ahead the earlier session rule required for loading them.

A6 (28 Oct 13:45). PlanTF and PLUTO released checkpoints are unavailable from their authors: the OneDrive links return HTTP
404 and the GitHub issues reporting this (planTF #39, pluto #44-46) are unanswered. Decision in A7.

A7 (28 Oct 14:40). PlanTF and PLUTO are restored from third-party copies instead of being replaced, which keeps Appendix H's
seven-family design. PLUTO (pluto_1M_aux_cil.ckpt, 51,431,136 B, sha256 ce60d7c8...e1): three independent copies that match
byte-for-byte on size, head/tail bytes and the full table of 1,723 internal members (Hugging Face rxinhkust/PlanScope_model_zoo,
whose owner is first author of PlanScope, co-authored by PLUTO's author, and whose README lists it as the original PLUTO
checkpoint; GitHub richie-live/SAH-Drive, committed 20 Jun 2025; a Google Drive folder of CurryChen77/RIFT). The Hugging Face
copy is used. PlanTF (planTF.ckpt, 8,184,077 B, git blob 091a5d65...54b06): one copy (SAH-Drive). Its layer names match
PlanTF's official model, it carries the NATTEN tensors that users of the official file report, and SAH-Drive reports Val14
84/77 with it (official 84.83/76.78); identity with the original is likely but not proven.
Both files were scanned statically before loading (only torch/collections/omegaconf/builtin references). Neither original
repository has a licence file, so the checkpoints are used for non-commercial research evaluation only and are not
redistributed; the README points to the mirrors. PlanTF's own RasterModel and UrbanDriver checkpoints have no copy; the raster
and Urban Driver baselines come from tuplan_garage's released PlanCNN and Urban Driver checkpoints instead.

A8 (28 Oct 13:45). The optional NAVSIM sensor arm of Appendix H (camera/lidar agents on navhard two-stage) is not run: it needs a
GPU and 31 GB of sensor data, and Appendix H marks it optional. Effect: conclusions cover planners nuPlan can simulate.

A9 (28 Oct 13:45). The user is taken to have accepted the nuPlan dataset terms (non-commercial research use, CC BY-NC-SA 4.0);
the public S3 bucket does not check registration.

A10 (28 Oct 14:25). Scenario set = 688, not 700. Val14 (1,118 scenarios) holds only 14 behind_long_vehicle and 15
following_lane_with_lead scenarios. 50 per type were drawn (seed 20261028) from Val14; the two short types were topped up
from the rest of the nuPlan val split with Val14's own filter settings (15 s spacing, valid goals). The whole val split has
only 38 + 23 such scenarios, so behind_long_vehicle reaches 50 and following_lane_with_lead 38. Every scenario row carries
its source (val14 / val_topup: 629 / 59) and the main results are re-checked on the 629 Val14 rows. Seed-variance subset:
10 per type (140). Lists: results/scenarios/val700.csv, seed140.csv, selection.json.

A11 (28 Oct 14:20). Severity weighting for the sensitivity analysis (Appendix H names severity weights without values).
Collision weight = the most vulnerable party hit at fault: vulnerable road user 1.0, vehicle 0.8, static object 0.4
(scheme S1); scheme S2 also scales it by min(1, (delta_v / 10 m/s)^2), where delta_v is nuPlan's recorded collision energy
(ego delta-v); plus 0.2 x TTC violation + 0.5 x drivable-area violation + 0.3 x driving-direction violation, capped at 1.
"Any safety violation" = at-fault collision OR TTC violation OR drivable-area violation OR driving-direction score 0.

A12 (28 Oct 14:35). NAVSIM scores at each scenario's first frame use NAVSIM v2.2's own simulator, scorer and non-reactive
log-replay traffic policy, with metric caches built from the nuPlan databases (A4). Two adaptations: (a) NAVSIM holds
OpenScene traffic-light frames (0.5 s apart) across its 0.1 s scoring steps; nuPlan databases are 0.05 s apart, so the light
state is read at every 0.1 s step instead; (b) the two-frame extended-comfort term of EPDMS needs consecutive frames, which a
first-frame evaluation lacks, so EPDMS = NC*DAC*DDC*TLC*(5 EP + 5 TTC + 2 LK + 2 HC)/14 with the human-penalty filter (v2), and
PDMS = NC*DAC*(5 EP + 5 TTC + 2 HC)/12 without the filter (v1 semantics; history comfort stands in for v1 comfort).

A13 (28 Oct 14:25). Planner variants (Appendix H gives counts, not settings). IDM (nuPlan IDMPlanner): default (v0 10 m/s,
s0 1 m, T 1.5 s, a 1.0, b 3.0); cautious (8, 3.0, 2.5, 0.8, 2.0); fast (15, 1.0, 1.0, 1.5, 3.0); aggressive (12, 0.5,
0.6, 2.5, 4.5); IDM+MOBIL = interPlan's IDMMobilPlanner with its published defaults (politeness 0.5, threshold 0.2).
PDM-Closed variants: without lateral offsets (centreline proposals only) and with a 2 s instead of 4 s proposal horizon,
chosen because both change a safety-relevant mechanism (evasive room, look-ahead). Constant velocity = nuPlan
SimplePlanner with the speed cap lifted (straight ahead at the current speed, as NAVSIM's constant-velocity agent).

A14 (28 Oct 14:15). Open-loop measures from the captured trajectories: L2 = mean over planning steps (every 0.5 s) of the
L2 error at 1, 2 and 3 s; collision with logged agents = mean over steps of the average of [first overlap with a logged
vehicle, pedestrian or cyclist at or before h], h = 1, 2, 3 s (the nuScenes/AD-MLP convention).

A15 (28 Oct 15:20). Diffusion Planner "three settings of guidance and refinement" (Appendix H): the released code has no
refinement option (the paper's "w/ refine" variant is not in the release), so the three settings are (1) default (EMA weights,
no guidance), (2) collision classifier guidance (the released GuidanceWrapper; guidance scale 0.5 fixed in the code) and
(3) raw (non-EMA) weights without guidance. All three need no code change. The released guidance is the repository's
tutorial example (documentation_guidance.md, sim_guidance_demo.sh on a 'val14-collision' subset), not a setting tuned or
reported for general driving.

A16 (28 Oct 15:20). Diffusion Planner samples stochastically and nuPlan seeds once per process. To make every scenario's
result independent of the other scenarios in the same process, the planner is reseeded at the start of each scenario from
(seed base, mission goal, route) (code/exp2_nuplan/planners.py). Seed base 0 is the main run; 1 and 2 are the two extra
sampling seeds of the seed analysis (seed subset only).

A17 (28 Oct 15:20). PLUTO "without post-processing": PLUTO at the pinned commit has no switch for it, so a 50-line subclass
(code/exp2_nuplan/pluto_no_post.py) returns the network's highest-probability trajectory and turns off all four steps of the
official post-processing: top-20 candidate trimming with the reference-free candidate, the rule-based TrajectoryEvaluator,
the 0.3 learned/rule score fusion and the EmergencyBrake. Same features, model and checkpoint.

A18 (28 Oct 15:20). Software: two conda environments (Python 3.9, torch 2.0.1 CPU). "nuplan" runs nuplan-devkit e924167 with
its locked third-party versions, tuplan_garage, interPlan, Diffusion Planner, PlanTF and PLUTO (pytorch-lightning 2.0.1,
timm 1.0.10, natten 0.14.6 built from source because the prebuilt-wheel server is offline). "navsim" runs NAVSIM v2.2 as its
requirements specify (its nuplan-devkit v1.2 code is byte-identical to e924167); it must be separate because NAVSIM's metric
cache fails with the scipy version nuPlan pins. One thread per simulation process (results change in the last digit with the
thread count, so OMP_NUM_THREADS=1 throughout). PlanCNN runs with model.pretrained=false (avoids 192 downloads of ImageNet
weights that the checkpoint overwrites anyway; identical result). A missing __init__.py was added to PLUTO's training package.

A19 (28 Oct 15:30). Before the full pass the data volume is raised from 3,000 IOPS / 250 MB/s to 6,000 IOPS / 500 MB/s and the
page cache is pre-warmed with every database the pass and the training read, so that 188 simulation processes starting
together are not throttled by cold random reads. Cost: well under US$1 per day; lowered again after the run. No effect on results.

A20 (28 Oct 15:40). Small baselines (details in env/README-training.md, decisions T1-T9):
- nuPlan's training framework runs under the environment's pytorch-lightning 2.0.1 through run-time shims instead of
  nuPlan's pinned 1.3.8, which would break PlanTF, PLUTO and Diffusion Planner in the shared environment.
- Training pool: train_pittsburgh only, nuPlan's training_scenarios filter, frames thinned to at least 2 s apart within each
  log (75,737 candidates), 50,000 drawn at random (seed 0), the same samples for all three models. 5 % of the logs (whole
  logs, by a salted hash of the name) are held out for validation; validation is reported, not used to pick checkpoints.
- EARLY checkpoint = weights after 15 % of the optimiser steps (about 1.5 of 10 epochs); FINAL = weights after the last step.
- Ego-status MLP: present + 4 past ego poses over 1.5 s, three hidden layers of 512, no augmentation; nuPlan has no
  driving-command feature, so unlike NAVSIM's ego-status MLP it has no command input.
- Raster network: nuPlan raster_model with a timm resnet18 initialised from ImageNet (nuPlan's default pretrained=true);
  vector network: nuPlan's training_simple_vector_model as is (with its kinematic augmentation).
- Optimisation, identical for all models: Adam, lr 1e-3 (chosen on a 4,000-sample check; nuPlan's 1e-4 learned far more
  slowly), x0.1 after epochs 7 and 9, weight decay 5e-4, batch 64, 10 epochs, fp32; seeds 0, 1, 2.
- Budget sized so that caching and all nine runs take about 25 min on the m8a.48xlarge.

A22 (28 Oct 16:10). The vector and MLP baselines trained with one PyTorch thread per run (a thread-variable inheritance
issue; MKL_NUM_THREADS=1), the raster baseline with 50 threads after the fix. Thread count changes speed and the last
floating-point digits, not the training procedure; all three use the same data, epochs and optimiser settings.

A23 (28 Oct 16:20). interPlan runs are split into one job per interPlan category (8 x 10 scenarios per planner) instead of
one 80-scenario job, so that no single process (30 s scenarios, up to 0.35 s per step) holds up the pass for 2.5 h. Each job
passes the category's ten official tokens with interPlan's unchanged interplan10 filter, whose 10-per-category cap then
removes nothing; the official 80 (results/scenarios/interplan80.csv) were listed with a fast constant-velocity run of the
full benchmark. The scenario set is therefore identical to interPlan's official split.

A24 (28 Oct 16:52). The full simulation pass was started while two raster-baseline seeds (1 and 2) were still training,
instead of after all training as bigrun.sh does, to use the idle cores of the large instance (code/pass_and_finish.sh).
Jobs whose checkpoint did not exist yet failed fast and were rerun once training had finished. At 17:15 the two training
processes were given higher CPU priority (renice -10); because their 50-thread OpenMP loops still stalled behind the
simulations (an epoch took over 40 min instead of 6), they were paused (SIGSTOP) at 18:01 and resumed automatically
(SIGCONT, code/resume_training_when_idle.sh) once fewer than 100 simulations were running. Pausing and resuming does
not change what is computed. This changes only the order and timing of
the runs: every job uses the final checkpoint, single-threaded simulation and the same settings as in bigrun.sh.

A25 (28 Oct 17:35). Parameter counts are measured from the checkpoints the planners load
(code/count_params.py on the simulation host, results/model_params.json): learnable tensors of the state dict, without
batch-norm statistics; for Diffusion Planner the model weights (its EMA copy has the same size).

A26 (28 Oct 18:55). interPlan ships its own model/gc_pgp_model.yaml, which comes first on interPlan's configuration search
path and shadows tuplan_garage's: its interpolations expect the key gc_pgp_model (so all eight GC-PGP interPlan jobs
failed at start-up) and it sets use_raster_feature_builder to false. For interPlan only, GC-PGP gets the overrides
+gc_pgp_model=${model} and model.use_raster_feature_builder=true (registry key interplan_overrides), which makes its model
configuration identical to the one used in the nuPlan protocols (checked: the two files differ only in that key prefix and
that flag; no other interPlan configuration file shadows a tuplan_garage or nuPlan one). The eight jobs were rerun in the
final sweep of the pass.

A27 (28 Oct 19:00). interPlan's pedestrian modifier (agents_modifier.py, get_pedestrians_at_iteration) raises an
AttributeError once the ego has left every lane, which ends that simulation as failed. Such runs are recorded as invalid
like any other simulation error: left out of the main analysis and counted as failures in the sensitivity analysis
(RQ4). interPlan's code is not patched, so the benchmark stays as
published.

A28 (28 Oct 19:05). PLUTO's released post-processing fails in scenarios that offer no reference line: it applies a
softmax to the empty set of reference-line candidates (src/planners/pluto_planner.py, line 286) before it adds its
reference-free trajectory, so the simulation ends with "ValueError: zero-size array to reduction operation maximum".
PLUTO is evaluated as released, so these runs are invalid (excluded in the main analysis, failures in the sensitivity
analysis); the variant without post-processing does not have this step.

A29 (28 Oct 20:15). interPlan's IDM+MOBIL planner (idm_mobil_planner.py) fails in 16 of its 2,064 nuPlan runs: 13 times
when it evaluates a lane change while a following vehicle has no previous state (line 626) and 3 times when it cannot
build its route plan (line 1115, an empty distance list). These runs are invalid like other planner failures; the
planner is run as published, without patches.

A30 (29 Oct 02:10, at the user's request). The small baselines' weights are published in the GitHub folder's models/:
the 12 checkpoints the planner registry loads (seed 0 early and final for the six baseline planners, seeds 1 and 2
final for the seed-variance check), copied from the SHA-256-checked backup of the training output; the 6 seed-1/2
early checkpoints, which no analysis used, are left out. Because they were trained on nuPlan data (CC BY-NC-SA 4.0),
the README and models/README.md release them for non-commercial research use under the nuPlan terms; the MIT
licence covers the code only. code/runner.py loads them from models/ when /data/exp2/baselines does not exist (a
fresh clone); the study's host is unaffected. tools/infer_baseline.py and a closed-loop harness run were checked on
the host with the training output hidden (results/models_check/).
