# Experiment 2 - data, models and code used

| Item | Source / link | Licence | Size | Used for |
|---|---|---|---|---|
| nuPlan maps v1.1 (map version nuplan-maps-v1.0) | https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip | nuPlan terms (CC BY-NC-SA 4.0) | 0.97 GB zip / 1.43 GB | all simulations and scoring |
| nuPlan v1.1 val split (1,381 SQLite logs) | https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_val.zip | nuPlan terms | 97.0 GB zip / 162.5 GB (1,381 logs) | 688 study scenarios: 629 Val14 + 59 val top-up (A10) |
| nuPlan v1.1 test split, 7 logs used by interPlan | https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip (members only) | nuPlan terms | 1.09 GB / 1.85 GB | interPlan 80-scenario split |
| nuPlan v1.1 train_pittsburgh split | https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_pittsburgh.zip | nuPlan terms | 30.6 GB zip / 55.7 GB (1,560 logs) | training the small baselines |
| tuplan_garage checkpoints: pdm_open_checkpoint.ckpt (PDM-Open), pdm_offset_checkpoint.ckpt (PDM-Hybrid), gc_pgp_checkpoint.ckpt (GC-PGP), plancnn_checkpoint.ckpt (PlanCNN), urbandriver_checkpoint.ckpt (Urban Driver) | Google Drive folder 1LLdunqyvQQuBuknzmf7KMIJiA2grLYB2 (tuplan_garage README) | see repo | 0.36 GB | released planners |
| PLUTO checkpoint pluto_1M_aux_cil.ckpt (third-party copy of the authors' release; A7) | https://huggingface.co/rxinhkust/PlanScope_model_zoo/resolve/main/pluto_1M_aux_cil_m12_ori.ckpt (sha256 ce60d7c854d10b310c4e1705099d799a6ea008baa4288e668626f4a2f22943e1) | no licence file in the original repo; research use | 51.4 MB | PLUTO family |
| PlanTF checkpoint planTF.ckpt (third-party copy; A7) | https://raw.githubusercontent.com/richie-live/SAH-Drive/master/SAH_Diffusion_Model/planTF.ckpt (git blob 091a5d65d0d50aa56725256c9057e55fc8054b06) | no licence file in the original repo; research use | 8.2 MB | released imitation planners |
| timm ResNet-18 ImageNet weights (resnet18.a1_in1k) | https://huggingface.co/timm/resnet18.a1_in1k | Apache-2.0 (timm); ImageNet-1k terms | 46.8 MB | initialisation of the small raster baseline (A20) |
| Diffusion Planner checkpoint (model.pth, args.json) | https://huggingface.co/ZhengYinan2001/Diffusion-Planner | MIT | 48.5 MB | Diffusion Planner family |

(All downloads finished 28 Oct 14:08 UTC, CRC-checked, 0 failures; code repositories and commits are listed in the README of For-GitHub.)

Produced here (not downloaded): the small baselines' checkpoints (ego MLP, vector and raster networks; early and
final, seeds 0-2) trained on the train_pittsburgh samples above (training configuration and manifests in
results/baselines/). The 12 the study evaluated (seed 0 early and final, seeds 1 and 2 final) are released in the
repository's models/ folder for non-commercial research use under the nuPlan terms (A30); all 18 are archived with
the study's records.
