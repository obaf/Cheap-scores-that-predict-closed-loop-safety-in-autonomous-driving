# Experiment 2 compute log (AWS us-east-1, on demand; list prices from the AWS Price List API, 28 Oct 2026)

| When (UTC, 28 Oct 2026) | Resource | Change | Price |
|---|---|---|---|
| 13:30 | EC2 i-<instance> | launched as m8a.2xlarge (8 vCPU, 32 GiB), Ubuntu 22.04, 100 GB gp3 root | US$0.48688/h |
| 13:49 | EBS vol-<volume> | 500 GB gp3, 3,000 IOPS, 250 MB/s, encrypted, attached at /data | US$0.08/GB-month + throughput above 125 MB/s |
| 13:50-14:08 | downloads | 130 GB from the public nuPlan S3 bucket and Hugging Face / Google Drive (18 min) | free ingress |
| 15:30 | EBS vol-<volume> | raised to 6,000 IOPS / 500 MB/s for the run day (A19) | about US$0.8/day extra |
| 15:38-15:39 | EC2 i-<instance> | stopped, resized to m8a.48xlarge (192 vCPU, 768 GiB), started; new public IP <ip> | US$11.68512/h |
| 15:39 | bigrun.sh | self-shutdown cap armed (8 h, stop at 23:39), pre-warm, baseline training, full pass, extraction, NAVSIM scoring | |
| 15:44-15:46 | bigrun.sh | first launch aborted: training ran single-CPU because GNU nproc honours OMP_NUM_THREADS=1; fixed (NPROC passed explicitly) | |
| 15:46 | bigrun.sh | relaunched; training in parallel mode on 192 vCPUs; cost cap re-armed for 23:46 | |
| 16:08 | bigrun.sh | raster training found single-threaded (PyTorch takes its pool size from MKL_NUM_THREADS=1); stopped; MLP and vector models (finished, data-loader bound) kept | |
| 16:09 | bigrun.sh | relaunched with all thread variables set; raster retrains at 50 threads x 3; cost cap re-armed for 00:09 UTC | |
| 16:52 | pass_and_finish.sh | full pass started (2,118 jobs, 61,248 simulations, 188 workers) while the last two raster baselines finished training | |
| 17:15 | training processes | raster seeds 1-2 reniced to -10 (they slowed from about 6 to about 30 min per epoch under the pass) | |
| 17:30 | memory | 493 of 741 GiB in use (190 simulation processes, 2.8 GB mean); training, runner and driver protected from the OOM killer | |
| 18:01 | training processes | raster seeds 1-2 paused (SIGSTOP): 50-thread OpenMP loops stalled under the pass (epoch >40 min); auto-resume (SIGCONT) when fewer than 100 simulations run (code/resume_training_when_idle.sh) | |
| 19:26 | training processes | raster seeds 1-2 resumed (SIGCONT) with 85 simulations left; seed2 final 19:43, seed1 final 20:02 | |
| 19:29 | pass_and_finish.sh | main loop finished (2,080 of 2,118 jobs); final sweep 20:02-20:12 ran the 30 raster-seed jobs and 8 fixed interPlan GC-PGP jobs (2,118 of 2,118) | |
| 20:12-20:14 | pass_and_finish.sh | metric extraction (61,248 runs, 61,203 valid) and NAVSIM scoring (19,684 first frames, 190 workers); BIGRUN_DONE 20:14:12 | |
| 20:14 | downloads | results, logs and baseline manifests to experiment2/results (tools/fetch_results.sh); reproduction smoke test log | |
| 20:15 | EC2 i-<instance> | stopped; instance type set back to m8a.2xlarge (left stopped; volumes kept) | m8a.48xlarge 15:39-20:15 = 4.6 h, about US$54 |
| 21:32 | EBS vol-<volume> | IOPS and throughput set back to 3,000 / 250 MB/s (after the 6 h modification wait) | gp3 rate; 250 MB/s is 125 MB/s above the free baseline, about US$5/month |
| 29 Oct 01:20 | EC2 i-<instance> | started (m8a.2xlarge) to back up what a re-download cannot recreate | US$0.48688/h |
| 29 Oct 01:23-01:31 | backup | make_backup.sh packed 5 archives (4.79 GB: trained baselines, run outputs, code, conda environments, third-party checkpoints); scp to experiment2/Artefacts-backup, all SHA-256 checks OK | about 4.8 GB out of AWS, inside the free 100 GB a month |
| 29 Oct 01:32 | EC2 i-<instance> | staging copies removed from /data; stopped again | 12 min, about US$0.10 |
| 29 Oct 02:01-02:05 | EC2 i-<instance> | started to test the released models as a fresh clone would use them (training output hidden by a temporary bind mount): tools/infer_baseline.py for all three model types and a closed-loop harness run, all valid (results/models_check/); test folder removed; stopped | 5 min, about US$0.04 |
| 29 Oct 04:51 | AWS clean-up | at the user's request: EC2 i-<instance> terminated (its 100 GiB root volume and network interface were deleted with it), then EBS vol-<volume> (500 GiB), security group av-exp2-ssh and key pair av-exp2-key deleted; checks afterwards found nothing from experiment 2 on AWS | storage charges of about US$53 a month end |
