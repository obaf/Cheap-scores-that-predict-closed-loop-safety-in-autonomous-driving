"""Collect one baselines run (train_baselines.sh) into <out>/manifest.json and <out>/manifest_runs.csv.

Per training run: model, seed, train/val sample counts, epochs, optimizer steps, batch size, threads, wall time,
throughput, train/val loss at the EARLY and FINAL checkpoints and the checkpoint paths. Also: run configuration,
data split (data_split.json), caching statistics (from the nuPlan cache logs) and the phase timeline.
  python write_manifest.py --out /data/exp2/baselines
"""
import argparse
import csv
import glob
import json
import os
import re
import time


def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def cache_stats(out, model):
    st = load(f"{out}/cache/{model}.timing.json") or {"model": model}
    log = f"{out}/logs/cache_{model}.log"
    if os.path.exists(log):
        with open(log, errors="replace") as f:
            text = f.read()
        m = re.search(r"ThinnedNuPlanScenarioBuilder: (\d+) candidate scenarios after ([\d.]+) s thinning", text)
        if m:
            st["candidate_pool"], st["min_gap_s"] = int(m.group(1)), float(m.group(2))
        m = re.search(r"Starting dataset caching of (\d+) files", text)
        if m:
            st["selected_samples"] = int(m.group(1))
        m = re.search(r"Failed features and targets: (\d+) out of (\d+)", text)
        if m:
            st["failed_features"], st["total_features"] = int(m.group(1)), int(m.group(2))
    st["done"] = os.path.exists(f"{out}/cache/{model}/DONE")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = a.out.rstrip("/")

    runs = []
    for f in sorted(glob.glob(f"{out}/*/seed*/train_manifest.json")):
        m = load(f)
        model, seed_dir = f.split("/")[-3], f.split("/")[-2]
        early, final = m.get("early") or {}, m.get("final") or {}
        timing = load(f"{out}/logs/train_{model}_{seed_dir}.timing.json") or {}
        runs.append({
            "model": model, "seed": int(seed_dir[4:]), "train_samples": m.get("train_samples"),
            "val_samples": m.get("val_samples"), "epochs": m.get("max_epochs"), "steps": m.get("total_steps"),
            "early_step": m.get("early_step"), "batch_size": m.get("batch_size"), "lr": m.get("lr"),
            "torch_threads": m.get("torch_threads"), "num_workers": m.get("num_workers"), "cpus": timing.get("cpus"),
            "process_wall_seconds": timing.get("process_wall_seconds"),
            "train_wall_seconds": m.get("train_wall_seconds"), "eval_wall_seconds": m.get("eval_wall_seconds"),
            "train_samples_per_second": m.get("train_samples_per_second"),
            "early_train_loss": early.get("train_loss_last100"), "early_val_loss": early.get("val_loss"),
            "final_train_loss": final.get("train_loss_last_epoch"), "final_val_loss": final.get("val_loss"),
            "early_ckpt": early.get("checkpoint"), "final_ckpt": final.get("checkpoint"), "run_manifest": f,
        })

    phases = []
    if os.path.exists(f"{out}/logs/phases.tsv"):
        with open(f"{out}/logs/phases.tsv") as f:
            for line in f:
                name, t0, t1 = line.split()[:3]
                phases.append({"phase": name, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(t0))),
                               "seconds": int(t1) - int(t0)})

    split = load(f"{out}/data_split.json") or {}
    config = load(f"{out}/run_config.json") or {}
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
        "data_split": {k: v for k, v in split.items() if k != "scenario_types"},
        "caches": [cache_stats(out, m) for m in config.get("models", ["mlp", "vector", "raster"])],
        "phases": phases,
        "runs": runs,
    }
    with open(f"{out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    if runs:
        with open(f"{out}/manifest_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(runs[0]))
            w.writeheader()
            w.writerows(runs)

    def f3(x):
        return "nan" if x is None else f"{x:.3f}"

    for r in runs:
        print(f"{r['model']:>6} seed{r['seed']}  samples train/val {r['train_samples']}/{r['val_samples']}  "
              f"epochs {r['epochs']}  steps {r['steps']}  wall: training loop {r['train_wall_seconds']} s, "
              f"process {r['process_wall_seconds']} s  {r['train_samples_per_second']} samples/s  "
              f"loss train/val: early {f3(r['early_train_loss'])}/{f3(r['early_val_loss'])}  "
              f"final {f3(r['final_train_loss'])}/{f3(r['final_val_loss'])}")
    print(f"wrote {out}/manifest.json ({len(runs)} runs)")


if __name__ == "__main__":
    main()
