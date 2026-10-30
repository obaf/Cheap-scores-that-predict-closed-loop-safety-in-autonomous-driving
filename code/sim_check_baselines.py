"""Drive planners of a registry on ONE scenario in nuPlan simulation (default: closed_loop_reactive_agents) to check
that each planner builds, loads its checkpoint and drives. Each command is made by runner.build_command, i.e. it is
exactly the command the experiment runner will use. Does not touch runner.py's progress.json or job files.

  python sim_check_baselines.py --registry planners_baselines.yaml                 # all entries, sequentially
  python sim_check_baselines.py --registry planners_baselines.yaml --only mlp_final,raster_early --parallel 2
  python sim_check_baselines.py --registry planners_baselines.yaml \
      --replace /data/exp2/baselines/=/data/exp2/train_smoke/baselines/            # smoke checkpoints
Writes <out>/<challenge>/<planner>/ (nuPlan outputs + log.txt) and <out>/sim_check.json.
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import subprocess
import sys
import time

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner  # noqa: E402

VAL = "/data/nuplan/dataset/nuplan-v1.1/splits/val"


def run_one(name, reg, a):
    job = {"id": f"simcheck__{name}", "planner_entry": name, "protocol": a.protocol,
           "dir": f"{a.out}/{runner.CHALLENGE[a.protocol]}/{name}", "db_files": [f"{VAL}/{a.log}.db"],
           "types": [a.type], "tokens": [a.token], "seed_override": ""}
    os.makedirs(job["dir"], exist_ok=True)
    cmd = runner.build_command(job, reg)
    t0 = time.time()
    with open(f"{job['dir']}/log.txt", "w") as log:
        log.write(cmd + "\n\n")
        log.flush()
        rc = subprocess.run(["bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout).returncode
    res = {"planner": name, "returncode": rc, "wall_s": round(time.time() - t0, 1), "has_metrics": runner.has_metrics(job)}
    rep = glob.glob(f"{job['dir']}/runner_report.parquet")
    if rep:
        r = pd.read_parquet(rep[0]).iloc[0]
        res.update(succeeded=bool(r.succeeded), error=r.error_message, planner_name=r.planner_name,
                   sim_duration_s=round(float(r.duration), 1),
                   planner_step_ms=round(1e3 * float(r.compute_trajectory_runtimes_mean), 1))
    agg = sorted(glob.glob(f"{job['dir']}/aggregator_metric/*.parquet"))
    if agg:
        df = pd.read_parquet(agg[-1])
        row = df[df.scenario == a.token]
        if len(row):
            row = row.iloc[0]
            res["score"] = float(row["score"])
            for m in ("no_ego_at_fault_collisions", "drivable_area_compliance", "ego_is_making_progress",
                      "ego_progress_along_expert_route", "time_to_collision_within_bound"):
                if m in row:
                    res[m] = None if pd.isna(row[m]) else round(float(row[m]), 3)
    res["ok"] = rc == 0 and res["has_metrics"] and res.get("succeeded", False)
    print(json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="/data/exp2/code/planners_baselines.yaml")
    ap.add_argument("--replace", default="", help="old=new, applied to every override string (e.g. checkpoint root)")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="/data/exp2/train_smoke/sim")
    ap.add_argument("--protocol", default="clsr", choices=["ols", "clsnr", "clsr"])
    ap.add_argument("--log", default="2021.07.24.00.58.02_veh-12_01831_03390")
    ap.add_argument("--type", default="behind_long_vehicle")
    ap.add_argument("--token", default="ac83bb7914fa5c49")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    reg = yaml.safe_load(open(a.registry))
    if a.replace:
        old, new = a.replace.split("=", 1)
        for p in reg["planners"].values():
            p["overrides"] = [o.replace(old, new) for o in p["overrides"]]
    names = [n for n in reg["planners"] if not a.only or n in a.only.split(",")]
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        results = list(ex.map(lambda n: run_one(n, reg, a), names))
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/sim_check.json", "w") as f:
        json.dump({"registry": os.path.abspath(a.registry), "replace": a.replace, "protocol": a.protocol,
                   "scenario": {"log": a.log, "type": a.type, "token": a.token}, "results": results}, f, indent=1)
    bad = [r["planner"] for r in results if not r["ok"]]
    print(f"{len(results) - len(bad)}/{len(results)} planners drove the scenario" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
