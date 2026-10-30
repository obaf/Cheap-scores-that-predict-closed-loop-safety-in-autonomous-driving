"""Light progress probe for the full pass, run on the host by tools/eta.py (one pass over jobs.jsonl, a few hundred
small directory listings). Work is weighted by each job's scheduling cost (scenarios x protocol x planner speed), so the
share done is not distorted by the heavy jobs that run first. A running job's finished scenarios are its
metrics/*.pickle.temp files (nuPlan writes one per scenario and merges them into parquet files at the end)."""
import glob
import json
import os
import time

ROOT = os.environ.get("EXP2_ROOT", "/data/exp2")  # where this repository lives on the host


def main():
    units_total = units_done = 0.0
    scen_total = scen_done = jobs_done = started = failed = 0
    by = {}
    for line in open(f"{ROOT}/jobs.jsonl"):
        j = json.loads(line)
        n, d = j["n"], j["dir"]
        if os.path.exists(f"{d}/DONE"):
            k = n
            jobs_done += 1
        elif os.path.isdir(f"{d}/metrics"):
            k = min(n, len(glob.glob(f"{d}/metrics/*.pickle.temp")))
            started += 1
        else:
            k = 0
        failed += os.path.exists(f"{d}/FAILED")
        units_total += j["cost"]
        units_done += j["cost"] * k / n
        scen_total += n
        scen_done += k
        p = by.setdefault(j["protocol"], [0, 0])
        p[0] += k
        p[1] += n
    print(json.dumps({"t": time.time(), "units_done": round(units_done, 1), "units_total": round(units_total, 1),
                      "scen_done": scen_done, "scen_total": scen_total, "jobs_done": jobs_done,
                      "jobs_started": started, "failed_markers": failed, "by_protocol": by}))


if __name__ == "__main__":
    main()
