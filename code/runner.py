"""Sharded, resumable simulation runner for experiment 2 (runs on the host).

  python runner.py make  --registry planners.yaml   # writes <root>/jobs.jsonl
  python runner.py run   --workers 188              # runs every job without a done marker
  python runner.py status

Job = one planner x one protocol x one shard of scenarios (shards group whole logs, about 32 scenarios, so each
nuPlan process opens only its own databases). Protocols: ols, clsnr, clsr on the 688 scenarios (val700.csv), seed
variants on seed140.csv, and interplan (interPlan's official 80-scenario filter, one job per planner).
Each job runs nuPlan's (or interPlan's) run_simulation.py with worker=sequential and one thread per process; a job
is done when it exits 0 and wrote metric files. /data/exp2/progress.json holds counters for tools/eta.py.
"""
import argparse
import concurrent.futures as cf
import json
import os
import shlex
import subprocess
import threading
import time
from collections import defaultdict

import pandas as pd
import yaml

ROOT = os.environ.get("EXP2_ROOT", "/data/exp2")  # where this repository lives on the host
SPLITS = "/data/nuplan/dataset/nuplan-v1.1/splits"
ENTRY = {"nuplan": "/data/code/nuplan-devkit/nuplan/planning/script/run_simulation.py",
         "interplan": "/data/code/interPlan/interplan/planning/script/run_simulation.py"}
NUPLAN_SEARCHPATH = ["pkg://nuplan.planning.script.config.common", "pkg://nuplan.planning.script.experiments"]
INTERPLAN_SEARCHPATH = ["pkg://interplan.planning.script.config.common", "pkg://interplan.planning.script.config.simulation",
                        "pkg://interplan.planning.script.experiments", "pkg://tuplan_garage.planning.script.config.common",
                        "pkg://tuplan_garage.planning.script.config.simulation", "pkg://nuplan.planning.script.config.common",
                        "pkg://nuplan.planning.script.config.simulation", "pkg://nuplan.planning.script.experiments"]
INTERPLAN_SMOKE_TOKEN = "2d62c3139aa95007-s0"
PROTOCOL = {"ols": "+simulation=open_loop_boxes", "clsnr": "+simulation=closed_loop_nonreactive_agents",
            "clsr": "+simulation=closed_loop_reactive_agents"}
# nuPlan's metric aggregator only reads metric files whose path contains the challenge name, so the protocol
# folders are named after the challenges.
CHALLENGE = {"ols": "open_loop_boxes", "clsnr": "closed_loop_nonreactive_agents", "clsr": "closed_loop_reactive_agents",
             "interplan": "interplan"}
COST = {"ols": 1.0, "clsnr": 1.0, "clsr": 1.15, "interplan": 2.8}  # interPlan scenarios last 30 s, not 15 s  # relative cost per scenario (scheduling only)
SHARD = 32
HOST_BASELINES = "/data/exp2/baselines/"  # where code/train_baselines.sh writes the small baselines
LOCK = threading.Lock()


def q(items):
    return "[" + ",".join("'" + str(x) + "'" for x in items) + "]"


def shards(df, size=SHARD):
    """Greedy packing of whole logs into shards of about `size` scenarios."""
    per_log = df.groupby("log_name").size().sort_values(ascending=False)
    bins = []
    for log, n in per_log.items():
        for b in bins:
            if b["n"] + n <= size:
                b["logs"].append(log)
                b["n"] += n
                break
        else:
            bins.append({"logs": [log], "n": n})
    return [df[df.log_name.isin(b["logs"])] for b in bins]


def common_overrides(job):
    out = ["worker=sequential", "~callback.simulation_log_callback", "~main_callback.metric_summary_callback",
           "experiment_name=exp2", f"job_name={job['id']}", f"output_dir={job['dir']}",
           "enable_simulation_progress_bar=false", "exit_on_failure=false"]
    return out


def resolve_checkpoint(override):
    """Point a registry override at the released copy in models/ when the training output is not on this machine
    (a fresh clone of the repository); the study's host, and anyone who retrains, keep the registry's path."""
    key, _, path = override.partition("=")
    if key.endswith(".checkpoint_path") and path.startswith(HOST_BASELINES) and not os.path.exists(path):
        released = os.path.join(ROOT, "models", path[len(HOST_BASELINES):])
        if os.path.exists(released):
            return f"{key}={released}"
    return override


def build_command(job, reg):
    p = reg["planners"][job["planner_entry"]]
    ov = [resolve_checkpoint(o) for o in p.get("overrides", [])]
    if job.get("seed_override"):
        ov.append(job["seed_override"])
    if job["protocol"] == "interplan":
        ov = ["+simulation=default_interplan_benchmark", "scenario_filter=interplan10",
              f"scenario_builder.db_files={SPLITS}/test"] + ov
        if job.get("smoke"):
            ov.append(f"scenario_filter.scenario_tokens=['{INTERPLAN_SMOKE_TOKEN}']")
        elif job["tokens"]:  # one category of the official 80 (<= 10 tokens, so the per-type cap removes nothing)
            ov.append(f"scenario_filter.scenario_tokens={q(job['tokens'])}")
        ov += list(p.get("interplan_overrides", []))  # planner-specific fixes for interPlan's config search path
    else:
        ov = [PROTOCOL[job["protocol"]], "scenario_builder=nuplan", "scenario_filter=all_scenarios",
              f"scenario_builder.db_files={q(job['db_files'])}",
              f"scenario_filter.scenario_types={q(job['types'])}", f"scenario_filter.scenario_tokens={q(job['tokens'])}",
              "scenario_filter.remove_invalid_goals=true"] + ov
        if job["protocol"] == "ols":
            ov += ["+callback.trajectory_capture._target_=exp2_nuplan.callbacks.TrajectoryCaptureCallback",
                   f"+callback.trajectory_capture.output_dir={job['dir']}/traj"]
    interplan = job["protocol"] == "interplan"
    extra = list(p.get("searchpath", [])) + (list(p.get("config_dirs", [])) if interplan else [])
    if interplan or not p.get("keep_searchpath"):
        # planners with their own run_simulation.py (PlanTF, PLUTO) keep their own search path in nuPlan protocols;
        # interPlan's script needs interPlan's search path plus the planner's own config folder
        base = INTERPLAN_SEARCHPATH if interplan else NUPLAN_SEARCHPATH
        sp = base + [x for x in extra if x not in base]
        ov.append("hydra.searchpath=[" + ",".join(sp) + "]")
    ov += common_overrides(job)
    pypath = ":".join([f"{ROOT}/code"] + list(p.get("pythonpath", [])))
    script = ENTRY["interplan"] if interplan else p.get("script", ENTRY["nuplan"])
    cwd = p.get("cwd", job["dir"])
    py = f"mkdir -p {job['dir']} && cd {cwd} && PYTHONPATH={pypath}:$PYTHONPATH python {script} " + \
        " ".join(shlex.quote(o) for o in ov)
    return f"source /data/code/envs/env.sh {p.get('env', 'nuplan')} && {py}"


def load_registry(paths):
    """Merge one or more registry files (comma-separated) into one {'planners': {...}} dict."""
    reg = {"planners": {}}
    for path in paths.split(","):
        part = yaml.safe_load(open(path)) or {}
        clash = set(part.get("planners", {})) & set(reg["planners"])
        if clash:
            raise ValueError(f"planner defined twice: {sorted(clash)}")
        reg["planners"].update(part.get("planners", {}))
    return reg


def make(a):
    reg = load_registry(a.registry)
    val = pd.read_csv(f"{ROOT}/scenarios/val700.csv")
    sub = pd.read_csv(f"{ROOT}/scenarios/seed140.csv")
    ip_path = f"{ROOT}/scenarios/interplan80.csv"
    ip = pd.read_csv(ip_path).rename(columns={"scenario": "token"}) if os.path.exists(ip_path) else None
    if a.limit:  # smoke tests: the first `limit` scenarios of one log
        log = val.log_name.value_counts().index[0]
        val = sub = val[val.log_name == log].head(a.limit)
    jobs = []
    for name, p in reg["planners"].items():
        seed_of, seed = p.get("seed_of"), p.get("seed", 0)
        speed = p.get("speed", 1.0)
        protos = p.get("protocols", ["ols", "clsnr", "clsr"] + ([] if seed_of else ["interplan"]))
        if a.protocols:
            protos = [x for x in a.protocols.split(",") if x in protos]
        scen = sub if seed_of else val
        for proto in protos:
            if proto == "interplan":
                # one job per interPlan category (10 scenarios each) when the official list is known
                parts = [g for _, g in ip.groupby("scenario_type")] if ip is not None and not a.limit else [None]
            else:
                parts = shards(scen)
            for k, sh in enumerate(parts):
                jid = f"{proto}__{name}__{k:03d}"
                job = {"id": jid, "planner_entry": name, "planner": seed_of or name, "seed": seed,
                       "family": p["family"], "protocol": proto, "dir": f"{ROOT}/{a.tag}/{CHALLENGE[proto]}/{name}/{k:03d}",
                       "seed_override": p.get("seed_override", "").format(seed=seed) if p.get("seed_override") else ""}
                if sh is None:
                    job.update(n=1 if a.limit else 80, tokens=[], types=[], db_files=[], smoke=bool(a.limit))
                elif proto == "interplan":
                    job.update(n=len(sh), tokens=list(sh.token), types=sorted(sh.scenario_type.unique()), db_files=[])
                else:
                    job.update(n=len(sh), tokens=list(sh.token), types=sorted(sh.scenario_type.unique()),
                               db_files=[f"{SPLITS}/val/{lg}.db" for lg in sorted(sh.log_name.unique())])
                job["cost"] = job["n"] * COST[proto] * speed
                job["cmd"] = build_command(job, reg)
                jobs.append(job)
    jobs.sort(key=lambda j: -j["cost"])  # longest first
    with open(a.jobs, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    by = defaultdict(int)
    for j in jobs:
        by[j["protocol"]] += j["n"]
    print(f"{len(jobs)} jobs, {sum(by.values()):,} scenario runs: {dict(by)}")


def load_jobs(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_done(job):
    return os.path.exists(f"{job['dir']}/DONE")


def has_metrics(job):
    for sub in ("metrics", "aggregator_metric"):
        d = f"{job['dir']}/{sub}"
        if os.path.isdir(d) and any(n.endswith(".parquet") for n in os.listdir(d)):
            return True
    return False


class Progress:
    def __init__(self, jobs):
        self.total = sum(j["n"] for j in jobs)
        self.done = sum(j["n"] for j in jobs if is_done(j))
        self.jobs_total, self.jobs_done = len(jobs), sum(is_done(j) for j in jobs)
        self.failed, self.t0, self.recent = [], time.time(), []

    def finish(self, job, ok, secs):
        with LOCK:
            if ok:
                self.done += job["n"]
                self.jobs_done += 1
                self.recent.append((time.time(), job["n"]))
            else:
                self.failed.append(job["id"])
            self.write()

    def write(self):
        now = time.time()
        win = [n for t, n in self.recent if now - t < 1800]
        span = min(1800, now - self.t0)
        rate = sum(win) / (span / 60) if span > 60 and win else None
        info = {"runs_total": self.total, "runs_done": self.done, "jobs_total": self.jobs_total,
                "jobs_done": self.jobs_done, "jobs_failed": len(self.failed), "failed": self.failed[-50:],
                "rate_runs_per_min": rate, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(f"{ROOT}/progress.json.tmp", "w") as f:
            json.dump(info, f)
        os.replace(f"{ROOT}/progress.json.tmp", f"{ROOT}/progress.json")


def run_job(job, prog, timeout):
    os.makedirs(job["dir"], exist_ok=True)
    t0 = time.time()
    with open(f"{job['dir']}/log.txt", "w") as log:
        log.write(job["cmd"] + "\n\n")
        log.flush()
        try:
            r = subprocess.run(["bash", "-c", job["cmd"]], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    ok = rc == 0 and has_metrics(job)
    secs = time.time() - t0
    if ok:
        with open(f"{job['dir']}/DONE", "w") as f:
            json.dump({"seconds": round(secs, 1), "n": job["n"]}, f)
    else:
        with open(f"{job['dir']}/FAILED", "w") as f:
            json.dump({"returncode": rc, "seconds": round(secs, 1)}, f)
    prog.finish(job, ok, secs)
    return job["id"], ok, rc, secs


def run(a):
    jobs = [j for j in load_jobs(a.jobs) if (not a.only or any(s in j["id"] for s in a.only.split(",")))]
    todo = [j for j in jobs if not is_done(j)]
    prog = Progress(jobs)
    prog.write()
    print(f"{len(todo)} of {len(jobs)} jobs to run with {a.workers} workers", flush=True)
    for attempt in range(1 + a.retries):
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = [ex.submit(run_job, j, prog, a.timeout) for j in todo]
            for fut in cf.as_completed(futs):
                jid, ok, rc, secs = fut.result()
                print(f"{time.strftime('%H:%M:%S')} {'ok ' if ok else 'ERR'} {jid} rc={rc} {secs:.0f}s "
                      f"[{prog.jobs_done}/{prog.jobs_total}]", flush=True)
        todo = [j for j in jobs if not is_done(j)]
        if not todo:
            break
        prog.failed = []
        print(f"retry {attempt + 1}: {len(todo)} jobs", flush=True)
    print(f"finished: {prog.jobs_done}/{prog.jobs_total} jobs done; {len(todo)} not done", flush=True)


def status(a):
    jobs = load_jobs(a.jobs)
    agg = defaultdict(lambda: [0, 0, 0])
    for j in jobs:
        k = (j["protocol"], j["planner"])
        agg[k][0] += 1
        agg[k][1] += is_done(j)
        agg[k][2] += os.path.exists(f"{j['dir']}/FAILED") and not is_done(j)
    for k in sorted(agg):
        print(k, "jobs %d done %d failed %d" % tuple(agg[k]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "run", "status"])
    ap.add_argument("--registry", default=f"{ROOT}/code/planners_core.yaml",
                    help="one or more registry files, comma-separated")
    ap.add_argument("--jobs", default=f"{ROOT}/jobs.jsonl")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--timeout", type=int, default=3 * 3600)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="runs", help="output folder under the root (runs, or e.g. smoke)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many scenarios")
    ap.add_argument("--protocols", default="", help="comma list overriding the registry's protocols")
    a = ap.parse_args()
    {"make": make, "run": run, "status": status}[a.cmd](a)
