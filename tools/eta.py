"""Rough ETA for experiment 2. Light by design: one short SSH call that runs a tiny probe on the host, then exits.

Phase plan (minutes) is a planning estimate; finished phases are recorded in tools/eta_state.json by the lead
(python tools/eta.py --done <phase>). During the simulation pass the remaining time comes from the host probe
code/pass_progress.py: cost-weighted work done (scenarios x protocol x planner speed, running jobs counted per finished
scenario), with the rate measured over the last 10-40 minutes of samples kept in tools/eta_samples.jsonl.
Usage: python tools/eta.py [--done PHASE] [--host IP]
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import time

PHASES = [  # (key, label, planned minutes)
    ("downloads", "data downloads", 45),
    ("envs", "environments and planner integration", 120),
    ("harness", "harness and smoke tests", 45),
    ("training", "small-baseline training on the big host", 75),
    ("pass", "full simulation pass (61,248 runs on 188 workers)", 150),
    ("scoring", "metric extraction, NAVSIM scoring, download and statistics", 45),
]
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "eta_state.json")
SAMPLES = os.path.join(HERE, "eta_samples.jsonl")
KEY = os.path.expanduser(os.environ.get("EXP2_SSH_KEY", "~/.ssh/id_ed25519"))  # key of the simulation host
PASS_START = dt.datetime(2026, 10, 28, 16, 52, 16, tzinfo=dt.timezone.utc).timestamp()


def load():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"done": {}}


def host_probe(host):
    cmd = ["ssh", "-i", KEY, "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", f"ubuntu@{host}",
           "python3 /data/exp2/code/pass_progress.py 2>/dev/null || echo {}"]
    try:
        return json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout or "{}")
    except Exception:
        return {}


def pass_minutes_left(pc):
    """Remaining pass minutes from the cost-weighted rate over a recent window (10-40 min), else since the start."""
    samples = []
    if os.path.exists(SAMPLES):
        with open(SAMPLES) as f:
            samples = [json.loads(x) for x in f if x.strip()]
    samples.append({"t": pc["t"], "u": pc["units_done"]})
    with open(SAMPLES, "w") as f:
        f.write("".join(json.dumps(s) + chr(10) for s in samples[-200:]))
    now = pc["t"]
    old = [s for s in samples if 600 <= now - s["t"] <= 2400]
    t0, u0 = (old[0]["t"], old[0]["u"]) if old else (PASS_START, 0.0)
    rate = (pc["units_done"] - u0) / max(now - t0, 60) * 60  # units per minute
    return (pc["units_total"] - pc["units_done"]) / max(rate, 1e-6) + 10  # + tail of the last jobs and final sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--done")
    ap.add_argument("--host", default=os.environ.get("EXP2_HOST", ""), help="simulation host (or set EXP2_HOST)")
    a = ap.parse_args()
    st = load()
    now = dt.datetime.now(dt.timezone.utc)
    if a.done:
        st["done"][a.done] = now.isoformat(timespec="minutes")
        with open(STATE, "w") as f:
            json.dump(st, f, indent=1)
    left, cur = 0.0, None
    for key, label, minutes in PHASES:
        if key in st["done"]:
            continue
        if cur is None:
            cur = label
            if key == "pass":
                pc = host_probe(a.host)
                if pc.get("units_total"):
                    minutes = pass_minutes_left(pc)
                    cur += (f" ({pc['scen_done']:,}/{pc['scen_total']:,} runs, "
                            f"{100 * pc['units_done'] / pc['units_total']:.0f}% of the cost-weighted work)")
            else:
                minutes *= 0.6  # the current phase is on average part-way through
        left += minutes
    docs_left = sum(m for k, _, m in PHASES if k not in st["done"] and k == "docs")
    fmt = "%a %d %b %H:%M UTC"
    print(f"current phase: {cur or 'finished'} | experiment results ~"
          f"{(now + dt.timedelta(minutes=left - docs_left)).strftime(fmt)} | all deliverables ~"
          f"{(now + dt.timedelta(minutes=left)).strftime(fmt)}")


if __name__ == "__main__":
    main()
