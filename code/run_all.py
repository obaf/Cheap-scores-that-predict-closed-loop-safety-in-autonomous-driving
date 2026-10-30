"""run_all.py - one entry point for reproducing experiment 2.

On the Linux simulation host (after code/data/download_all.sh and the env recipes in env/):
    python code/run_all.py host          # scenario lists, then bigrun.sh: baseline training, ~61,000 nuPlan/interPlan
                                         # simulations, metric extraction, NAVSIM scoring (hours on 192 vCPUs;
                                         # the study's timings are in results/summary.json)
On any machine with the analysis packages (requirements.txt), after copying the host's results folder:
    python code/run_all.py analysis      # merge -> statistics -> figures -> summary.json
    python code/run_all.py analysis --quick   # small bootstraps, for a fast check
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, env_name=None):
    print(">>", cmd, flush=True)
    if env_name:
        cmd = f"source /data/code/envs/env.sh {env_name} && {cmd}"
        r = subprocess.run(["bash", "-c", cmd], cwd=ROOT)
    else:
        r = subprocess.run(cmd, shell=True, cwd=ROOT)
    if r.returncode:
        sys.exit(f"step failed ({r.returncode}): {cmd}")


def host():
    sh(f"PYTHONPATH=code python code/select_scenarios.py --out {os.environ.get('EXP2_ROOT', '/data/exp2')}/scenarios", env_name="nuplan")
    sh("bash code/bigrun.sh")


def analysis(quick):
    py = sys.executable
    sh(f'"{py}" code/summarize.py merge')
    sh(f'"{py}" code/stats.py --runs results/runs_merged.csv --out results/stats' + (" --quick" if quick else ""))
    sh(f'"{py}" code/figures.py --stats results/stats --out figures')
    sh(f'"{py}" code/summarize.py summary')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["host", "analysis"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    host() if a.stage == "host" else analysis(a.quick)
