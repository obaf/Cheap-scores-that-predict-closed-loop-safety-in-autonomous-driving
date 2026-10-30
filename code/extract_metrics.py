"""Collect every finished job into one tidy table: results/runs.csv (one row per planner x protocol x scenario x seed).

Per run it records validity (nuPlan runner report), the official scenario score from nuPlan's weighted-average
aggregator (CLS-NR / CLS-R / OLS / interPlan), the safety outcomes, and for open-loop runs the measures computed from the
captured trajectories (exp2_nuplan.callbacks):
- l2_avg: mean over planning steps (every 0.5 s) of the L2 error at 1, 2 and 3 s (nuScenes convention);
- ade_8s: mean over steps of the average L2 error over the 8 s horizon;
- log_collision_rate: mean over steps of the average of [first collision with a logged agent at or before h], h=1,2,3 s;
- log_collision_8s: share of steps whose plan overlaps a logged agent within 8 s.
Severity (sensitivity analysis, assumption A11): collision weight by the most vulnerable party hit (VRU 1.0, vehicle 0.8,
object 0.4) - S1 - or additionally scaled by min(1, (delta_v / 10 m/s)^2) from nuPlan's collision energy - S2; plus
0.2 x TTC violation + 0.5 x drivable-area violation + 0.3 x driving-direction violation, capped at 1.
Scenarios a job was given but did not simulate are written with valid = False.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("EXP2_ROOT", "/data/exp2")  # where this repository lives on the host
PROTO_OF = {"open_loop_boxes": "ols", "closed_loop_nonreactive_agents": "clsnr",
            "closed_loop_reactive_agents": "clsr", "interplan": "interplan"}
TYPE_W = {"VRUs": 1.0, "vehicles": 0.8, "objects": 0.4}
CLS_COLS = {"no_ego_at_fault_collisions": "m_collision", "drivable_area_compliance": "m_drivable",
            "driving_direction_compliance": "m_direction", "ego_is_making_progress": "m_progress_made",
            "time_to_collision_within_bound": "m_ttc", "ego_progress_along_expert_route": "m_progress_ratio",
            "speed_limit_compliance": "m_speed_limit", "ego_is_comfortable": "m_comfort"}
OLS_COLS = {"planner_expert_average_l2_error_within_bound": "m_avg_l2", "planner_expert_final_l2_error_within_bound":
            "m_final_l2", "planner_miss_rate_within_bound": "m_miss_rate",
            "planner_expert_average_heading_error_within_bound": "m_avg_heading",
            "planner_expert_final_heading_error_within_bound": "m_final_heading"}


def read_one(pattern):
    files = sorted(glob.glob(pattern))
    return pd.read_parquet(files[-1]) if files else None


def last_line(msg):
    """The exception line of a traceback (its last non-empty line), which names the cause."""
    lines = [x.strip() for x in str(msg).strip().splitlines() if x.strip().strip("'")]
    return (lines[-1] if lines else str(msg))[:300]


def collision_table(job_dir):
    c = read_one(f"{job_dir}/metrics/no_ego_at_fault_collisions.parquet")
    if c is None:
        return pd.DataFrame()
    out = pd.DataFrame({"scenario": c.scenario_name})
    for k in TYPE_W:
        out[f"n_col_{k}"] = c.get(f"number_of_at_fault_collisions_with_{k}_stat_value", 0)
        out[f"energy_{k}"] = c.get(f"max_collision_energy_with_{k}_stat_value", np.nan)
    out["n_col_all"] = c.get("number_of_all_at_fault_collisions_stat_value", 0)
    return out


def traj_measures(job_dir):
    rows = []
    for f in glob.glob(f"{job_dir}/traj/*.json"):
        j = json.load(open(f))
        l2_123, ade8, col123, col8 = [], [], [], []
        for st in j["steps"]:
            l2 = st["l2"]
            if len(l2) >= 6:  # samples every 0.5 s: indices 1, 3, 5 are 1, 2, 3 s
                l2_123.append(np.mean([l2[1], l2[3], l2[5]]))
            if l2:
                ade8.append(np.mean(l2))
            t = st["col"][0] if st.get("col") else None
            col123.append(np.mean([t is not None and t <= h for h in (1.0, 2.0, 3.0)]))
            col8.append(t is not None)
        rows.append({"scenario": j["token"], "l2_avg": np.mean(l2_123) if l2_123 else np.nan,
                     "ade_8s": np.mean(ade8) if ade8 else np.nan,
                     "log_collision_rate": np.mean(col123) if col123 else np.nan,
                     "log_collision_8s": np.mean(col8) if col8 else np.nan,
                     "first_frame_ok": bool(j.get("first") and len(j["first"].get("poses_0p5s", [])) >= 8)})
    return pd.DataFrame(rows)


def severity(r, energy):
    w = 0.0
    for k, tw in TYPE_W.items():
        if r.get(f"n_col_{k}", 0) > 0:
            e = r.get(f"energy_{k}", np.nan)
            f = min(1.0, (e / 10.0) ** 2) if energy and e == e else 1.0
            w = max(w, tw * f)
    other = 0.2 * r["ttc_violation"] + 0.5 * r["drivable_violation"] + 0.3 * r["direction_violation"]
    return min(1.0, w + other)


def extract_job(job):
    d, proto = job["dir"], job["protocol"]
    expected = job["tokens"] or None
    rep = read_one(f"{d}/runner_report.parquet")
    agg = read_one(f"{d}/aggregator_metric/*.parquet")
    base = {"planner": job["planner"], "family": job["family"], "protocol": proto, "seed": job["seed"],
            "planner_entry": job["planner_entry"], "job": job["id"]}
    if agg is not None:
        agg = agg[agg.log_name.notna()].copy()  # drop the per-type and final_score rows
    rows = {}
    if agg is not None:
        cols = OLS_COLS if proto == "ols" else CLS_COLS
        for _, a in agg.iterrows():
            r = {**base, "scenario": a.scenario, "scenario_type": a.scenario_type, "log_name": a.log_name,
                 "valid": True, ("ols_score" if proto == "ols" else
                                 "interplan_score" if proto == "interplan" else "cls_score"): float(a.score)}
            for src, dst in cols.items():
                if src in a:
                    r[dst] = float(a[src]) if a[src] is not None else np.nan
            rows[a.scenario] = r
    if rep is not None:
        for _, x in rep.iterrows():
            r = rows.setdefault(x.scenario_name, {**base, "scenario": x.scenario_name, "log_name": x.log_name,
                                                   "valid": False})
            r["valid"] = bool(r.get("valid") and x.succeeded)
            r["error"] = None if x.succeeded else last_line(x.error_message)
            r["planner_runtime_s"] = float(x.compute_trajectory_runtimes_mean)
            r["sim_duration_s"] = float(x.duration)
    for tok in expected or []:
        rows.setdefault(tok, {**base, "scenario": tok, "valid": False, "error": "not simulated"})
    df = pd.DataFrame(list(rows.values()))
    if proto != "ols":
        col = collision_table(d)
        if len(col):
            df = df.merge(col, on="scenario", how="left")
        for k in TYPE_W:
            df[f"n_col_{k}"] = df.get(f"n_col_{k}", pd.Series(0, index=df.index)).fillna(0)
        def col(name, default):
            return df[name].astype(float) if name in df else pd.Series(default, index=df.index, dtype=float)
        df["at_fault_collision"] = (col("n_col_all", 0) > 0).astype(float)
        df["ttc_violation"] = (col("m_ttc", 1.0) < 1).astype(float)
        df["drivable_violation"] = (col("m_drivable", 1.0) < 1).astype(float)
        df["direction_violation"] = 1.0 - col("m_direction", 1.0)
        df["any_violation"] = ((df.at_fault_collision + df.ttc_violation + df.drivable_violation +
                                (df.direction_violation >= 1)) > 0).astype(float)
        # safety outcomes alone reward planners that barely move; this outcome also fails a scenario without progress
        df["no_progress"] = (col("m_progress_made", 1.0) < 1).astype(float)
        df["violation_or_no_progress"] = ((df.any_violation + df.no_progress) > 0).astype(float)
        df["severity_s1"] = df.apply(lambda r: severity(r, False), axis=1)
        df["severity_s2"] = df.apply(lambda r: severity(r, True), axis=1)
        outcome = ["at_fault_collision", "ttc_violation", "drivable_violation", "direction_violation",
                   "any_violation", "no_progress", "violation_or_no_progress", "severity_s1", "severity_s2"]
        df.loc[~df.valid.astype(bool), outcome] = np.nan
    else:
        tm = traj_measures(d)
        if len(tm):
            df = df.merge(tm, on="scenario", how="left")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=f"{ROOT}/jobs.jsonl")
    ap.add_argument("--out", default=f"{ROOT}/results/runs.csv")
    ap.add_argument("--scenarios", default=f"{ROOT}/scenarios/val700.csv")
    a = ap.parse_args()
    jobs = [json.loads(x) for x in open(a.jobs) if x.strip()]
    parts, missing = [], []
    for j in jobs:
        if os.path.exists(f"{j['dir']}/DONE") or os.path.exists(f"{j['dir']}/FAILED"):
            parts.append(extract_job(j))
        else:
            missing.append(j["id"])
    df = pd.concat(parts, ignore_index=True)
    sc = pd.read_csv(a.scenarios)[["token", "scenario_type", "log_name", "source"]]
    meta = sc.set_index("token")
    for c in ("scenario_type", "log_name"):
        fill = df.scenario.map(meta[c]) if c in meta else None
        df[c] = df[c].fillna(fill) if c in df else fill
    df["source"] = df.scenario.map(meta.source).fillna("interplan")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    summ = df.groupby(["protocol"]).agg(runs=("valid", "size"), valid=("valid", "sum")).to_dict("index")
    print(json.dumps({"rows": len(df), "by_protocol": summ, "jobs_not_run": len(missing)}, indent=1, default=str))


if __name__ == "__main__":
    main()
