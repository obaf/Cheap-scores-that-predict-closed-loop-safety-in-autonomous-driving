"""NAVSIM v2.2 PDMS and EPDMS of every planner's first-frame trajectory on the study scenarios (host, `navsim` env).

NAVSIM normally scores OpenScene frames. Here its own MetricCacheProcessor is fed nuPlan database scenarios directly
(the same val logs OpenScene was converted from; assumption A4), so no OpenScene pickles are needed. Per scenario the
metric cache is built once and every planner's trajectory planned at the scenario's first frame (captured in the
open-loop runs by exp2_nuplan.callbacks) is scored with NAVSIM's PDMSimulator + PDMScorer and the non-reactive
log-replay traffic policy.
- epdms: single-stage EPDMS with the human-penalty filter, computed by NAVSIM's own compute_final_scores with the
  two-frame extended-comfort term set to NaN (it needs consecutive frames; NAVSIM then drops it from the weights):
  NC*DAC*DDC*TLC*(5 EP + 5 TTC + 2 LK + 2 HC)/14 (assumption A12).
- pdms: NAVSIM v1 PDMS from a second pass without the filter = NC*DAC*(5 TTC + 5 EP + 2 C)/12, with v1 comfort C =
  ego_is_comfortable on that pass's simulated 4 s states.
Output: CSV rows with protocol "navsim" (merged into runs.csv by extract_metrics.py).
"""
import argparse
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

VAL = "/data/nuplan/dataset/nuplan-v1.1/splits/val"
ROOT = os.environ.get("EXP2_ROOT", "/data/exp2")  # where this repository lives on the host
SUB = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
       "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort"]
SHORT = dict(zip(SUB, ["nc", "dac", "ddc", "tlc", "ep", "ttc", "lk", "hc"]))


class ScenarioAdapter:
    """A nuPlan scenario with the two public attributes NAVSIM's metric caching reads."""

    def __init__(self, scenario):
        self._sc = scenario

    def __getattr__(self, name):
        return getattr(self._sc, name)

    @property
    def map_root(self):
        return self._sc._map_root

    @property
    def map_version(self):
        return self._sc._map_version


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_local(poses, ego):
    x0, y0, h0 = ego
    c, s = math.cos(h0), math.sin(h0)
    return [[c * (x - x0) + s * (y - y0), -s * (x - x0) + c * (y - y0), wrap(h - h0)] for x, y, h in poses]


def first_frames(jobs_file):
    """token -> list of (job metadata, first-frame dict) from the open-loop jobs' captured trajectories."""
    out = defaultdict(list)
    for line in open(jobs_file):
        j = json.loads(line)
        if j["protocol"] != "ols":
            continue
        tdir = f"{j['dir']}/traj"
        if not os.path.isdir(tdir):
            continue
        meta = {k: j[k] for k in ("planner", "planner_entry", "family", "seed")}
        for name in os.listdir(tdir):
            with open(f"{tdir}/{name}") as f:
                d = json.load(f)
            out[d["token"]].append((meta, d.get("first")))
    return out


def score_log(log, tokens, frames, maps_root, map_version):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.script.run_pdm_score_one_stage import compute_final_scores
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_comfortable
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    class NuPlanMetricCacheProcessor(MetricCacheProcessor):
        """NAVSIM holds each OpenScene traffic-light frame (0.5 s apart) for the 0.1 s scoring steps; nuPlan databases
        are 0.05 s apart, where that integer ratio is 0. Read the light state at every scoring step instead."""

        def _interpolate_traffic_light_status(self, scenario):
            step, horizon = self._proposal_sampling.interval_length, self._proposal_sampling.time_horizon
            db_step = scenario.database_interval
            return [list(scenario.get_traffic_light_status_at_iteration(iteration=int(round(k * step / db_step))))
                    for k in range(int(round(horizon / step)) + 1)]

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    out_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    processor = NuPlanMetricCacheProcessor(cache_path=None, force_feature_computation=True, proposal_sampling=sampling)
    simulator = PDMSimulator(sampling)
    scorers = {f: PDMScorer(sampling, PDMScorerConfig(human_penalty_filter=f)) for f in (True, False)}
    traffic = LogReplayTrafficAgents(future_trajectory_sampling=sampling)
    from exp2_nuplan.scenarios import builder as sim_builder

    builder = sim_builder(VAL, [f"{VAL}/{log}.db"])  # same scenario mapping as the simulation (same first frame)
    flt = ScenarioFilter(scenario_types=None, scenario_tokens=list(tokens), log_names=[log], map_names=None,
                         num_scenarios_per_type=None, limit_total_scenarios=None, timestamp_threshold_s=None,
                         ego_displacement_minimum_m=None, expand_scenarios=False, remove_invalid_goals=True,
                         shuffle=False)
    rows, seen = [], set()
    for sc in builder.get_scenarios(flt, Sequential()):
        if sc.token in seen:
            continue
        seen.add(sc.token)
        t0 = sc.get_time_point(0).time_us
        try:
            cache = processor.compute_metric_cache(ScenarioAdapter(sc))
        except Exception as e:  # the scenario cannot be scored at all: every planner gets an invalid row
            for meta, _ in frames.get(sc.token, []):
                rows.append({**meta, "scenario": sc.token, "protocol": "navsim", "valid": False,
                             "error": f"metric cache: {e!r}"[:300]})
            continue
        for meta, first in frames.get(sc.token, []):
            row = {**meta, "scenario": sc.token, "protocol": "navsim", "valid": False}
            try:
                if abs(first["t0_us"] - t0) > 1000:  # the captured first frame must be the cache's first frame
                    raise ValueError(f"first-frame time mismatch {(first['t0_us'] - t0) / 1e6:.3f} s")
                poses = first["poses_0p5s"][:8]
                if len(poses) < 8:
                    raise ValueError(f"trajectory shorter than 4 s ({len(poses)} poses)")
                traj = Trajectory(np.array(to_local(poses, first["ego_pose"]), dtype=np.float32), out_sampling)
                df_f, _ = pdm_score(cache, traj, sampling, simulator, scorers[True], traffic)
                df_r, states_r = pdm_score(cache, traj, sampling, simulator, scorers[False], traffic)
                fil, raw = df_f.iloc[0], df_r.iloc[0]
                for k in SUB:
                    row[SHORT[k]] = float(fil[k])
                    row[SHORT[k] + "_raw"] = float(raw[k])
                df = df_f.copy()
                df["two_frame_extended_comfort"] = np.nan  # single frame: NAVSIM drops EC from the weights
                row["epdms"] = float(compute_final_scores(df)["score"].iloc[0])
                t = np.arange(states_r.shape[0], dtype=np.float64) * sampling.interval_length
                row["comfort_v1"] = float(ego_is_comfortable(states_r[None], t).all(axis=-1)[0])
                row["pdms"] = (raw.no_at_fault_collisions * raw.drivable_area_compliance *
                               (5 * raw.time_to_collision_within_bound + 5 * raw.ego_progress +
                                2 * row["comfort_v1"]) / 12)
                row["valid"] = True
            except Exception as e:
                row["error"] = repr(e)[:300]
            rows.append(row)
    for tok in set(tokens) - seen:
        for meta, _ in frames.get(tok, []):
            rows.append({**meta, "scenario": tok, "protocol": "navsim", "valid": False, "error": "scenario not built"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=f"{ROOT}/jobs.jsonl")
    ap.add_argument("--scenarios", default=f"{ROOT}/scenarios/val700.csv")
    ap.add_argument("--out", default=f"{ROOT}/results/navsim_scores.csv")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit-logs", type=int, default=0)
    a = ap.parse_args()
    frames = first_frames(a.jobs)
    sc = pd.read_csv(a.scenarios)
    sc = sc[sc.token.isin(frames)]
    logs = sorted(sc.log_name.unique())[: a.limit_logs or None]
    maps_root, mv = os.environ["NUPLAN_MAPS_ROOT"], os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(score_log, lg, list(sc[sc.log_name == lg].token),
                          {t: frames[t] for t in sc[sc.log_name == lg].token}, maps_root, mv) for lg in logs]
        for k, f in enumerate(as_completed(futs)):
            rows += f.result()
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(logs)} logs scored", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    ok = df[df.valid]
    print(json.dumps({"rows": len(df), "valid": int(df.valid.sum()), "scenarios": int(df.scenario.nunique()),
                      "mean_pdms": float(ok.pdms.mean()) if len(ok) else None,
                      "mean_epdms": float(ok.epdms.mean()) if len(ok) else None}, indent=1))


if __name__ == "__main__":
    main()
