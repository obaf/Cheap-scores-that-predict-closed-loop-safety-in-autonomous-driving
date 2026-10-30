"""infer_baseline.py - plan with one of the released small-baseline models on nuPlan scenarios (no simulation).

The planner is configured exactly as in the study (its overrides in code/planners_baselines.yaml; the checkpoint comes
from models/ unless this machine has the training output under /data/exp2/baselines), loaded by nuPlan's own planner
builder, given the scenario's first 2 s of history and asked for one plan: 16 poses over 8 s. The plan is printed with
its distance to the trajectory the human driver actually drove, and can be saved as JSON.

Run on a machine set up as in "Steps to reproduce" (nuplan environment, nuPlan maps and validation logs), from the
repository root:
    source env/env.sh nuplan && export PYTHONPATH=$(pwd)/code:$PYTHONPATH
    python tools/infer_baseline.py --planner raster_final
    python tools/infer_baseline.py --planner mlp_final --n 5 --out plans.json
    python tools/infer_baseline.py --planner vector_final --tokens 63ef5aab300c5f09,470cb336d24754b4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("EXP2_ROOT", REPO)  # runner.ROOT: this repository (models/, scenarios/)
sys.path.insert(0, os.path.join(REPO, "code"))

import nuplan.planning.script as nuplan_script  # noqa: E402
import pandas as pd  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from nuplan.planning.script.builders.planner_builder import build_planners  # noqa: E402
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder  # noqa: E402
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter  # noqa: E402
from nuplan.planning.script.builders.worker_pool_builder import build_worker  # noqa: E402
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer  # noqa: E402
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks  # noqa: E402
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput  # noqa: E402
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration  # noqa: E402

import runner  # noqa: E402

HORIZON_S, POSES = 8.0, 16  # the three models predict 16 poses at 0.5 s


def compose_config(entry, sel, checkpoint=""):
    """nuPlan's simulation configuration with the planner overrides the study used (as runner.build_command)."""
    ov = [runner.resolve_checkpoint(o) for o in entry["overrides"]]
    if checkpoint:
        ov = [o.split("=", 1)[0] + "=" + os.path.abspath(checkpoint) if o.split("=", 1)[0].endswith(".checkpoint_path")
              else o for o in ov]
    sp = runner.NUPLAN_SEARCHPATH + [x for x in entry.get("searchpath", []) if x not in runner.NUPLAN_SEARCHPATH]
    db_files = [f"{runner.SPLITS}/val/{lg}.db" for lg in sorted(sel.log_name.unique())]
    overrides = [runner.PROTOCOL["clsnr"], "scenario_builder=nuplan", "scenario_filter=all_scenarios",
                 f"scenario_builder.db_files={runner.q(db_files)}",
                 f"scenario_filter.scenario_types={runner.q(sorted(sel.scenario_type.unique()))}",
                 f"scenario_filter.scenario_tokens={runner.q(sel.token)}", "scenario_filter.remove_invalid_goals=true",
                 "worker=sequential", "hydra.searchpath=[" + ",".join(sp) + "]"] + ov
    config_dir = os.path.join(os.path.dirname(nuplan_script.__file__), "config", "simulation")
    with initialize_config_dir(config_dir=config_dir):
        return compose(config_name="default_simulation", overrides=overrides)


def plan_once(planner, scenario, buffer_s):
    """One planning step at the scenario's start, fed as nuPlan's simulator feeds it (2 s of logged history)."""
    planner.initialize(PlannerInitialization(route_roadblock_ids=scenario.get_route_roadblock_ids(),
                                             mission_goal=scenario.get_mission_goal(), map_api=scenario.map_api))
    size = int(buffer_s / scenario.database_interval) + 1
    history = SimulationHistoryBuffer.initialize_from_scenario(size, scenario, DetectionsTracks)
    step = PlannerInput(iteration=SimulationIteration(time_point=scenario.start_time, index=0), history=history,
                        traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0)))
    t0 = time.perf_counter()
    trajectory = planner.compute_trajectory(step)
    return trajectory, time.perf_counter() - t0


def pose(state, start_s):
    return [round(state.time_point.time_s - start_s, 2), round(state.rear_axle.x, 3), round(state.rear_axle.y, 3),
            round(state.rear_axle.heading, 4)]


def main():
    ap = argparse.ArgumentParser(description="Plan with a released small-baseline model on nuPlan scenarios.")
    ap.add_argument("--planner", default="raster_final", help="a key of code/planners_baselines.yaml, e.g. mlp_final, "
                    "vector_early, raster_final_s2")
    ap.add_argument("--tokens", default="", help="comma-separated scenario tokens from scenarios/val700.csv")
    ap.add_argument("--n", type=int, default=1, help="without --tokens: the first n study scenarios")
    ap.add_argument("--checkpoint", default="", help="a .ckpt to use instead of the registry's")
    ap.add_argument("--out", default="", help="write the plans and the human trajectories as JSON")
    a = ap.parse_args()

    reg = runner.load_registry(os.path.join(REPO, "code", "planners_baselines.yaml"))
    if a.planner not in reg["planners"]:
        raise SystemExit(f"unknown planner {a.planner}; choose from {', '.join(reg['planners'])}")
    entry = reg["planners"][a.planner]
    scen = pd.read_csv(os.path.join(REPO, "scenarios", "val700.csv"))
    sel = scen[scen.token.isin(a.tokens.split(","))] if a.tokens else scen.head(a.n)
    if sel.empty:
        raise SystemExit("none of these tokens is in scenarios/val700.csv")
    cfg = compose_config(entry, sel, a.checkpoint)
    ckpt = cfg.planner.ml_planner.checkpoint_path
    scenarios = build_scenario_builder(cfg).get_scenarios(build_scenario_filter(cfg.scenario_filter), build_worker(cfg))
    by_token = {s.token: s for s in scenarios}
    planner = build_planners(cfg.planner, scenarios[0] if scenarios else None)[0]  # loads the checkpoint
    print(f"{a.planner}: {os.path.relpath(ckpt, REPO) if ckpt.startswith(REPO) else ckpt}")

    out = []
    for row in sel.itertuples():
        sc = by_token.get(row.token)
        if sc is None:
            print(f"{row.token}: not built (check that {row.log_name}.db is under {runner.SPLITS}/val)")
            continue
        trajectory, secs = plan_once(planner, sc, cfg.simulation_history_buffer_duration)
        start = trajectory.start_time.time_s  # the ego state the plan starts from (stamped just before the scenario start)
        plan = [pose(s, start) for s in trajectory.get_sampled_trajectory()]
        human, dist = [], []
        for s in sc.get_ego_future_trajectory(iteration=0, time_horizon=HORIZON_S, num_samples=POSES):
            if s.time_point.time_us > trajectory.end_time.time_us:
                break
            p = trajectory.get_state_at_time(s.time_point)
            human.append(pose(s, start))
            dist.append(math.hypot(p.rear_axle.x - s.rear_axle.x, p.rear_axle.y - s.rear_axle.y))
        print(f"{row.token} ({row.scenario_type}, {row.map_name}): planned in {secs:.2f} s; distance to the human "
              f"trajectory {sum(dist) / len(dist):.2f} m on average, {dist[-1]:.2f} m at {human[-1][0]:.1f} s")
        print("    t [s]       x [m]       y [m]   heading [rad]")
        for t, x, y, h in plan[::2]:
            print(f"    {t:5.1f}  {x:10.2f}  {y:10.2f}  {h:8.3f}")
        out.append({"planner": a.planner, "checkpoint": ckpt, "token": row.token, "scenario_type": row.scenario_type,
                    "log_name": row.log_name, "seconds": round(secs, 3), "plan_t_x_y_heading": plan,
                    "human_t_x_y_heading": human, "mean_distance_m": round(sum(dist) / len(dist), 3)})
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
