"""Simulation callback that captures each planner's open-loop output in nuPlan's open_loop_boxes runs.

Per scenario it writes <output_dir>/<token>.json with
- steps: every `every_n_steps` iterations (default 5 = 0.5 s), the planned trajectory compared with the logged expert
  future (L2 at each 0.5 s sample, ADE/FDE over 3 s and 8 s) and checked for overlap with the LOGGED agents' boxes at
  the same future times (collision with logged agents, the open-loop collision measure used on nuScenes/NAVSIM);
- first_frame: the trajectory planned at the first frame (0.1 s samples to 4 s and 0.5 s samples to 8 s, global frame),
  plus the initial ego state, used later for NAVSIM PDMS/EPDMS scoring.
The expert future and the logged agents are queried once per scenario (to 8 s past the scenario end) to keep the
database load small; with every_n_steps=5 the step times fall on the same 0.5 s grid.
"""
import json
import math
import os
from typing import Dict, List

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback

AGENT_TYPES = {TrackedObjectType.VEHICLE: "vehicle", TrackedObjectType.PEDESTRIAN: "pedestrian",
               TrackedObjectType.BICYCLE: "bicycle"}
HORIZON_S, DT_S = 8.0, 0.5


def _pose(state):
    p = state.rear_axle
    return [round(p.x, 3), round(p.y, 3), round(p.heading, 4)]


class TrajectoryCaptureCallback(AbstractCallback):
    def __init__(self, output_dir: str, every_n_steps: int = 5):
        self._out = output_dir
        self._every = every_n_steps
        self._s: Dict[str, dict] = {}

    # -------------------------------------------------------------------------------------------------- helpers
    def _prepare(self, setup) -> dict:
        sc = setup.scenario
        n_it = sc.get_number_of_iterations()
        span = (n_it - 1) * sc.database_interval + HORIZON_S
        n = int(round(span / DT_S))
        expert = list(sc.get_ego_future_trajectory(0, span, n))
        tracks = list(sc.get_future_tracked_objects(0, span, n))
        times = [e.time_point.time_us for e in expert]
        boxes = []
        for det in tracks[: len(times)]:
            boxes.append([(AGENT_TYPES.get(o.tracked_object_type, "static"), o.box.geometry)
                          for o in det.tracked_objects.tracked_objects])
        return {"expert": expert[: len(boxes)], "times": times[: len(boxes)], "boxes": boxes, "steps": [],
                "first": None, "token": sc.token, "log": sc.log_name, "type": sc.scenario_type}

    @staticmethod
    def _nearest(times: List[int], t: int) -> int:
        lo, hi = 0, len(times) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if times[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0 and abs(times[lo - 1] - t) < abs(times[lo] - t):
            lo -= 1
        return lo

    # -------------------------------------------------------------------------------------------------- callbacks
    def on_initialization_start(self, setup, planner) -> None:
        pass

    def on_initialization_end(self, setup, planner) -> None:
        pass

    def on_step_start(self, setup, planner) -> None:
        pass

    def on_step_end(self, setup, planner, sample) -> None:
        pass

    def on_planner_start(self, setup, planner) -> None:
        pass

    def on_simulation_start(self, setup) -> None:
        self._s[setup.scenario.token] = self._prepare(setup)

    def on_planner_end(self, setup, planner, trajectory) -> None:
        st = self._s.get(setup.scenario.token)
        if st is None or trajectory is None:
            return
        it = setup.time_controller.get_iteration()
        idx, t0 = it.index, it.time_point.time_us
        start, end = trajectory.start_time.time_us, trajectory.end_time.time_us
        if idx == 0:
            fine = [_pose(trajectory.get_state_at_time(TimePoint(t0 + k * 100_000)))
                    for k in range(41) if t0 + k * 100_000 <= end]
            coarse = [_pose(trajectory.get_state_at_time(TimePoint(t0 + k * 500_000)))
                      for k in range(1, 17) if t0 + k * 500_000 <= end]
            ego = setup.ego_controller.get_state()
            dyn = ego.dynamic_car_state
            st["first"] = {"t0_us": t0, "poses_0p1s": fine, "poses_0p5s": coarse, "ego_pose": _pose(ego),
                           "ego_velocity": [dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y],
                           "ego_acceleration": [dyn.rear_axle_acceleration_2d.x, dyn.rear_axle_acceleration_2d.y],
                           "trajectory_end_s": (end - t0) / 1e6}
        if idx % self._every:
            return
        rec = {"it": idx, "l2": [], "col": None, "col_static": None}
        k0 = self._nearest(st["times"], t0)
        for k in range(1, int(HORIZON_S / DT_S) + 1):
            j = k0 + k
            if j >= len(st["times"]):
                break
            t = st["times"][j]
            if t < start or t > end:
                break
            plan = trajectory.get_state_at_time(TimePoint(t))
            ex = st["expert"][j].rear_axle
            rec["l2"].append(round(math.hypot(plan.rear_axle.x - ex.x, plan.rear_axle.y - ex.y), 3))
            ego_poly = plan.car_footprint.oriented_box.geometry
            for kind, poly in st["boxes"][j]:
                if ego_poly.intersects(poly):
                    if kind == "static":
                        rec["col_static"] = rec["col_static"] or round(k * DT_S, 1)
                    elif rec["col"] is None:
                        rec["col"] = [round(k * DT_S, 1), kind]
        st["steps"].append(rec)

    def on_simulation_end(self, setup, planner, history) -> None:
        st = self._s.pop(setup.scenario.token, None)
        if st is None:
            return
        os.makedirs(self._out, exist_ok=True)
        out = {k: st[k] for k in ("token", "log", "type", "steps", "first")}
        out["planner"] = planner.name()
        with open(os.path.join(self._out, f"{st['token']}.json"), "w") as f:
            json.dump(out, f, separators=(",", ":"))
