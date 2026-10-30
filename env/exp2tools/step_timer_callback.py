"""nuPlan simulation callback that records per-step wall-clock times and appends one JSON line per simulation.

Add to any run_simulation.py command with
  +callback.step_timer._target_=exp2tools.step_timer_callback.StepTimerCallback
  +callback.step_timer.output_file=/some/dir/step_timing.jsonl
(requires /data/code/envs/tools on PYTHONPATH; env.sh does that).

Fields written per simulation:
  init_s            planner.initialize() wall time (model loading etc.)
  n_steps           number of simulation steps (planner calls)
  step_mean_s       mean wall time of a full step (observation update + planner + controller + propagate)
  step_median_s, step_p95_s, step_max_s
  planner_mean_s    mean wall time of planner.compute_trajectory() alone
  loop_s            total wall time of the step loop (sum of steps)
"""
import json
import os
import socket
import time
from typing import List, Optional

import numpy as np

from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class StepTimerCallback(AbstractCallback):
    """Collects step and planner wall times for each simulation."""

    def __init__(self, output_file: str, tag: Optional[str] = None):
        self._output_file = output_file
        self._tag = tag
        self._reset()

    def _reset(self) -> None:
        self._t_init: Optional[float] = None
        self._init_s: Optional[float] = None
        self._t_step: Optional[float] = None
        self._t_plan: Optional[float] = None
        self._steps: List[float] = []
        self._plans: List[float] = []

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        self._reset()

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        self._t_init = time.perf_counter()

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        if self._t_init is not None:
            self._init_s = time.perf_counter() - self._t_init

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        self._t_step = time.perf_counter()

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        self._t_plan = time.perf_counter()

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        if self._t_plan is not None:
            self._plans.append(time.perf_counter() - self._t_plan)

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        if self._t_step is not None:
            self._steps.append(time.perf_counter() - self._t_step)

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        steps = np.asarray(self._steps, dtype=float)
        plans = np.asarray(self._plans, dtype=float)
        scenario = setup.scenario
        rec = {
            "tag": self._tag,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "planner": planner.name(),
            "scenario_token": scenario.token,
            "scenario_type": scenario.scenario_type,
            "log_name": scenario.log_name,
            "init_s": self._init_s,
            "n_steps": int(steps.size),
            "loop_s": float(steps.sum()) if steps.size else None,
            "step_mean_s": float(steps.mean()) if steps.size else None,
            "step_median_s": float(np.median(steps)) if steps.size else None,
            "step_p95_s": float(np.percentile(steps, 95)) if steps.size else None,
            "step_max_s": float(steps.max()) if steps.size else None,
            "planner_mean_s": float(plans.mean()) if plans.size else None,
            "planner_median_s": float(np.median(plans)) if plans.size else None,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self._output_file)), exist_ok=True)
        with open(self._output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
