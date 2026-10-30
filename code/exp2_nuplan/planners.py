"""Planner wrappers used in experiment 2.

SeededDiffusionPlanner: Diffusion Planner samples stochastically and nuPlan seeds once per process, so with several
scenarios per process a scenario's result would depend on the scenarios simulated before it. This subclass reseeds
Python, NumPy and torch at the start of every scenario from (seed_base, mission goal, route), which makes every
scenario's result independent of the process it ran in. seed_base 0 is the main run; 1 and 2 are the seed-variance runs.
"""
import random
import zlib

import numpy as np
import torch
from diffusion_planner.planner.planner import DiffusionPlanner


class SeededDiffusionPlanner(DiffusionPlanner):
    def __init__(self, *args, seed_base: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._seed_base = int(seed_base)

    def initialize(self, initialization) -> None:
        g = initialization.mission_goal
        goal = f"{g.x:.2f},{g.y:.2f},{g.heading:.3f}" if g is not None else "none"
        key = goal + "|" + ",".join(initialization.route_roadblock_ids[:8])
        seed = (self._seed_base * 1_000_003 + zlib.crc32(key.encode())) % (2 ** 31 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        super().initialize(initialization)
