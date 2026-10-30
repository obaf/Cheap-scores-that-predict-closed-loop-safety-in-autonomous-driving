"""NuPlanScenarioBuilder whose per-log candidate frames are thinned in the worker processes.

With expand_scenarios=true (nuPlan's training_scenarios filter) every lidar frame of a valid scene is a candidate
scenario: about 5.9 million frames in train_pittsburgh. nuPlan builds a NuPlanScenario object for every candidate and
ships all of them to the main process before limit_total_scenarios samples from them, which costs many GB of memory
and minutes of single-threaded unpickling. This builder keeps, inside each worker and for each log, at most one frame
every `min_gap_s` seconds (the first frame of each gap, any scenario type), so only the thinned pool (about 73 k frames
for min_gap_s = 4 s) is returned. The standard nuPlan filters (limit_total_scenarios etc.) then run unchanged.

The candidate query, the validity rules (scenes with 2 scenes of margin, remove_invalid_goals) and the scenario objects
are nuPlan's own: the parent's _create_scenarios is reused with its per-log function wrapped.
"""
import logging
from functools import partial
from typing import Any, List, Optional, Tuple, Type, Union

import nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder as nuplan_scenario_builder_module
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import (
    GetScenariosFromDbFileParams,
    ScenarioDict,
    get_scenarios_from_db_file,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool

logger = logging.getLogger(__name__)


def thin_scenario_dict(scenario_dict: ScenarioDict, min_gap_s: float) -> ScenarioDict:
    """Keep at most one scenario every min_gap_s seconds (by initial lidar timestamp, across all types)."""
    scenarios = sorted((s for lst in scenario_dict.values() for s in lst), key=lambda s: s._initial_lidar_timestamp)
    gap_us = int(round(min_gap_s * 1e6))
    out: ScenarioDict = {}
    next_us: Optional[int] = None
    for scenario in scenarios:
        t_us = scenario._initial_lidar_timestamp
        if next_us is None or t_us >= next_us:
            out.setdefault(scenario.scenario_type, []).append(scenario)
            next_us = t_us + gap_us
    return out


def get_thinned_scenarios_from_log_file(
    parameters: List[GetScenariosFromDbFileParams], min_gap_s: float
) -> List[ScenarioDict]:
    """Drop-in replacement of nuPlan's get_scenarios_from_log_file that thins each log before returning."""
    output: ScenarioDict = {}
    for parameter in parameters:
        for key, value in thin_scenario_dict(get_scenarios_from_db_file(parameter), min_gap_s).items():
            output.setdefault(key, []).extend(value)
    return [output]


class ThinnedNuPlanScenarioBuilder(NuPlanScenarioBuilder):
    """NuPlanScenarioBuilder with per-log time thinning of the candidate frames (see module docstring)."""

    def __init__(
        self,
        data_root: str,
        map_root: str,
        sensor_root: str,
        db_files: Optional[Union[List[str], str]],
        map_version: str,
        include_cameras: bool = False,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        scenario_mapping: Optional[ScenarioMapping] = None,
        vehicle_parameters: Optional[VehicleParameters] = None,
        min_gap_s: float = 4.0,
    ):
        """See NuPlanScenarioBuilder; min_gap_s: minimum time [s] between two kept frames of a log."""
        super().__init__(data_root, map_root, sensor_root, db_files, map_version, include_cameras, max_workers,
                         verbose, scenario_mapping, vehicle_parameters)
        assert min_gap_s >= 0.0, f"min_gap_s must be >= 0, got {min_gap_s}"
        self._min_gap_s = float(min_gap_s)

    def __reduce__(self) -> Tuple[Type["ThinnedNuPlanScenarioBuilder"], Tuple[Any, ...]]:
        """Pickle with the extra constructor argument."""
        return self.__class__, (
            self._data_root, self._map_root, self._sensor_root, self._db_files, self._map_version,
            self._include_cameras, self._max_workers, self._verbose, self._scenario_mapping,
            self._vehicle_parameters, self._min_gap_s,
        )

    def _create_scenarios(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> ScenarioDict:
        """nuPlan's _create_scenarios with the per-log function replaced by its thinning wrapper."""
        original = nuplan_scenario_builder_module.get_scenarios_from_log_file
        nuplan_scenario_builder_module.get_scenarios_from_log_file = partial(
            get_thinned_scenarios_from_log_file, min_gap_s=self._min_gap_s
        )
        try:
            scenario_dict = super()._create_scenarios(scenario_filter, worker)
        finally:
            nuplan_scenario_builder_module.get_scenarios_from_log_file = original
        total = sum(len(v) for v in scenario_dict.values())
        logger.info("ThinnedNuPlanScenarioBuilder: %d candidate scenarios after %.1f s thinning (%d types, %d dbs)",
                    total, self._min_gap_s, len(scenario_dict), len(self._db_files))
        return scenario_dict
