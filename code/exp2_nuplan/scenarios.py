"""Scenario construction that matches nuPlan's simulation exactly.

nuPlan's simulation builds scenarios through the scenario_builder=nuplan config, which applies nuplan_scenario_mapping
(per-type duration and start offset, 10 Hz subsampling). Any scenario built outside Hydra (scenario selection, NAVSIM
scoring) must use the same mapping, or iteration 0 falls at a different time from the simulation's first frame.
"""
import os

import yaml

DEVKIT = os.environ.get("NUPLAN_DEVKIT_ROOT", "/data/code/nuplan-devkit")
MAPPING_YAML = f"{DEVKIT}/nuplan/planning/script/config/common/scenario_builder/scenario_mapping/nuplan_scenario_mapping.yaml"


def simulation_scenario_mapping():
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

    cfg = yaml.safe_load(open(MAPPING_YAML))
    return ScenarioMapping(scenario_map=cfg["scenario_map"], subsample_ratio_override=cfg.get("subsample_ratio_override"))


def builder(data_root, db_files=None, max_workers=1):
    """NuPlanScenarioBuilder configured as nuPlan's simulation configures it (mapping + default vehicle)."""
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder

    return NuPlanScenarioBuilder(data_root=data_root, map_root=os.environ["NUPLAN_MAPS_ROOT"], sensor_root=None,
                                 db_files=db_files, map_version=os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"),
                                 include_cameras=False, max_workers=max_workers, verbose=False,
                                 scenario_mapping=simulation_scenario_mapping())
