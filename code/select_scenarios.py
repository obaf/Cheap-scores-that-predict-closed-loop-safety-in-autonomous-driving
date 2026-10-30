"""Build the scenario lists of experiment 2 (runs on the host in the `nuplan` conda env).

1. Val14: tuplan_garage's val14_split filter (14 types, 1,118 tokens, logs from nuPlan's val split list,
   num_scenarios_per_type=100, timestamp_threshold_s=15, remove_invalid_goals=True) resolved against the val databases.
2. val700: 50 scenarios per type drawn from Val14 with a fixed seed (all of a type if it has fewer than 50).
3. seed140: the first 10 of each type in val700's random order (the seed-variance subset).
Writes <out>/val14_all.csv, val700.csv, seed140.csv and selection.json (counts, seed, source commits).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

from exp2_nuplan.scenarios import builder as sim_builder

SEED = 20261028
PER_TYPE, SEED_PER_TYPE = 50, 10
VAL14 = "/data/code/tuplan_garage/tuplan_garage/planning/script/config/common/scenario_filter/val14_split.yaml"
SPLITTER = "/data/code/nuplan-devkit/nuplan/planning/script/config/common/splitter/nuplan.yaml"


def val14_filter():
    cfg = yaml.safe_load(open(VAL14))
    val_logs = yaml.safe_load(open(SPLITTER))["log_splits"]["val"]
    fields = {k: v for k, v in cfg.items() if not k.startswith("_")}
    fields["log_names"] = val_logs
    return ScenarioFilter(**fields), len(cfg["scenario_tokens"])


def to_frame(scenarios, source):
    rows = [{"token": s.token, "scenario_type": s.scenario_type, "log_name": s.log_name,
             "map_name": s.map_api.map_name, "t0_us": s.get_time_point(0).time_us, "source": source}
            for s in scenarios]
    return pd.DataFrame(rows).sort_values(["scenario_type", "token"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data/nuplan/dataset/nuplan-v1.1/splits/val")
    ap.add_argument("--out", default="/data/exp2/scenarios")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    flt, n_tokens = val14_filter()
    builder = sim_builder(a.data_root, None, a.workers)  # same scenario mapping as the simulation
    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=a.workers)
    scenarios = builder.get_scenarios(flt, worker)
    allv = to_frame(scenarios, "val14")
    allv.to_csv(f"{a.out}/val14_all.csv", index=False)
    # Two Val14 types hold fewer than 50 scenarios; top them up from the same val logs with the same filter settings
    # (15 s spacing, valid goals), excluding Val14 tokens. Each row keeps its source so results can be re-checked on
    # the Val14 rows alone.
    short = [t for t, n in allv.scenario_type.value_counts().items() if n < PER_TYPE]
    extra = pd.DataFrame(columns=allv.columns)
    if short:
        top = ScenarioFilter(**{**flt.__dict__, "scenario_types": short, "scenario_tokens": None,
                                "num_scenarios_per_type": None})
        extra = to_frame(builder.get_scenarios(top, worker), "val_topup")
        extra = extra[~extra.token.isin(allv.token)].drop_duplicates("token")
        extra.to_csv(f"{a.out}/topup_candidates.csv", index=False)
    rng = np.random.default_rng(SEED)
    picks, seeds = [], []
    for t in sorted(allv.scenario_type.unique()):
        g = allv[allv.scenario_type == t]
        g = g.iloc[rng.permutation(len(g))]
        if len(g) < PER_TYPE:
            e = extra[extra.scenario_type == t]
            g = pd.concat([g, e.iloc[rng.permutation(len(e))[: PER_TYPE - len(g)]]])
        chosen = g.iloc[:PER_TYPE]
        picks.append(chosen)
        seeds.append(chosen.iloc[:SEED_PER_TYPE])
    val700 = pd.concat(picks).reset_index(drop=True)
    seed140 = pd.concat(seeds).reset_index(drop=True)
    val700.to_csv(f"{a.out}/val700.csv", index=False)
    seed140.to_csv(f"{a.out}/seed140.csv", index=False)
    info = {"seed": SEED, "val14_tokens_in_yaml": n_tokens, "val14_resolved": len(allv),
            "per_type_available": allv.scenario_type.value_counts().sort_index().to_dict(),
            "topped_up_types": short, "topup_candidates": int(len(extra)),
            "val700": len(val700), "val700_by_source": val700.source.value_counts().to_dict(),
            "seed140": len(seed140), "n_logs_val700": int(val700.log_name.nunique()),
            "duplicate_tokens_val700": int(val700.token.duplicated().sum())}
    json.dump(info, open(f"{a.out}/selection.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
