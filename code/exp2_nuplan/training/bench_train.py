"""Per-sample training cost of an experiment-2 baseline on this CPU (used to size train_baselines.sh).

Measures, on samples of an existing nuPlan feature cache:
  load    : ScenarioDataset.__getitem__ (gzip/pickle load + tensor conversion, plus the experiment's data
            augmentation if any) followed by FeatureCollate, in one process  -> ms per sample per loader worker;
  compute : forward + objective + backward + Adam step of the model on collated batches with T torch threads
            -> ms per sample at T threads.
  python bench_train.py --model raster --cache /data/exp2/train_smoke/baselines/cache/raster --threads 1,2,4
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

NUPLAN_CFG = "/data/code/nuplan-devkit/nuplan/planning/script/config"
EXP2_CFG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
MODEL_CFG = {"mlp": f"{EXP2_CFG}/model/ego_mlp_model.yaml",
             "vector": f"{NUPLAN_CFG}/common/model/simple_vector_model.yaml",
             "raster": f"{NUPLAN_CFG}/common/model/raster_model.yaml"}
AUGMENTATION = {"vector": f"{NUPLAN_CFG}/training/data_augmentation/kinematic_agent_augmentation.yaml"}


def build(model: str, pretrained: bool) -> Any:
    cfg = OmegaConf.create({"model": OmegaConf.load(MODEL_CFG[model])})
    if model == "raster":
        cfg.model.model_name = "resnet18"
        cfg.model.pretrained = pretrained
    OmegaConf.resolve(cfg)
    return instantiate(cfg.model)


def cached_scenarios(cache: str, limit: int) -> List[Any]:
    from nuplan.planning.scenario_builder.cache.cached_scenario import CachedScenario

    out = []
    for log in sorted(os.listdir(cache)):
        p = os.path.join(cache, log)
        if not os.path.isdir(p) or log == "metadata":
            continue
        for stype in sorted(os.listdir(p)):
            for tok in sorted(os.listdir(os.path.join(p, stype))):
                out.append(CachedScenario(log_name=log, token=tok, scenario_type=stype))
    return out[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_CFG))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--samples", type=int, default=448)
    ap.add_argument("--steps", type=int, default=4, help="timed optimizer steps per thread count (after 1 warm-up)")
    ap.add_argument("--no-augmentation", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
    from nuplan.planning.training.modeling.objectives.imitation_objective import ImitationObjective
    from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
    from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor

    torch.manual_seed(0)
    model = build(a.model, pretrained=False)
    pre = FeaturePreprocessor(cache_path=a.cache, force_feature_computation=False,
                              feature_builders=model.get_list_of_required_feature(),
                              target_builders=model.get_list_of_computed_target())
    augmentors = None
    if a.model in AUGMENTATION and not a.no_augmentation:
        augmentors = [instantiate(c) for c in OmegaConf.load(AUGMENTATION[a.model]).values()]
    scen = cached_scenarios(a.cache, a.samples)
    ds = ScenarioDataset(scenarios=scen, feature_preprocessor=pre, augmentors=augmentors)
    collate = FeatureCollate()

    # ---- loading (single process, like one DataLoader worker)
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    batches = []
    for i in range(0, len(ds) - a.batch + 1, a.batch):
        batches.append(collate([ds[j] for j in range(i, i + a.batch)]))
    n_loaded = len(batches) * a.batch
    load_ms = (time.perf_counter() - t0) / n_loaded * 1e3

    # ---- compute
    objective = ImitationObjective(scenario_type_loss_weighting={}, weight=1.0) \
        if "scenario_type_loss_weighting" in ImitationObjective.__init__.__code__.co_varnames \
        else ImitationObjective(weight=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    model.train()
    compute: Dict[str, float] = {}
    for t in [int(x) for x in a.threads.split(",")]:
        torch.set_num_threads(t)
        times = []
        for k in range(a.steps + 1):
            features, targets, scenarios = batches[k % len(batches)]
            s = time.perf_counter()
            opt.zero_grad()
            pred = model(features)
            loss = objective.compute(pred, targets, scenarios)
            loss.backward()
            opt.step()
            if k > 0:
                times.append(time.perf_counter() - s)
        compute[str(t)] = sum(times) / len(times) / a.batch * 1e3
    res = {"model": a.model, "batch": a.batch, "samples_loaded": n_loaded,
           "augmentation": bool(augmentors), "load_ms_per_sample": round(load_ms, 3),
           "compute_ms_per_sample": {k: round(v, 3) for k, v in compute.items()},
           "params": sum(p.numel() for p in model.parameters())}
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
