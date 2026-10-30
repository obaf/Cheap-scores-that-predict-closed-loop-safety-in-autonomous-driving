"""Count the parameters of every checkpoint the planners load (run on the simulation host only: checkpoints are pickles).

  python code/count_params.py   # -> /data/exp2/results/model_params.json

Counts learnable tensors in the checkpoint's state dict; batch-norm statistics and counters are not parameters. For a
checkpoint that stores both raw and EMA weights only one copy is counted.
"""
import json
import os

import torch

CKPTS = {
    "pdm_open": "/data/checkpoints/tuplan_garage/pdm_open_checkpoint.ckpt",
    "pdm_hybrid": "/data/checkpoints/tuplan_garage/pdm_offset_checkpoint.ckpt",
    "gc_pgp": "/data/checkpoints/tuplan_garage/gc_pgp_checkpoint.ckpt",
    "plancnn": "/data/checkpoints/tuplan_garage/plancnn_checkpoint.ckpt",
    "urban_driver": "/data/checkpoints/tuplan_garage/urbandriver_checkpoint.ckpt",
    "plantf": "/data/checkpoints/plantf/planTF.ckpt",
    "pluto": "/data/checkpoints/pluto/pluto_1M_aux_cil.ckpt",
    "diffusion_planner": "/data/checkpoints/diffusion_planner/model.pth",
    "mlp": "/data/exp2/baselines/mlp/seed0/final.ckpt",
    "vector": "/data/exp2/baselines/vector/seed0/final.ckpt",
    "raster": "/data/exp2/baselines/raster/seed0/final.ckpt",
}
NOT_PARAMS = ("running_mean", "running_var", "num_batches_tracked")


def state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "ema_state_dict", "ema"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k], k
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj, "(top level)"
    raise ValueError("no state dict found; keys: " + str(list(obj)[:10]))


def main():
    out = {}
    for name, path in CKPTS.items():
        obj = torch.load(path, map_location="cpu")
        sd, key = state_dict(obj)
        n = sum(v.numel() for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()
                and not k.endswith(NOT_PARAMS))
        top = list(obj) if isinstance(obj, dict) else []
        out[name] = {"params": int(n), "params_m": round(n / 1e6, 2), "state_dict_key": key, "file": os.path.basename(path),
                     "top_level_keys": [str(t) for t in top[:8]]}
        print(name, out[name]["params_m"], "M", key, top[:6])
    os.makedirs("/data/exp2/results", exist_ok=True)
    json.dump(out, open("/data/exp2/results/model_params.json", "w"), indent=1)


if __name__ == "__main__":
    main()
