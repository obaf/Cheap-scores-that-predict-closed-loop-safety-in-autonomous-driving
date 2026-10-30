"""nuPlan's run_training.py (py_func=cache | train) run under pytorch-lightning 2.x.

Same command line as nuplan-devkit/nuplan/planning/script/run_training.py, e.g.
  PYTHONPATH=/data/exp2/code:$PYTHONPATH python /data/exp2/code/exp2_nuplan/training/run_training.py \
      py_func=train +training=training_raster_model ...
This file only installs the shims of pl2_compat.py and then calls nuPlan's own main() (Hydra config path, config and
all builders are nuPlan's). Caching does not need the shims; they are harmless there.
"""
import logging
import os
import sys

# nuPlan's Hydra entry point resolves its config relative to its own package, so the working directory is irrelevant.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from exp2_nuplan.training import pl2_compat  # noqa: E402

pl2_compat.apply()

# DataLoader workers hand every tensor of a batch to the main process through shared memory. nuPlan's VectorMap
# feature holds about a hundred small tensors per sample, so with PyTorch's default "file_descriptor" strategy a
# batch of 64 needs thousands of open descriptors (the default soft limit is 1024: "Too many open files").
# train_baselines.sh raises the soft limit; EXP2_SHARING_STRATEGY=file_system is the alternative.
if os.environ.get("EXP2_SHARING_STRATEGY"):
    import torch.multiprocessing  # noqa: E402

    torch.multiprocessing.set_sharing_strategy(os.environ["EXP2_SHARING_STRATEGY"])

from nuplan.planning.script import run_training as nuplan_run_training  # noqa: E402

logging.getLogger("numba").setLevel(logging.WARNING)

if __name__ == "__main__":
    nuplan_run_training.main()
