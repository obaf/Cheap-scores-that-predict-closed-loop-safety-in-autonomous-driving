"""pytorch-lightning 2.x shims for the training framework of nuplan-devkit e924167 (written for PL 1.3.8).

The shared conda env "nuplan" has pytorch-lightning 2.0.1 (required by planTF, pluto and Diffusion-Planner), and the
env must not be downgraded. nuPlan's feature caching (py_func=cache) runs unchanged under PL 2.0.1; nuPlan's training
(py_func=train) fails in three places, which apply() patches at run time (nuplan-devkit's files are not modified):

1. training_builder.build_trainer passes PL-1-only Trainer arguments (gpus, amp_level, terminate_on_nan,
   track_grad_norm, checkpoint_callback, auto_lr_find, auto_scale_batch_size, ...) and pl.plugins.DDPPlugin, both
   removed in PL 2. -> build_trainer_pl2: the Trainer arguments of cfg.lightning.trainer.params that PL 2 still has,
   CPU accelerator, fp32, and the callbacks of cfg.callbacks (nuPlan's default_callbacks use the removed on_epoch_end
   hook, so train_baselines.sh selects callbacks=exp2_baseline_callbacks).
2. DataModule.transfer_batch_to_device(batch, device) lacks the dataloader_idx argument PL 2 passes.
3. LightningModuleWrapper logs without batch_size; PL 2 then guesses the batch size from the first tensor it finds
   (e.g. the number of lanes of the first sample for the vector model), which skews epoch means. The patched _step is
   nuPlan's _step with batch_size=len(scenarios) and on_epoch=True for the training metrics too.

Optional, performance only (EXP2_CHANNELS_LAST=1, set by train_baselines.sh for the raster model): the CNN and its
raster input use PyTorch's channels_last memory layout, about 1.35x faster for resnet18 training on the EPYC CPUs.
The computation is the same (up to float rounding) and checkpoints load into nuPlan's unchanged RasterModel.
"""
import logging
import os
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Arguments of nuPlan's lightning.trainer.params that still exist in pytorch-lightning 2.x.
PL2_TRAINER_KEYS = (
    "max_epochs", "min_epochs", "max_steps", "min_steps", "check_val_every_n_epoch", "val_check_interval",
    "limit_train_batches", "limit_val_batches", "limit_test_batches", "num_sanity_val_steps",
    "accumulate_grad_batches", "gradient_clip_val", "gradient_clip_algorithm", "log_every_n_steps",
    "fast_dev_run", "deterministic", "benchmark", "detect_anomaly",
)


def build_trainer_pl2(cfg: DictConfig) -> pl.Trainer:
    """PL-2 replacement of nuplan.planning.script.builders.training_builder.build_trainer (CPU, single process)."""
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    params = cfg.lightning.trainer.params
    kwargs: Dict[str, Any] = {}
    for key in PL2_TRAINER_KEYS:
        if key in params and params[key] is not None:
            kwargs[key] = params[key]
    if not kwargs.get("gradient_clip_val"):
        kwargs.pop("gradient_clip_val", None)
        kwargs.pop("gradient_clip_algorithm", None)
    dropped = sorted(k for k in params.keys() if k not in PL2_TRAINER_KEYS)
    logger.info("PL2 trainer kwargs %s (ignored PL-1 params: %s)", kwargs, dropped)

    callbacks: List[pl.Callback] = []
    for name, callback_cfg in cfg.callbacks.items():
        callback = instantiate(callback_cfg)
        assert isinstance(callback, pl.Callback), f"callbacks.{name} is not a pytorch_lightning Callback"
        callbacks.append(callback)
    if "data_augmentation_scheduler" in cfg and cfg.data_augmentation_scheduler:
        callbacks.extend(instantiate(s) for s in cfg.data_augmentation_scheduler.values())

    loggers = [
        TensorBoardLogger(save_dir=cfg.output_dir, name="tensorboard", version="", prefix="", log_graph=False),
        CSVLogger(save_dir=cfg.output_dir, name="csv", version=""),
    ]
    return pl.Trainer(accelerator="cpu", devices=1, precision=32, logger=loggers, callbacks=callbacks,
                      enable_checkpointing=False, enable_progress_bar=False, **kwargs)


def _transfer_batch_to_device(self: Any, batch: Tuple[Any, ...], device: torch.device,
                              dataloader_idx: int = 0) -> Tuple[Any, ...]:
    """nuPlan DataModule.transfer_batch_to_device with the PL-2 signature."""
    from nuplan.planning.training.modeling.types import move_features_type_to_device

    return tuple(
        (move_features_type_to_device(batch[0], device), move_features_type_to_device(batch[1], device), batch[2])
    )


def _step(self: Any, batch: Tuple[Any, Any, List[Any]], prefix: str) -> torch.Tensor:
    """nuPlan LightningModuleWrapper._step, logging with an explicit batch size (len(scenarios))."""
    from nuplan.planning.training.modeling.objectives.abstract_objective import aggregate_objectives

    features, targets, scenarios = batch
    predictions = self.forward(features)
    objectives = self._compute_objectives(predictions, targets, scenarios)
    metrics = self._compute_metrics(predictions, targets)
    loss = aggregate_objectives(objectives, agg_mode=self.objective_aggregate_mode)

    log_kwargs = dict(batch_size=len(scenarios), on_step=(prefix == "train"), on_epoch=True)
    self.log(f"loss/{prefix}_loss", loss, **log_kwargs)
    for key, value in objectives.items():
        self.log(f"objectives/{prefix}_{key}", value, **log_kwargs)
    for key, value in metrics.items():
        self.log(f"metrics/{prefix}_{key}", value, **log_kwargs)
    return loss


def _forward(self: Any, features: Dict[str, Any]) -> Dict[str, Any]:
    """nuPlan LightningModuleWrapper.forward, with the optional channels_last layout for raster inputs."""
    if "raster" in features and os.environ.get("EXP2_CHANNELS_LAST", "0") == "1":
        if not getattr(self, "_exp2_channels_last", False):
            self.model.to(memory_format=torch.channels_last)
            self._exp2_channels_last = True
            logger.info("exp2: raster model and inputs use the channels_last memory layout")
        features["raster"].data = features["raster"].data.contiguous(memory_format=torch.channels_last)
    return self.model(features)


_APPLIED = False


def apply() -> None:
    """Install the patches (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return
    import nuplan.planning.script.builders.training_builder as training_builder
    import nuplan.planning.training.experiments.training as training_experiment
    from nuplan.planning.training.data_loader.datamodule import DataModule
    from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper

    training_builder.build_trainer = build_trainer_pl2
    training_experiment.build_trainer = build_trainer_pl2  # imported by name in training.py
    DataModule.transfer_batch_to_device = _transfer_batch_to_device
    LightningModuleWrapper._step = _step
    LightningModuleWrapper.forward = _forward
    _APPLIED = True
    logger.info("exp2 PL2 shims applied (pytorch-lightning %s)", pl.__version__)
