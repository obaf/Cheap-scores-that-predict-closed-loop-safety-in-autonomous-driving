"""EARLY/FINAL checkpoints and a per-run manifest for the experiment-2 baselines (pytorch-lightning 2.x callback).

- early.ckpt: saved right after the optimizer step at which global_step reaches round(early_fraction * total_steps),
  total_steps = trainer.estimated_stepping_batches (max_epochs x batches per epoch);
- final.ckpt: saved at the end of training (last step), not the best-validation model;
- both are Lightning checkpoints of nuPlan's LightningModuleWrapper (weights + hyper-parameters, no optimizer state),
  loadable by nuPlan's MLPlanner (planner.ml_planner.checkpoint_path);
- train_manifest.json: sample counts, epochs, steps, seed, threads, wall times, train/val losses at both checkpoints
  (val loss = mean imitation objective over the whole validation loader, computed the same way for both) and the
  per-epoch history.
Progress lines ("exp2 epoch ...", "exp2 EARLY ...", "exp2 FINAL ...") go to the run's log, e.g.
  grep exp2 /data/exp2/baselines/logs/train_raster_seed0.log
"""
import json
import logging
import os
import socket
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)


class BaselineCheckpointCallback(pl.Callback):
    """Saves <dirpath>/early.ckpt and <dirpath>/final.ckpt and writes <dirpath>/train_manifest.json."""

    def __init__(self, dirpath: str, early_fraction: float = 0.15, run_name: Optional[str] = None,
                 info: Optional[Dict[str, Any]] = None) -> None:
        """
        :param dirpath: output folder of the two checkpoints and the manifest.
        :param early_fraction: fraction of the optimizer steps after which the EARLY checkpoint is saved.
        :param run_name: label stored in the manifest.
        :param info: extra key/values stored in the manifest (resolved Hydra interpolations).
        """
        super().__init__()
        assert 0.0 < early_fraction < 1.0
        self.dirpath = dirpath
        self.early_fraction = early_fraction
        self.run_name = run_name
        self.info = dict(info or {})
        self.total_steps = 0
        self.early_step = 0
        self.early: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        self._recent: Deque[float] = deque(maxlen=100)
        self._t_fit = 0.0
        self._t_epoch = 0.0
        self._eval_seconds = 0.0
        self._samples_seen = 0
        self._epoch_loss_sum = 0.0
        self._epoch_samples = 0
        self._epoch_lr: Optional[float] = None
        self._last_val: Optional[float] = None

    # ------------------------------------------------------------------------------------------------ helpers
    def _save(self, trainer: pl.Trainer, name: str) -> str:
        path = os.path.join(self.dirpath, f"{name}.ckpt")
        tmp = path + ".tmp"
        trainer.save_checkpoint(tmp, weights_only=True)
        os.replace(tmp, path)
        return path

    def _validation_loss(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Optional[float]:
        """Mean aggregated objective over the validation loader (eval mode, no grad)."""
        from nuplan.planning.training.modeling.objectives.abstract_objective import aggregate_objectives

        t0 = time.time()
        was_training = pl_module.training
        pl_module.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for features, targets, scenarios in trainer.datamodule.val_dataloader():
                predictions = pl_module.forward(features)
                objectives = pl_module._compute_objectives(predictions, targets, scenarios)
                loss = aggregate_objectives(objectives, agg_mode=pl_module.objective_aggregate_mode)
                total += float(loss) * len(scenarios)
                count += len(scenarios)
        if was_training:
            pl_module.train()
        self._eval_seconds += time.time() - t0
        return total / count if count else None

    def _train_seconds(self) -> float:
        return time.time() - self._t_fit - self._eval_seconds

    # ------------------------------------------------------------------------------------------------ hooks
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        os.makedirs(self.dirpath, exist_ok=True)
        self.total_steps = int(trainer.estimated_stepping_batches)
        self.early_step = max(1, int(round(self.early_fraction * self.total_steps)))
        self._t_fit = time.time()
        logger.info("exp2 start: %d optimizer steps (%d epochs x %d batches), EARLY at step %d, torch threads %d",
                    self.total_steps, trainer.max_epochs or 0, int(trainer.num_training_batches), self.early_step,
                    torch.get_num_threads())

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._t_epoch = time.time()
        self._epoch_loss_sum, self._epoch_samples = 0.0, 0
        self._epoch_lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else None

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any,
                           batch_idx: int) -> None:
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        value = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        n = len(batch[2])
        self._samples_seen += n
        self._epoch_loss_sum += value * n
        self._epoch_samples += n
        self._recent.append(value)
        if self.early is None and trainer.global_step >= self.early_step:
            path = self._save(trainer, "early")
            self.early = {
                "checkpoint": path, "global_step": int(trainer.global_step), "epoch": int(trainer.current_epoch),
                "epochs_completed": round(trainer.global_step / max(1, self.total_steps) *
                                          float(trainer.max_epochs or 0), 3),
                "samples_seen": self._samples_seen,
                "train_loss_last100": sum(self._recent) / len(self._recent),
                "train_seconds": round(self._train_seconds(), 1),
            }
            self.early["val_loss"] = self._validation_loss(trainer, pl_module)
            logger.info("exp2 EARLY saved at step %d (%.0f s): train loss (last 100) %.3f, val loss %s",
                        trainer.global_step, self.early["train_seconds"], self.early["train_loss_last100"],
                        self.early["val_loss"])

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.sanity_checking:
            value = trainer.callback_metrics.get("loss/val_loss")
            self._last_val = float(value) if value is not None else None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # lr = learning rate used in this epoch (read at epoch start: PL 2 steps epoch schedulers before this hook)
        self.history.append({
            "epoch": int(trainer.current_epoch), "global_step": int(trainer.global_step), "lr": self._epoch_lr,
            "train_loss": self._epoch_loss_sum / self._epoch_samples if self._epoch_samples else None,
            "val_loss": self._last_val, "epoch_seconds": round(time.time() - self._t_epoch, 1),
        })
        h = self.history[-1]
        logger.info("exp2 epoch %d/%d done in %.0f s (lr %s): train loss %s, val loss %s, %d samples so far",
                    h["epoch"] + 1, trainer.max_epochs or 0, h["epoch_seconds"], h["lr"], h["train_loss"],
                    h["val_loss"], self._samples_seen)
        self._last_val = None

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        train_seconds = self._train_seconds()
        path = self._save(trainer, "final")
        dm = trainer.datamodule
        n_train = len(dm._train_set) if getattr(dm, "_train_set", None) is not None else None
        n_val = len(dm._val_set) if getattr(dm, "_val_set", None) is not None else None
        final = {
            "checkpoint": path, "global_step": int(trainer.global_step),
            "epochs_completed": int(trainer.current_epoch), "samples_seen": self._samples_seen,
            "train_loss_last_epoch": self.history[-1]["train_loss"] if self.history else None,
            "train_loss_last100": sum(self._recent) / len(self._recent) if self._recent else None,
            "train_seconds": round(train_seconds, 1),
            "val_loss": self._validation_loss(trainer, pl_module),
        }
        logger.info("exp2 FINAL saved at step %d (%.0f s training): train loss (last epoch) %s, val loss %s",
                    trainer.global_step, train_seconds, final["train_loss_last_epoch"], final["val_loss"])
        manifest = {
            "run_name": self.run_name, **self.info,
            "train_samples": n_train, "val_samples": n_val,
            "batch_size": trainer.train_dataloader.batch_size if trainer.train_dataloader is not None else None,
            "batches_per_epoch": int(trainer.num_training_batches),
            "max_epochs": trainer.max_epochs, "total_steps": self.total_steps,
            "early_fraction": self.early_fraction, "early_step": self.early_step,
            "early": self.early, "final": final, "history": self.history,
            "train_wall_seconds": round(train_seconds, 1), "eval_wall_seconds": round(self._eval_seconds, 1),
            "train_samples_per_second": round(self._samples_seen / train_seconds, 2) if train_seconds > 0 else None,
            "torch_threads": torch.get_num_threads(), "host": socket.gethostname(), "nproc": os.cpu_count(),
            "torch": torch.__version__, "pytorch_lightning": pl.__version__,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = os.path.join(self.dirpath, "train_manifest.json.tmp")
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=1)
        os.replace(tmp, os.path.join(self.dirpath, "train_manifest.json"))
