from utils.image_utils import t2im

from pytorch_lightning import Callback, Trainer, LightningModule
from typing import Any, Dict
from torch import Tensor
from os.path import join as path_join
from os import makedirs

import re


# Class to save predicted images after predict step.
class SaveFeatures(Callback):
    def __init__(self, save_dir: str) -> None:
        super().__init__()
        
        self.save_dir = save_dir
        makedirs(self.save_dir, exist_ok=True)

    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any,
                             batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:

        for name, mask_tensor in zip(batch['names'], outputs):
            save_path = path_join(self.save_dir, name)
            pil_mask = t2im(mask_tensor)
            pil_mask.save(save_path)


class LrLogger(Callback):
    """Log learning rate in each epoch start."""

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for i, optimizer in enumerate(trainer.optimizers):
            for j, params in enumerate(optimizer.param_groups):
                key = f"opt{i}_lr{j}"
                value = params["lr"]
                pl_module.logger.log_metrics({key: value}, step=trainer.global_step)
                pl_module.log(key, value, logger=False, sync_dist=pl_module.distributed)


class EarlyStoppingLR(Callback):
    """Early stop model training when the LR is lower than threshold."""

    def __init__(self, lr_threshold: float, mode="all"):
        self.lr_threshold = lr_threshold

        if mode in ("any", "all"):
            self.mode = mode
        else:
            raise ValueError(f"mode must be one of ('any', 'all')")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._run_early_stop_checking(trainer)

    def _run_early_stop_checking(self, trainer: Trainer) -> None:
        metrics = trainer._logger_connector.callback_metrics
        if len(metrics) == 0:
            return
        all_lr = []
        for key, value in metrics.items():
            if re.match(r"opt\d+_lr\d+", key):
                all_lr.append(value)

        if len(all_lr) == 0:
            return

        if self.mode == "all":
            if all(lr <= self.lr_threshold for lr in all_lr):
                trainer.should_stop = True
                print(f"Early stopping due to LR [{all_lr[-1]}] / Threshold: [{self.lr_threshold}]")
        elif self.mode == "any":
            if any(lr <= self.lr_threshold for lr in all_lr):
                trainer.should_stop = True
                print(f"Early stopping due to LR [{all_lr[-1]}] / Threshold: [{self.lr_threshold}]")
