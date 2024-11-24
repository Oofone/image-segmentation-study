from model.segformer.segformer import HFSegformerModule, _SegformerTrainMethod

from torchmetrics import JaccardIndex, Dice, F1Score, Accuracy
from typing import Optional, Union, Dict, List, Any
from pytorch_lightning import LightningModule
from torch import Tensor, nn, argmax, optim

import torch.nn.functional as F


class HFSegFormerPtLightning(LightningModule):
    
    def __init__(self, n_class: int, d_model: int, segformer_train_method: _SegformerTrainMethod,
                 train_batch_size: int, val_batch_size: int, test_batch_size: int, infer_resolution: int = 512,
                 hf_model_checkpoint_path: str = "nvidia/segformer-b0-finetuned-ade-512-512", 
                 loss_fn: Optional[nn.Module] = None, initial_learning_rate: float = 1e-4,
                 weight_decay: float = 1e-2, distributed: bool = False, **kwargs):
        super().__init__()

        self.n_class = n_class
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.infer_resolution = infer_resolution
        self.initial_learning_rate = initial_learning_rate
        self.weight_decay = weight_decay
        self.distributed = distributed
        self.segformer = HFSegformerModule(
            n_class=n_class,
            segformer_train_method=segformer_train_method,
            d_model=d_model,
            hf_model_checkpoint_path=hf_model_checkpoint_path,
            **kwargs)
        self.save_hyperparameters()

        self.loss_fn = loss_fn
        self.experiment_dir = f"segformer-dchannel_{d_model}-n_class_{n_class}-lr_{initial_learning_rate:.3f}-train_strategy_{segformer_train_method}"

        # Validation metrics
        self.accuracy = Accuracy(task='multiclass', num_classes=n_class)
        self.mIoU = JaccardIndex(task='multiclass', num_classes=n_class)
        self.f1 = F1Score(task='multiclass', num_classes=n_class)
        self.dice = Dice(num_classes=n_class)

    def reset_metrics(self):
        self.accuracy.reset()
        self.mIoU.reset()
        self.f1.reset()
        self.dice.reset()

    def train_val_step(self, batch: Dict[str, Union[Tensor, List[str]]]) -> Dict[Any, Any]:
        Y_cap = loss = None
        if self.loss_fn is None:
            Y_cap, loss = self.segformer(batch['X'], batch['Y'].long())
        else:
            Y_cap, _ = self.segformer(batch['X'], None)
            Y_cap = F.interpolate(
                Y_cap, size=batch['Y'].shape[-2:], mode="bilinear", align_corners=False)
            loss = self.loss_fn(Y_cap, batch['Y'].long())
        del batch
        return Y_cap, loss

    def training_step(self, batch: Dict[str, Union[Tensor, List[str]]], batch_idx: Optional[int] = None) -> Tensor:
        _, loss = self.train_val_step(batch=batch)
        del batch
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=self.distributed, batch_size=self.train_batch_size)
        return loss
    
    def validation_step(self, batch: Dict[str, Union[Tensor, List[str]]], batch_idx: Optional[int] = None) -> Tensor:
        Y_cap, val_loss = self.train_val_step(batch=batch)
        Y = batch["Y"]
        del batch
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=self.distributed, batch_size=self.val_batch_size)

        # B x C x H x W
        Y_cap = F.interpolate(
            Y_cap, size=Y.shape[-2:], mode="bilinear", align_corners=False)
        # B x sm(C) x H x W
        Y_cap = F.softmax(Y_cap, dim=1)
        # B x H x W
        Y_cap = argmax(Y_cap, dim=1)

        self.accuracy.update(Y_cap, Y)
        self.mIoU.update(Y_cap, Y)
        self.dice.update(Y_cap, Y)
        self.f1.update(Y_cap, Y)
        return val_loss

    def predict_step(self, batch: Dict[str, Tensor], batch_idx: Optional[int] = None) -> Tensor:
        # Y_cap_logits -> B x C x H x W
        Y_cap, _ = self.segformer(batch["X"], None)
        # Y_cap -> B x C x H x W
        Y_cap = F.interpolate(
            Y_cap, size=(self.infer_resolution, self.infer_resolution),
            mode="bilinear", align_corners=False)
        # Y_cap_probs -> B x sm(C) x H x W
        Y_cap = nn.functional.softmax(Y_cap, dim=1)
        # masks -> B x H x W
        masks = argmax(Y_cap, dim=1)
        return masks

    def on_validation_epoch_end(self) -> None:
        metrics = {
            "f1_score": self.f1.compute(),
            "accuracy": self.accuracy.compute(),
            "mIoU": self.mIoU.compute(),
            "dice": self.dice.compute(),
        }
        self.log_dict(metrics, prog_bar=True, sync_dist=self.distributed)
        self.reset_metrics()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.initial_learning_rate, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-5),
                "monitor": "mIoU"
            }
        }

    def get_experiment_path_suffix(self) -> str:
        return self.experiment_dir
