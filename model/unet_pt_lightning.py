from model.conv.conv_blocks import ConvDownsamplingBlock
from model.unet.encoder import UnetEncoder
from model.unet.decoder import UnetDecoder

from torchmetrics import JaccardIndex, Dice, F1Score, Accuracy
from typing import Union, Tuple, Dict, List, Any, Optional
from pytorch_lightning import LightningModule
from torch import nn, Tensor, optim, argmax

import torch.nn.functional as F

# Unet model for image segmentation 
class UnetPtLightning(LightningModule):

    def __init__(self, input_channels: int, starting_channel: int, depth: int, n_class: int, downsampling_factor: int,
                 train_batch_size: int, val_batch_size: int, test_batch_size: int, kernel_size: int, stride: int,
                 padding: Union[int, Tuple[input]], loss_fn: nn.Module, distributed: bool = False, infer_resolution: int = 512,
                 initial_learning_rate: float = 1e-4, weight_decay: float = 1e-2, **kwargs):
        super().__init__()

        # Attrs
        channels = [starting_channel * (downsampling_factor ** i) for i in range(depth)]
        self.initial_learning_rate = initial_learning_rate
        self.weight_decay = weight_decay
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.infer_resolution = infer_resolution
        self.save_hyperparameters(ignore=['loss_fn'])

        self.experiment_dir = f"unet-schannel_{starting_channel}-depth_{depth}-kernel_{kernel_size}-lr_{initial_learning_rate:.3f}-dwseperable_{kwargs.get('depthwise_pointwise', False)}"

        # nn.Module parts
        self.distributed = distributed
        self.encoder = UnetEncoder(
            channels_list=[input_channels] + channels,
            downsampling_factor=downsampling_factor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs)
        bottleneck_c_in = channels[-1]
        bottleneck_c_out = bottleneck_c_in * downsampling_factor
        self.bottleneck = ConvDownsamplingBlock(
            c_in=bottleneck_c_in,
            c_out=bottleneck_c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_fn=None,
            downsampling_factor=0,
            **kwargs)
        self.decoder = UnetDecoder(
            channels_list=[starting_channel * (downsampling_factor ** depth)] + list(reversed(channels)),
            upsampling_factor=downsampling_factor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs)
        self.final_conv = nn.Conv2d(
            in_channels=starting_channel,
            out_channels=n_class,
            kernel_size=1)
        self.loss_fn = loss_fn
        
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

    def forward(self, X: Tensor) -> Tensor:
        # X - B x C x H x W
        X, skips = self.encoder(X)
        X, _ = self.bottleneck(X)
        X = self.decoder(X, skips)
        return self.final_conv(X)

    def train_val_step(self, batch: Dict[str, Union[Tensor, List[str]]]) -> Dict[Any, Any]:
        Y_cap = self.forward(batch['X'])
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
        # X -> B x C x H x W, 
        # Y_cap_logits -> B x C x H x W
        Y_cap_logits = self.forward(batch["X"])

        # Y_cap_probs -> B x sm(C) x H x W
        Y_cap_probs = nn.functional.softmax(Y_cap_logits, dim=1)

        # masks -> B x H x W
        masks = argmax(Y_cap_probs, dim=1)
        return masks

    def on_validation_epoch_end(self) -> None:
        metrics = {
            "f1_score": self.f1.compute(),
            "accuracy": self.accuracy.compute(),
            "mIoU": self.mIoU.compute(),
            "dice": self.dice.compute()
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

    def get_experiment_path_suffix(self):
        return self.experiment_dir
