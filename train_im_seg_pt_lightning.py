from constants import ModelType, LossFunction, _MODEL_TYPE_PARAM, _LOSS_FUNCTION_PARAM, MULTICLASS_MODE, CHECKPOINT_PATH, DEFAULT_LOGS_FILE
from datasets.mapillary_vistas import MapillaryVistasDataModule
from utils.callbacks import EarlyStoppingLR, LrLogger
from utils.log_utils import link_slurm_logs
from model.unet_pt_lightning import UnetPtLightning
from model.loss import FocalLoss

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT
from typing import Literal, Tuple, Optional
from pytorch_lightning import Trainer
from torch.nn import CrossEntropyLoss

import fire
import os


def build_experiment_dir(dataset_prefix: str, model_infix: str, loss_suffix: str):
    return f"{dataset_prefix}--{model_infix}--{loss_suffix}"


def main(
        dataset_root_path: str,
        experiment_root_dir: str = "experiments",
        image_processing_resolution: Tuple[int] = (512, 512),
        dataset_version: str = "v2.0",
        train_batch_size: int = 32,
        val_batch_size: int = 64,
        model_type: Literal[_MODEL_TYPE_PARAM] = "Unet",
        loss_function: Literal[_LOSS_FUNCTION_PARAM] = "CrossEntropyLoss",
        model_depth: int = 4,
        input_channels: int = 3,
        unet_downsampling_factor: int = 2,
        unet_starting_channel: int = 64,
        unet_kernel_size: int = 5,
        unet_padding: int = 2,
        unet_stride: int = 1,
        initial_learning_rate: float = 1e-4,
        gradient_clip_val: float = 1.0,
        gradient_accumulation_steps: int = 1,
        depthwise_pointwise_conv: bool = False,
        precision: _PRECISION_INPUT = "32",
        n_gpu: int = 1,
        log_file: Optional[str] = None) -> None:

    # Validations and assertions
    assert model_type in ModelType._values(), f"model [{model_type}] unsupported; Must be one of {ModelType._values()}"
    assert loss_function in LossFunction._values(), f"loss_function [{loss_function}] unsupported; Must be one of {LossFunction._values()}"

    # Load data module and setup dataset
    dm = MapillaryVistasDataModule(
        root=dataset_root_path,
        version=dataset_version,
        batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        resolution=image_processing_resolution)

    # Get CMAP from data module and identify num classes
    CMAP = dm.get_CMAP()
    num_classes = int(len(CMAP) / 3)

    # Build loss functions
    loss_fn = None
    if loss_function == str(LossFunction.CrossEntropyLoss.value):
        loss_fn = CrossEntropyLoss(ignore_index=-1, reduction="mean")
    elif loss_function == str(LossFunction.FocalLoss.value):
        loss_fn = FocalLoss(mode=MULTICLASS_MODE, ignore_index=-1, reduction="mean")
    else:
        raise NotImplementedError(f"{loss_function} not implemented yet")

    # Build model
    if model_type == str(ModelType.Unet.value):
        model = UnetPtLightning(
            input_channels=input_channels,
            starting_channel=unet_starting_channel,
            depth=model_depth,
            n_class=num_classes,
            downsampling_factor=unet_downsampling_factor,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=val_batch_size,
            kernel_size=unet_kernel_size,
            stride=unet_stride,
            padding=unet_padding,
            loss_fn=loss_fn,
            initial_learning_rate=initial_learning_rate,
            distributed=n_gpu>1,
            depthwise_pointwise=depthwise_pointwise_conv)
    else:
        raise NotImplementedError(f"{model} not implemented yet")

    # Setup experiment path and log linking
    experiment_dir = build_experiment_dir("mv2.0", model.get_experiment_path_suffix(), loss_function)
    experiment_path = os.path.join(experiment_root_dir, experiment_dir)
    os.makedirs(experiment_path, exist_ok=True)
    if log_file is not None:
        link_slurm_logs(log_file, os.path.join(experiment_path, DEFAULT_LOGS_FILE))
    ckpt_path = os.path.join(experiment_path, CHECKPOINT_PATH)

    # Initialize trainer
    monitor = "val_loss"
    trainer = Trainer(log_every_n_steps=50, precision=precision, gradient_clip_val=gradient_clip_val,
                      max_epochs=100, accumulate_grad_batches=gradient_accumulation_steps,
                      callbacks=[
                          ModelCheckpoint(dirpath=ckpt_path, save_last=True, filename=f'seg-{model_type}' + "-{epoch}-{val_loss:.3f}--{mIoU:.3f}", monitor=monitor, mode="min"),
                          EarlyStopping(monitor=monitor, mode='min', verbose=False, patience=7),
                          EarlyStoppingLR(lr_threshold=1e-6),
                          StochasticWeightAveraging(1e-2),
                          LrLogger(),],
                      enable_checkpointing=True,
                      benchmark=True,
                      accelerator="gpu",
                      devices=n_gpu,
                      strategy="auto" if n_gpu < 2 else "ddp",)

    # Start training
    print(f"Model built; Training for [{num_classes}] classes:")
    print(model)
    trainer.fit(model=model, datamodule=dm)
    print("Done")


if __name__ == "__main__":
    fire.Fire(main)
