from constants import ModelType, LossFunction, _MODEL_TYPE_PARAM, _LOSS_FUNCTION_PARAM, MULTICLASS_MODE, CHECKPOINT_PATH, DEFAULT_LOGS_FILE
from datasets.mapillary_vistas import MapillaryVistasDataModule
from utils.callbacks import EarlyStoppingLR, LrLogger
from utils.log_utils import link_slurm_logs
from model.unet_pt_lightning import UnetPtLightning
from model.hf_segformer_pt_lightning import HFSegFormerPtLightning, _SegformerTrainMethod
from model.loss import FocalLoss

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from typing import Literal, Tuple, Optional, List
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
        train_batch_size: int = 8,
        val_batch_size: int = 64,
        model_type: Literal[_MODEL_TYPE_PARAM] = "Unet",
        loss_function: Literal[_LOSS_FUNCTION_PARAM] = "CrossEntropyLoss",
        model_depth: int = 4,
        input_channels: int = 3,
        segformer_train_method: _SegformerTrainMethod = "full_train",
        segformer_d_model: int = 768,
        segformer_hf_model_checkpoint_path: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        full_lora_target_modules: List[str] = ["query", "key", "value"],
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
        n_cpu: int = 4,
        n_gpu: int = 1,
        max_epochs: int = 50,
        log_file: Optional[str] = None,
        hparam_tune: bool = False) -> None:

    # Validations and assertions
    assert model_type in ModelType._values(), f"model [{model_type}] unsupported; Must be one of {ModelType._values()}"
    assert loss_function in LossFunction._values(), f"loss_function [{loss_function}] unsupported; Must be one of {LossFunction._values()}"

    # Load data module and setup dataset
    dm = MapillaryVistasDataModule(
        root=dataset_root_path,
        version=dataset_version,
        batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        resolution=image_processing_resolution,
        num_workers=n_cpu)

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
    elif model_type == str(ModelType.SegFormerHF.value):
        model = HFSegFormerPtLightning(
            n_class=num_classes,
            d_model=segformer_d_model,
            segformer_train_method=segformer_train_method,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=val_batch_size,
            hf_model_checkpoint_path=segformer_hf_model_checkpoint_path,
            loss_fn=loss_fn,
            initial_learning_rate=initial_learning_rate,
            distributed=n_gpu>1,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            full_lora_target_modules=full_lora_target_modules)
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
    monitor = "mIoU"
    callbacks=[
        ModelCheckpoint(dirpath=ckpt_path, save_last=True, filename=f'seg-{model_type}' + "-{epoch}-{val_loss:.3f}--{mIoU:.3f}", monitor=monitor, mode="max"),
        EarlyStopping(monitor=monitor, mode='max', verbose=False, patience=7),
        EarlyStoppingLR(lr_threshold=1e-6),
        StochasticWeightAveraging(1e-2),
        LrLogger()]
    if hparam_tune:
        callbacks.append(TuneReportCheckpointCallback(
            metrics=["mIoU", "val_loss"],
            on="validation_end"
        ))
    trainer = Trainer(log_every_n_steps=50, precision=precision, gradient_clip_val=gradient_clip_val,
                      max_epochs=max_epochs, accumulate_grad_batches=gradient_accumulation_steps,
                      callbacks=callbacks,
                      enable_checkpointing=True,
                      benchmark=True,
                      accelerator="auto", #"gpu",
                      devices=n_gpu,
                      num_nodes=1,
                      strategy="auto",)#"auto" if n_gpu < 2 else "ddp",)

    # Start training
    print(f"Model built; Training for [{num_classes}] classes with [{n_gpu} GPUs]")
    print(model)
    trainer.fit(model=model, datamodule=dm)
    print("Done")


if __name__ == "__main__":
    fire.Fire(main)
