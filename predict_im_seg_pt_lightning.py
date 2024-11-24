from constants import ModelType, LossFunction, _MODEL_TYPE_PARAM, _LOSS_FUNCTION_PARAM, MULTICLASS_MODE, INFERENCE_LOGS_FILE, PREDICTIONS_DIR
from datasets.mapillary_vistas import MapillaryVistasDataModule
from model.hf_segformer_pt_lightning import HFSegFormerPtLightning
from model.unet_pt_lightning import UnetPtLightning
from model.loss import FocalLoss
from utils.callbacks import SavePredictedMasks
from utils.log_utils import link_slurm_logs

from typing import Literal, Tuple, Optional
from pytorch_lightning import Trainer
from torch.nn import CrossEntropyLoss

import fire
import os


def build_experiment_dir(dataset_prefix: str, model_infix: str, loss_suffix: str):
    return f"{dataset_prefix}--{model_infix}--{loss_suffix}"


def main(
        dataset_root_path: str,
        checkpoint_path: str,
        experiment_path: str = "experiments/example",
        inference_resolution: int = 512,
        dataset_version: str = "v2.0",
        test_batch_size: int = 8,
        model_type: Literal[_MODEL_TYPE_PARAM] = "Unet",
        # val_loss_function: Optional[_LOSS_FUNCTION_PARAM] = None,
        n_cpu: int = 4,
        n_gpu: int = 1,
        log_file: Optional[str] = None) -> None:

    # Validations and assertions
    assert model_type in ModelType._values(), f"model [{model_type}] unsupported; Must be one of {ModelType._values()}"

    # Load data module and setup dataset
    dm = MapillaryVistasDataModule(
        root=dataset_root_path,
        version=dataset_version,
        val_batch_size=test_batch_size,
        resolution=(inference_resolution, inference_resolution),
        num_workers=n_cpu)
    dm.setup()

    # Get CMAP from data module and identify num classes
    CMAP = dm.get_CMAP()
    num_classes = int(len(CMAP) / 3)

    # Build loss functions
    # loss_fn = None
    # if val_loss_function == str(LossFunction.CrossEntropyLoss.value):
    #     loss_fn = CrossEntropyLoss(ignore_index=-1, reduction="mean")
    # elif val_loss_function == str(LossFunction.FocalLoss.value):
    #     loss_fn = FocalLoss(mode=MULTICLASS_MODE, ignore_index=-1, reduction="mean")
    # else:
    #     "Validation Loss will not be computed"

    # Build model
    if model_type == str(ModelType.Unet.value):
        model = UnetPtLightning.load_from_checkpoint(checkpoint_path=checkpoint_path, loss_fn=None)
    elif model_type == str(ModelType.SegFormerHF.value):
        model = HFSegFormerPtLightning.load_from_checkpoint(checkpoint_path=checkpoint_path)
        model.infer_resolution = inference_resolution
    else:
        raise NotImplementedError(f"{model} not implemented yet")
    # if loss_fn is not None:
    #     model.loss_fn = loss_fn

    # Setup experiment path and log linking
    os.makedirs(experiment_path, exist_ok=True)
    if log_file is not None:
        link_slurm_logs(log_file, os.path.join(experiment_path, INFERENCE_LOGS_FILE))
    out_dir = os.path.join(experiment_path, PREDICTIONS_DIR)

    # Initialize trainer
    trainer = Trainer(
        enable_checkpointing=False, devices=n_gpu if n_gpu > 0 else None,
        accelerator="gpu" if n_gpu > 0 else "cpu",
        callbacks=[SavePredictedMasks(save_dir=out_dir, CMAP=CMAP)]
    )

    # Start training
    print(f"Model built; Predicting for [{num_classes}] classes with [{n_gpu} GPUs]")
    print(model)
    trainer.predict(model=model, dataloaders=dm.val_dataloader())
    print("Done")


if __name__ == "__main__":
    fire.Fire(main)
