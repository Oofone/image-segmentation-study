from constants import _LOSS_FUNCTION_PARAM, _MODEL_TYPE_PARAM, DEFAULT_LOGS_FILE
from train_im_seg_pt_lightning import main as trainable
from utils.log_utils import link_slurm_logs

from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT
from typing import Union, Optional, Dict, Any
from ray import tune

import fire
import os


# Search space

unet_search_space: Dict[Any, Any] = {
    "unet_kernel_size": tune.grid_search([3, 5]),
    "model_depth": tune.grid_search([4, 5]),
    "initial_learning_rate": tune.grid_search([1e-4, 1e-3]),
    "train_batch_size": tune.grid_search([8, 16])
}

unet_search_space2: Dict[Any, Any] = {
    "unet_kernel_size": tune.grid_search([3, 5]),
    "model_depth": tune.grid_search([4, 5]),
}

segformer_search_space: Dict[Any, Any] = {
    "initial_learning_rate": tune.grid_search([1e-4, 1e-3, 1e-2]),
    "train_batch_size": tune.grid_search([16, 32]),
    "loss_function": tune.grid_search(["CrossEntropyLoss", "FocalLoss"])
}

# Search/Train function

unet_padding_map = {
    3: 1,
    5: 2,
    7: 3,
}

def train_hook(
        config: Dict[str, Union[int, float]],
        dataset_root_path: str,
        experiment_root_dir: str,
        model_type: _MODEL_TYPE_PARAM,
        loss_function: _LOSS_FUNCTION_PARAM,
        max_epochs: int,
        n_cpu: int,
        n_gpu: int,
        precision: _PRECISION_INPUT) -> None:

    unet_padding = unet_padding_map[config['unet_kernel_size']]

    trainable(
        dataset_root_path=dataset_root_path,
        experiment_root_dir=experiment_root_dir,
        val_batch_size=32,
        model_type=model_type,
        loss_function=loss_function,
        unet_padding=unet_padding,
        n_cpu=n_cpu,
        n_gpu=n_gpu,
        precision=precision,
        max_epochs=max_epochs,
        hparam_tune=True,
        **config)

# Entry point

def main(
        dataset_root_path: str,
        experiment_root_dir: str,
        model_type: _MODEL_TYPE_PARAM,
        loss_function: _LOSS_FUNCTION_PARAM,
        max_epochs_per_trial: int,
        n_cpu_per_trial: int,
        n_gpu_per_trial: int,
        max_concurrent_trials: int,
        precision: _PRECISION_INPUT,
        log_file: Optional[str] = None,
        num_samples:int = 1) -> None:

    experiment_root_dir = os.path.abspath(experiment_root_dir)
    os.makedirs(experiment_root_dir, exist_ok=True)
    trainable_hook = tune.with_parameters(
        train_hook,
        dataset_root_path = dataset_root_path,
        experiment_root_dir = experiment_root_dir,
        model_type = model_type,
        loss_function = loss_function,
        max_epochs = max_epochs_per_trial,
        n_cpu = n_cpu_per_trial,
        n_gpu = n_gpu_per_trial,
        precision = precision)
    
    if log_file is not None:
        link_slurm_logs(log_file, os.path.join(experiment_root_dir, DEFAULT_LOGS_FILE))

    analysis = tune.run(
        trainable_hook,
        max_concurrent_trials=max_concurrent_trials,
        resources_per_trial={
            "cpu": n_cpu_per_trial,
            "gpu": n_gpu_per_trial
        },
        metric="mIoU",
        mode="max",
        num_samples=num_samples,
        config=unet_search_space2,
        name="tune_unet_kernel_mapillary_vistas",
        storage_path=experiment_root_dir)

    print("Completed Tuning")
    print(analysis.best_config)


if __name__ == "__main__":
    fire.Fire(main)
