from torch import from_numpy as tensor_from_numpy, Tensor
from torchvision import transforms
from typing import Tuple, Callable

import numpy as np


def get_default_image_transform(resolution: Tuple, normalize: bool = False) -> Callable[[Tensor], Tensor]:
    if normalize:
        return transforms.Compose([
            transforms.Resize( # Resize the image
                resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), # Convert PIL image to tensor
            transforms.Normalize(
                mean = [0.0131, 0.0143, 0.0147], # Normalizing with training mean and std
                std = [0.0081, 0.0084, 0.0093] # Refer scratch/dataset_analysis.py
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize( # Resize the image
                resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), # Convert PIL image to tensor
        ])


def get_default_mask_transform(resolution: Tuple):
    return transforms.Compose([
        transforms.Resize( # Resize the mask
            resolution, interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        transforms.Lambda( # Mask is not normalized since it's categorical labels
            lambda mask_im: tensor_from_numpy(np.array(mask_im)))
    ])
