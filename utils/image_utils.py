from typing import Optional, Tuple, List
from torch import Tensor
from PIL import Image

import numpy as np


def get_image_resolution(image_path: str) -> Tuple[str]:
    return Image.open(image_path).size


def open_image(image_path: str) -> np.array:
    return np.array(Image.open(image_path))


def open_image_as_PIL(image_path: str) -> Image:
    return Image.open(image_path)


def ar2im(image_arr: np.array, CMAP: Optional[List[int]]) -> Image.Image:
    img = Image.fromarray(image_arr)
    if CMAP is not None:
        img.putpalette(CMAP)
    return img


def t2im(image_tensor: Tensor, CMAP: Optional[List[int]]) -> Image.Image:
    return ar2im(image_arr=image_tensor.numpy(), CMAP=CMAP)
