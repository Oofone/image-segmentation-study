from model.conv.conv_blocks import ConvDownsamplingBlock

from typing import Union, Tuple, List
from torch import Tensor, nn


class UnetEncoder(nn.Module):

    def __init__(self, channels_list: List[int], downsampling_factor: int,
                 kernel_size: int, stride: int, padding: Union[int, Tuple[input]],
                 **kwargs) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            ConvDownsamplingBlock(
                c_in=channels_list[i],
                c_out=channels_list[i+1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                downsampling_factor=downsampling_factor,
                **kwargs)
            for i in range(len(channels_list) - 1)
        ])

    def forward(self, X: Tensor) -> Tuple[Tensor, List[Tensor]]:
        skips = []
        for layer in self.layers:
            skip, X = layer(X)
            skips.append(skip)
        return X, list(reversed(skips))
