from model.conv_blocks import ConvUpsamplingBlock

from typing import Union, Tuple, List
from torch import Tensor, nn


class UnetDecoder(nn.Module):

    def __init__(self, channels_list: List[int], upsampling_factor: int,
                 kernel_size: int, stride: int, padding: Union[int, Tuple[input]],
                 **kwargs) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList([
            ConvUpsamplingBlock(
                c_in=channels_list[i],
                c_out=channels_list[i+1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                upsampling_factor=upsampling_factor,
                **kwargs)
            for i in range(len(channels_list) - 1)
        ])

    def forward(self, X: Tensor, skips: List[Tensor]) -> Tensor:
        for skip, layer in zip(skips, self.layers):
            X = layer(X, skip)
        return X
