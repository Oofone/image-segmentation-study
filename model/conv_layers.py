from typing import Union, Tuple
from torch import nn


# Custom Convolution  
class Conv2d(nn.Module):

    def __init__(self, c_in: int, c_out: int, kernel_size: int,
                 stride: int, padding: Union[int, Tuple[int]],
                 depthwise_pointwise: bool = False) -> None:
        super().__init__()

        if depthwise_pointwise:
            self.conv_operation = nn.Sequential(
                # Depthwise
                nn.Conv2d(
                    in_channels=c_in,
                    out_channels=c_in,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=c_in),
                # Pointwise
                nn.Conv2d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=1,
                    stride=1,
                    padding=0)
            )
        else:
            self.conv_operation = nn.Conv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)

    def forward(self, feature_map):
        return self.conv_operation(feature_map)
