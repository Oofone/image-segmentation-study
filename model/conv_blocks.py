from model.conv_layers import Conv2d

from typing import Union, Tuple
from torch import nn, cat as t_cat, Tensor


# Convolutional downsampling block 
# Channel c_in --down_sample--> c_out
class ConvDownsamplingBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int, kernel_size: int,
                 stride: int, padding: Union[int, Tuple[input]],
                 pooling_fn: Union[nn.Module, None] = nn.MaxPool2d,
                 normalize: nn.Module = nn.BatchNorm2d,
                 activation: nn.Module = nn.LeakyReLU,
                 depthwise_pointwise: bool = False,
                 downsampling_factor: int = 2,
                 **kwargs) -> None:
        super().__init__()

        self.conv_module = nn.Sequential(
            Conv2d(c_in=c_in, c_out=c_out, kernel_size=kernel_size, stride=stride,
                   padding=padding, depthwise_pointwise=depthwise_pointwise),
            normalize(c_out),
            activation(),
            Conv2d(c_in=c_out, c_out=c_out, kernel_size=kernel_size, stride=stride,
                   padding=padding, depthwise_pointwise=depthwise_pointwise),
            normalize(c_out),
            activation())

        if pooling_fn is not None:
            self.do_pool = True
            self.pool = pooling_fn(kernel_size=downsampling_factor, stride=downsampling_factor)
        else:
            self.do_pool = False

    def forward(self, X: Tensor) -> Tensor:
        X = self.conv_module(X)
        if self.do_pool:
            return X, self.pool(X)
        else:
            return X, None


# Convolutional upsampling block 
# Channel c_in --up_sample--> c_out
class ConvUpsamplingBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int, kernel_size: int,
                 stride: int, padding: Union[int, Tuple[input]],
                 pooling_fn: Union[nn.Module, None] = None,
                 normalize: nn.Module = nn.BatchNorm2d,
                 activation: nn.Module = nn.LeakyReLU,
                 depthwise_pointwise: bool = False,
                 upsampling_factor: int = 2,
                 **kwargs) -> None:
        super().__init__()

        self.u_conv = nn.ConvTranspose2d(
            in_channels=c_in, out_channels=c_out, kernel_size=upsampling_factor,
            stride=upsampling_factor)
        self.cat_conv = ConvDownsamplingBlock(
            c_in=c_in, c_out=c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, normalize=normalize, activation=activation,
            pooling_fn=pooling_fn, depthwise_pointwise=depthwise_pointwise,
            downsampling_factor=upsampling_factor)

    def forward(self, X: Tensor, skip: Tensor) -> Tensor:
        X = self.u_conv(X)
        X = t_cat((X, skip),dim=1)
        X, _ = self.cat_conv(X)
        return X
