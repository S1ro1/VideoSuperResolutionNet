# Author: Matej Sirovatka

from typing import Callable
import torch.nn as nn


def make_conv_relu(
    in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, num_scales: int = 3
) -> nn.Sequential:
    """Creates a sequence of convolutional layers followed by ReLU activation

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int, optional): Kernel size. Defaults to 3.
        stride (int, optional): Stride. Defaults to 1.
        padding (int, optional): Padding. Defaults to 1.
        num_scales (int, optional): Number of blocks. Defaults to 3.

    Returns:
        nn.Sequential: Sequence of convolutional layers followed by ReLU activation
    """
    convs = []
    for _ in range(num_scales):
        convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        convs.append(nn.ReLU(inplace=True))
    return nn.Sequential(*convs)


def make_layer(layer: Callable, num_layers: int, *args, **kwargs) -> nn.Sequential:
    """Creates a sequence of layers

    Args:
        layer (Callable): Layer function
        num_layers (int): Number of layers

    Returns:
        nn.Sequential: Sequence of layers
    """
    layers = []
    for _ in range(num_layers):
        layers.append(layer(*args, **kwargs))
    return nn.Sequential(*layers)
