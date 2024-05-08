# Author: Matej Sirovatka

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int):
        """Upsample block, upsamples the output 4x

        Args:
            in_channels (int): Input channels
        """
        super(UpsampleBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upconv1 = nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1)
        self.conv_hr = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(in_channels, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        out = F.relu(self.pixel_shuffle(self.upconv1(x)))
        out = F.relu(self.pixel_shuffle(self.upconv2(out)))
        out = F.relu(self.conv_hr(out))
        return self.conv_last(out)
