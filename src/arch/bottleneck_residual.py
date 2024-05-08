# Author: Matej Sirovatka

import torch.nn as nn
import torch.nn.functional as F


class BottleneckResidualBlock(nn.Module):
    def __init__(self, num_features):
        """Bottleneck residual block, ---1x1 conv---3x3 conv---1x1 conv---+--->
                                       |__________________________________|
        Args:
            num_features (_type_): Number of input features
        """
        super(BottleneckResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1)

    def forward(self, x):
        i = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return F.relu(out + i)
