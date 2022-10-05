"""
SIMPLE RESIDUAL BLOCK
"""

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2))
        self.batchnorm_conv1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2))
        self.batchnorm_conv2 = nn.BatchNorm2d(channels)

    def forward(self, t):
        res_t = t
        t = self.conv1(t)
        t = self.batchnorm_conv1(t)
        t = F.relu(t)
        t = self.conv2(t)
        t = self.batchnorm_conv2(t)
        t += res_t
        t = F.relu(t)

        return t