# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#
# Downsampling and upsampling blocks for the U-Net encoder and decoder paths.
# DownSample halves spatial resolution via strided convolution.
# UpSample doubles spatial resolution via nearest-neighbour interpolation followed by a conv.

import torch.nn as nn
from torch.nn import functional as F

"""Downsampling block."""
class DownSample(nn.Module):
    def __init__(self, channels, out_channels = None):
        super().__init__()

        # Store the input and output channel sizes
        self.channels = channels
        self.out_channels = out_channels or channels

        # Strided convolution reduces the spatial resolution by a factor of 2
        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        # Check that the input has the expected number of channels
        assert x.shape[1] == self.channels
        return self.conv(x)

"""Upsampling block."""
class UpSample(nn.Module):
    def __init__(self, channels, out_channels = None):
        super().__init__()

        # Store the input and output channel sizes
        self.channels = channels
        self.out_channels = out_channels or channels

        # Convolution is applied after upsampling to refine the features
        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Check that the input has the expected number of channels
        assert x.shape[1] == self.channels

        # Increase the spatial resolution by a factor of 2
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)