"""
StyleForge - Convolutional Blocks

Basic building blocks for encoder/decoder in style transfer network.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Conv2d + InstanceNorm + ReLU

    Used in encoder for downsampling and feature extraction.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class DeconvBlock(nn.Module):
    """
    ConvTranspose2d + InstanceNorm + ReLU

    Used in decoder for upsampling back to original resolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutions.

    Used for deeper feature refinement in the encoder.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + residual
