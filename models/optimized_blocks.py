"""
StyleForge - Optimized Convolutional Blocks

Convolutional blocks with fused Instance Normalization.
"""

import torch
import torch.nn as nn

from ..kernels import FusedInstanceNorm2d


class OptimizedConvBlock(nn.Module):
    """
    Conv2d + Fused InstanceNorm + ReLU

    Uses custom CUDA kernel for Instance Normalization.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Stride for convolution
        padding: Padding for convolution

    Example:
        >>> block = OptimizedConvBlock(3, 32, kernel_size=9, stride=1, padding=4)
        >>> x = torch.randn(1, 3, 512, 512).cuda()
        >>> y = block(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = FusedInstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class OptimizedDeconvBlock(nn.Module):
    """
    ConvTranspose2d + Fused InstanceNorm + ReLU

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Stride for transposed convolution
        padding: Padding for transposed convolution
        output_padding: Additional output padding

    Example:
        >>> block = OptimizedDeconvBlock(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        >>> x = torch.randn(1, 128, 128, 128).cuda()
        >>> y = block(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int
    ):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.norm = FusedInstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class OptimizedResidualBlock(nn.Module):
    """
    Residual block with fused Instance Normalization.

    Args:
        channels: Number of channels
    """

    def __init__(self, channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = FusedInstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = FusedInstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        return x + residual
