"""
Fast Neural Style Transfer Transformer Network

Based on Johnson et al. "Perceptual Losses for Real-Time Style Transfer"
https://arxiv.org/abs/1603.08155

Architecture:
    Input (3, H, W)
        ↓
    Encoder: 3 conv layers with InstanceNorm
        ↓
    Residual blocks: 5-10 residual blocks
        ↓
    Decoder: 3 upsampling conv layers with InstanceNorm
        ↓
    Output (3, H, W)

This network is designed to be trained for each specific style.
Pre-trained weights can be loaded from the fast-neural-style repository.
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvLayer(nn.Module):
    """Convolution -> InstanceNorm -> ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        relu: bool = True,
        norm: bool = True,
    ):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        ]

        if norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))

        if relu:
            layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Residual block with two ConvLayers and skip connection"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=0)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=0, relu=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return residual + out


class UpsampleConvLayer(nn.Module):
    """Upsample (nearest neighbor) -> Conv -> InstanceNorm -> ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        upsample: int = 2,
    ):
        super().__init__()
        layers = []

        if upsample > 1:
            layers.append(nn.Upsample(scale_factor=upsample, mode='nearest'))

        layers.extend([
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerNet(nn.Module):
    """
    Fast Neural Style Transfer Network

    Args:
        num_residual_blocks: Number of residual blocks (default: 5)
            - Original paper uses 5 for faster training
            - Use 10 for better quality results
        checkpoint_path: Optional path to load pre-trained weights

    Example:
        >>> model = TransformerNet()
        >>> model.load_state_dict(torch.load('candy.pth'))
        >>> output = model(input_image)
    """

    def __init__(self, num_residual_blocks: int = 5):
        super().__init__()

        # Initial convolution layers (encoder)
        # Input: (3, H, W) -> (32, H, W)
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1, padding=4)
        # (32, H, W) -> (64, H/2, W/2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, padding=1)
        # (64, H/2, W/2) -> (128, H/4, W/4)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_residual_blocks)]
        )

        # Upsampling layers (decoder)
        # (128, H/4, W/4) -> (64, H/2, W/2)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, padding=1, upsample=2)
        # (64, H/2, W/2) -> (32, H, W)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, padding=1, upsample=2)
        # (32, H, W) -> (3, H, W)
        self.deconv3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, kernel_size=9, stride=1)
        )

    def forward(self, x):
        """
        Args:
            x: Input image tensor (B, 3, H, W) in range [-1, 1] or [0, 1]

        Returns:
            Stylized image (B, 3, H, W) in range [-1, 1]
        """
        # Encoder
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # Residual blocks
        out = self.residual_blocks(out)

        # Decoder
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)

        return out

    def load_checkpoint(self, checkpoint_path: str, device: Optional[torch.device] = None) -> None:
        """
        Load pre-trained weights from checkpoint file.

        Args:
            checkpoint_path: Path to .pth file containing state_dict
            device: Device to load weights onto (defaults to current device of model)

        Example:
            >>> model = TransformerNet()
            >>> model.load_checkpoint('models/pretrained/candy.pth')
        """
        if device is None:
            device = next(self.parameters()).device

        state_dict = torch.load(checkpoint_path, map_location=device)

        # Handle different state dict formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        self.load_state_dict(new_state_dict, strict=True)
        print(f"✅ Loaded checkpoint from {checkpoint_path}")

    def get_model_size(self) -> float:
        """Return model size in MB (FP32)"""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / 1e6

    def get_parameter_count(self) -> tuple[int, int]:
        """Return (total_params, trainable_params)"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# Pre-trained style names from fast-neural-style repository
AVAILABLE_STYLES = [
    "candy",      # Candy style
    "composition", # Composition VII style
    "la_muse",    # La Muse style
    "mosaic",     # Mosaic style
    "starry",     # Starry Night style
    "udnie",      # Udnie style
    "wave",       # The Great Wave off Kanagawa style
]


def get_style_url(style_name: str) -> str:
    """Get download URL for pre-trained style weights"""
    base_url = "https://github.com/jcjohnson/fast-neural-style/raw/master/models"
    return f"{base_url}/{style_name}.pth"
