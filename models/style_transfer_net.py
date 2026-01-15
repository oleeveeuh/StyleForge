"""
StyleForge - Fast Neural Style Transfer Network

Architecture:
    Input (3, 512, 512)
        ↓
    Encoder: 3 conv layers → (128, 128, 128)
        ↓
    Transformer: 5 blocks → (128, 128, 128)
        ↓
    Decoder: 3 deconv layers → (3, 512, 512)

Total Parameters: ~1.6M
FLOPs per forward: ~12 GFLOPs
"""

import torch
import torch.nn as nn

from .conv_blocks import ConvBlock, DeconvBlock
from .transformer import TransformerBlock


class StyleTransferNetwork(nn.Module):
    """
    Fast Neural Style Transfer Network

    Args:
        use_custom_cuda: Whether to use custom CUDA kernels (default: False)
        num_transformer_blocks: Number of transformer blocks (default: 5)
        in_channels: Input image channels (default: 3)
        embed_dim: Transformer embedding dimension (default: 128)

    Forward:
        x: (B, 3, 512, 512) - Input image normalized to [-1, 1]
        Returns: (B, 3, 512, 512) - Styled output in [-1, 1]
    """

    def __init__(
        self,
        use_custom_cuda=False,
        num_transformer_blocks=5,
        in_channels=3,
        embed_dim=128
    ):
        super().__init__()
        self.use_custom_cuda = use_custom_cuda
        self.embed_dim = embed_dim

        # Encoder: Downsample 512→256→128
        self.encoder = nn.ModuleList([
            ConvBlock(in_channels, 32, kernel_size=9, stride=1, padding=4),      # 512×512
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),              # 256×256
            ConvBlock(64, embed_dim, kernel_size=3, stride=2, padding=1),       # 128×128
        ])

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=4, ffn_dim=512)
            for _ in range(num_transformer_blocks)
        ])

        # Decoder: Upsample 128→256→512
        self.decoder = nn.ModuleList([
            DeconvBlock(embed_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256×256
            DeconvBlock(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),         # 512×512
            nn.Conv2d(32, in_channels, kernel_size=9, stride=1, padding=4),
        ])

        self.final_activation = nn.Tanh()  # Output in [-1, 1]

    def forward(self, x):
        """
        Args:
            x: (B, 3, 512, 512) - Input image normalized to [-1, 1]
        Returns:
            (B, 3, 512, 512) - Styled output in [-1, 1]
        """
        # Encode
        for layer in self.encoder:
            x = layer(x)

        # Reshape for transformer: (B, C, H, W) → (B, H*W, C)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Reshape back: (B, H*W, C) → (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Decode
        for layer in self.decoder:
            x = layer(x)

        # Final activation
        x = self.final_activation(x)

        return x

    def get_model_size(self):
        """Return model size in MB (FP32)"""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / 1e6

    def get_parameter_count(self):
        """Return total and trainable parameter counts"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
