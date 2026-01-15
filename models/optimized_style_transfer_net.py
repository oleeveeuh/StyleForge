"""
StyleForge - Optimized Style Transfer Network

Fully optimized StyleTransferNetwork using all custom CUDA kernels:

- FusedAttentionV2: ~15-20x faster attention
- FusedFFN: ~4-5x faster feed-forward
- FusedInstanceNorm2d: ~3-5x faster normalization

Target: 50-100x speedup over baseline PyTorch model
"""

import torch
import torch.nn as nn

from .optimized_blocks import OptimizedConvBlock, OptimizedDeconvBlock
from .optimized_transformer import OptimizedTransformerBlock
from .conv_blocks import DeconvBlock


class OptimizedStyleTransferNetwork(nn.Module):
    """
    Fully Optimized Style Transfer Network

    Uses custom CUDA kernels for all compute-intensive operations:

    Architecture:
        Input (3, 512, 512)
            ↓
        Encoder: 3 ConvBlocks with Fused InstanceNorm
            ↓
        Transformer: 5 OptimizedTransformerBlocks
            ↓
        Decoder: 3 DeconvBlocks
            ↓
        Output (3, 512, 512)

    Args:
        num_transformer_blocks: Number of transformer blocks (default: 5)
        embed_dim: Transformer embedding dimension (default: 128)
        num_heads: Number of attention heads (default: 4)
        ffn_dim: Feed-forward hidden dimension (default: 512)
        use_cuda: Use CUDA kernels (default: True)

    Example:
        >>> model = OptimizedStyleTransferNetwork().cuda()
        >>> x = torch.randn(1, 3, 512, 512).cuda()
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([1, 3, 512, 512])
    """

    def __init__(
        self,
        num_transformer_blocks: int = 5,
        embed_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 512,
        use_cuda: bool = True,
        in_channels: int = 3
    ):
        super().__init__()

        self.num_transformer_blocks = num_transformer_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_cuda = use_cuda

        # ============================================
        # Encoder: Downsample 512→256→128
        # ============================================
        if use_cuda:
            self.encoder = nn.ModuleList([
                OptimizedConvBlock(in_channels, 32, kernel_size=9, stride=1, padding=4),      # 512×512
                OptimizedConvBlock(32, 64, kernel_size=3, stride=2, padding=1),              # 256×256
                OptimizedConvBlock(64, embed_dim, kernel_size=3, stride=2, padding=1),       # 128×128
            ])
        else:
            from .conv_blocks import ConvBlock
            self.encoder = nn.ModuleList([
                ConvBlock(in_channels, 32, kernel_size=9, stride=1, padding=4),
                ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
                ConvBlock(64, embed_dim, kernel_size=3, stride=2, padding=1),
            ])

        # ============================================
        # Transformer Blocks
        # ============================================
        self.transformer_blocks = nn.ModuleList([
            OptimizedTransformerBlock(embed_dim, num_heads, ffn_dim, use_cuda=use_cuda)
            for _ in range(num_transformer_blocks)
        ])

        # ============================================
        # Decoder: Upsample 128→256→512
        # ============================================
        if use_cuda:
            self.decoder = nn.ModuleList([
                OptimizedDeconvBlock(embed_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256×256
                OptimizedDeconvBlock(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # 512×512
                nn.Conv2d(32, in_channels, kernel_size=9, stride=1, padding=4),
            ])
        else:
            self.decoder = nn.ModuleList([
                DeconvBlock(embed_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                DeconvBlock(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Conv2d(32, in_channels, kernel_size=9, stride=1, padding=4),
            ])

        self.final_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the optimized network.

        Args:
            x: Input image [batch, 3, 512, 512] normalized to [-1, 1]

        Returns:
            Styled output [batch, 3, 512, 512] in [-1, 1]
        """
        # Encoder
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

        # Decoder
        for layer in self.decoder:
            x = layer(x)

        # Final activation
        x = self.final_activation(x)

        return x

    def get_model_size(self) -> float:
        """Return model size in MB (FP32)"""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / 1e6

    def get_parameter_count(self) -> tuple[int, int]:
        """Return total and trainable parameter counts"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_optimized_model(use_cuda: bool = True, **kwargs) -> OptimizedStyleTransferNetwork:
    """
    Helper function to create an optimized model.

    Args:
        use_cuda: Use CUDA kernels
        **kwargs: Additional arguments for OptimizedStyleTransferNetwork

    Returns:
        OptimizedStyleTransferNetwork instance
    """
    model = OptimizedStyleTransferNetwork(use_cuda=use_cuda, **kwargs)
    return model
