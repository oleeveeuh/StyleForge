"""
StyleForge - Optimized Transformer Block

Transformer block using all custom CUDA kernels:
- Fused Attention V2
- Fused FFN
- Fused Instance Normalization
"""

import torch
import torch.nn as nn
from typing import Optional

from ..kernels import FusedAttentionV2, FusedFFN


class OptimizedTransformerBlock(nn.Module):
    """
    Transformer block using all custom CUDA kernels.

    Replaces PyTorch operations with fused kernels:
    - FusedAttentionV2 instead of nn.MultiheadAttention
    - FusedFFN instead of Sequential(Linear, GELU, Linear)
    - LayerNorm (can be replaced with FusedInstanceNorm)

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network hidden dimension
        dropout: Dropout probability
        use_cuda: Use CUDA kernels (default: True)

    Example:
        >>> block = OptimizedTransformerBlock(embed_dim=128, num_heads=4).cuda()
        >>> x = torch.randn(2, 256, 128).cuda()
        >>> y = block(x)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        use_cuda: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_cuda = use_cuda

        if use_cuda:
            # Fused Attention V2
            self.attention = FusedAttentionV2(embed_dim, num_heads)

            # Fused FFN
            self.ffn = FusedFFN(embed_dim, ffn_dim)
        else:
            # Fallback to PyTorch
            from .transformer import TransformerBlock
            self._pytorch_block = TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)

        # Layer normalization (still use PyTorch for now)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with CUDA kernels.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        if not self.use_cuda:
            return self._pytorch_block(x)

        # Attention block with residual
        residual = x
        x = self.norm1(x)

        attn_output = self.attention(x)
        x = residual + self.dropout(attn_output)

        # FFN block with residual
        residual = x
        x = self.norm2(x)

        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)

        return x

    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, ffn_dim={self.ffn_dim}, use_cuda={self.use_cuda}'
