"""
StyleForge - Transformer Block Implementation

Standard transformer block using PyTorch nn.MultiheadAttention

Later we'll replace with custom CUDA kernels for:
- Fused QKV projection + attention + output projection
- Fused FFN (Linear + GELU + Linear)
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Standard transformer block using PyTorch nn.MultiheadAttention

    Args:
        embed_dim: Dimension of embeddings (default: 128)
        num_heads: Number of attention heads (default: 4)
        ffn_dim: Hidden dimension of feed-forward network (default: 512)
        dropout: Dropout probability (default: 0.0)

    Forward:
        x: (batch, seq_len, embed_dim)
        Returns: (batch, seq_len, embed_dim)
    """

    def __init__(self, embed_dim=128, num_heads=4, ffn_dim=512, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input: (batch, seq, embed)
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            (batch, seq_len, embed_dim)
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x, need_weights=False)
        x = residual + self.dropout(attn_output)

        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)

        return x
