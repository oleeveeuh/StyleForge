"""
Wrapper for custom attention kernel compatible with LLM workloads

Adapts your existing attention_v3 kernel to work with Llama-2 style inputs.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for kernel imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import your existing kernel
try:
    from kernels.attention_v3_wrapper import FusedAttentionV3
    CUSTOM_KERNEL_AVAILABLE = True
except ImportError:
    print("Warning: FusedAttentionV3 not found.")
    CUSTOM_KERNEL_AVAILABLE = False


class CustomMultiHeadAttention(nn.Module):
    """
    Custom multi-head attention using your optimized CUDA kernel

    Features:
    - Online softmax (no attention matrix materialization)
    - Register-based value accumulation
    - Warp-level parallel reductions
    - O(N) memory complexity

    Compatible with Llama-2 architecture.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        if CUSTOM_KERNEL_AVAILABLE:
            # Use the existing fused attention kernel
            self.attn = FusedAttentionV3(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
            )
        else:
            # Fallback: create standard components
            self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
            self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.attn = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.use_custom_kernel = CUSTOM_KERNEL_AVAILABLE

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass using custom CUDA kernel

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional mask (not supported in v1)
            return_attention_weights: If True, return weights (not supported in v1)

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        if self.use_custom_kernel and self.attn is not None:
            # Use custom kernel
            output = self.attn(hidden_states)
        else:
            # Fallback to PyTorch implementation
            output = self._pytorch_attention(hidden_states)

        if self.dropout is not None:
            output = self.dropout(output)

        return output

    def _pytorch_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fallback PyTorch implementation"""
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(hidden_states)  # [batch, seq, 3*hidden]

        # Reshape and split: [batch, seq, 3*hidden] -> 3 x [batch, heads, seq, head_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]

        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Standard scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # Reshape: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.out_proj(output)

        return output


def create_pytorch_baseline_attention(
    hidden_size: int,
    num_heads: int,
    bias: bool = False,
) -> nn.Module:
    """
    Create PyTorch baseline using nn.MultiheadAttention

    This is the standard PyTorch implementation for comparison.
    """
    return nn.MultiheadAttention(
        embed_dim=hidden_size,
        num_heads=num_heads,
        dropout=0.0,
        bias=bias,
        batch_first=True,
    )
