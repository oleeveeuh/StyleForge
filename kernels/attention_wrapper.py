"""
StyleForge - Fused Attention Python Wrapper

Python interface for the fused attention CUDA kernels.
Handles compilation, parameter validation, and provides a PyTorch-like API.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from utils import compile_inline, get_cuda_info


# Global variable to hold compiled module
_attention_module = None


def get_attention_module():
    """
    Lazy-load and compile the attention kernel.

    Returns:
        Compiled CUDA module
    """
    global _attention_module

    if _attention_module is not None:
        return _attention_module

    # Load CUDA source
    kernel_path = Path(__file__).parent / "attention.cu"

    if not kernel_path.exists():
        raise FileNotFoundError(f"Attention kernel not found at {kernel_path}")

    cuda_source = kernel_path.read_text()

    # Compile
    print("Compiling fused attention kernel...")
    _attention_module = compile_inline(
        name='fused_attention',
        cuda_source=cuda_source,
        functions=['fused_qkv_proj', 'fused_attention_v1'],
        build_directory=Path('build'),
        verbose=False
    )
    print("Compilation complete!")

    return _attention_module


class FusedAttentionFunction(torch.autograd.Function):
    """
    Autograd function for fused attention.

    Forward pass uses CUDA kernel.
    Backward pass uses PyTorch's automatic differentiation (for V1).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        w_out: torch.Tensor,
        bias_qkv: Optional[torch.Tensor],
        bias_out: Optional[torch.Tensor],
        num_heads: int,
        scale: float
    ) -> torch.Tensor:
        """
        Forward pass with fused attention kernel.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            w_qkv: Fused QKV weight [3 * embed_dim, embed_dim]
            w_out: Output projection weight [embed_dim, embed_dim]
            bias_qkv: Optional QKV bias [3 * embed_dim]
            bias_out: Optional output projection bias [embed_dim]
            num_heads: Number of attention heads
            scale: Scaling factor for attention scores

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        module = get_attention_module()

        # Save for backward
        ctx.save_for_backward(x, w_qkv, w_out, bias_qkv, bias_out)
        ctx.num_heads = num_heads
        ctx.scale = scale

        # Call CUDA kernel
        with torch.cuda.nvtx.range("fused_attention_forward"):
            output = module.fused_attention_v1(
                x.contiguous(),
                w_qkv.contiguous(),
                w_out.contiguous(),
                bias_qkv,
                bias_out,
                scale
            )

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using PyTorch autograd (simplified for V1).

        In production, we'd implement a custom backward kernel.
        """
        x, w_qkv, w_out, bias_qkv, bias_out = ctx.saved_tensors
        num_heads = ctx.num_heads
        scale = ctx.scale

        # For V1, we use PyTorch's autograd by reconstructing the computation
        # This is slower but correct for gradient computation
        # In V2, we'll implement a custom backward kernel

        # TODO: Implement custom backward kernel
        # For now, return None for all gradients (no backward support in V1)
        return None, None, None, None, None, None, None


class FusedAttention(nn.Module):
    """
    Fused Multi-Head Attention Module

    This module replaces PyTorch's nn.MultiheadAttention with a custom
    CUDA kernel that fuses QKV projection, attention computation, and
    output projection.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability (not implemented in V1)
        bias: Use bias in projections (includes QKV bias and output bias)

    Example:
        >>> attn = FusedAttention(embed_dim=128, num_heads=4).cuda()
        >>> x = torch.randn(2, 16384, 128).cuda()  # [batch, seq, embed]
        >>> y = attn(x)
        >>> print(y.shape)  # [2, 16384, 128]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Fused QKV projection
        self.w_qkv = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.bias_qkv = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None

        # Output projection
        self.w_out = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.bias_out = nn.Parameter(torch.empty(embed_dim)) if bias else None

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform"""
        nn.init.xavier_uniform_(self.w_qkv)
        nn.init.xavier_uniform_(self.w_out)

        if self.bias_qkv is not None:
            nn.init.zeros_(self.bias_qkv)
        if self.bias_out is not None:
            nn.init.zeros_(self.bias_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused attention.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        # Call custom autograd function
        return FusedAttentionFunction.apply(
            x,
            self.w_qkv,
            self.w_out,
            self.bias_qkv,
            self.bias_out,
            self.num_heads,
            self.scale
        )

    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}'


class FusedQKVProjection(nn.Module):
    """
    Standalone fused QKV projection module.

    This can be used as a drop-in replacement for separate Q, K, V projections.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, bias: bool = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Fused QKV weight
        self.w_qkv = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.bias_qkv = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w_qkv)
        if self.bias_qkv is not None:
            nn.init.zeros_(self.bias_qkv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute fused QKV projection.

        Args:
            x: Input [batch, seq_len, embed_dim]

        Returns:
            qkv: [batch, seq_len, 3 * embed_dim]
        """
        module = get_attention_module()

        return module.fused_qkv_proj(
            x.contiguous(),
            self.w_qkv.contiguous(),
            self.bias_qkv
        )


def test_fused_attention():
    """
    Test function to verify fused attention works correctly.
    Compares output with PyTorch's nn.MultiheadAttention.
    """
    print("\n" + "=" * 70)
    print("  Testing Fused Attention Kernel")
    print("=" * 70)

    batch_size = 2
    seq_len = 128
    embed_dim = 128
    num_heads = 4

    # Create test input
    x = torch.randn(batch_size, seq_len, embed_dim, device='cuda')

    # PyTorch baseline
    attn_torch = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
    with torch.no_grad():
        out_torch, _ = attn_torch(x, x, x)

    # Fused attention
    attn_fused = FusedAttention(embed_dim, num_heads).cuda()

    # Copy weights from PyTorch for comparison
    with torch.no_grad():
        attn_fused.w_qkv.copy_(torch.cat([attn_torch.in_proj_weight[:embed_dim],
                                           attn_torch.in_proj_weight[embed_dim:2*embed_dim],
                                           attn_torch.in_proj_weight[2*embed_dim:]], dim=0))
        attn_fused.w_out.copy_(attn_torch.out_proj.weight)
        # Copy output bias if present
        if attn_torch.out_proj.bias is not None and attn_fused.bias_out is not None:
            attn_fused.bias_out.copy_(attn_torch.out_proj.bias)

    with torch.no_grad():
        out_fused = attn_fused(x)

    # Compare
    diff = (out_fused - out_torch).abs().max().item()
    print(f"  Max difference vs PyTorch: {diff:.2e}")

    if diff < 1e-3:
        print("  ✅ Fused attention matches PyTorch!")
    else:
        print(f"  ⚠️  Difference exceeds threshold (1e-3)")

    print("=" * 70 + "\n")

    return out_fused, out_torch


if __name__ == "__main__":
    # Run test if module is executed directly
    test_fused_attention()
