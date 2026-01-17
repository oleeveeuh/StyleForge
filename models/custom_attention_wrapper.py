"""
Custom MultiheadAttention Wrapper for StyleForge CUDA Kernels

This module provides a PyTorch-compatible wrapper around the fused_attention_v1
CUDA kernel, enabling drop-in replacement for nn.MultiheadAttention.

Usage:
    from models.custom_attention_wrapper import CustomMultiheadAttention

    # Replace nn.MultiheadAttention with CustomMultiheadAttention
    attn = CustomMultiheadAttention(embed_dim=512, num_heads=8)
    output = attn(input_tensor)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# Try to import the CUDA kernels
CUDA_ATTENTION_AVAILABLE = False
CUDA_ATTENTION_V3_AVAILABLE = False
FusedAttentionFunction = None
FusedAttentionV3Function = None

if torch.cuda.is_available():
    # Try V3 first (register-based, minimal shared memory)
    try:
        from kernels.attention_v3_wrapper import FusedAttentionV3Function
        CUDA_ATTENTION_V3_AVAILABLE = True
    except (ImportError, RuntimeError):
        CUDA_ATTENTION_V3_AVAILABLE = False
        FusedAttentionV3Function = None

    # V1 as fallback
    try:
        from kernels.attention_wrapper import FusedAttentionFunction
        CUDA_ATTENTION_AVAILABLE = True
    except (ImportError, RuntimeError):
        CUDA_ATTENTION_AVAILABLE = False
        FusedAttentionFunction = None

# NOTE: These kernels are EDUCATIONAL. They demonstrate CUDA programming
# (kernel fusion, warp reductions, register-based accumulation) but cannot
# compete with Flash Attention 2 which uses tensor cores.
#
# Fundamental limitation: Computing QKV with element-wise operations is
# much slower than tensor-core GEMM. V3 improves on V1/V2 by using register
# accumulation instead of shared memory, but the O(seq_len² × embed_dim)
# complexity remains.
#
# For production: Use torch.nn.functional.scaled_dot_product_attention.


class CustomMultiheadAttention(nn.Module):
    """
    PyTorch-compatible MultiheadAttention wrapper.

    EDUCATIONAL: Demonstrates CUDA kernel design (fusion, warp reductions).

    V3 KERNEL (Default): Register-based accumulation
    - Minimal shared memory (only Q vector: ~256 bytes for head_dim=64)
    - Online softmax algorithm
    - No shared memory for V accumulation

    STILL SLOWER THAN FLASH ATTENTION 2 because:
    - Element-wise matmul vs tensor cores
    - O(seq_len² × embed_dim) vs O(seq_len²) with better constants

    For production: Use torch.nn.functional.scaled_dot_product_attention

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        bias: If True, adds learnable bias to QKV and output projections
        dropout: Dropout probability (not supported by CUDA kernel)
        use_cuda_kernel: If True, prefer V3 CUDA kernel when available
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
        use_cuda_kernel: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_cuda_kernel = use_cuda_kernel and CUDA_ATTENTION_AVAILABLE

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # QKV projection weight [3 * embed_dim, embed_dim]
        self.w_qkv = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.bias_qkv = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None

        # Output projection weight [embed_dim, embed_dim]
        self.w_out = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.bias_out = nn.Parameter(torch.empty(embed_dim)) if bias else None

        # Scaling factor
        self.scale = (self.head_dim) ** -0.5

        # Initialize parameters
        self._reset_parameters()

        # Always create PyTorch fallback for cases where CUDA kernel fails (e.g., shared memory)
        self._pytorch_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
            batch_first=True
        )

        # Statistics tracking
        self._kernel_call_count = 0
        self._pytorch_call_count = 0

    def _reset_parameters(self) -> None:
        """Initialize parameters following PyTorch's MultiheadAttention."""
        nn.init.xavier_uniform_(self.w_qkv)
        nn.init.xavier_uniform_(self.w_out)
        if self.bias_qkv is not None:
            nn.init.zeros_(self.bias_qkv)
        if self.bias_out is not None:
            nn.init.zeros_(self.bias_out)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with CUDA kernel or PyTorch fallback.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            key_padding_mask: Not supported by CUDA kernel
            need_weights: If True, returns attention weights
            attn_mask: Not supported by CUDA kernel
            average_attn_weights: For PyTorch fallback

        Returns:
            output: [batch, seq_len, embed_dim]
            attention_weights: Returned if need_weights=True, else None
        """
        batch_size, seq_len, embed_dim = x.shape

        # Check for unsupported features with CUDA kernel
        max_seq = FusedAttentionV3Function.MAX_SEQ_LEN if CUDA_ATTENTION_V3_AVAILABLE else (
            FusedAttentionFunction.MAX_SEQ_LEN if CUDA_ATTENTION_AVAILABLE else 0)

        use_cuda = (
            self.use_cuda_kernel and
            x.is_cuda and
            key_padding_mask is None and
            attn_mask is None and
            not need_weights and
            seq_len <= max_seq
        )

        if use_cuda:
            return self._forward_cuda(x)
        else:
            return self._forward_pytorch(x, key_padding_mask, need_weights, attn_mask, average_attn_weights)

    def _forward_cuda(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Forward pass using CUDA V3 kernel (register-based, educational)."""
        try:
            self._kernel_call_count += 1

            # Prefer V3 (register-based, minimal shared memory)
            if CUDA_ATTENTION_V3_AVAILABLE:
                output = FusedAttentionV3Function.apply(
                    x,
                    self.w_qkv,
                    self.w_out,
                    self.bias_qkv,
                    self.bias_out,
                    self.num_heads,
                    self.scale
                )
            else:
                # Fall back to V1
                output = FusedAttentionFunction.apply(
                    x,
                    self.w_qkv,
                    self.w_out,
                    self.bias_qkv,
                    self.bias_out,
                    self.num_heads,
                    self.scale
                )

            return output, None
        except Exception as e:
            # Fallback to PyTorch on any error
            self._kernel_call_count -= 1
            self._pytorch_call_count += 1
            return self._forward_pytorch(x, None, False, None, True)

    def _forward_pytorch(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        need_weights: bool,
        attn_mask: Optional[torch.Tensor],
        average_attn_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using PyTorch nn.MultiheadAttention."""
        self._pytorch_call_count += 1

        # Reshape weights to match PyTorch's MultiheadAttention format
        # PyTorch expects: [embed_dim, embed_dim] for each of q, k, v, out
        in_proj_weight = self.w_qkv
        in_proj_bias = self.bias_qkv
        out_proj_weight = self.w_out
        out_proj_bias = self.bias_out

        # Create a temporary MHA module with our weights
        # This is a bit inefficient but ensures compatibility
        with torch.no_grad():
            self._pytorch_attn.in_proj_weight.copy_(in_proj_weight)
            self._pytorch_attn.in_proj_bias.copy_(in_proj_bias)
            self._pytorch_attn.out_proj.weight.copy_(out_proj_weight.T)
            self._pytorch_attn.out_proj.bias.copy_(out_proj_bias)

        output, attn_weights = self._pytorch_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights
        )

        return output, attn_weights

    def get_stats(self) -> dict:
        """Get statistics about kernel usage."""
        total = self._kernel_call_count + self._pytorch_call_count
        cuda_pct = (self._kernel_call_count / total * 100) if total > 0 else 0
        return {
            "cuda_kernel_calls": self._kernel_call_count,
            "pytorch_fallback_calls": self._pytorch_call_count,
            "cuda_percentage": cuda_pct,
            "total_calls": total
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._kernel_call_count = 0
        self._pytorch_call_count = 0


class FusedFFNWrapper(nn.Module):
    """
    Wrapper for the fused FFN CUDA kernel.

    Args:
        embed_dim: Input dimension
        hidden_dim: Hidden layer dimension (typically 4x embed_dim)
        bias: If True, adds bias to both linear layers
        use_cuda_kernel: If True, prefer CUDA kernel when available
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        bias: bool = True,
        use_cuda_kernel: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_cuda_kernel = use_cuda_kernel

        # Try to import CUDA kernel
        self.cuda_available = False
        if use_cuda_kernel and torch.cuda.is_available():
            try:
                from kernels.ffn_wrapper import FusedFFNFunction
                self.FusedFFNFunction = FusedFFNFunction
                self.cuda_available = True
            except (ImportError, RuntimeError):
                self.cuda_available = False

        # Parameters for PyTorch fallback or CUDA kernel
        self.fc1_weight = nn.Parameter(torch.empty(hidden_dim, embed_dim))
        self.fc1_bias = nn.Parameter(torch.empty(hidden_dim)) if bias else None
        self.fc2_weight = nn.Parameter(torch.empty(embed_dim, hidden_dim))
        self.fc2_bias = nn.Parameter(torch.empty(embed_dim)) if bias else None

        self._reset_parameters()

        self._kernel_call_count = 0
        self._pytorch_call_count = 0

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1_weight)
        nn.init.xavier_uniform_(self.fc2_weight)
        if self.fc1_bias is not None:
            nn.init.zeros_(self.fc1_bias)
        if self.fc2_bias is not None:
            nn.init.zeros_(self.fc2_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cuda_available and x.is_cuda:
            self._kernel_call_count += 1
            return self.FusedFFNFunction.apply(
                x,
                self.fc1_weight,
                self.fc1_bias,
                self.fc2_weight,
                self.fc2_bias
            )
        else:
            self._pytorch_call_count += 1
            # PyTorch fallback: Linear -> GELU -> Linear
            x = torch.nn.functional.linear(x, self.fc1_weight, self.fc1_bias)
            x = torch.nn.functional.gelu(x)
            x = torch.nn.functional.linear(x, self.fc2_weight, self.fc2_bias)
            return x

    def get_stats(self) -> dict:
        total = self._kernel_call_count + self._pytorch_call_count
        cuda_pct = (self._kernel_call_count / total * 100) if total > 0 else 0
        return {
            "cuda_kernel_calls": self._kernel_call_count,
            "pytorch_fallback_calls": self._pytorch_call_count,
            "cuda_percentage": cuda_pct,
            "total_calls": total
        }


def get_attention_kernel_stats(model: nn.Module) -> dict:
    """
    Collect statistics from all CustomMultiheadAttention modules in a model.

    Args:
        model: PyTorch model containing CustomMultiheadAttention modules

    Returns:
        Dictionary with aggregated statistics
    """
    total_kernel_calls = 0
    total_pytorch_calls = 0
    module_count = 0

    for name, module in model.named_modules():
        if isinstance(module, CustomMultiheadAttention):
            stats = module.get_stats()
            total_kernel_calls += stats["cuda_kernel_calls"]
            total_pytorch_calls += stats["pytorch_fallback_calls"]
            module_count += 1

    total = total_kernel_calls + total_pytorch_calls
    cuda_pct = (total_kernel_calls / total * 100) if total > 0 else 0

    return {
        "attention_modules": module_count,
        "cuda_kernel_calls": total_kernel_calls,
        "pytorch_fallback_calls": total_pytorch_calls,
        "cuda_percentage": cuda_pct,
        "total_calls": total
    }


def print_attention_stats(model: nn.Module) -> None:
    """Print attention kernel statistics for a model."""
    stats = get_attention_kernel_stats(model)

    print("\n" + "=" * 60)
    print("CUDA Attention Kernel Statistics")
    print("=" * 60)
    print(f"Attention modules: {stats['attention_modules']}")
    print(f"CUDA kernel calls:  {stats['cuda_kernel_calls']}")
    print(f"PyTorch fallbacks:   {stats['pytorch_fallback_calls']}")
    print(f"CUDA usage:          {stats['cuda_percentage']:.1f}%")
    print("=" * 60)
