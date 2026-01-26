"""
StyleForge - Fused Attention V5 Python Wrapper

V5 is the PERFORMANCE-FIXED version with proper grid configuration:
- One block per head (NOT per query position)
- 16 blocks instead of 8,192 blocks for seq_len=512
- Should be much faster than V4
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from utils import compile_inline

_attention_v5_module = None

def get_attention_v5_module():
    global _attention_v5_module

    if _attention_v5_module is not None:
        return _attention_v5_module

    kernel_path = Path(__file__).parent / "attention_v5.cu"

    if not kernel_path.exists():
        raise FileNotFoundError(f"V5 kernel not found at {kernel_path}")

    cuda_source = kernel_path.read_text()

    print("Compiling fused attention V5 kernel (PERFORMANCE FIXED version)...")
    _attention_v5_module = compile_inline(
        name='fused_attention_v5',
        cuda_source=cuda_source,
        functions=['fused_attention_v5'],
        build_directory=Path('build_v5'),
        verbose=False
    )
    print("V5 Compilation complete!")

    return _attention_v5_module


class FusedAttentionV5Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,  # [batch, num_heads, seq_len, head_dim]
        K: torch.Tensor,
        V: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        module = get_attention_v5_module()

        ctx.save_for_backward(Q, K, V)
        ctx.scale = scale

        output = module.fused_attention_v5(
            Q.contiguous(),
            K.contiguous(),
            V.contiguous(),
            scale
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # No autograd support
        return None, None, None, None


class FusedAttentionV5(nn.Module):
    """
    Fused Multi-Head Attention V5 (PERFORMANCE FIXED)

    V5 fixes the catastrophic grid configuration bug in V4:
    - V4: (batch, heads, seq_len) blocks = 8,192 launches for seq_len=512
    - V5: (batch, heads) blocks = 16 launches for seq_len=512

    Usage:
        >>> attn = FusedAttentionV5(embed_dim=128, num_heads=4)
        >>> x = torch.randn(1, 256, 128)  # [batch, seq, embed_dim]
        >>> output = attn(x)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        bias: bool = True
    ):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass: input -> output

        Args:
            x: [batch, seq_len, embed_dim]

        Returns:
            [batch, seq_len, embed_dim]
        """
        B, S, E = x.shape

        # Compute QKV
        qkv = self.qkv_proj(x)  # [B, S, 3*E]
        qkv = qkv.reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, S, head_dim]

        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Compute attention
        attn_output = self.forward_qkv(Q, K, V)  # [B, heads, S, head_dim]

        # Reshape and project
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [B, S, heads, head_dim]
        attn_output = attn_output.reshape(B, S, E)  # [B, S, E]
        output = self.out_proj(attn_output)

        return output

    def forward_qkv(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-computed Q, K, V tensors

        Args:
            Q: [batch, num_heads, seq_len, head_dim]
            K: [batch, num_heads, seq_len, head_dim]
            V: [batch, num_heads, seq_len, head_dim]

        Returns:
            [batch, num_heads, seq_len, head_dim]
        """
        return FusedAttentionV5Function.apply(Q, K, V, self.scale)


def benchmark_attention_v5_vs_pytorch(
    batch_size: int = 1,
    seq_len: int = 256,
    embed_dim: int = 128,
    num_heads: int = 4,
    iterations: int = 100
):
    """
    Benchmark V5 (performance fixed) against PyTorch
    """
    import numpy as np

    print(f"\nBenchmarking Attention V5 (B={batch_size}, S={seq_len}, E={embed_dim}, H={num_heads})...")
    print("=" * 70)

    x = torch.randn(batch_size, seq_len, embed_dim, device='cuda')

    # PyTorch baseline
    pytorch_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda().eval()

    times = []
    for _ in range(10):
        with torch.no_grad():
            _ = pytorch_attn(x, x, x)
    torch.cuda.synchronize()

    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = pytorch_attn(x, x, x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    pytorch_mean = np.mean(times)
    pytorch_std = np.std(times)
    print(f"PyTorch:  {pytorch_mean:.3f} ± {pytorch_std:.3f} ms")

    # V5 Custom
    custom_attn = FusedAttentionV5(embed_dim, num_heads).cuda().eval()

    times = []
    for _ in range(10):
        with torch.no_grad():
            _ = custom_attn(x)
    torch.cuda.synchronize()

    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = custom_attn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    custom_mean = np.mean(times)
    custom_std = np.std(times)
    print(f"V5 Fixed:  {custom_mean:.3f} ± {custom_std:.3f} ms")

    speedup = pytorch_mean / custom_mean
    print(f"\nSpeedup: {speedup:.2f}x")

    if speedup >= 1.0:
        print(" ✓ V5 is FASTER than PyTorch!")
    else:
        print(f" V5 is {1/speedup:.2f}x slower - needs more optimization")

    return {
        'pytorch_ms': pytorch_mean,
        'v5_ms': custom_mean,
        'speedup': speedup
    }


if __name__ == "__main__":
    benchmark_attention_v5_vs_pytorch()
