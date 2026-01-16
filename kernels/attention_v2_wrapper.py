"""
StyleForge - Optimized Attention Kernel V2 Wrapper

Python interface for the V2 optimized attention kernel with:
- Shared memory tiling
- Vectorized memory access
- Warp-level reductions
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from utils import compile_inline

# Global module cache
_attention_v2_module = None


def get_attention_v2_module():
    """Lazy-load and compile the V2 attention kernel."""
    global _attention_v2_module

    if _attention_v2_module is not None:
        return _attention_v2_module

    kernel_path = Path(__file__).parent / "attention_v2.cu"

    if not kernel_path.exists():
        raise FileNotFoundError(f"V2 kernel not found at {kernel_path}")

    cuda_source = kernel_path.read_text()

    print("Compiling optimized attention kernel V2...")
    _attention_v2_module = compile_inline(
        name='attention_v2',
        cuda_source=cuda_source,
        functions=['forward'],
        build_directory=Path('build'),
        verbose=False
    )
    print("V2 compilation complete!")

    return _attention_v2_module


class FusedAttentionV2(nn.Module):
    """
    Optimized Fused Multi-Head Attention (V2)

    Improvements over V1:
    - Shared memory tiling for QKV projection
    - Vectorized memory access (float4)
    - Warp-level reductions for softmax
    - Fused output projection

    Target: 15-20x speedup over PyTorch baseline

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        bias: Use bias in projections
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        bias: bool = True
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Fused QKV projection [3 * embed_dim, embed_dim]
        self.w_qkv = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.bias_qkv = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None

        # Output projection [embed_dim, embed_dim]
        self.w_out = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.bias_out = nn.Parameter(torch.empty(embed_dim)) if bias else None

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
        Forward pass with optimized fused attention.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        module = get_attention_v2_module()

        # Transpose weights for kernel layout (our kernel expects [out, in])
        w_qkv_t = self.w_qkv.T.contiguous()
        w_out_t = self.w_out.T.contiguous()

        with torch.cuda.nvtx.range("attention_v2_forward"):
            output = module.forward(
                x.contiguous(),
                w_qkv_t,
                self.bias_qkv if self.bias_qkv is not None else torch.zeros(
                    3 * self.embed_dim, device=x.device
                ),
                w_out_t,
                self.bias_out if self.bias_out is not None else torch.zeros(
                    self.embed_dim, device=x.device
                )
            )

        return output

    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}'


def benchmark_v2_vs_others(
    batch_size: int = 2,
    seq_len: int = 256,
    embed_dim: int = 128,
    num_heads: int = 4,
    iterations: int = 100
):
    """
    Benchmark V2 against PyTorch and V1.

    Returns:
        Dictionary with benchmark results
    """
    import torch.nn.functional as F
    import numpy as np

    print(f"\nBenchmarking Attention Kernels ({batch_size}x{seq_len}x{embed_dim})...")
    print("=" * 70)

    # Create test input
    x = torch.randn(batch_size, seq_len, embed_dim, device='cuda')

    results = {}

    # ----------------------------------------
    # PyTorch Baseline
    # ----------------------------------------
    print("\n1. PyTorch MultiheadAttention...")

    attn_pytorch = nn.MultiheadAttention(
        embed_dim, num_heads, batch_first=True
    ).cuda().eval()

    times = []
    for _ in range(10):  # Warmup
        with torch.no_grad():
            _ = attn_pytorch(x, x, x)[0]

    torch.cuda.synchronize()
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = attn_pytorch(x, x, x)[0]
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    results['pytorch'] = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'name': 'PyTorch'
    }
    print(f"   {results['pytorch']['mean_ms']:.2f} ± {results['pytorch']['std_ms']:.2f} ms")

    # ----------------------------------------
    # V1 Kernel
    # ----------------------------------------
    print("\n2. Fused Attention V1...")

    try:
        from .attention_wrapper import FusedAttention

        attn_v1 = FusedAttention(embed_dim, num_heads).cuda().eval()

        times = []
        for _ in range(10):
            with torch.no_grad():
                _ = attn_v1(x)

        torch.cuda.synchronize()
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                _ = attn_v1(x)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        results['v1'] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'name': 'V1 Fused'
        }
        print(f"   {results['v1']['mean_ms']:.2f} ± {results['v1']['std_ms']:.2f} ms")
    except Exception as e:
        print(f"   Skipped: {e}")
        results['v1'] = None

    # ----------------------------------------
    # V2 Kernel
    # ----------------------------------------
    print("\n3. Fused Attention V2 (Optimized)...")

    attn_v2 = FusedAttentionV2(embed_dim, num_heads).cuda().eval()

    times = []
    for _ in range(10):
        with torch.no_grad():
            _ = attn_v2(x)

    torch.cuda.synchronize()
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = attn_v2(x)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    results['v2'] = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'name': 'V2 Optimized'
    }
    print(f"   {results['v2']['mean_ms']:.2f} ± {results['v2']['std_ms']:.2f} ms")

    # ----------------------------------------
    # Summary
    # ----------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline = results['pytorch']['mean_ms']
    v2_time = results['v2']['mean_ms']

    print(f"\nPyTorch:  {baseline:.2f} ms")
    if results['v1']:
        print(f"V1:       {results['v1']['mean_ms']:.2f} ms  ({baseline/results['v1']['mean_ms']:.2f}x)")
    print(f"V2:       {v2_time:.2f} ms  ({baseline/v2_time:.2f}x)")

    print(f"\n🚀 V2 is {baseline/v2_time:.2f}x faster than PyTorch!")

    return results


if __name__ == "__main__":
    # Run benchmark if executed directly
    benchmark_v2_vs_others()
